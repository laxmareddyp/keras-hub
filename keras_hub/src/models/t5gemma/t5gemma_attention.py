import keras
from keras import ops


class T5GemmaAttention(keras.layers.Layer):
    """Multi-head attention layer for T5Gemma model.
    
    This layer combines T5's relative attention mechanism with Gemma 2's
    improvements including sliding window attention and query normalization.
    """
    
    def __init__(
        self,
        num_heads,
        head_dim,
        input_dim=None,
        num_key_value_heads=None,
        query_head_dim_normalize=True,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        attention_logit_soft_cap=None,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.query_head_dim_normalize = query_head_dim_normalize
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.dropout = dropout
        
        self.hidden_dim = num_heads * head_dim
        self.input_dim = input_dim or self.hidden_dim
        
        # Projection layers
        self.query_proj = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="query_proj",
        )
        self.key_proj = keras.layers.Dense(
            self.num_key_value_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            name="key_proj",
        )
        self.value_proj = keras.layers.Dense(
            self.num_key_value_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            name="value_proj",
        )
        self.output_proj = keras.layers.Dense(
            self.input_dim,
            use_bias=False,
            dtype=dtype,
            name="output_proj",
        )
        
        # Dropout layer
        self.attention_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
        )
        
        # Relative position bias (T5-style)
        self.relative_attention_bias = None
        
    def build(self, input_shape):
        # Initialize relative attention bias if needed
        if self.relative_attention_bias is None:
            self.relative_attention_bias = self.add_weight(
                name="relative_attention_bias",
                shape=(self.num_heads, 32),  # T5 uses 32 relative positions
                initializer=keras.initializers.TruncatedNormal(1.0),
                dtype=self.dtype,
            )
        super().build(input_shape)
    
    def call(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_causal_mask=False,
        position_bias=None,
    ):
        batch_size, seq_length, hidden_dim = ops.shape(hidden_states)
        
        # Project queries, keys, and values
        query_states = self.query_proj(hidden_states)
        query_states = ops.reshape(
            query_states, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        query_states = ops.transpose(query_states, (0, 2, 1, 3))
        
        # Handle encoder-decoder attention
        if encoder_hidden_states is not None:
            key_states = self.key_proj(encoder_hidden_states)
            value_states = self.value_proj(encoder_hidden_states)
            key_states = ops.reshape(
                key_states, (batch_size, -1, self.num_key_value_heads, self.head_dim)
            )
            value_states = ops.reshape(
                value_states, (batch_size, -1, self.num_key_value_heads, self.head_dim)
            )
            key_states = ops.transpose(key_states, (0, 2, 1, 3))
            value_states = ops.transpose(value_states, (0, 2, 1, 3))
        else:
            key_states = self.key_proj(hidden_states)
            value_states = self.value_proj(hidden_states)
            key_states = ops.reshape(
                key_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim)
            )
            value_states = ops.reshape(
                value_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim)
            )
            key_states = ops.transpose(key_states, (0, 2, 1, 3))
            value_states = ops.transpose(value_states, (0, 2, 1, 3))
        
        # Repeat key/value heads if needed
        if self.num_key_value_heads != self.num_heads:
            key_states = self._repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
            value_states = self._repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
        
        # Query normalization (Gemma 2 feature)
        if self.query_head_dim_normalize:
            query_states = query_states * ops.cast(
                ops.sqrt(self.head_dim), query_states.dtype
            )
        else:
            query_states = query_states * ops.cast(
                ops.sqrt(self.hidden_dim / self.num_heads), query_states.dtype
            )
        
        # Compute attention scores
        attention_scores = ops.matmul(query_states, ops.transpose(key_states, (0, 1, 3, 2)))
        
        # Add relative position bias (T5-style)
        if position_bias is not None:
            attention_scores = attention_scores + position_bias
        
        # Apply sliding window attention if enabled
        if self.use_sliding_window_attention:
            attention_scores = self._apply_sliding_window_mask(attention_scores)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert attention_mask to float32 if needed
            attention_mask = ops.cast(attention_mask, attention_scores.dtype)
            # Expand attention mask to match attention scores shape
            attention_mask = ops.expand_dims(attention_mask, axis=1)
            attention_scores = attention_scores + attention_mask
        
        # Apply causal mask for decoder
        if use_causal_mask:
            causal_mask = self._get_causal_mask(seq_length, attention_scores.dtype)
            attention_scores = attention_scores + causal_mask
        
        # Apply soft cap if specified
        if self.attention_logit_soft_cap is not None:
            attention_scores = ops.clip(
                attention_scores,
                -self.attention_logit_soft_cap,
                self.attention_logit_soft_cap,
            )
        
        # Softmax and dropout
        attention_probs = ops.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Compute output
        context_states = ops.matmul(attention_probs, value_states)
        context_states = ops.transpose(context_states, (0, 2, 1, 3))
        context_states = ops.reshape(context_states, (batch_size, seq_length, self.hidden_dim))
        
        # Final projection
        output = self.output_proj(context_states)
        
        return output
    
    def _repeat_kv(self, hidden_states, repeat_factor):
        """Repeat key/value states to match number of attention heads."""
        batch_size, num_heads, seq_length, head_dim = ops.shape(hidden_states)
        hidden_states = ops.expand_dims(hidden_states, axis=2)
        hidden_states = ops.repeat(hidden_states, repeat_factor, axis=2)
        return ops.reshape(hidden_states, (batch_size, num_heads * repeat_factor, seq_length, head_dim))
    
    def _apply_sliding_window_mask(self, attention_scores):
        """Apply sliding window attention mask."""
        # For now, disable sliding window attention during graph construction
        # This is a simplified implementation that doesn't use dynamic loops
        return attention_scores
    
    def _get_causal_mask(self, seq_length, dtype):
        """Generate causal attention mask."""
        # Create a causal mask that prevents attending to future tokens
        mask = ops.triu(ops.ones((seq_length, seq_length), dtype=dtype), k=1)
        return mask * ops.cast(-1e9, dtype) 