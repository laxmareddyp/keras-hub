import keras
from keras import ops

from keras_hub.src.models.t5gemma.t5gemma_attention import T5GemmaAttention
from keras_hub.src.models.t5gemma.t5gemma_rms_normalization import T5GemmaRMSNormalization


class T5GemmaTransformerLayer(keras.layers.Layer):
    """Transformer layer for T5Gemma model.
    
    This layer combines T5's encoder-decoder structure with Gemma 2's
    improvements including RMS normalization and sliding window attention.
    """
    
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        intermediate_dim,
        num_heads,
        head_dim,
        num_key_value_heads=None,
        query_head_dim_normalize=True,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        attention_logit_soft_cap=None,
        use_post_attention_norm=False,
        use_post_ffw_norm=False,
        activation="gelu",
        dropout=0.0,
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.is_decoder = is_decoder
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.query_head_dim_normalize = query_head_dim_normalize
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.use_post_attention_norm = use_post_attention_norm
        self.use_post_ffw_norm = use_post_ffw_norm
        self.activation = activation
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        
        # Self-attention layer
        self.self_attention = T5GemmaAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            input_dim=hidden_dim,
            num_key_value_heads=self.num_key_value_heads,
            query_head_dim_normalize=query_head_dim_normalize,
            use_sliding_window_attention=use_sliding_window_attention,
            sliding_window_size=sliding_window_size,
            attention_logit_soft_cap=attention_logit_soft_cap,
            dropout=dropout,
            dtype=dtype,
            name="self_attention",
        )
        
        # Cross-attention layer (for decoder)
        if is_decoder:
            self.cross_attention = T5GemmaAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                input_dim=hidden_dim,
                num_key_value_heads=self.num_key_value_heads,
                query_head_dim_normalize=query_head_dim_normalize,
                use_sliding_window_attention=False,  # No sliding window for cross-attention
                attention_logit_soft_cap=attention_logit_soft_cap,
                dropout=dropout,
                dtype=dtype,
                name="cross_attention",
            )
        
        # Feed-forward network
        self.ffw = self._build_ffw()
        
        # Normalization layers
        self.self_attention_norm = T5GemmaRMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="self_attention_norm",
        )
        
        if is_decoder:
            self.cross_attention_norm = T5GemmaRMSNormalization(
                epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="cross_attention_norm",
            )
        
        self.ffw_norm = T5GemmaRMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="ffw_norm",
        )
        
        # Dropout layers
        self.attention_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
        )
        self.ffw_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
        )
        
    def _build_ffw(self):
        """Build the feed-forward network."""
        if self.activation == "gelu":
            activation = keras.activations.gelu
        elif self.activation == "relu":
            activation = keras.activations.relu
        elif self.activation == "swish":
            activation = keras.activations.swish
        else:
            activation = self.activation
        
        return keras.Sequential(
            [
                keras.layers.Dense(
                    self.intermediate_dim,
                    activation=activation,
                    dtype=self.dtype,
                    name="ffw_dense_1",
                ),
                keras.layers.Dense(
                    self.hidden_dim,
                    dtype=self.dtype,
                    name="ffw_dense_2",
                ),
            ],
            name="ffw",
        )
    
    def call(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_causal_mask=False,
        position_bias=None,
    ):
        # Self-attention
        residual = hidden_states
        if self.use_post_attention_norm:
            hidden_states = self.self_attention_norm(hidden_states)
        
        attention_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            position_bias=position_bias,
        )
        
        if not self.use_post_attention_norm:
            attention_output = self.self_attention_norm(attention_output)
        
        attention_output = self.attention_dropout(attention_output)
        hidden_states = residual + attention_output
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            residual = hidden_states
            if self.use_post_attention_norm:
                hidden_states = self.cross_attention_norm(hidden_states)
            
            cross_attention_output = self.cross_attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            
            if not self.use_post_attention_norm:
                cross_attention_output = self.cross_attention_norm(cross_attention_output)
            
            cross_attention_output = self.attention_dropout(cross_attention_output)
            hidden_states = residual + cross_attention_output
        
        # Feed-forward network
        residual = hidden_states
        if self.use_post_ffw_norm:
            hidden_states = self.ffw_norm(hidden_states)
        
        ffw_output = self.ffw(hidden_states)
        
        if not self.use_post_ffw_norm:
            ffw_output = self.ffw_norm(ffw_output)
        
        ffw_output = self.ffw_dropout(ffw_output)
        hidden_states = residual + ffw_output
        
        return hidden_states 