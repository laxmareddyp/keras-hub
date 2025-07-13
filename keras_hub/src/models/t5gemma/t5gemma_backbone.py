import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.t5gemma.t5gemma_rms_normalization import T5GemmaRMSNormalization
from keras_hub.src.models.t5gemma.t5gemma_transformer_layer import T5GemmaTransformerLayer


@keras_hub_export("keras_hub.models.T5GemmaBackbone")
class T5GemmaBackbone(Backbone):
    """T5Gemma encoder-decoder backbone model.

    T5Gemma combines T5's encoder-decoder architecture with Gemma 2's improvements
    including RMS normalization, sliding window attention, and query normalization.
    This model is designed for sequence-to-sequence tasks like translation,
    summarization, and text generation.

    The default constructor gives a fully customizable, randomly initialized T5Gemma
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of Transformer layers for both encoder and decoder.
        num_heads: int. The number of attention heads for each Transformer.
        head_dim: int. The dimension of each attention head.
        num_key_value_heads: int. The number of heads for key/value projections.
            Defaults to num_heads.
        hidden_dim: int. The hidden size of the Transformer layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each Transformer layer.
        query_head_dim_normalize: boolean. If True, normalize the query before
            attention with head_dim. Defaults to True.
        use_sliding_window_attention: boolean. Whether to use sliding local
            window attention. Defaults to False.
        sliding_window_size: int. Size of the sliding local window. Defaults to 4096.
        attention_logit_soft_cap: None or int. Soft cap for the attention logits.
            Defaults to None.
        use_post_attention_norm: boolean. Whether to normalize after the attention
            block. Defaults to False.
        use_post_ffw_norm: boolean. Whether to normalize after the feedforward
            block. Defaults to False.
        activation: string. The activation function to use in the dense blocks
            of the Transformer Layers. Defaults to "gelu".
        dropout: float. Dropout probability for the Transformer layers.
        layer_norm_epsilon: float. Epsilon factor to be used in the
            layer normalization layers.
        tie_embedding_weights: boolean. If True, the weights of the token
            embedding and the weights projecting language model outputs from
            hidden_dim are tied.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Example:
    ```python
    input_data = {
        "encoder_token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "encoder_padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
        "decoder_token_ids": np.ones(shape=(1, 8), dtype="int32"),
        "decoder_padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]]),
    }

    # Pretrained T5Gemma model.
    model = keras_hub.models.T5GemmaBackbone.from_preset("t5gemma_2b")
    model(input_data)

    # Randomly initialized T5Gemma model with custom config.
    model = keras_hub.models.T5GemmaBackbone(
        vocabulary_size=32000,
        num_layers=12,
        num_heads=12,
        head_dim=64,
        hidden_dim=768,
        intermediate_dim=3072,
        use_sliding_window_attention=True,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        num_key_value_heads=None,
        query_head_dim_normalize=True,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        attention_logit_soft_cap=None,
        use_post_attention_norm=False,
        use_post_ffw_norm=False,
        activation="gelu",
        dropout=0.1,
        layer_norm_epsilon=1e-6,
        tie_embedding_weights=True,
        dtype=None,
        **kwargs,
    ):
        # Token embedding layer. This layer is shared by encoder and decoder.
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_embedding_weights,
            embeddings_initializer=keras.initializers.TruncatedNormal(1.0),
            dtype=dtype,
            name="token_embedding",
        )
        
        # Encoder layers
        self.encoder_embedding_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_embedding_dropout",
        )
        self.encoder_transformer_layers = []
        for i in range(num_layers):
            sliding_window = use_sliding_window_attention and (i % 2 == 0)
            layer = T5GemmaTransformerLayer(
                is_decoder=False,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
                query_head_dim_normalize=query_head_dim_normalize,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
                attention_logit_soft_cap=attention_logit_soft_cap,
                use_post_attention_norm=use_post_attention_norm,
                use_post_ffw_norm=use_post_ffw_norm,
                activation=activation,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"encoder_transformer_layer_{i}",
            )
            self.encoder_transformer_layers.append(layer)
        
        self.encoder_layer_norm = T5GemmaRMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="encoder_output_layer_norm",
        )
        self.encoder_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_output_dropout",
        )
        
        # Decoder layers
        self.decoder_embedding_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="decoder_embedding_dropout",
        )
        self.decoder_transformer_layers = []
        for i in range(num_layers):
            sliding_window = use_sliding_window_attention and (i % 2 == 0)
            layer = T5GemmaTransformerLayer(
                is_decoder=True,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
                query_head_dim_normalize=query_head_dim_normalize,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
                attention_logit_soft_cap=attention_logit_soft_cap,
                use_post_attention_norm=use_post_attention_norm,
                use_post_ffw_norm=use_post_ffw_norm,
                activation=activation,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"decoder_transformer_layer_{i}",
            )
            self.decoder_transformer_layers.append(layer)
        
        self.decoder_layer_norm = T5GemmaRMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="decoder_output_layer_norm",
        )
        self.decoder_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="decoder_output_dropout",
        )

        # === Functional Model ===
        encoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        encoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )
        decoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )
        
        # Encoder
        x = self.token_embedding(encoder_token_id_input)
        x = self.encoder_embedding_dropout(x)
        encoder_attention_mask = encoder_padding_mask_input[:, None, :]
        
        for transformer_layer in self.encoder_transformer_layers:
            x = transformer_layer(
                x,
                attention_mask=encoder_attention_mask,
                use_causal_mask=False,
            )
        
        x = self.encoder_layer_norm(x)
        x = self.encoder_dropout(x)
        encoder_output = x
        
        # Decoder
        x = self.token_embedding(decoder_token_id_input)
        x = self.decoder_embedding_dropout(x)
        decoder_attention_mask = decoder_padding_mask_input[:, None, :]
        
        for transformer_layer in self.decoder_transformer_layers:
            x = transformer_layer(
                x,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_attention_mask,
                use_causal_mask=True,
            )
        
        x = self.decoder_layer_norm(x)
        x = self.decoder_dropout(x)
        decoder_output = x

        super().__init__(
            inputs={
                "encoder_token_ids": encoder_token_id_input,
                "encoder_padding_mask": encoder_padding_mask_input,
                "decoder_token_ids": decoder_token_id_input,
                "decoder_padding_mask": decoder_padding_mask_input,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.query_head_dim_normalize = query_head_dim_normalize
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.use_post_attention_norm = use_post_attention_norm
        self.use_post_ffw_norm = use_post_ffw_norm
        self.activation = activation
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_embedding_weights = tie_embedding_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "query_head_dim_normalize": self.query_head_dim_normalize,
                "use_sliding_window_attention": self.use_sliding_window_attention,
                "sliding_window_size": self.sliding_window_size,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "use_post_attention_norm": self.use_post_attention_norm,
                "use_post_ffw_norm": self.use_post_ffw_norm,
                "activation": self.activation,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "tie_embedding_weights": self.tie_embedding_weights,
            }
        )
        return config 