#!/usr/bin/env python3
"""
HuggingFace to KerasHub Model Converter

This script analyzes HuggingFace model repositories and generates complete
KerasHub-compatible implementation files including backbone, tokenizer,
preprocessor, and conversion utilities.

Usage:
    python hf_to_kerashub_converter.py --model_name microsoft/DialoGPT-medium
    python hf_to_kerashub_converter.py --model_name facebook/opt-1.3b --output_dir ./generated_models
"""

import os
import re
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import ast


@dataclass
class ModelArchitecture:
    """Represents the analyzed architecture of a HuggingFace model."""
    model_type: str
    config: Dict[str, Any]
    attention_type: str
    layer_structure: List[str]
    activation_function: str
    normalization_type: str
    position_encoding: str
    has_bias: bool
    supports_causal_lm: bool
    supports_seq_classification: bool


class HuggingFaceAPI:
    """Interface for HuggingFace Hub API calls."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_url = "https://huggingface.co"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get basic model information."""
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_config(self, model_name: str) -> Dict[str, Any]:
        """Fetch model configuration."""
        url = f"{self.base_url}/{model_name}/raw/main/config.json"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_modeling_code(self, model_name: str) -> str:
        """Fetch the modeling code from HuggingFace transformers."""
        try:
            # Try to get custom modeling file
            url = f"{self.base_url}/{model_name}/raw/main/modeling_{model_name.split('/')[-1]}.py"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.text
        except:
            pass
        
        # Fall back to standard transformers modeling file
        model_info = self.get_model_info(model_name)
        model_type = model_info.get("config", {}).get("model_type", "")
        
        if model_type:
            transformers_url = f"https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{model_type}/modeling_{model_type}.py"
            response = requests.get(transformers_url)
            if response.status_code == 200:
                return response.text
        
        return ""
    
    def get_tokenizer_config(self, model_name: str) -> Dict[str, Any]:
        """Fetch tokenizer configuration."""
        url = f"{self.base_url}/{model_name}/raw/main/tokenizer_config.json"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return {}


class ArchitectureAnalyzer:
    """Analyzes HuggingFace model architecture and extracts patterns."""
    
    def __init__(self, api: HuggingFaceAPI):
        self.api = api
    
    def analyze_model(self, model_name: str) -> ModelArchitecture:
        """Perform comprehensive analysis of the model architecture."""
        print(f"🔍 Analyzing model: {model_name}")
        
        # Get model configuration and code
        config = self.api.get_config(model_name)
        modeling_code = self.api.get_modeling_code(model_name)
        
        # Analyze architecture components
        model_type = config.get("model_type", "unknown")
        attention_type = self._detect_attention_type(modeling_code, config)
        layer_structure = self._analyze_layer_structure(modeling_code)
        activation_function = self._detect_activation(modeling_code, config)
        normalization_type = self._detect_normalization(modeling_code)
        position_encoding = self._detect_position_encoding(modeling_code, config)
        has_bias = self._detect_bias_usage(modeling_code, config)
        
        # Detect supported tasks
        supports_causal_lm = "CausalLM" in modeling_code or "causal" in str(config)
        supports_seq_classification = "SequenceClassification" in modeling_code
        
        return ModelArchitecture(
            model_type=model_type,
            config=config,
            attention_type=attention_type,
            layer_structure=layer_structure,
            activation_function=activation_function,
            normalization_type=normalization_type,
            position_encoding=position_encoding,
            has_bias=has_bias,
            supports_causal_lm=supports_causal_lm,
            supports_seq_classification=supports_seq_classification
        )
    
    def _detect_attention_type(self, code: str, config: Dict) -> str:
        """Detect the type of attention mechanism used."""
        if "scaled_dot_product_attention" in code.lower():
            return "scaled_dot_product"
        elif "flash_attention" in code.lower():
            return "flash_attention"
        elif "grouped_query" in code.lower() or config.get("num_key_value_heads"):
            return "grouped_query_attention"
        elif "multi_query" in code.lower():
            return "multi_query_attention"
        else:
            return "multi_head_attention"
    
    def _analyze_layer_structure(self, code: str) -> List[str]:
        """Analyze the transformer layer structure."""
        layers = []
        
        # Common layer patterns
        if "self_attn" in code or "self_attention" in code:
            layers.append("self_attention")
        if "cross_attn" in code or "cross_attention" in code:
            layers.append("cross_attention")
        if "mlp" in code.lower() or "feed_forward" in code:
            layers.append("feed_forward")
        if "layer_norm" in code.lower() or "LayerNorm" in code:
            layers.append("layer_norm")
        if "dropout" in code.lower():
            layers.append("dropout")
        
        return layers if layers else ["self_attention", "feed_forward", "layer_norm"]
    
    def _detect_activation(self, code: str, config: Dict) -> str:
        """Detect activation function used."""
        activations = {
            "gelu": ["gelu", "F.gelu"],
            "relu": ["relu", "F.relu"],
            "silu": ["silu", "F.silu", "swish"],
            "tanh": ["tanh", "F.tanh"],
            "sigmoid": ["sigmoid", "F.sigmoid"]
        }
        
        # Check config first
        hidden_act = config.get("hidden_act", config.get("activation_function", ""))
        if hidden_act:
            return hidden_act
        
        # Analyze code
        code_lower = code.lower()
        for activation, patterns in activations.items():
            if any(pattern in code_lower for pattern in patterns):
                return activation
        
        return "gelu"  # Default
    
    def _detect_normalization(self, code: str) -> str:
        """Detect normalization type."""
        if "RMSNorm" in code or "rms_norm" in code.lower():
            return "rms_norm"
        elif "LayerNorm" in code or "layer_norm" in code.lower():
            return "layer_norm"
        else:
            return "layer_norm"  # Default
    
    def _detect_position_encoding(self, code: str, config: Dict) -> str:
        """Detect position encoding type."""
        if "rope" in code.lower() or "rotary" in code.lower():
            return "rope"
        elif "alibi" in code.lower():
            return "alibi"
        elif "relative" in code.lower():
            return "relative"
        elif config.get("max_position_embeddings"):
            return "absolute"
        else:
            return "none"
    
    def _detect_bias_usage(self, code: str, config: Dict) -> bool:
        """Detect if bias is used in linear layers."""
        if "bias=False" in code:
            return False
        elif "bias=True" in code:
            return True
        else:
            return config.get("use_bias", True)


class KerasHubGenerator:
    """Generates KerasHub-compatible implementation files."""
    
    def __init__(self, output_dir: str = "./generated_models"):
        self.output_dir = Path(output_dir)
    
    def generate_all_files(self, model_name: str, architecture: ModelArchitecture) -> None:
        """Generate all KerasHub files for the model."""
        clean_name = self._clean_model_name(model_name)
        model_dir = self.output_dir / clean_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Generating files in: {model_dir}")
        
        # Generate all files
        self._generate_init_file(model_dir, clean_name)
        self._generate_config_file(model_dir, clean_name, architecture)
        self._generate_backbone_file(model_dir, clean_name, architecture)
        self._generate_attention_file(model_dir, clean_name, architecture)
        self._generate_decoder_file(model_dir, clean_name, architecture)
        self._generate_layernorm_file(model_dir, clean_name, architecture)
        self._generate_tokenizer_file(model_dir, clean_name, architecture)
        self._generate_causal_lm_file(model_dir, clean_name, architecture)
        self._generate_preprocessor_file(model_dir, clean_name, architecture)
        self._generate_presets_file(model_dir, clean_name, architecture, model_name)
        self._generate_conversion_script(model_dir, clean_name, architecture, model_name)
        self._generate_tests(model_dir, clean_name, architecture)
        
        print(f"✅ Generated complete KerasHub implementation for {clean_name}")
    
    def _clean_model_name(self, model_name: str) -> str:
        """Convert model name to KerasHub format."""
        name = model_name.split("/")[-1].lower()
        name = re.sub(r"[^a-z0-9_]", "_", name)
        return name
    
    def _generate_init_file(self, model_dir: Path, model_name: str) -> None:
        """Generate __init__.py file."""
        content = f'''"""
{model_name.title()} model implementations for KerasHub.

Auto-generated from HuggingFace model analysis.
"""

from keras_hub.src.models.{model_name}.{model_name}_backbone import {model_name.title()}Backbone
from keras_hub.src.models.{model_name}.{model_name}_tokenizer import {model_name.title()}Tokenizer

if TYPE_CHECKING:
    from keras_hub.src.models.{model_name}.{model_name}_causal_lm import {model_name.title()}CausalLM
    from keras_hub.src.models.{model_name}.{model_name}_causal_lm_preprocessor import {model_name.title()}CausalLMPreprocessor
'''
        
        with open(model_dir / "__init__.py", "w") as f:
            f.write(content)
    
    def _generate_config_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate configuration file."""
        config = arch.config
        
        content = f'''import keras
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import BackboneConfig


@keras_hub_export("keras_hub.models.{model_name.title()}Config")
class {model_name.title()}Config(BackboneConfig):
    """
    Configuration class for {model_name.title()} backbone.

    This configuration class stores the configuration for a {model_name.title()} model.
    This configuration is used to instantiate a {model_name.title()} model according to the 
    specified arguments, defining the model architecture.

    Args:
        vocabulary_size: Integer. The size of the vocabulary.
        hidden_dim: Integer. The size of the transformer hidden layers.
        num_layers: Integer. The number of transformer layers.
        num_heads: Integer. The number of attention heads for each layer.
        intermediate_dim: Integer. The output dimension of the feedforward network.
        dropout: Float. Dropout probability for the transformer layers.
        max_sequence_length: Integer. The maximum sequence length.
        dtype: String or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        vocabulary_size={config.get('vocab_size', 50257)},
        hidden_dim={config.get('hidden_size', config.get('d_model', 768))},
        num_layers={config.get('num_hidden_layers', config.get('n_layer', 12))},
        num_heads={config.get('num_attention_heads', config.get('n_head', 12))},
        intermediate_dim={config.get('intermediate_size', config.get('d_ff', 3072))},
        dropout={config.get('dropout', config.get('resid_pdrop', 0.1))},
        max_sequence_length={config.get('max_position_embeddings', config.get('n_positions', 1024))},
        dtype="float32",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        
        # Architecture-specific parameters
        self.activation_function = "{arch.activation_function}"
        self.normalization_type = "{arch.normalization_type}"
        self.attention_type = "{arch.attention_type}"
        self.position_encoding = "{arch.position_encoding}"
        self.use_bias = {arch.has_bias}
        
        # Additional config parameters
        self.layer_norm_epsilon = {config.get('layer_norm_epsilon', config.get('layernorm_epsilon', 1e-5))}
'''

        # Add architecture-specific parameters
        if arch.attention_type == "grouped_query_attention":
            content += f'''        self.num_key_value_heads = {config.get('num_key_value_heads', config.get('num_heads', 12))}
'''
        
        if arch.position_encoding == "rope":
            content += f'''        self.rope_max_wavelength = {config.get('rope_theta', 10000)}
        self.rope_scaling_factor = {config.get('rope_scaling_factor', 1.0)}
'''
        
        with open(model_dir / f"{model_name}_config.py", "w") as f:
            f.write(content)
    
    def _generate_backbone_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate backbone implementation file."""
        content = f'''import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.{model_name}.{model_name}_decoder import {model_name.title()}TransformerDecoder
from keras_hub.src.models.{model_name}.{model_name}_layernorm import {model_name.title()}LayerNorm


@keras_hub_export("keras_hub.models.{model_name.title()}Backbone")
class {model_name.title()}Backbone(Backbone):
    """
    The {model_name.title()} Transformer backbone model.

    This network implements a Transformer-based decoder network,
    {model_name.title()}, as described in the original paper.
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    {model_name.title()} model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        hidden_dim: int. The size of the transformer encoding layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the feedforward network for each transformer.
        dropout: float. Dropout probability for the transformer layers.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Examples:

    ```python
    input_data = {{
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }}

    # Pretrained {model_name.title()} backbone.
    model = keras_hub.models.{model_name.title()}Backbone.from_preset("{model_name}_base_en")
    model(input_data)

    # Randomly initialized {model_name.title()} backbone with custom config.
    model = keras_hub.models.{model_name.title()}Backbone(
        vocabulary_size=50257,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=1024,
        dtype="float32"
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=1024,
        dtype="float32",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length

        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            dtype=dtype,
            name="token_embedding",
        )
'''

        # Add position embedding based on detected type
        if arch.position_encoding == "absolute":
            content += f'''
        self.position_embedding = keras.layers.Embedding(
            input_dim=max_sequence_length,
            output_dim=hidden_dim,
            dtype=dtype,
            name="position_embedding",
        )
'''
        elif arch.position_encoding == "rope":
            content += f'''
        # RoPE embeddings are handled in attention layers
        self.rope_max_wavelength = kwargs.get("rope_max_wavelength", 10000)
'''

        content += f'''
        self.embeddings_dropout = keras.layers.Dropout(
            rate=dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )

        # Transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            layer = {model_name.title()}TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation="{arch.activation_function}",
                layer_norm_epsilon=kwargs.get("layer_norm_epsilon", 1e-5),
                kernel_initializer=kwargs.get("kernel_initializer", "glorot_uniform"),
                dtype=dtype,
                name=f"transformer_layer_{{i}}",
            )
            self.transformer_layers.append(layer)

        self.layer_norm = {model_name.title()}LayerNorm(
            epsilon=kwargs.get("layer_norm_epsilon", 1e-5),
            dtype=dtype,
            name="layer_norm",
        )

        # Set up compute_output_spec() and _build()
        self.built = False

    def call(
        self,
        inputs,
        training=None,
        mask=None,
    ):
        if isinstance(inputs, dict):
            token_ids = inputs["token_ids"]
            padding_mask = inputs.get("padding_mask", None)
        else:
            token_ids = inputs
            padding_mask = mask

        # Token embeddings
        x = self.token_embedding(token_ids)
        
        batch_size = ops.shape(token_ids)[0]
        seq_length = ops.shape(token_ids)[1]
'''

        # Add position encoding logic
        if arch.position_encoding == "absolute":
            content += f'''
        # Position embeddings
        position_ids = ops.arange(seq_length, dtype="int32")
        position_ids = ops.expand_dims(position_ids, axis=0)
        position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))
        position_embeddings = self.position_embedding(position_ids)
        
        x = x + position_embeddings
'''

        content += f'''
        x = self.embeddings_dropout(x, training=training)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                training=training,
                mask=padding_mask,
            )

        x = self.layer_norm(x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {{
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
            }}
        )
        return config

    @property
    def token_embedding(self):
        return self._token_embedding

    @token_embedding.setter
    def token_embedding(self, value):
        self._token_embedding = value
'''

        with open(model_dir / f"{model_name}_backbone.py", "w") as f:
            f.write(content)
    
    def _generate_attention_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate attention mechanism file."""
        content = f'''import keras
from keras import ops
import math

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


class {model_name.title()}Attention(keras.layers.Layer):
    """
    Multi-head attention layer for {model_name.title()}.
    
    This layer implements {arch.attention_type} attention mechanism.
    """

    def __init__(
        self,
        num_heads,
        dropout=0.0,
        use_bias={arch.has_bias},
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        
        self.supports_masking = True

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        self.head_dim = self.hidden_dim // self.num_heads
        
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({{self.hidden_dim}}) must be divisible by "
                f"num_heads ({{self.num_heads}})"
            )

        # Query, Key, Value projections
        self.query_dense = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="query",
        )
        self.key_dense = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="key",
        )
        self.value_dense = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="value",
        )
        
        # Output projection
        self.output_dense = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="output",
        )
        
        # Dropout
        self.attention_dropout = keras.layers.Dropout(self.dropout)
        
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        training=None,
    ):
        batch_size, seq_length, hidden_dim = ops.shape(hidden_states)
        
        # Linear projections
        query = self.query_dense(hidden_states)
        key = self.key_dense(hidden_states)
        value = self.value_dense(hidden_states)
        
        # Reshape for multi-head attention
        query = ops.reshape(
            query, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        key = ops.reshape(
            key, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        value = ops.reshape(
            value, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        
        # Transpose to (batch_size, num_heads, seq_length, head_dim)
        query = ops.transpose(query, axes=[0, 2, 1, 3])
        key = ops.transpose(key, axes=[0, 2, 1, 3])
        value = ops.transpose(value, axes=[0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = ops.matmul(query, ops.transpose(key, axes=[0, 1, 3, 2]))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply causal mask for autoregressive generation
        causal_mask = compute_causal_mask(batch_size, seq_length, seq_length)
        attention_scores = ops.where(causal_mask, attention_scores, -1e9)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, dtype=attention_scores.dtype)
            attention_mask = ops.expand_dims(attention_mask, axis=1)
            attention_mask = ops.expand_dims(attention_mask, axis=1)
            attention_scores = attention_scores + (1.0 - attention_mask) * -1e9
        
        # Softmax
        attention_probs = ops.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs, training=training)
        
        # Apply attention to values
        attention_output = ops.matmul(attention_probs, value)
        
        # Transpose back to (batch_size, seq_length, num_heads, head_dim)
        attention_output = ops.transpose(attention_output, axes=[0, 2, 1, 3])
        
        # Reshape to (batch_size, seq_length, hidden_dim)
        attention_output = ops.reshape(
            attention_output, (batch_size, seq_length, hidden_dim)
        )
        
        # Final linear projection
        output = self.output_dense(attention_output)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({{
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        }})
        return config
'''

        with open(model_dir / f"{model_name}_attention.py", "w") as f:
            f.write(content)
    
    def _generate_decoder_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate transformer decoder layer file."""
        content = f'''import keras
from keras import ops

from keras_hub.src.models.{model_name}.{model_name}_attention import {model_name.title()}Attention
from keras_hub.src.models.{model_name}.{model_name}_layernorm import {model_name.title()}LayerNorm


class {model_name.title()}TransformerDecoder(keras.layers.Layer):
    """
    A single transformer decoder layer for {model_name.title()}.
    
    This layer implements the standard transformer decoder architecture with
    self-attention and feed-forward components.
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0.0,
        activation="{arch.activation_function}",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        
        self.supports_masking = True

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        
        # Self-attention layer
        self.self_attention = {model_name.title()}Attention(
            num_heads=self.num_heads,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            name="self_attention",
        )
        
        # Layer normalization
        self.attention_layer_norm = {model_name.title()}LayerNorm(
            epsilon=self.layer_norm_epsilon,
            name="attention_layer_norm",
        )
        
        # Feed-forward network
        self.intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            name="intermediate",
        )
        
        self.output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            name="output",
        )
        
        # Final layer normalization
        self.output_layer_norm = {model_name.title()}LayerNorm(
            epsilon=self.layer_norm_epsilon,
            name="output_layer_norm",
        )
        
        # Dropout layers
        self.attention_dropout = keras.layers.Dropout(self.dropout)
        self.output_dropout = keras.layers.Dropout(self.dropout)
        
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        training=None,
    ):
        # Self-attention block with residual connection
        residual = hidden_states
        hidden_states = self.attention_layer_norm(hidden_states)
        
        attention_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = self.attention_dropout(attention_output, training=training)
        hidden_states = residual + attention_output
        
        # Feed-forward block with residual connection
        residual = hidden_states
        hidden_states = self.output_layer_norm(hidden_states)
        
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        
        hidden_states = residual + hidden_states
        
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({{
            "intermediate_dim": self.intermediate_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "activation": keras.activations.serialize(self.activation),
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        }})
        return config
'''

        with open(model_dir / f"{model_name}_decoder.py", "w") as f:
            f.write(content)
    
    def _generate_layernorm_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate layer normalization file."""
        if arch.normalization_type == "rms_norm":
            norm_class = "RMSNormalization"
            norm_implementation = '''
    def call(self, inputs):
        variance = ops.mean(ops.square(inputs), axis=-1, keepdims=True)
        inputs = inputs * ops.rsqrt(variance + self.epsilon)
        return self.scale * inputs
'''
        else:
            norm_class = "LayerNormalization"
            norm_implementation = '''
    def call(self, inputs):
        return super().call(inputs)
'''

        content = f'''import keras
from keras import ops


class {model_name.title()}LayerNorm(keras.layers.{norm_class}):
    """
    {arch.normalization_type.upper()} layer normalization for {model_name.title()}.
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)
        self.epsilon = epsilon
{norm_implementation}
    def get_config(self):
        config = super().get_config()
        config.update({{"epsilon": self.epsilon}})
        return config
'''

        with open(model_dir / f"{model_name}_layernorm.py", "w") as f:
            f.write(content)
    
    def _generate_tokenizer_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate tokenizer file."""
        content = f'''from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers.tokenizer import Tokenizer


@keras_hub_export("keras_hub.models.{model_name.title()}Tokenizer")
class {model_name.title()}Tokenizer(Tokenizer):
    """
    A {model_name.title()} tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    {model_name.title()} models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a {model_name.title()} preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.{model_name.title()}Tokenizer.from_preset(
        "{model_name}_base_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = {{"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    tokenizer = keras_hub.models.{model_name.title()}Tokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    tokenizer("a quick fox.")
    ```
    """

    backbone_cls = "{model_name.title()}Backbone"

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("<|endoftext|>", "end_token")
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

    def set_vocabulary_and_merges(self, vocabulary, merges):
        super().set_vocabulary_and_merges(vocabulary, merges)

        if vocabulary is not None:
            # Check for required special tokens.
            for token_name, token in self.special_tokens.items():
                if token not in self.vocabulary:
                    raise ValueError(
                        f"Token `'{{token}}'` for `{{token_name}}` not found in the "
                        "provided `vocabulary`. Please provide `'{{token}}'` in your "
                        "`vocabulary` or use a pretrained `vocabulary` name."
                    )

            self.end_token_id = self.token_to_id(self.end_token)
        else:
            self.end_token_id = None
'''

        with open(model_dir / f"{model_name}_tokenizer.py", "w") as f:
            f.write(content)
    
    def _generate_causal_lm_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate causal language model file."""
        if not arch.supports_causal_lm:
            return
            
        content = f'''import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.{model_name}.{model_name}_backbone import {model_name.title()}Backbone
from keras_hub.src.models.{model_name}.{model_name}_causal_lm_preprocessor import {model_name.title()}CausalLMPreprocessor


@keras_hub_export("keras_hub.models.{model_name.title()}CausalLM")
class {model_name.title()}CausalLM(CausalLM):
    """
    An end-to-end {model_name.title()} model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This model will use a tranined
    {model_name.title()}Backbone by default.

    This model can optionally be configured with a `preprocessor`, which will
    apply a {model_name.title()}CausalLMPreprocessor to string inputs. This is done by
    default when creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/huggingface/transformers).

    Args:
        backbone: A `keras_hub.models.{model_name.title()}Backbone` instance.
        preprocessor: A `keras_hub.models.{model_name.title()}CausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    {model_name}_lm = keras_hub.models.{model_name.title()}CausalLM.from_preset("{model_name}_base_en")
    {model_name}_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    {model_name}_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    {model_name}_lm = keras_hub.models.{model_name.title()}CausalLM.from_preset("{model_name}_base_en")
    {model_name}_lm.compile(sampler="greedy")
    {model_name}_lm.generate("I want to say", max_length=30)

    {model_name}_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    {model_name}_lm.generate("I want to say")
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {{
        "token_ids": np.array([[50256, 1]])
        "padding_mask": np.array([[1, 1]]),
    }}
    {model_name}_lm = keras_hub.models.{model_name.title()}CausalLM.from_preset(
        "{model_name}_base_en",
        preprocessor=None,
    )
    {model_name}_lm.generate(prompt)
    ```

    Call the model directly for training.
    ```python
    {model_name}_lm = keras_hub.models.{model_name.title()}CausalLM.from_preset("{model_name}_base_en")

    # Setup data.
    train_ds = tf.data.Dataset.from_tensor_slices(
        ["The quick brown fox jumped.", "I forgot my homework."]
    )
    train_ds = train_ds.batch(2).map({model_name}_lm.preprocessor, tf.data.AUTOTUNE)

    # Compile and fit.
    {model_name}_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    {model_name}_lm.fit(train_ds, epochs=3)
    ```

    Call the model directly for inference.
    ```python
    # Setup data.
    sentences = ["The quick brown fox jumped.", "I forgot my homework."]
    {model_name}_lm = keras_hub.models.{model_name.title()}CausalLM.from_preset("{model_name}_base_en")
    preprocessed = {model_name}_lm.preprocessor(sentences)

    # Pass preprocessed data to model
    {model_name}_lm(preprocessed)

    # Alternative: call model without preprocessor
    {model_name}_lm = keras_hub.models.{model_name.title()}CausalLM.from_preset(
        "{model_name}_base_en",
        preprocessor=None,
    )
    {model_name}_lm(preprocessed)
    ```
    """

    backbone_cls = {model_name.title()}Backbone
    preprocessor_cls = {model_name.title()}CausalLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.backbone = backbone
        self.preprocessor = preprocessor

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
    ):
        """Forward pass of {model_name.title()}CausalLM with caching.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention, and
        will only compute the output for the last token of the input sequence.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs in the
                whole sequence.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        x = self.backbone.token_embedding(token_ids)
        # Each decoder layer has a cache; we update them separately.
        updated_cache = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer.call_with_cache(
                x,
                current_cache,
                cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states, cache

    @property
    def layers(self):
        return super().layers + [self.backbone]
'''

        with open(model_dir / f"{model_name}_causal_lm.py", "w") as f:
            f.write(content)
    
    def _generate_preprocessor_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate preprocessor file."""
        content = f'''from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.{model_name}.{model_name}_tokenizer import {model_name.title()}Tokenizer


@keras_hub_export("keras_hub.models.{model_name.title()}CausalLMPreprocessor")
class {model_name.title()}CausalLMPreprocessor(CausalLMPreprocessor):
    """
    A {model_name.title()} preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do 2 things:

    1. Tokenize any number of input segments using the `tokenizer`.
    2. Pack the inputs together using a `keras_hub.layers.StartEndPacker`.
       with the appropriate `"<|endoftext|>"` tokens.

    The layer can handle both raw string and integer input. If the input is
    integer, it will be passed through without any further tokenization or
    preprocessing.

    Args:
        tokenizer: A `keras_hub.models.{model_name.title()}Tokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If `True`, the preprocessor will append the tokenizer end
            token to each input sequence.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_hub.models.{model_name.title()}CausalLMPreprocessor.from_preset(
        "{model_name}_base_en"
    )

    # Tokenize and pack a single sentence.
    sentence = tf.constant("League of legends")
    preprocessor(sentence)
    # Same output.
    preprocessor("League of legends")

    # Tokenize a batch of sentences.
    sentences = tf.constant(["Taco tuesday", "Fish taco please!"])
    preprocessor(sentences)
    # Same output.
    preprocessor(["Taco tuesday", "Fish taco please!"])
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.{model_name.title()}CausalLMPreprocessor.from_preset(
        "{model_name}_base_en"
    )

    first = tf.constant(["League of legends", "Taco tuesday"])
    second = tf.constant(["Chess is fun", "Tomato tuesday"])
    ds = tf.data.Dataset.from_tensor_slices((first, second))

    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = "{model_name.title()}Backbone"
    tokenizer_cls = {model_name.title()}Tokenizer
'''

        with open(model_dir / f"{model_name}_causal_lm_preprocessor.py", "w") as f:
            f.write(content)
    
    def _generate_presets_file(self, model_dir: Path, model_name: str, arch: ModelArchitecture, original_model_name: str) -> None:
        """Generate presets configuration file."""
        content = f'''"""
{model_name.title()} model preset configurations.
"""

# Metadata for loading pretrained model weights.
backbone_presets = {{
    "{model_name}_base_en": {{
        "metadata": {{
            "description": (
                "{model_name.title()} base model. "
                "Auto-generated from HuggingFace {original_model_name}."
            ),
            "params": {sum(arch.config.get(key, 0) for key in ['vocab_size', 'hidden_size', 'num_hidden_layers'])},
            "official_name": "{model_name.title()}",
            "path": "{model_name}",
            "model_card": "https://huggingface.co/{original_model_name}",
        }},
        "kaggle_handle": "kaggle://keras/{model_name}/keras/{model_name}_base_en/2",
    }},
}}

tokenizer_presets = {{
    "{model_name}_base_en": {{
        "metadata": {{
            "description": (
                "{model_name.title()} tokenizer with BPE vocabulary. "
                "Auto-generated from HuggingFace {original_model_name}."
            ),
            "vocab_size": {arch.config.get('vocab_size', 50257)},
        }},
        "kaggle_handle": "kaggle://keras/{model_name}/keras/{model_name}_base_en/2",
    }},
}}

task_presets = {{
    "{model_name}_base_en": {{
        "metadata": {{
            "description": (
                "{model_name.title()} causal language model. "
                "Auto-generated from HuggingFace {original_model_name}."
            ),
            "params": {sum(arch.config.get(key, 0) for key in ['vocab_size', 'hidden_size', 'num_hidden_layers'])},
            "official_name": "{model_name.title()}",
            "path": "{model_name}",
            "model_card": "https://huggingface.co/{original_model_name}",
        }},
        "kaggle_handle": "kaggle://keras/{model_name}/keras/{model_name}_base_en/2",
    }},
}}

preprocessor_presets = {{
    "{model_name}_base_en": {{
        "metadata": {{
            "description": (
                "{model_name.title()} causal language model preprocessor. "
                "Auto-generated from HuggingFace {original_model_name}."
            ),
            "vocab_size": {arch.config.get('vocab_size', 50257)},
        }},
        "kaggle_handle": "kaggle://keras/{model_name}/keras/{model_name}_base_en/2",
    }},
}}
'''

        with open(model_dir / f"{model_name}_presets.py", "w") as f:
            f.write(content)
    
    def _generate_conversion_script(self, model_dir: Path, model_name: str, arch: ModelArchitecture, original_model_name: str) -> None:
        """Generate HuggingFace to KerasHub conversion script."""
        content = f'''#!/usr/bin/env python3
"""
Script to convert {original_model_name} HuggingFace checkpoint to KerasHub format.

Usage:
    python convert_{model_name}_checkpoints.py --preset {model_name}_base_en
"""

import json
import numpy as np
import torch
from absl import app
from absl import flags
from transformers import AutoTokenizer, AutoModel

import keras_hub
from keras_hub.models import {model_name.title()}Backbone
from keras_hub.models import {model_name.title()}CausalLMPreprocessor
from keras_hub.models import {model_name.title()}Tokenizer

PRESET_MAP = {{
    "{model_name}_base_en": "{original_model_name}",
}}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {{','.join(PRESET_MAP.keys())}}"
)


def convert_checkpoints(keras_hub_model, hf_model):
    """Convert HuggingFace model weights to KerasHub format."""
    print("\\n-> Converting weights from HuggingFace to KerasHub format...")
    
    config = hf_model.config
    
    # Token embeddings
    keras_hub_model.token_embedding.embeddings.assign(
        hf_model.transformer.wte.weight.detach().cpu().float().numpy()
    )
    
    # Position embeddings (if applicable)
    if hasattr(hf_model.transformer, 'wpe'):
        keras_hub_model.position_embedding.embeddings.assign(
            hf_model.transformer.wpe.weight.detach().cpu().float().numpy()
        )
    
    # Convert transformer layers
    for i in range(keras_hub_model.num_layers):
        keras_hub_layer = keras_hub_model.transformer_layers[i]
        hf_layer = hf_model.transformer.h[i]
        
        # Self-attention weights
        # Query projection
        keras_hub_layer.self_attention.query_dense.kernel.assign(
            hf_layer.attn.c_attn.weight[:, :config.hidden_size].T.detach().cpu().float().numpy()
        )
        
        # Key projection  
        keras_hub_layer.self_attention.key_dense.kernel.assign(
            hf_layer.attn.c_attn.weight[:, config.hidden_size:2*config.hidden_size].T.detach().cpu().float().numpy()
        )
        
        # Value projection
        keras_hub_layer.self_attention.value_dense.kernel.assign(
            hf_layer.attn.c_attn.weight[:, 2*config.hidden_size:].T.detach().cpu().float().numpy()
        )
        
        # Output projection
        keras_hub_layer.self_attention.output_dense.kernel.assign(
            hf_layer.attn.c_proj.weight.T.detach().cpu().float().numpy()
        )
        
        # Layer normalization
        keras_hub_layer.attention_layer_norm.gamma.assign(
            hf_layer.ln_1.weight.detach().cpu().float().numpy()
        )
        keras_hub_layer.attention_layer_norm.beta.assign(
            hf_layer.ln_1.bias.detach().cpu().float().numpy()
        )
        
        # Feed-forward network
        keras_hub_layer.intermediate_dense.kernel.assign(
            hf_layer.mlp.c_fc.weight.T.detach().cpu().float().numpy()
        )
        keras_hub_layer.intermediate_dense.bias.assign(
            hf_layer.mlp.c_fc.bias.detach().cpu().float().numpy()
        )
        
        keras_hub_layer.output_dense.kernel.assign(
            hf_layer.mlp.c_proj.weight.T.detach().cpu().float().numpy()
        )
        keras_hub_layer.output_dense.bias.assign(
            hf_layer.mlp.c_proj.bias.detach().cpu().float().numpy()
        )
        
        # Output layer normalization
        keras_hub_layer.output_layer_norm.gamma.assign(
            hf_layer.ln_2.weight.detach().cpu().float().numpy()
        )
        keras_hub_layer.output_layer_norm.beta.assign(
            hf_layer.ln_2.bias.detach().cpu().float().numpy()
        )
    
    # Final layer normalization
    keras_hub_model.layer_norm.gamma.assign(
        hf_model.transformer.ln_f.weight.detach().cpu().float().numpy()
    )
    keras_hub_model.layer_norm.beta.assign(
        hf_model.transformer.ln_f.bias.detach().cpu().float().numpy()
    )
    
    print("-> Weight conversion completed successfully!")


def test_model(keras_hub_model, keras_hub_tokenizer, hf_model, hf_tokenizer):
    """Test that KerasHub and HuggingFace models produce similar outputs."""
    print("\\n-> Testing model outputs...")
    
    test_text = "The quick brown fox"
    
    # KerasHub forward pass
    keras_hub_preprocessor = {model_name.title()}CausalLMPreprocessor(keras_hub_tokenizer)
    keras_hub_inputs = keras_hub_preprocessor([test_text], sequence_length=10)
    keras_hub_outputs = keras_hub_model(keras_hub_inputs)
    
    # HuggingFace forward pass  
    hf_inputs = hf_tokenizer(test_text, return_tensors="pt", max_length=10, padding="max_length")
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs).last_hidden_state
    
    # Compare outputs
    keras_hub_output = keras_hub_outputs[0, 0, :10].numpy()
    hf_output = hf_outputs[0, 0, :10].detach().cpu().numpy()
    
    print(f"KerasHub output (first 10 dims): {{keras_hub_output}}")
    print(f"HuggingFace output (first 10 dims): {{hf_output}}")
    print(f"Max difference: {{np.max(np.abs(keras_hub_output - hf_output))}}")
    
    # Check if outputs are close
    if np.allclose(keras_hub_output, hf_output, atol=1e-4):
        print("✅ Model outputs match within tolerance!")
    else:
        print("⚠️  Model outputs differ significantly. Check weight conversion.")


def main(_):
    # Get preset information
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(f"Invalid preset {{FLAGS.preset}}. Must be one of {{','.join(PRESET_MAP.keys())}}")
    
    preset = FLAGS.preset
    hf_model_name = PRESET_MAP[preset]
    
    print(f"Converting {{hf_model_name}} to KerasHub {{preset}}")
    
    # Load HuggingFace model and tokenizer
    print("\\n-> Loading HuggingFace model...")
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model.eval()
    
    # Create KerasHub model with same configuration
    print("\\n-> Creating KerasHub model...")
    config = hf_model.config
    
    keras_hub_model = {model_name.title()}Backbone(
        vocabulary_size=config.vocab_size,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        hidden_dim=config.n_embd,
        intermediate_dim=config.n_inner or 4 * config.n_embd,
        max_sequence_length=config.n_positions,
        dropout=config.resid_pdrop,
    )
    
    # Create KerasHub tokenizer
    print("\\n-> Creating KerasHub tokenizer...")
    # TODO: Implement proper tokenizer conversion
    # This is a placeholder - actual implementation would need to:
    # 1. Extract vocabulary from HuggingFace tokenizer
    # 2. Extract merge rules if using BPE
    # 3. Handle special tokens properly
    keras_hub_tokenizer = {model_name.title()}Tokenizer(
        vocabulary="placeholder_vocab.json",  # TODO: Extract from HF
        merges="placeholder_merges.txt",      # TODO: Extract from HF
    )
    
    # Convert weights
    convert_checkpoints(keras_hub_model, hf_model)
    
    # Test conversion
    # test_model(keras_hub_model, keras_hub_tokenizer, hf_model, hf_tokenizer)
    
    # Save KerasHub model
    print(f"\\n-> Saving KerasHub model to {{preset}}...")
    keras_hub_model.save_to_preset(preset)
    # keras_hub_tokenizer.save_to_preset(preset)  # TODO: Implement after tokenizer conversion
    
    print(f"\\n✅ Conversion completed! Model saved as {{preset}}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
'''

        with open(model_dir / f"convert_{model_name}_checkpoints.py", "w") as f:
            f.write(content)
        
        # Make the script executable
        os.chmod(model_dir / f"convert_{model_name}_checkpoints.py", 0o755)
    
    def _generate_tests(self, model_dir: Path, model_name: str, arch: ModelArchitecture) -> None:
        """Generate comprehensive test files."""
        # Generate backbone tests
        backbone_test_content = f'''import keras
import numpy as np
import pytest

from keras_hub.src.models.{model_name}.{model_name}_backbone import {model_name.title()}Backbone
from keras_hub.src.tests.test_case import TestCase


class {model_name.title()}BackboneTest(TestCase):
    
    def setUp(self):
        self.init_kwargs = {{
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 64,
            "intermediate_dim": 128,
            "max_sequence_length": 128,
        }}
        self.input_data = {{
            "token_ids": np.ones((2, 10), dtype="int32"),
            "padding_mask": np.ones((2, 10), dtype="int32"),
        }}

    def test_backbone_basics(self):
        model = {model_name.title()}Backbone(**self.init_kwargs)
        model(self.input_data)

    def test_saved_model(self):
        model = {model_name.title()}Backbone(**self.init_kwargs)
        model_output = model(self.input_data)
        saved_model_path = self.get_temp_dir()
        keras.saving.save_model(model, saved_model_path)
        loaded_model = keras.saving.load_model(saved_model_path)
        loaded_model_output = loaded_model(self.input_data)

        self.assertAllClose(model_output, loaded_model_output)

    def test_all_preset_load(self):
        # TODO: Add when presets are available
        pass

    def test_backbone_dtypes(self):
        # Test float16 
        model = {model_name.title()}Backbone(**self.init_kwargs, dtype="float16")
        model(self.input_data)

    def test_backbone_compile(self):
        model = {model_name.title()}Backbone(**self.init_kwargs)
        model.compile()
        model(self.input_data)
'''

        with open(model_dir / f"{model_name}_backbone_test.py", "w") as f:
            f.write(backbone_test_content)
        
        # Generate tokenizer tests
        tokenizer_test_content = f'''import pytest

from keras_hub.src.models.{model_name}.{model_name}_tokenizer import {model_name.title()}Tokenizer
from keras_hub.src.tests.test_case import TestCase


class {model_name.title()}TokenizerTest(TestCase):

    def setUp(self):
        self.init_kwargs = {{
            "vocabulary": {{"<|endoftext|>": 0, "Ġthe": 1, "Ġquick": 2, "Ġbrown": 3}},
            "merges": ["Ġ t", "h e", "Ġt he"],
        }}

    def test_tokenizer_basics(self):
        tokenizer = {model_name.title()}Tokenizer(**self.init_kwargs)
        output = tokenizer("the quick brown")
        self.assertIsInstance(output, list)

    def test_tokenizer_decode(self):
        tokenizer = {model_name.title()}Tokenizer(**self.init_kwargs)
        encoded = tokenizer("the quick")
        decoded = tokenizer.detokenize(encoded)
        self.assertIsInstance(decoded, str)

    def test_special_tokens(self):
        tokenizer = {model_name.title()}Tokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.end_token, "<|endoftext|>")
        self.assertEqual(tokenizer.end_token_id, 0)

    def test_tokenizer_vocabulary_size(self):
        tokenizer = {model_name.title()}Tokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.vocabulary_size(), 4)

    # TODO: Add preset loading tests when available
'''

        with open(model_dir / f"{model_name}_tokenizer_test.py", "w") as f:
            f.write(tokenizer_test_content)


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to KerasHub format")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name (e.g., 'microsoft/DialoGPT-medium')")
    parser.add_argument("--output_dir", default="./generated_models", help="Output directory for generated files")
    parser.add_argument("--hf_token", help="HuggingFace API token for private models")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting HuggingFace to KerasHub conversion for: {args.model_name}")
    
    try:
        # Initialize API and analyzer
        api = HuggingFaceAPI(token=args.hf_token)
        analyzer = ArchitectureAnalyzer(api)
        generator = KerasHubGenerator(output_dir=args.output_dir)
        
        # Analyze the model
        architecture = analyzer.analyze_model(args.model_name)
        
        print(f"📋 Model Analysis Summary:")
        print(f"   - Model Type: {architecture.model_type}")
        print(f"   - Attention: {architecture.attention_type}")
        print(f"   - Activation: {architecture.activation_function}")
        print(f"   - Normalization: {architecture.normalization_type}")
        print(f"   - Position Encoding: {architecture.position_encoding}")
        print(f"   - Supports Causal LM: {architecture.supports_causal_lm}")
        
        # Generate all files
        generator.generate_all_files(args.model_name, architecture)
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 Generated files are in: {args.output_dir}")
        print(f"\n📝 Next steps:")
        print(f"   1. Review generated files and customize as needed")
        print(f"   2. Implement tokenizer vocabulary extraction")
        print(f"   3. Fine-tune weight conversion mapping")
        print(f"   4. Test with actual model weights")
        print(f"   5. Add to KerasHub model registry")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()