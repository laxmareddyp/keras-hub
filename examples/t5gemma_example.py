#!/usr/bin/env python3
"""
T5Gemma Model Example

This script demonstrates how to use the T5Gemma model for sequence-to-sequence tasks.
"""

import numpy as np
import keras
from keras_hub.models import T5GemmaBackbone


def basic_usage_example():
    """Demonstrate basic usage of T5Gemma model."""
    print("=== T5Gemma Basic Usage Example ===")
    
    # Create a small T5Gemma model for demonstration
    model = T5GemmaBackbone(
        vocabulary_size=32000,
        num_layers=4,
        num_heads=8,
        head_dim=64,
        hidden_dim=512,
        intermediate_dim=2048,
        use_sliding_window_attention=True,
        sliding_window_size=1024,
    )
    
    print(f"Model created with {model.count_params():,} parameters")
    
    # Prepare sample input data
    batch_size = 2
    encoder_seq_length = 10
    decoder_seq_length = 8
    
    # Create random token IDs (in practice, you'd use a proper tokenizer)
    encoder_token_ids = np.random.randint(0, 32000, (batch_size, encoder_seq_length))
    encoder_padding_mask = np.ones((batch_size, encoder_seq_length))
    decoder_token_ids = np.random.randint(0, 32000, (batch_size, decoder_seq_length))
    decoder_padding_mask = np.ones((batch_size, decoder_seq_length))
    
    # Set some padding tokens to 0
    encoder_padding_mask[:, 8:] = 0
    decoder_padding_mask[:, 6:] = 0
    
    inputs = {
        "encoder_token_ids": encoder_token_ids,
        "encoder_padding_mask": encoder_padding_mask,
        "decoder_token_ids": decoder_token_ids,
        "decoder_padding_mask": decoder_padding_mask,
    }
    
    # Run inference
    outputs = model(inputs)
    
    print(f"Input shapes:")
    print(f"  Encoder tokens: {encoder_token_ids.shape}")
    print(f"  Decoder tokens: {decoder_token_ids.shape}")
    print(f"Output shapes:")
    print(f"  Encoder output: {outputs['encoder_sequence_output'].shape}")
    print(f"  Decoder output: {outputs['decoder_sequence_output'].shape}")
    
    # Show some statistics
    encoder_output = outputs['encoder_sequence_output']
    decoder_output = outputs['decoder_sequence_output']
    
    print(f"\nOutput statistics:")
    print(f"  Encoder output mean: {np.mean(encoder_output):.4f}")
    print(f"  Encoder output std: {np.std(encoder_output):.4f}")
    print(f"  Decoder output mean: {np.mean(decoder_output):.4f}")
    print(f"  Decoder output std: {np.std(decoder_output):.4f}")
    
    return model, outputs


def training_example():
    """Demonstrate training a T5Gemma model."""
    print("\n=== T5Gemma Training Example ===")
    
    # Create a small model for training
    model = T5GemmaBackbone(
        vocabulary_size=32000,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        hidden_dim=256,
        intermediate_dim=1024,
        dropout=0.1,
    )
    
    print(f"Training model with {model.count_params():,} parameters")
    
    # For demonstration, we'll create a simple training setup
    # In practice, you'd want to create a proper Seq2Seq task model
    print("Note: This is a demonstration of the backbone model.")
    print("For actual training, you would typically:")
    print("1. Create a Seq2SeqLM task model using this backbone")
    print("2. Use proper sequence-to-sequence loss functions")
    print("3. Implement teacher forcing and proper decoding")
    
    # Generate synthetic training data
    batch_size = 4
    encoder_seq_length = 32
    decoder_seq_length = 16
    
    # Create random training data
    encoder_token_ids = np.random.randint(0, 32000, (batch_size, encoder_seq_length))
    encoder_padding_mask = np.ones((batch_size, encoder_seq_length))
    decoder_token_ids = np.random.randint(0, 32000, (batch_size, decoder_seq_length))
    decoder_padding_mask = np.ones((batch_size, decoder_seq_length))
    
    # Test the model with sample data
    inputs = {
        "encoder_token_ids": encoder_token_ids,
        "encoder_padding_mask": encoder_padding_mask,
        "decoder_token_ids": decoder_token_ids,
        "decoder_padding_mask": decoder_padding_mask,
    }
    
    outputs = model(inputs)
    
    print(f"Model test successful!")
    print(f"Encoder output shape: {outputs['encoder_sequence_output'].shape}")
    print(f"Decoder output shape: {outputs['decoder_sequence_output'].shape}")
    
    # Show that the model can process different sequence lengths
    print(f"\nTesting with different sequence lengths...")
    
    # Test with shorter sequences
    short_encoder = np.random.randint(0, 32000, (batch_size, 8))
    short_decoder = np.random.randint(0, 32000, (batch_size, 6))
    
    short_inputs = {
        "encoder_token_ids": short_encoder,
        "encoder_padding_mask": np.ones((batch_size, 8)),
        "decoder_token_ids": short_decoder,
        "decoder_padding_mask": np.ones((batch_size, 6)),
    }
    
    short_outputs = model(short_inputs)
    print(f"Short sequences - Encoder: {short_outputs['encoder_sequence_output'].shape}")
    print(f"Short sequences - Decoder: {short_outputs['decoder_sequence_output'].shape}")
    
    return model


def model_configuration_example():
    """Demonstrate different model configurations."""
    print("\n=== T5Gemma Configuration Examples ===")
    
    # Example 1: Small model with sliding window attention
    small_model = T5GemmaBackbone(
        vocabulary_size=32000,
        num_layers=6,
        num_heads=8,
        head_dim=64,
        hidden_dim=512,
        intermediate_dim=2048,
        use_sliding_window_attention=True,
        sliding_window_size=1024,
        attention_logit_soft_cap=50,
    )
    
    # Example 2: Medium model with post-normalization
    medium_model = T5GemmaBackbone(
        vocabulary_size=32000,
        num_layers=12,
        num_heads=12,
        head_dim=64,
        hidden_dim=768,
        intermediate_dim=3072,
        use_post_attention_norm=True,
        use_post_ffw_norm=True,
        activation="gelu",
        dropout=0.1,
    )
    
    # Example 3: Large model with all Gemma 2 features
    large_model = T5GemmaBackbone(
        vocabulary_size=32000,
        num_layers=24,
        num_heads=16,
        head_dim=128,
        hidden_dim=1024,
        intermediate_dim=4096,
        use_sliding_window_attention=True,
        sliding_window_size=4096,
        attention_logit_soft_cap=100,
        use_post_attention_norm=True,
        use_post_ffw_norm=True,
        query_head_dim_normalize=True,
        activation="gelu",
        dropout=0.1,
        layer_norm_epsilon=1e-6,
    )
    
    print(f"Small model: {small_model.count_params():,} parameters")
    print(f"Medium model: {medium_model.count_params():,} parameters")
    print(f"Large model: {large_model.count_params():,} parameters")
    
    # Test all models with sample input
    batch_size = 1
    encoder_seq_length = 8
    decoder_seq_length = 6
    
    encoder_token_ids = np.random.randint(0, 32000, (batch_size, encoder_seq_length))
    encoder_padding_mask = np.ones((batch_size, encoder_seq_length))
    decoder_token_ids = np.random.randint(0, 32000, (batch_size, decoder_seq_length))
    decoder_padding_mask = np.ones((batch_size, decoder_seq_length))
    
    inputs = {
        "encoder_token_ids": encoder_token_ids,
        "encoder_padding_mask": encoder_padding_mask,
        "decoder_token_ids": decoder_token_ids,
        "decoder_padding_mask": decoder_padding_mask,
    }
    
    for name, model in [("Small", small_model), ("Medium", medium_model), ("Large", large_model)]:
        outputs = model(inputs)
        print(f"{name} model output shapes:")
        print(f"  Encoder: {outputs['encoder_sequence_output'].shape}")
        print(f"  Decoder: {outputs['decoder_sequence_output'].shape}")


def main():
    """Run all T5Gemma examples."""
    print("T5Gemma Model Examples")
    print("=" * 50)
    
    try:
        # Basic usage
        model, outputs = basic_usage_example()
        
        # Training example
        trained_model = training_example()
        
        # Configuration examples
        model_configuration_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Encoder-decoder architecture")
        print("- RMS normalization")
        print("- Sliding window attention")
        print("- Query normalization")
        print("- Post-attention and post-ffw normalization")
        print("- Custom model configurations")
        print("- Training and inference")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 