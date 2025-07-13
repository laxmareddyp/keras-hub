import numpy as np
import unittest
import keras

from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone


class T5GemmaBackboneTest(unittest.TestCase):
    def test_backbone_basics(self):
        """Test the T5Gemma backbone basic functionality."""
        model = T5GemmaBackbone(
            vocabulary_size=32000,
            num_layers=6,
            num_heads=8,
            head_dim=64,
            hidden_dim=512,
            intermediate_dim=2048,
            use_sliding_window_attention=False,  # Disable for testing
        )
        
        # Test input shapes
        batch_size = 2
        encoder_seq_length = 10
        decoder_seq_length = 8
        
        encoder_token_ids = np.random.randint(
            0, 32000, (batch_size, encoder_seq_length)
        )
        encoder_padding_mask = np.ones((batch_size, encoder_seq_length))
        decoder_token_ids = np.random.randint(
            0, 32000, (batch_size, decoder_seq_length)
        )
        decoder_padding_mask = np.ones((batch_size, decoder_seq_length))
        
        inputs = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
        
        outputs = model(inputs)
        
        # Check output shapes
        self.assertEqual(
            outputs["encoder_sequence_output"].shape,
            (batch_size, encoder_seq_length, 512),
        )
        self.assertEqual(
            outputs["decoder_sequence_output"].shape,
            (batch_size, decoder_seq_length, 512),
        )
        
        # Check that outputs have the expected shape and are not None
        self.assertIsNotNone(outputs["encoder_sequence_output"])
        self.assertIsNotNone(outputs["decoder_sequence_output"])
    
    def test_backbone_with_preset(self):
        """Test the T5Gemma backbone with preset configuration."""
        # For now, test with a custom model since presets aren't set up yet
        model = T5GemmaBackbone(
            vocabulary_size=32000,
            num_layers=6,
            num_heads=8,
            head_dim=64,
            hidden_dim=512,
            intermediate_dim=2048,
            use_sliding_window_attention=False,  # Disable for testing
        )
        
        batch_size = 1
        encoder_seq_length = 12
        decoder_seq_length = 8
        
        encoder_token_ids = np.random.randint(
            0, 32000, (batch_size, encoder_seq_length)
        )
        encoder_padding_mask = np.ones((batch_size, encoder_seq_length))
        decoder_token_ids = np.random.randint(
            0, 32000, (batch_size, decoder_seq_length)
        )
        decoder_padding_mask = np.ones((batch_size, decoder_seq_length))
        
        inputs = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
        
        outputs = model(inputs)
        
        # Check output shapes
        self.assertEqual(
            outputs["encoder_sequence_output"].shape,
            (batch_size, encoder_seq_length, 512),
        )
        self.assertEqual(
            outputs["decoder_sequence_output"].shape,
            (batch_size, decoder_seq_length, 512),
        )
    
    def test_backbone_config(self):
        """Test that the backbone configuration is preserved."""
        model = T5GemmaBackbone(
            vocabulary_size=32000,
            num_layers=6,
            num_heads=8,
            head_dim=64,
            hidden_dim=512,
            intermediate_dim=2048,
            use_sliding_window_attention=False,  # Disable for testing
            sliding_window_size=2048,
            attention_logit_soft_cap=50,
            use_post_attention_norm=True,
            use_post_ffw_norm=True,
            activation="gelu",
            dropout=0.1,
            layer_norm_epsilon=1e-6,
            tie_embedding_weights=True,
        )
        
        config = model.get_config()
        
        self.assertEqual(config["vocabulary_size"], 32000)
        self.assertEqual(config["num_layers"], 6)
        self.assertEqual(config["num_heads"], 8)
        self.assertEqual(config["head_dim"], 64)
        self.assertEqual(config["hidden_dim"], 512)
        self.assertEqual(config["intermediate_dim"], 2048)
        self.assertFalse(config["use_sliding_window_attention"])  # We set it to False in the test
        self.assertEqual(config["sliding_window_size"], 2048)
        self.assertEqual(config["attention_logit_soft_cap"], 50)
        self.assertTrue(config["use_post_attention_norm"])
        self.assertTrue(config["use_post_ffw_norm"])
        self.assertEqual(config["activation"], "gelu")
        self.assertEqual(config["dropout"], 0.1)
        self.assertEqual(config["layer_norm_epsilon"], 1e-6)
        self.assertTrue(config["tie_embedding_weights"])
    
    def test_backbone_with_different_activations(self):
        """Test the backbone with different activation functions."""
        activations = ["gelu", "relu", "swish"]
        
        for activation in activations:
            model = T5GemmaBackbone(
                vocabulary_size=32000,
                num_layers=2,
                num_heads=4,
                head_dim=32,
                hidden_dim=256,
                intermediate_dim=1024,
                activation=activation,
            )
            
            batch_size = 1
            encoder_seq_length = 8
            decoder_seq_length = 6
            
            encoder_token_ids = np.random.randint(
                0, 32000, (batch_size, encoder_seq_length)
            )
            encoder_padding_mask = np.ones((batch_size, encoder_seq_length))
            decoder_token_ids = np.random.randint(
                0, 32000, (batch_size, decoder_seq_length)
            )
            decoder_padding_mask = np.ones((batch_size, decoder_seq_length))
            
            inputs = {
                "encoder_token_ids": encoder_token_ids,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_token_ids": decoder_token_ids,
                "decoder_padding_mask": decoder_padding_mask,
            }
            
            outputs = model(inputs)
            
            # Check that outputs are valid
            self.assertEqual(
                outputs["encoder_sequence_output"].shape,
                (batch_size, encoder_seq_length, 256),
            )
            self.assertEqual(
                outputs["decoder_sequence_output"].shape,
                (batch_size, decoder_seq_length, 256),
            ) 