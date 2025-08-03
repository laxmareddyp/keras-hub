# HuggingFace to KerasHub Model Converter

A comprehensive tool that analyzes HuggingFace model repositories via API calls and generates complete KerasHub-compatible implementation files including backbone, tokenizer, preprocessor, and conversion utilities.

## 🚀 Features

- **Automated Architecture Analysis**: Analyzes HF models via API to detect attention mechanisms, layer structures, activations, and more
- **Complete File Generation**: Generates all necessary KerasHub files including:
  - `{model}_backbone.py` - Core model architecture
  - `{model}_attention.py` - Attention mechanism implementation  
  - `{model}_decoder.py` - Transformer decoder layers
  - `{model}_tokenizer.py` - Tokenizer wrapper
  - `{model}_causal_lm.py` - Causal language model implementation
  - `{model}_preprocessor.py` - Input preprocessing
  - `{model}_config.py` - Model configuration
  - `{model}_presets.py` - Model variant definitions
  - `convert_{model}_checkpoints.py` - Weight conversion script
  - Comprehensive test files
- **Framework Translation**: Converts PyTorch/HF patterns to Keras/KerasHub patterns
- **Weight Conversion**: Generates scripts to convert HF weights to KerasHub format
- **Architecture Detection**: Automatically detects and handles different model architectures

## 📋 Requirements

```bash
pip install requests pathlib dataclasses argparse
```

## 🔧 Usage

### Basic Usage

```bash
python hf_to_kerashub_converter.py --model_name microsoft/DialoGPT-medium
```

### Advanced Usage

```bash
python hf_to_kerashub_converter.py \
    --model_name facebook/opt-1.3b \
    --output_dir ./my_generated_models \
    --hf_token your_hf_token_here
```

### Batch Conversion Example

```bash
python example_usage.py
```

## 📊 Supported Model Types

The converter currently supports analysis and generation for:

- **GPT-style models**: GPT-2, DialoGPT, CodeGPT, etc.
- **BERT-style models**: BERT, DistilBERT, RoBERTa, etc.
- **Modern LLMs**: OPT, GPT-Neo, Llama, Mistral, etc.
- **Specialized models**: CodeBERT, BioBERT, etc.

## 🎯 Generated File Structure

For a model named `my_model`, the converter generates:

```
generated_models/my_model/
├── __init__.py                           # Package initialization
├── my_model_config.py                    # Model configuration class
├── my_model_backbone.py                  # Core backbone implementation
├── my_model_attention.py                 # Attention mechanism
├── my_model_decoder.py                   # Transformer decoder layer
├── my_model_layernorm.py                 # Layer normalization
├── my_model_tokenizer.py                 # Tokenizer wrapper
├── my_model_causal_lm.py                 # Causal language model
├── my_model_causal_lm_preprocessor.py    # Input preprocessor
├── my_model_presets.py                   # Model presets/variants
├── convert_my_model_checkpoints.py       # Weight conversion script
├── my_model_backbone_test.py             # Backbone tests
└── my_model_tokenizer_test.py            # Tokenizer tests
```

## 🔍 Architecture Analysis

The tool automatically detects:

- **Attention Type**: Multi-head, grouped-query, flash attention, etc.
- **Activation Functions**: GELU, ReLU, SiLU, etc.
- **Normalization**: LayerNorm, RMSNorm
- **Position Encoding**: Absolute, RoPE, ALiBi, relative
- **Layer Structure**: Self-attention, feed-forward, cross-attention
- **Model Capabilities**: Causal LM, sequence classification, etc.

## 📝 Example Output

```bash
🚀 Starting HuggingFace to KerasHub conversion for: microsoft/DialoGPT-medium

🔍 Analyzing model: microsoft/DialoGPT-medium

📋 Model Analysis Summary:
   - Model Type: gpt2
   - Attention: multi_head_attention
   - Activation: gelu
   - Normalization: layer_norm
   - Position Encoding: absolute
   - Supports Causal LM: True

📁 Generating files in: ./generated_models/dialogpt_medium

✅ Generated complete KerasHub implementation for dialogpt_medium

🎉 Conversion completed successfully!
📁 Generated files are in: ./generated_models

📝 Next steps:
   1. Review generated files and customize as needed
   2. Implement tokenizer vocabulary extraction
   3. Fine-tune weight conversion mapping
   4. Test with actual model weights
   5. Add to KerasHub model registry
```

## 🛠️ Manual Customization Required

While the converter generates 70-90% of the implementation automatically, some manual work is typically needed:

### 1. Tokenizer Implementation
- Extract vocabulary from HuggingFace tokenizer
- Handle BPE merge rules properly  
- Implement special token handling

### 2. Weight Conversion Fine-tuning
- Adjust layer name mappings between HF and KerasHub
- Handle architecture-specific weight reshaping
- Test weight conversion accuracy

### 3. Architecture-specific Logic
- Custom attention mechanisms
- Unique activation functions
- Special preprocessing requirements

## 🧪 Testing Generated Models

Each generated model includes test files. To run tests:

```bash
cd generated_models/my_model
python -m pytest my_model_backbone_test.py -v
python -m pytest my_model_tokenizer_test.py -v
```

## 🔄 Weight Conversion Process

1. **Use the generated conversion script**:
   ```bash
   cd generated_models/my_model
   python convert_my_model_checkpoints.py --preset my_model_base_en
   ```

2. **Manual steps required**:
   - Implement tokenizer vocabulary extraction
   - Fine-tune weight layer mappings
   - Test output equivalence with HF model

## 📚 Integration with KerasHub

To integrate the generated model with KerasHub:

1. **Copy files to KerasHub source**:
   ```bash
   cp -r generated_models/my_model keras_hub/src/models/
   ```

2. **Update KerasHub registry**:
   - Add imports to `keras_hub/src/models/__init__.py`
   - Register model classes in appropriate modules

3. **Add to preset system**:
   - Upload weights to Kaggle/HF Hub
   - Update preset configurations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Enhanced Architecture Detection**: Support for more model types
- **Better Weight Mapping**: More sophisticated conversion logic
- **Tokenizer Automation**: Automatic vocabulary extraction
- **Testing Framework**: More comprehensive test generation

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace team for the Transformers library
- Google KerasHub team for the KerasHub framework
- Open source community for model implementations

## 🐛 Issues and Support

If you encounter issues:

1. Check that the HuggingFace model is publicly accessible
2. Verify the model has a standard configuration format
3. Review generated files for any obvious errors
4. Open an issue with the model name and error details

---

**Note**: This tool generates starting implementations that require manual refinement. The generated code should be reviewed and tested before production use.
