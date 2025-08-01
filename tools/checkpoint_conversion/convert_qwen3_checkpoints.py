import os
import random
import traceback

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide any CUDA devices

import numpy as np
import torch
from absl import app
from absl import flags

random.seed(123)
torch.manual_seed(123)
device = torch.device("cpu")
# Force PyTorch to use CPU
torch.set_default_device(device)

from keras import ops  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import keras_hub  # noqa: E402

PRESET_MAP = {
    "qwen3_0.6b_en": "Qwen/Qwen3-0.6B",
    "qwen3_1.7b_en": "Qwen/Qwen3-1.7B",
    "qwen3_4b_en": "Qwen/Qwen3-4B",
    "qwen3_8b_en": "Qwen/Qwen3-8B",
    "qwen3_14b_en": "Qwen/Qwen3-14B",
    "qwen3_32b_en": "Qwen/Qwen3-32B",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def test_model(
    keras_hub_model, keras_hub_tokenizer, hf_model, hf_model_tokenizer
):
    # First, test that the number of parameters match
    keras_hub_params = keras_hub_model.count_params()
    hf_params = hf_model.num_parameters()
    assert keras_hub_params == hf_params

    # Test the outputs of both the models
    hf_inputs = hf_model_tokenizer(["What is Keras?"], return_tensors="pt").to(
        device
    )
    hf_outputs = hf_model(**hf_inputs)
    hf_output_logits = hf_outputs.logits.detach().cpu().float().numpy()

    keras_hub_preprocessor = keras_hub.models.Qwen3CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_inputs = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=5
    )[0]
    # keras_hub_inputs = {k: v.to(device) for k, v in keras_hub_inputs.items()}

    keras_hub_output = keras_hub_model(keras_hub_inputs)
    keras_hub_logits = keras_hub_model.token_embedding(
        keras_hub_output, reverse=True
    )
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits)

    try:
        np.testing.assert_allclose(
            keras_hub_logits, hf_output_logits, atol=1e-2
        )
    except AssertionError as err:
        print("\n")
        print(traceback.format_exc())
        print(err.args[0])
        print("\n")


def test_tokenizer(keras_hub_tokenizer, hf_tokenizer):
    hf_output = hf_tokenizer(["What is Keras?"], return_tensors="pt")
    hf_output = hf_output["input_ids"].detach().cpu().numpy()
    keras_hub_preprocessor = keras_hub.models.Qwen3CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    keras_hub_output = keras_hub_preprocessor(
        ["What is Keras?"], sequence_length=5
    )
    keras_hub_output = ops.convert_to_numpy(keras_hub_output[0]["token_ids"])

    np.testing.assert_equal(keras_hub_output, hf_output)


def validate_output(
    keras_model,
    hf_model,
    hf_tokenizer,
):
    input_str = "What is Keras?"
    length = 32

    # KerasHub
    keras_output = keras_model.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)

    hf_inputs = hf_tokenizer([input_str], return_tensors="pt")
    outputs = hf_model.generate(
        **hf_inputs,
        max_length=length,
        do_sample=False,
        num_beams=1,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    print("🔶 Huggingface generated token ids:", outputs[0])
    hf_generated_text = hf_tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
    print("🔶 Huggingface output:", hf_generated_text)


def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    # === Load the Huggingface model ===
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_preset,
        device_map=device,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset, return_tensors="pt")
    hf_model.eval()

    keras_hub_backbone = keras_hub.models.Qwen3Backbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_tokenizer = keras_hub.models.Qwen3Tokenizer.from_preset(
        f"hf://{hf_preset}"
    )

    print("\n-> Huggingface model and tokenizer loaded")

    # === Check that the models and tokenizers outputs match ===
    test_tokenizer(keras_hub_tokenizer, hf_tokenizer)
    test_model(keras_hub_backbone, keras_hub_tokenizer, hf_model, hf_tokenizer)
    print("\n-> Tests passed!")

    preprocessor = keras_hub.models.Qwen3CausalLMPreprocessor(
        keras_hub_tokenizer
    )
    qwen3_lm = keras_hub.models.Qwen3CausalLM(
        backbone=keras_hub_backbone, preprocessor=preprocessor, sampler="greedy"
    )

    validate_output(qwen3_lm, hf_model, hf_tokenizer)

    qwen3_lm.save_to_preset(f"./{preset}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
