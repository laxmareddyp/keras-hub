"""Preprocessor for OpenAI Privacy Filter models."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.models.preprocessor import Preprocessor


@keras_hub_export(
    "keras_hub.models.OpenAIPrivacyFilterPreprocessor",
)
class OpenAIPrivacyFilterPreprocessor(Preprocessor):
    """Preprocessor for OpenAI Privacy Filter token classification.

    This preprocessing layer tokenizes and pads/truncates input text into
    `token_ids` and `padding_mask` tensors suitable for the backbone.

    Args:
        tokenizer: A `OpenAIPrivacyFilterTokenizer` instance.
        sequence_length: int. The length of the packed inputs.
            Defaults to `512`.
        truncate: string. The algorithm to truncate a list of batched
            segments. Defaults to `"round_robin"`.
    """

    backbone_cls = OpenAIPrivacyFilterBackbone
    tokenizer_cls = OpenAIPrivacyFilterTokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self.sequence_length = sequence_length
        self.truncate = truncate

    def build(self, input_shape):
        super().build(input_shape)
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.pad_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length
        token_ids, segment_ids = self.packer(
            self.tokenizer(x), sequence_length=sequence_length
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
        return x, y, sample_weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "truncate": self.truncate,
            }
        )
        return config
