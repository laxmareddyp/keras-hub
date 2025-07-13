"""T5Gemma model preset configurations."""

backbone_presets = {
    "t5gemma_2b": {
        "metadata": {
            "description": (
                "2B parameter T5Gemma model combining T5 encoder-decoder "
                "architecture with Gemma 2 improvements including RMS normalization "
                "and sliding window attention."
            ),
            "params": 2000000000,
            "path": "t5gemma",
        },
        "kaggle_handle": "kaggle://keras/t5gemma/keras/t5gemma_2b/1",
    },
    "t5gemma_7b": {
        "metadata": {
            "description": (
                "7B parameter T5Gemma model combining T5 encoder-decoder "
                "architecture with Gemma 2 improvements including RMS normalization "
                "and sliding window attention."
            ),
            "params": 7000000000,
            "path": "t5gemma",
        },
        "kaggle_handle": "kaggle://keras/t5gemma/keras/t5gemma_7b/1",
    },
    "t5gemma_27b": {
        "metadata": {
            "description": (
                "27B parameter T5Gemma model combining T5 encoder-decoder "
                "architecture with Gemma 2 improvements including RMS normalization "
                "and sliding window attention."
            ),
            "params": 27000000000,
            "path": "t5gemma",
        },
        "kaggle_handle": "kaggle://keras/t5gemma/keras/t5gemma_27b/1",
    },
} 