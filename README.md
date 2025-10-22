# MedSAM2 HF

This package wraps the MedSAM2 preprocessing and segmentation flow with a Hugging Face compatible API. It exposes:

- `MedSam2Tokenizer.from_pretrained(...)` to load preprocessing defaults and convert DeepLesion style inputs into tensors.
- `MedSam2ForSegmentation.from_pretrained(...)` to build the video predictor and generate volumetric masks.

Both components fall back to Hugging Face Hub assets (`wanglab/MedSAM2`) when local checkpoints or configuration files are not available.

## Installation

```
pip install .
```

Or, for editable development:

```
pip install -e .
```

## Quickstart

```python
from medsam2_hf import MedSam2Tokenizer, MedSam2ForSegmentation

tokenizer = MedSam2Tokenizer.from_pretrained("medsam2_hf")
model = MedSam2ForSegmentation.from_pretrained("medsam2_hf")

inputs = tokenizer("path/to/case.nii.gz", dataset_info="DeepLesion_Dataset_Info.csv")
output = model(**inputs)
```
