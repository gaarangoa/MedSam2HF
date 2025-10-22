from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_hf_download(
    repo_or_path: Optional[Union[str, Path]],
    filename: str,
    *,
    local_files_only: bool = False,
) -> Optional[Path]:
    if repo_or_path is None:
        return None
    repo_path = Path(repo_or_path)
    if repo_path.is_file() and repo_path.name == filename:
        return repo_path
    if repo_path.is_dir():
        candidate = repo_path / filename
        if candidate.exists():
            return candidate
    try:
        resolved = hf_hub_download(
            repo_id=str(repo_or_path),
            filename=filename,
            local_files_only=local_files_only,
        )
        return Path(resolved)
    except Exception:
        return None


def _resize_grayscale_volume_to_rgb(array: np.ndarray, image_size: int) -> np.ndarray:
    depth, _, _ = array.shape
    resized = np.zeros((depth, 3, image_size, image_size), dtype=np.float32)
    for idx in range(depth):
        img = Image.fromarray(array[idx].astype(np.uint8))
        img_rgb = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        resized[idx] = np.asarray(img_rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return resized


class MedSam2Tokenizer:
    """
    Hugging Face-style tokenizer that prepares MedSAM2 inputs.

    The tokenizer takes care of reading the NIfTI volume, applying the
    DeepLesion windowing, resizing the frames, normalising them, and
    packaging everything as tensors ready to be consumed by the model.
    """

    config_filename = "tokenizer_config.json"

    def __init__(
        self,
        *,
        image_size: int = 512,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        if dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(f"Unsupported dtype '{dtype}'.")
        self.image_size = int(image_size)
        self.device = torch.device(device if device else "cpu")
        self.dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32)[:, None, None])
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32)[:, None, None])
        self.pretrained_model_name_or_path: Optional[Union[str, Path]] = None

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    @classmethod
    def _default_config(cls) -> Dict[str, Any]:
        default_path = Path(__file__).with_name(cls.config_filename)
        return _load_json(default_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, Path]] = None,
        *,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> "MedSam2Tokenizer":
        config = cls._default_config()
        if pretrained_model_name_or_path is not None:
            cfg_path = _maybe_hf_download(
                pretrained_model_name_or_path,
                cls.config_filename,
                local_files_only=local_files_only,
            )
            if cfg_path is not None:
                config.update(_load_json(cfg_path))
        config.update(kwargs)
        tokenizer = cls(**config)
        tokenizer.pretrained_model_name_or_path = pretrained_model_name_or_path
        return tokenizer

    def __call__(
        self,
        nii_path: Union[str, Path],
        *,
        dataset_info: Optional[Union[str, Path, pd.DataFrame]] = None,
        case_selector: Optional[Dict[str, Any]] = None,
        bbox: Optional[Union[np.ndarray, torch.Tensor, Tuple[int, ...], Tuple[float, ...], list]] = None,
        slice_range: Optional[Tuple[int, int]] = None,
        key_slice_index: Optional[int] = None,
        dicom_window: Optional[Tuple[float, float]] = None,
        return_tensors: str = "pt",
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        nii_path = Path(nii_path)
        if not nii_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nii_path}")

        (
            bbox_xyxy,
            (slice_idx_start, slice_idx_end),
            key_slice_index,
            window_bounds,
            case_metadata,
        ) = self._resolve_metadata(
            nii_path,
            dataset_info=dataset_info,
            case_selector=case_selector,
            bbox=bbox,
            slice_range=slice_range,
            key_slice_index=key_slice_index,
            dicom_window=dicom_window,
        )

        reference_image = sitk.ReadImage(str(nii_path))
        volume = sitk.GetArrayFromImage(reference_image)

        lower, upper = window_bounds
        clipped = np.clip(volume, lower, upper)
        min_val = float(clipped.min())
        max_val = float(clipped.max())
        if max_val - min_val < 1e-6:
            normalised = np.zeros_like(clipped, dtype=np.uint8)
        else:
            normalised = ((clipped - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

        key_slice_offset = int(key_slice_index - slice_idx_start)
        if not 0 <= key_slice_offset < normalised.shape[0]:
            raise ValueError("Key slice index is outside of the requested slice range.")

        resized = _resize_grayscale_volume_to_rgb(normalised, self.image_size)
        pixel_values = torch.from_numpy(resized)
        if return_tensors == "pt":
            target_device = torch.device(device) if device is not None else self.device
            pixel_values = pixel_values.to(dtype=torch.float32, device=target_device)
            mean = self.mean.to(device=target_device, dtype=torch.float32)
            std = self.std.to(device=target_device, dtype=torch.float32)
            pixel_values.sub_(mean).div_(std)
            pixel_values = pixel_values.to(self.dtype)
        elif return_tensors is None:
            pixel_values = resized
        else:
            raise ValueError(f"Unsupported return_tensors '{return_tensors}'.")

        bbox_tensor = torch.as_tensor(bbox_xyxy, dtype=torch.float32)
        if return_tensors == "pt":
            bbox_tensor = bbox_tensor.to(device=pixel_values.device)

        video_height = int(normalised.shape[1])
        video_width = int(normalised.shape[2])

        return {
            "pixel_values": pixel_values,
            "video_height": video_height,
            "video_width": video_width,
            "key_slice_idx_offset": key_slice_offset,
            "bbox": bbox_tensor,
            "preprocessed_volume": normalised,
            "reference_image": reference_image,
            "slice_range": (int(slice_idx_start), int(slice_idx_end)),
            "key_slice_index": int(key_slice_index),
            "case_metadata": case_metadata,
        }

    def _resolve_metadata(
        self,
        nii_path: Path,
        *,
        dataset_info: Optional[Union[str, Path, pd.DataFrame]],
        case_selector: Optional[Dict[str, Any]],
        bbox: Optional[Union[np.ndarray, torch.Tensor, Tuple[int, ...], Tuple[float, ...], list]],
        slice_range: Optional[Tuple[int, int]],
        key_slice_index: Optional[int],
        dicom_window: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, Tuple[int, int], int, Tuple[float, float], Dict[str, Any]]:
        if dataset_info is not None:
            df = self._load_dataset_info(dataset_info)
            row = self._match_deeplesion_row(nii_path.name, df, case_selector)
            bbox = self._parse_bbox(row["Bounding_boxes"])
            slice_range = self._parse_slice_range(row["Slice_range"])
            key_slice_index = int(row["Key_slice_index"])
            dicom_window = self._parse_window(row["DICOM_windows"])
            metadata = row.to_dict()
        else:
            if bbox is None or slice_range is None or key_slice_index is None or dicom_window is None:
                raise ValueError(
                    "bbox, slice_range, key_slice_index, and dicom_window must be provided when dataset_info is missing."
                )
            bbox = np.asarray(bbox, dtype=np.float32)
            metadata = {"nii_name": nii_path.name}

        bbox_xyxy = np.array([bbox[1], bbox[0], bbox[3], bbox[2]], dtype=np.float32)
        return bbox_xyxy, slice_range, int(key_slice_index), dicom_window, metadata

    @staticmethod
    def _load_dataset_info(dataset_info: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(dataset_info, pd.DataFrame):
            return dataset_info
        dataset_path = Path(dataset_info)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset info file not found: {dataset_path}")
        return pd.read_csv(dataset_path)

    @staticmethod
    def _match_deeplesion_row(
        nii_name: str,
        df: pd.DataFrame,
        case_selector: Optional[Dict[str, Any]],
    ) -> pd.Series:
        if case_selector is not None:
            if "index" in case_selector:
                return df.loc[int(case_selector["index"])]
            if "File_name" in case_selector:
                matches = df[df["File_name"] == case_selector["File_name"]]
                if matches.empty:
                    raise ValueError("No rows match the provided File_name selector.")
                return matches.iloc[0]

        range_match = re.findall(r"\d{3}-\d{3}", nii_name)
        case_match = re.findall(r"^(\d{6}_\d{2}_\d{2})", nii_name)
        if not range_match or not case_match:
            raise ValueError("Cannot infer case identifiers from NIfTI filename.")

        slice_range = ", ".join(str(int(val)) for val in range_match[0].split("-"))
        case_name = case_match[0]

        matches = df[
            df["File_name"].str.contains(case_name, na=False)
            & df["Slice_range"].str.contains(slice_range, na=False)
        ]
        if matches.empty:
            raise ValueError("No dataset info rows match the detected case identifiers.")
        return matches.iloc[0]

    @staticmethod
    def _parse_bbox(raw_bbox: str) -> np.ndarray:
        coords = [float(coord) for coord in raw_bbox.split(",")]
        if len(coords) != 4:
            raise ValueError("Bounding box must contain four coordinates.")
        return np.array(coords, dtype=np.float32)

    @staticmethod
    def _parse_slice_range(raw: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(raw, tuple):
            return int(raw[0]), int(raw[1])
        start, end = raw.split(",")
        return int(float(start)), int(float(end))

    @staticmethod
    def _parse_window(raw: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
        if isinstance(raw, tuple):
            return float(raw[0]), float(raw[1])
        lower, upper = raw.split(",")
        return float(lower), float(upper)
