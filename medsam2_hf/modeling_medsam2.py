from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
from huggingface_hub import hf_hub_download
from skimage import measure

from sam2.build_sam import build_sam2_video_predictor_npz


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


def _largest_connected_component(segmentation: np.ndarray) -> np.ndarray:
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return (labels == largest_label).astype(np.uint8)


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return mapping[name]


@dataclass
class MedSam2SegmentationOutput:
    mask_volume: np.ndarray
    logits_volume: Optional[np.ndarray] = None
    sitk_mask: Optional[sitk.Image] = None
    sitk_image: Optional[sitk.Image] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    largest_component_applied: bool = False


class MedSam2ForSegmentation(torch.nn.Module):
    """
    Hugging Face-style wrapper around the MedSAM2 predictor.

    The forward pass expects the dictionary returned by ``MedSam2Tokenizer``
    and yields a ``MedSam2SegmentationOutput`` with the predicted mask volume.
    """

    config_filename = "model_config.json"

    def __init__(
        self,
        *,
        model_cfg: Union[str, Path],
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        predict_dtype: str = "bfloat16",
        postprocess_largest_cc: bool = True,
    ) -> None:
        super().__init__()
        self.model_cfg = Path(model_cfg)
        self.checkpoint_path = Path(checkpoint_path)
        requested_device = torch.device(device if device else "cpu")
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.predict_dtype = _resolve_dtype(predict_dtype)
        self.postprocess_largest_cc = postprocess_largest_cc
        self.predictor = build_sam2_video_predictor_npz(str(self.model_cfg), str(self.checkpoint_path))
        self.pretrained_model_name_or_path: Optional[Union[str, Path]] = None

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
    ) -> "MedSam2ForSegmentation":
        config = cls._default_config()
        base_path: Optional[Path] = None
        if pretrained_model_name_or_path is not None:
            resolved_config = _maybe_hf_download(
                pretrained_model_name_or_path,
                cls.config_filename,
                local_files_only=local_files_only,
            )
            if resolved_config is not None:
                config.update(_load_json(resolved_config))
            repo_path = Path(pretrained_model_name_or_path)
            if repo_path.exists():
                base_path = repo_path if repo_path.is_dir() else repo_path.parent
        config.update(kwargs)

        model_cfg_value = config.pop("model_cfg", None)
        model_cfg_filename = config.pop("model_cfg_filename", None)
        model_cfg_repo_id = config.pop("model_cfg_repo_id", None)
        config["model_cfg"] = cls._resolve_model_cfg(
            model_cfg_value,
            model_cfg_filename,
            pretrained_model_name_or_path,
            base_path,
            model_cfg_repo_id,
            local_files_only=local_files_only,
        )
        checkpoint_path = config.pop("checkpoint_path", None)
        if checkpoint_path is not None:
            config["checkpoint_path"] = cls._resolve_checkpoint_path(
                checkpoint_path,
                pretrained_model_name_or_path,
                base_path,
                local_files_only=local_files_only,
            )
        else:
            checkpoint_filename = config.pop("checkpoint_filename", None)
            if checkpoint_filename is None:
                raise ValueError("checkpoint_filename must be provided when checkpoint_path is absent.")
            checkpoint_repo_id = config.pop("checkpoint_repo_id", None)
            config["checkpoint_path"] = cls._resolve_checkpoint_file(
                checkpoint_filename,
                pretrained_model_name_or_path,
                base_path,
                checkpoint_repo_id,
                local_files_only=local_files_only,
            )

        model = cls(**config)
        model.pretrained_model_name_or_path = pretrained_model_name_or_path
        return model

    @staticmethod
    def _resolve_model_cfg(
        model_cfg: Optional[Union[str, Path]],
        model_cfg_filename: Optional[str],
        repo_or_path: Optional[Union[str, Path]],
        base_path: Optional[Path],
        model_cfg_repo_id: Optional[str],
        local_files_only: bool,
    ) -> Path:
        candidates = []
        if model_cfg is not None:
            candidates.append(Path(model_cfg))
        if model_cfg_filename is not None:
            candidates.append(Path(model_cfg_filename))

        for candidate in candidates:
            if candidate.is_file():
                return candidate
            if base_path is not None:
                in_base = (base_path / candidate).resolve()
                if in_base.exists():
                    return in_base
                in_parent = (base_path.parent / candidate).resolve()
                if in_parent.exists():
                    return in_parent

        repo_id = model_cfg_repo_id or (str(repo_or_path) if repo_or_path else None)
        filename = None
        if candidates:
            filename = candidates[0].name
        if filename is not None and repo_id is not None:
            resolved = _maybe_hf_download(
                repo_id,
                filename,
                local_files_only=local_files_only,
            )
            if resolved is not None:
                return resolved

        raise FileNotFoundError(
            "Model configuration file could not be resolved. Provide a valid 'model_cfg' path or set "
            "'model_cfg_filename' with an accessible Hugging Face repo."
        )

    @staticmethod
    def _resolve_checkpoint_path(
        checkpoint: Union[str, Path],
        repo_or_path: Optional[Union[str, Path]],
        base_path: Optional[Path],
        local_files_only: bool,
    ) -> Path:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_file():
            return checkpoint_path
        if base_path is not None:
            candidate = (base_path / checkpoint_path).resolve()
            if candidate.exists():
                return candidate
            parent_candidate = (base_path.parent / checkpoint_path).resolve()
            if parent_candidate.exists():
                return parent_candidate
        resolved = _maybe_hf_download(
            repo_or_path,
            checkpoint_path.name,
            local_files_only=local_files_only,
        )
        if resolved is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return resolved

    @staticmethod
    def _resolve_checkpoint_file(
        filename: str,
        repo_or_path: Optional[Union[str, Path]],
        base_path: Optional[Path],
        checkpoint_repo_id: Optional[str],
        local_files_only: bool,
    ) -> Path:
        if base_path is not None:
            candidate = (base_path / filename).resolve()
            if candidate.exists():
                return candidate
            parent_candidate = (base_path.parent / filename).resolve()
            if parent_candidate.exists():
                return parent_candidate
        repo_id = checkpoint_repo_id or (str(repo_or_path) if repo_or_path else None)
        resolved = _maybe_hf_download(
            repo_id,
            filename,
            local_files_only=local_files_only,
        )
        if resolved is None:
            raise FileNotFoundError(f"Checkpoint file '{filename}' not found in local paths or Hugging Face hub.")
        return resolved

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        video_height: int,
        video_width: int,
        key_slice_idx_offset: int,
        bbox: Union[torch.Tensor, np.ndarray],
        preprocessed_volume: Optional[np.ndarray] = None,
        reference_image: Optional[sitk.Image] = None,
        case_metadata: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> MedSam2SegmentationOutput:
        pixel_values = pixel_values.to(device=self.device)
        bbox_np = bbox.detach().cpu().numpy() if isinstance(bbox, torch.Tensor) else np.asarray(bbox, dtype=np.float32)
        mask_volume = np.zeros((pixel_values.shape[0], video_height, video_width), dtype=np.uint8)
        logits_volume = np.zeros((pixel_values.shape[0], 1, video_height, video_width), dtype=np.float32)

        autocast_enabled = self.device.type != "cpu"
        dtype = self.predict_dtype if autocast_enabled else torch.float32

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            inference_state = self.predictor.init_state(pixel_values, int(video_height), int(video_width))
            _, _, _ = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(key_slice_idx_offset),
                obj_id=1,
                box=bbox_np,
            )
            for frame_idx, _, mask_logits in self.predictor.propagate_in_video(inference_state):
                logits = mask_logits[0].cpu().float().numpy()
                logits_volume[frame_idx] = logits
                mask_volume[frame_idx, logits[0] > 0.0] = 1
            self.predictor.reset_state(inference_state)

            _, _, _ = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(key_slice_idx_offset),
                obj_id=1,
                box=bbox_np,
            )
            for frame_idx, _, mask_logits in self.predictor.propagate_in_video(inference_state, reverse=True):
                logits = mask_logits[0].cpu().float().numpy()
                logits_volume[frame_idx] = logits
                mask_volume[frame_idx, logits[0] > 0.0] = 1
            self.predictor.reset_state(inference_state)

        largest_component_applied = False
        if self.postprocess_largest_cc and mask_volume.max() > 0:
            mask_volume = _largest_connected_component(mask_volume)
            largest_component_applied = True

        img_sitk = None
        mask_sitk = None
        if preprocessed_volume is not None:
            img_sitk = sitk.GetImageFromArray(preprocessed_volume.astype(np.uint8))
            if reference_image is not None:
                img_sitk.CopyInformation(reference_image)
        if reference_image is not None:
            mask_sitk = sitk.GetImageFromArray(mask_volume.astype(np.uint8))
            mask_sitk.CopyInformation(reference_image)

        return MedSam2SegmentationOutput(
            mask_volume=mask_volume,
            logits_volume=logits_volume,
            sitk_mask=mask_sitk,
            sitk_image=img_sitk,
            metadata=case_metadata or {},
            largest_component_applied=largest_component_applied,
        )
