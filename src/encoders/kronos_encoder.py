"""
Kronos-mini 编码器: 使用 Kronos Tokenizer + Kronos-mini Transformer 将 OHLCV 窗口编码为固定维度向量
"""
import os
import sys
import json
import numpy as np
import torch
from safetensors.torch import load_file

from .base import BaseEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KRONOS_ROOT = os.path.join(PROJECT_ROOT, "Kronos")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "Kronos-Tokenizer-2k")
KRONOS_MINI_DIR = os.path.join(PROJECT_ROOT, "Kronos-mini")

sys.path.insert(0, KRONOS_ROOT)
from model.kronos import KronosTokenizer, Kronos


class KronosEncoder(BaseEncoder):
    def __init__(self, device=None):
        # --- Tokenizer ---
        tokenizer_config = dict(
            d_in=6,
            d_model=256,
            n_heads=4,
            ff_dim=512,
            n_enc_layers=4,
            n_dec_layers=4,
            ffn_dropout_p=0.0,
            attn_dropout_p=0.0,
            resid_dropout_p=0.0,
            s1_bits=10,
            s2_bits=10,
            beta=0.05,
            gamma0=1.0,
            gamma=1.1,
            zeta=0.05,
            group_size=5,
        )
        self.tokenizer = KronosTokenizer(**tokenizer_config)
        tokenizer_weights = os.path.join(TOKENIZER_DIR, "model.safetensors")
        self.tokenizer.load_state_dict(load_file(tokenizer_weights))
        self.tokenizer.eval()

        # --- Kronos-mini model ---
        kronos_config_path = os.path.join(KRONOS_MINI_DIR, "config.json")
        with open(kronos_config_path, "r") as f:
            kronos_config = json.load(f)
        self.model = Kronos(**kronos_config)
        model_weights = os.path.join(KRONOS_MINI_DIR, "model.safetensors")
        self.model.load_state_dict(load_file(model_weights))
        self.model.eval()

        # --- Device ---
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    @property
    def dim(self) -> int:
        return 256

    @property
    def name(self) -> str:
        return "KronosMini"

    def encode(self, daily_data: np.ndarray) -> np.ndarray:
        """
        daily_data: shape (T, F) — 列顺序与 ENCODER_COLS 一致
        提取 OHLC + volume + amount (6 列), 归一化后用 Tokenizer 编码为 token indices,
        再通过 Kronos-mini Transformer 获取上下文隐层, mean pooling 后返回 256 维向量.
        """
        # 提取 Kronos 需要的 6 维特征: open(0), high(1), low(2), close(3), vol(7), amount(8)
        cols = [0, 1, 2, 3, 7, 8]
        x = daily_data[:, cols].astype(np.float32)

        # NaN / Inf → 0
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 过滤全零行
        valid = ~(np.abs(x[:, :4]).sum(axis=1) == 0)
        if valid.sum() < 5:
            return np.zeros(self.dim, dtype=np.float32)
        x = x[valid]

        # Z-score 归一化
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x_norm = (x - mean) / (std + 1e-5)
        x_norm = np.clip(x_norm, -5.0, 5.0)

        # 编码
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(self.device)  # (1, T, 6)

        with torch.no_grad():
            # Step 1: Tokenizer 编码 → s1/s2 token indices
            z_indices = self.tokenizer.encode(x_tensor, half=True)
            s1_ids, s2_ids = z_indices  # each (1, T)

            # Step 2: Kronos-mini Transformer 获取上下文隐层
            _, hidden = self.model.decode_s1(s1_ids, s2_ids)  # hidden: (1, T, 256)

            # Step 3: Mean pooling
            vec = hidden.mean(dim=1).squeeze(0)  # (256,)

        return vec.cpu().numpy().astype(np.float32)
