"""
Kronos-mini 编码器: 使用 Kronos Tokenizer + Kronos-mini Transformer 将 OHLCV 窗口编码为固定维度向量
"""
import os
import sys
import json
from collections import defaultdict

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

    def encode(self, daily_data: list[np.ndarray]) -> np.ndarray:
        """
        daily_data: list of (T_i, F) arrays — 每个窗口形状可能不同
        提取 OHLC + volume + amount (6 列), 归一化后用 Tokenizer 编码为 token indices,
        再通过 Kronos-mini Transformer 获取上下文隐层, mean pooling 后返回 256 维向量.
        returns: (B, 256)
        """
        B = len(daily_data)
        if B == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        cols = [0, 1, 2, 3, 7, 8]

        # 按序列长度分组, 同长度的才能 stack 成 batch
        groups = defaultdict(list)  # T_len -> [(original_idx, cleaned_array), ...]
        for orig_idx, arr in enumerate(daily_data):
            x = arr[:, cols].astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # 过滤全零行 (OHLC 全为零的停牌日)
            row_valid = ~(np.abs(x[:, :4]).sum(axis=1) == 0)
            if row_valid.sum() < 5:
                continue
            x = x[row_valid]
            groups[x.shape[0]].append((orig_idx, x))

        # 初始化输出, 未覆盖到的保留零向量
        vecs = np.zeros((B, self.dim), dtype=np.float32)

        # 逐组批量编码
        for T_len, items in groups.items():
            indices = [item[0] for item in items]
            batch = np.stack([item[1] for item in items], axis=0)  # (b, T, 6)

            # 逐样本 Z-score 归一化
            mean = batch.mean(axis=1, keepdims=True)   # (b, 1, 6)
            std = batch.std(axis=1, keepdims=True)     # (b, 1, 6)
            batch_norm = (batch - mean) / (std + 1e-5)
            batch_norm = np.clip(batch_norm, -5.0, 5.0)

            x_tensor = torch.from_numpy(batch_norm).to(self.device)  # (b, T, 6)

            with torch.no_grad():
                # Step 1: Tokenizer → s1/s2 token indices
                z_indices = self.tokenizer.encode(x_tensor, half=True)
                s1_ids, s2_ids = z_indices  # each (b, T)

                # Step 2: Kronos-mini Transformer → 上下文隐层
                _, hidden = self.model.decode_s1(s1_ids, s2_ids)  # (b, T, 256)

                # Step 3: 时间维度 mean pooling
                batch_vecs = hidden.mean(dim=1)  # (b, 256)

            vecs[indices] = batch_vecs.cpu().numpy().astype(np.float32)

        return vecs
