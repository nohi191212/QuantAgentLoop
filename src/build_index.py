"""
向量库构建: 加载预编码向量, L2归一化后建 Faiss 索引 (余弦相似度)
"""
import os
import sys
import time
import numpy as np
import faiss

# ========== 配置 ==========
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORS_DIR = os.path.join(PROJECT_DIR, "data", "vectors")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "data", "samples")
WINDOW_SIZES = [1, 2, 4, 8, 12, 24, 52]
INDEX_TYPE = "IVF256,Flat"   # IVF nlist=256, 适合百万级
# ==========================


def build_one_window(ws):
    """为单个窗口大小构建 Faiss 索引"""
    vec_path = os.path.join(VECTORS_DIR, f"window_{ws}w.npy")
    if not os.path.exists(vec_path):
        print(f"  {ws}周: 向量文件不存在, 跳过")
        return

    print(f"  {ws}周: 加载向量...")
    vectors = np.load(vec_path)
    n, d = vectors.shape
    print(f"    {n:,} 个向量, dim={d}")

    # 过滤包含 NaN 的向量
    nan_mask = np.isnan(vectors).any(axis=1)
    if nan_mask.any():
        print(f"    移除 {nan_mask.sum()} 个含 NaN 的向量")
        vectors = vectors[~nan_mask]

    n, d = vectors.shape
    if n == 0:
        print(f"    无有效向量, 跳过")
        return

    # L2 归一化 → 内积等价于余弦相似度
    print(f"    L2 归一化...")
    faiss.normalize_L2(vectors)

    # 构建 IVF 索引
    print(f"    构建索引 ({INDEX_TYPE})...")
    nlist = min(256, max(4, int(np.sqrt(n))))
    quantizer = faiss.IndexFlatIP(d)  # 内积检索
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    if n < 10000:
        # 样本少直接用暴力检索
        index = faiss.IndexFlatIP(d)
        print(f"    少量样本, 使用暴力检索")
    else:
        # 训练 IVF
        train_n = min(n, 100000)
        print(f"    训练 IVF (nlist={nlist}, train={train_n})...")
        index.train(vectors[:train_n])
        print(f"    nprobe 设为 32")
        index.nprobe = 32

    index.add(vectors)

    # 保存
    index_dir = os.path.join(VECTORS_DIR, f"window_{ws}w")
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "index.faiss")
    faiss.write_index(index, index_path)

    # 保存有效样本的 ID 映射 (faiss_id → sample_id)
    import pandas as pd
    meta_path = os.path.join(SAMPLES_DIR, f"window_{ws}w", "meta.parquet")
    if os.path.exists(meta_path):
        meta = pd.read_parquet(meta_path)
        if len(meta) != n:
            # 有 NaN 被移除的情况
            full_meta = meta
            valid_meta = full_meta[~nan_mask].reset_index(drop=True) if nan_mask.sum() > 0 else full_meta
        else:
            valid_meta = meta
        id_map = valid_meta[["sample_id", "ts_code", "window_size", "anchor_date"]].copy()
        id_map["faiss_id"] = range(len(id_map))
        id_map_path = os.path.join(index_dir, "id_map.parquet")
        id_map.to_parquet(id_map_path, index=False)

    print(f"    索引已保存: {index_path}")
    print(f"    索引大小: {os.path.getsize(index_path)/1024/1024:.1f}MB")


def main():
    print(f"向量目录: {VECTORS_DIR}")
    print(f"窗口大小: {WINDOW_SIZES}")

    for ws in WINDOW_SIZES:
        t0 = time.time()
        build_one_window(ws)
        print(f"    耗时: {time.time() - t0:.1f}s")
        print()


if __name__ == "__main__":
    main()
