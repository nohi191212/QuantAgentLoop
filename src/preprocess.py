"""
数据预处理: 将所有股票日线按 N 周窗口切片, 编码为向量, 计算未来 1~30 日收益
一次处理一只股票, 所有窗口大小一起产出, 避免重复加载 CSV

支持多进程并行: --workers N  (默认 1 = 顺序)
"""
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoders.kronos_encoder import KronosEncoder

# ========== 配置 ==========
STOCK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_daily_data"
)
SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples"
)
VECTORS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vectors"
)

WINDOW_SIZES = [2, 4, 12, 52]      # 周
FUTURE_DAYS = 30                                # 未来收益天数
# 编码器需要的列, 按固定顺序; CSV 中可能缺失部分列, 自动补 NaN
ENCODER_COLS = [
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
    "turnover_rate", "volume_ratio",
    "ma5", "ma_v_5", "ma10", "ma_v_10",
    "ma20", "ma_v_20", "ma60", "ma_v_60",
]
# ==========================


def load_and_clean(csv_path):
    """加载单只股票日线, 去全NaN行, 返回按日期排序的DataFrame"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df.sort_values("trade_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # 去除 OHLC 全 NaN 的行（停牌日）
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    df.reset_index(drop=True, inplace=True)
    return df


def assign_iso_week(df):
    """为每个交易日分配 ISO 周标签 (year, week)"""
    df = df.copy()
    # ISO week: Monday=1, Sunday=7
    iso = df["trade_date"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    return df


def make_windows(df, window_size):
    """
    对每天数据按周分组, 按 window_size 滑动生成窗口.
    返回窗口列表: [(window_idx, anchor_date_str, start_idx, end_idx), ...]
    start_idx/end_idx 是 df 的行索引（窗口内所有交易日的起止）
    """
    # 按 (iso_year, iso_week) 分组
    groups = df.groupby(["iso_year", "iso_week"])
    week_keys = sorted(groups.groups.keys())
    if len(week_keys) < window_size + 1:
        return []

    windows = []
    for anchor_pos in range(len(week_keys) - window_size):
        target_weeks = week_keys[anchor_pos: anchor_pos + window_size]
        # 找到这些周的第一天和最后一天的行索引
        rows_in_window = []
        for (y, w) in target_weeks:
            rows_in_window.extend(groups.groups[(y, w)].tolist())
        if len(rows_in_window) < 5:  # 窗口太小无效
            continue
        start_idx = min(rows_in_window)
        end_idx = max(rows_in_window)
        anchor_date_str = df.loc[start_idx, "trade_date"].strftime("%Y%m%d")
        windows.append((anchor_pos, anchor_date_str, start_idx, end_idx))
    return windows


def precompute_all_future_rets(df, future_days=30):
    """
    一次性预计算所有位置的未来累计收益矩阵。
    用 cumsum(log(1+r)) 向量化, O(N * future_days) → O(N + future_days)。
    返回 (N, future_days) 的 float32 数组, rets[i, d] = 从 i 之后第 d+1 天的累计收益。
    """
    pct = df["pct_chg"].to_numpy(dtype=np.float64)
    pct = np.nan_to_num(pct, nan=0.0) / 100.0
    n = len(pct)

    log_r = np.log(1.0 + pct)
    # prefix[t] = sum_{k=0}^{t-1} log(1+r_k),   prefix[0] = 0
    prefix = np.zeros(n + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(log_r)

    i_idx = np.arange(n)[:, None]            # (N, 1)
    d_idx = np.arange(future_days)[None, :]   # (1, 30)

    start = i_idx + 1                         # 第一个未来日的 prefix 起点
    end = i_idx + d_idx + 2                   # 最后未来日 + 1 的 prefix 终点

    valid = end <= n
    end_safe = np.clip(end, 0, n)

    log_cum = np.where(valid, prefix[end_safe] - prefix[start], np.nan)
    rets = np.exp(log_cum) - 1.0
    return rets.astype(np.float32)


# 预构建 ret 列名, 避免循环内重复 f-string
_RET_COLS = {d: f"ret_{d}d" for d in range(1, FUTURE_DAYS + 1)}


def process_one_stock(csv_path, encoder):
    """
    处理单只股票: 对所有窗口大小生成样本 + 向量 + 未来收益.
    返回 (n_windows, results_dict) 其中 results_dict: {ws: {"vectors": [...], "meta": [...]}}
    """
    ts_code = os.path.basename(csv_path).replace(".csv", "")

    try:
        df = load_and_clean(csv_path)
    except Exception:
        return 0, {}

    if len(df) < 30:  # 太短的股票跳过
        return 0, {}

    df = assign_iso_week(df)

    # 将需要的列转为 numpy 加速, 缺失列填 NaN
    for c in ENCODER_COLS:
        if c not in df.columns:
            df[c] = np.nan
    arr = df[ENCODER_COLS].to_numpy(dtype=np.float32)
    dates = df["trade_date"].dt.strftime("%Y%m%d").to_numpy()

    # 预计算所有位置的未来收益矩阵 (N, 30), O(1) 查表
    future_mat = precompute_all_future_rets(df, FUTURE_DAYS)

    total_windows = 0
    results = {}
    max_batch_size = 256  # 可根据显存/Gpu内存调整

    for ws in WINDOW_SIZES:
        windows = make_windows(df, ws)
        if not windows:
            continue

        # 批量准备：收集所有窗口数据和元信息
        batch_windows = []
        batch_metas = []

        for _, anchor_date, start_idx, end_idx in windows:
            win_arr = arr[start_idx: end_idx + 1]
            batch_windows.append(win_arr)

            # 未来收益 — O(1) 查表
            row = future_mat[end_idx]
            meta = {
                "sample_id": f"{ts_code}_{ws}w_{anchor_date}",
                "ts_code": ts_code,
                "window_size": ws,
                "anchor_date": anchor_date,
                "window_start": dates[start_idx],
                "window_end": dates[end_idx],
            }
            for d in range(1, FUTURE_DAYS + 1):
                meta[_RET_COLS[d]] = row[d - 1]
            batch_metas.append(meta)

        ws_vectors = []
        ws_metas = []

        # 分批编码
        for batch_start in range(0, len(batch_windows), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(batch_windows))
            current_batch = batch_windows[batch_start:batch_end]
            current_metas = batch_metas[batch_start:batch_end]

            vecs = encoder.encode(current_batch)  # (b, 256)
            for i in range(vecs.shape[0]):
                ws_vectors.append(vecs[i])
            ws_metas.extend(current_metas)
            total_windows += len(current_batch)

        results[ws] = {"vectors": ws_vectors, "meta": ws_metas}

    return total_windows, results


# ======================== 多进程支持 ========================

_worker_encoder = None


def _worker_init():
    """多进程 worker 初始化：每个进程加载自己的编码器。"""
    global _worker_encoder
    import torch
    torch.set_num_threads(1)  # 限制 PyTorch 内部线程数, 靠进程数来并行
    _worker_encoder = KronosEncoder(device='cpu')


def _worker_process_stock(csv_path):
    """多进程 worker: 处理一只股票, 返回 (n_windows, results_dict)。"""
    global _worker_encoder
    return process_one_stock(csv_path, _worker_encoder)


# ======================== 主函数 ========================

def main(limit_stocks=None, n_workers=1):
    """
    主函数, 处理所有股票.
    limit_stocks: 限制股票数量（None=全部, 用于测试）
    n_workers: 并行进程数 (1 = 顺序, >1 = 多进程)
    """
    # 获取股票列表
    csv_files = sorted(
        f for f in os.listdir(STOCK_DATA_DIR) if f.endswith(".csv")
    )
    if limit_stocks:
        csv_files = csv_files[:limit_stocks]

    csv_paths = [os.path.join(STOCK_DATA_DIR, f) for f in csv_files]
    total_csv = len(csv_paths)

    print(f"编码器: KronosMini, 维度: 256")
    print(f"窗口大小: {WINDOW_SIZES}")
    print(f"股票数据目录: {STOCK_DATA_DIR}")
    print(f"股票数量: {total_csv}")
    print(f"并行进程: {n_workers}")
    print()

    # 初始化累加器
    accumulators = {ws: {"vectors": [], "meta": []} for ws in WINDOW_SIZES}

    total_windows = 0
    start_time = time.time()

    if n_workers <= 1:
        # ——— 顺序模式 ———
        encoder = KronosEncoder(device='cpu')
        from tqdm import tqdm
        for i, csv_path in tqdm(enumerate(csv_paths)):
            n_windows, results = process_one_stock(csv_path, encoder)
            total_windows += n_windows
            for ws, data in results.items():
                accumulators[ws]["vectors"].extend(data["vectors"])
                accumulators[ws]["meta"].extend(data["meta"])

            if (i + 1) % 500 == 0 or (i + 1) == total_csv:
                elapsed = time.time() - start_time
                print(
                    f"  股票: {i + 1}/{total_csv} "
                    f"| 累计窗口: {total_windows:,} "
                    f"| 耗时: {elapsed/60:.1f}min "
                    f"| 速度: {(i+1)/elapsed:.1f} 只/秒"
                )
    else:
        # ——— 多进程模式 ———
        from multiprocessing import get_context
        from tqdm import tqdm

        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
            with tqdm(total=total_csv, desc="处理股票") as pbar:
                for n_windows, results in pool.imap_unordered(
                    _worker_process_stock, csv_paths
                ):
                    total_windows += n_windows
                    for ws, data in results.items():
                        accumulators[ws]["vectors"].extend(data["vectors"])
                        accumulators[ws]["meta"].extend(data["meta"])

                    pbar.update(1)
                    if pbar.n % 500 == 0 or pbar.n == total_csv:
                        elapsed = time.time() - start_time
                        pbar.set_postfix({
                            "windows": f"{total_windows:,}",
                            "time": f"{elapsed/60:.1f}min",
                        })

    # ——— 保存 ———
    print(f"\n处理完成, 共 {total_windows:,} 个窗口, 开始保存...")

    for ws in WINDOW_SIZES:
        vectors = accumulators[ws]["vectors"]
        metas = accumulators[ws]["meta"]

        if not vectors:
            print(f"  {ws}周: 0 样本, 跳过")
            continue

        vec_arr = np.array(vectors, dtype=np.float32)
        meta_df = pd.DataFrame(metas)

        ws_dir = os.path.join(SAMPLES_DIR, f"window_{ws}w")
        os.makedirs(ws_dir, exist_ok=True)

        # 向量单独存为 npy (后续建索引用)
        vec_path = os.path.join(VECTORS_DIR, f"window_{ws}w.npy")
        np.save(vec_path, vec_arr)

        # 元数据存为 parquet
        meta_path = os.path.join(ws_dir, "meta.parquet")
        meta_df.to_parquet(meta_path, index=False)

        print(f"  {ws}周: {len(vectors):,} 样本, dim={vec_arr.shape[1]}, "
              f"向量 {vec_arr.nbytes/1024/1024:.1f}MB, "
              f"元数据 {os.path.getsize(meta_path)/1024/1024:.1f}MB")

    elapsed = time.time() - start_time
    print(f"\n全部完成! 总耗时: {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="限制股票数量（调试用）")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行进程数 (默认 1 = 顺序)")
    args = parser.parse_args()
    main(limit_stocks=args.limit, n_workers=args.workers)
