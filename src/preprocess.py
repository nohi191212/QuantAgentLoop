"""
数据预处理: 将所有股票日线按 N 周窗口切片, 编码为向量, 计算未来 1~30 日收益
一次处理一只股票, 所有窗口大小一起产出, 避免重复加载 CSV
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


def compute_future_rets(df, after_idx):
    """计算窗口之后 1~30 个交易日的累计收益率"""
    future = df.iloc[after_idx + 1: after_idx + 1 + FUTURE_DAYS]
    n_actual = len(future)
    last_close = df.iloc[after_idx]["close"]

    rets = np.full(FUTURE_DAYS, np.nan, dtype=np.float32)
    if n_actual > 0 and last_close > 0:
        cum_ret = 1.0
        for i, (_, row) in enumerate(future.iterrows()):
            if i >= FUTURE_DAYS:
                break
            daily_r = row["pct_chg"] / 100.0 if not pd.isna(row["pct_chg"]) else 0.0
            cum_ret *= (1.0 + daily_r)
            rets[i] = cum_ret - 1.0
    return rets


def process_one_stock(csv_path, encoder, accumulators):
    """
    处理单只股票: 对所有窗口大小生成样本 + 向量 + 未来收益
    accumulators: dict[window_size -> {"vectors":[], "meta":[]}]
    """
    ts_code = os.path.basename(csv_path).replace(".csv", "")

    try:
        df = load_and_clean(csv_path)
    except Exception:
        return 0

    if len(df) < 30:  # 太短的股票跳过
        return 0

    df = assign_iso_week(df)

    # 将需要的列转为 numpy 加速, 缺失列填 NaN
    avail_cols = [c for c in ENCODER_COLS if c in df.columns]
    missing = [c for c in ENCODER_COLS if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    arr = df[ENCODER_COLS].to_numpy(dtype=np.float32)
    dates = df["trade_date"].dt.strftime("%Y%m%d").to_numpy()

    total_windows = 0

    for ws in WINDOW_SIZES:
        windows = make_windows(df, ws)
        for _, anchor_date, start_idx, end_idx in windows:
            # 窗口内数据
            win_arr = arr[start_idx: end_idx + 1]

            # 编码
            vec = encoder.encode(win_arr)

            # 未来收益
            future_rets = compute_future_rets(df, end_idx)

            # 样本ID
            sample_id = f"{ts_code}_{ws}w_{anchor_date}"

            accumulators[ws]["vectors"].append(vec)
            accumulators[ws]["meta"].append({
                "sample_id": sample_id,
                "ts_code": ts_code,
                "window_size": ws,
                "anchor_date": anchor_date,
                "window_start": dates[start_idx],
                "window_end": dates[end_idx],
                **{f"ret_{d}d": future_rets[d - 1] for d in range(1, FUTURE_DAYS + 1)},
            })
            total_windows += 1

    return total_windows


def main(limit_stocks=None):
    """
    主函数, 处理所有股票.
    limit_stocks: 限制股票数量（None=全部, 用于测试）
    """
    encoder = KronosEncoder()
    print(f"编码器: {encoder.name}, 维度: {encoder.dim}")
    print(f"窗口大小: {WINDOW_SIZES}")
    print(f"股票数据目录: {STOCK_DATA_DIR}")
    print()

    # 初始化累加器
    accumulators = {ws: {"vectors": [], "meta": []} for ws in WINDOW_SIZES}

    # 获取股票列表
    csv_files = sorted(
        f for f in os.listdir(STOCK_DATA_DIR) if f.endswith(".csv")
    )
    if limit_stocks:
        csv_files = csv_files[:limit_stocks]

    total_csv = len(csv_files)
    total_windows = 0
    start_time = time.time()

    from tqdm import tqdm
    for i, fname in tqdm(enumerate(csv_files)):
        csv_path = os.path.join(STOCK_DATA_DIR, fname)
        n_windows = process_one_stock(csv_path, encoder, accumulators)
        total_windows += n_windows

        if (i + 1) % 500 == 0 or (i + 1) == total_csv:
            elapsed = time.time() - start_time
            print(
                f"  股票: {i + 1}/{total_csv} "
                f"| 累计窗口: {total_windows:,} "
                f"| 耗时: {elapsed/60:.1f}min "
                f"| 速度: {(i+1)/elapsed:.1f} 只/秒"
            )

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
    args = parser.parse_args()
    main(limit_stocks=args.limit)
