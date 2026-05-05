"""
Backtest: 验证历史相似窗口的收益率预测能力
从已有编码数据中随机抽取 N 个最近一年的 12 周窗口，
用预计算向量直接检索 FAISS，比较 TOP-K 平均收益 vs 实际收益，
生成图文报告到 output/ 目录
"""
import os, sys, time, random, warnings
import numpy as np
import pandas as pd
import faiss

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query import SimilarityEngine, PROJECT_DIR, VECTORS_DIR, SAMPLES_DIR

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 12
N_SAMPLES = 1000
TOP_K_LIST = [10, 20]
MIN_FUTURE = 30
SEARCH_PAD = 10  # 多搜几条用于过滤自匹配
SEED = 42

# ---- matplotlib ----
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
    for fn in ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']:
        try:
            matplotlib.font_manager.findfont(fn, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [fn]
            plt.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue
except ImportError:
    HAS_MPL = False


def load_precomputed():
    """加载预编码向量和元数据，返回 vecs(N,256), meta_df, sample_id→row_idx 映射"""
    vec_path = os.path.join(VECTORS_DIR, f"window_{WINDOW_SIZE}w.npy")
    meta_path = os.path.join(SAMPLES_DIR, f"window_{WINDOW_SIZE}w", "meta.parquet")

    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"向量文件不存在: {vec_path}\n请先运行 preprocess.py")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"元数据不存在: {meta_path}\n请先运行 preprocess.py")

    print("加载预编码数据...")
    t0 = time.time()
    vecs = np.load(vec_path)
    meta = pd.read_parquet(meta_path)
    print(f"  向量: {vecs.shape}  |  元数据: {len(meta):,} 行  ({time.time() - t0:.1f}s)")

    # 向量和元数据逐行对齐
    assert len(vecs) == len(meta), "向量和元数据行数不一致!"

    # sample_id → row index in vecs/meta
    sid_to_idx = {sid: i for i, sid in enumerate(meta["sample_id"].values)}
    return vecs, meta, sid_to_idx


def filter_candidates(meta, sid_to_idx, id_map_sids):
    """筛选候选窗口：最近一年 + 有效30天未来收益 + 在FAISS索引中"""
    one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)

    meta["anchor_dt"] = pd.to_datetime(meta["anchor_date"].astype(str), format="%Y%m%d")
    recent = meta["anchor_dt"] >= one_year_ago

    # 检查未来收益：至少有一半非 NaN
    ret_cols = [f"ret_{d}d" for d in range(1, MIN_FUTURE + 1)]
    rets_mat = meta[ret_cols].to_numpy(dtype=np.float32)
    valid_future = ~np.isnan(rets_mat).all(axis=1)

    # 检查在 FAISS 索引中 (预处理时可能因 NaN 被过滤)
    in_index = meta["sample_id"].isin(id_map_sids)

    mask = recent.values & valid_future & in_index.values
    candidates = meta[mask].copy()
    candidates["_row_idx"] = candidates["sample_id"].map(sid_to_idx)

    print(f"  候选窗口: {len(candidates):,}  (最近一年 + 有效未来 + 在索引中)")
    return candidates


def sample_candidates(candidates, n, seed):
    """分层随机采样：每只股票最多取 5 个窗口"""
    random.seed(seed)
    by_stock = candidates.groupby("ts_code")
    pool = []
    for _, grp in by_stock:
        rows = grp.to_dict("records")
        random.shuffle(rows)
        pool.extend(rows[:5])
    random.shuffle(pool)
    sampled = pool[:n]
    stocks = len(set(s["ts_code"] for s in sampled))
    print(f"  采样: {len(sampled)} 窗口, 覆盖 {stocks} 只股票")
    return sampled


def build_id_lookups(engine):
    """构建双向 faiss_id ↔ sample_id 和 sample_id→future_rets 快速查询"""
    print("构建查询表...")
    t0 = time.time()

    # 双向映射
    faiss_ids = engine.id_map["faiss_id"].values
    sample_ids = engine.id_map["sample_id"].values
    faiss_to_sid = dict(zip(faiss_ids, sample_ids))
    sid_to_faiss = dict(zip(sample_ids, faiss_ids))

    # sample_id → (30,) future returns
    ret_cols = [f"ret_{d}d" for d in range(1, MIN_FUTURE + 1)]
    rets_mat = engine.meta[ret_cols].to_numpy(dtype=np.float32)
    sid_to_rets = dict(zip(engine.meta["sample_id"].values, rets_mat))

    print(f"  faiss↔sid: {len(faiss_to_sid):,}  |  sid→rets: {len(sid_to_rets):,}  ({time.time() - t0:.1f}s)")
    return faiss_to_sid, sid_to_faiss, sid_to_rets, set(sample_ids)


def run_backtest(sampled, vecs, engine, faiss_to_sid, sid_to_faiss, sid_to_rets):
    """批量 FAISS 检索 + 预测 vs 实际对比"""
    max_k = max(TOP_K_LIST)
    search_k = max_k + SEARCH_PAD
    N = len(sampled)

    # 1. 从预编码向量中提取查询向量
    query_vecs = np.zeros((N, 256), dtype=np.float32)
    sampled_faiss_ids = np.zeros(N, dtype=np.int64)
    actual_rets = np.zeros((N, MIN_FUTURE), dtype=np.float32)

    for i, s in enumerate(sampled):
        row_idx = s["_row_idx"]
        query_vecs[i] = vecs[row_idx]
        sampled_faiss_ids[i] = sid_to_faiss.get(s["sample_id"], -1)
        for d in range(1, MIN_FUTURE + 1):
            actual_rets[i, d - 1] = s.get(f"ret_{d}d", np.nan)

    # 2. L2 归一化 + FAISS 批量检索
    print(f"\nFAISS 批量检索 ({N} queries, top_k={search_k})...")
    faiss.normalize_L2(query_vecs)
    distances, indices = engine.index.search(query_vecs, search_k)

    # 3. 逐查询处理匹配结果
    print("处理匹配结果...")
    predictions = {k: np.full((N, MIN_FUTURE), np.nan, dtype=np.float32) for k in TOP_K_LIST}
    match_counts = np.zeros(N, dtype=int)

    for i in range(N):
        query_faiss = sampled_faiss_ids[i]
        collected = {k: [] for k in TOP_K_LIST}
        seen = set()

        for j in range(search_k):
            fid = int(indices[i, j])
            if fid < 0 or fid in seen:
                continue
            seen.add(fid)

            if fid == query_faiss:
                continue

            sid = faiss_to_sid.get(fid)
            if sid is None:
                continue
            rets = sid_to_rets.get(sid)
            if rets is None or np.isnan(rets).all():
                continue

            for k in TOP_K_LIST:
                if len(collected[k]) < k:
                    collected[k].append(rets)

            if all(len(collected[k]) >= k for k in TOP_K_LIST):
                break

        for k in TOP_K_LIST:
            if len(collected[k]) >= k:
                predictions[k][i] = np.nanmean(np.stack(collected[k][:k], axis=0), axis=0)
        match_counts[i] = len(collected[TOP_K_LIST[0]])

    # 4. 过滤匹配不足的样本
    min_k = min(TOP_K_LIST)
    valid_mask = match_counts >= min_k
    print(f"  有效样本: {valid_mask.sum()}/{N}  (匹配数 >= {min_k})")

    return predictions, actual_rets, valid_mask


# ========== 指标 & 打印 & 绘图 (沿用) ==========

def compute_metrics(actual, predicted):
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() < 5:
        return {"corr": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "n": 0}
    a, p = actual[mask], predicted[mask]
    n = len(a)
    corr = np.corrcoef(a, p)[0, 1] if n > 2 else np.nan
    ss_res = ((a - p) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"corr": corr, "r2": r2, "rmse": np.sqrt(np.mean((a - p) ** 2)),
            "mae": np.mean(np.abs(a - p)), "n": n}


def print_summary(predictions, actual_rets, valid_mask):
    actual = actual_rets[valid_mask]
    day_labels = [1, 2, 3, 5, 10, 15, 20, 30]
    day_indices = [d - 1 for d in day_labels]

    print("\n" + "=" * 90)
    print("  回测结果: 历史相似窗口预测能力评估")
    print("=" * 90)
    print(f"  有效样本: {len(actual)}")
    print(f"  窗口大小: {WINDOW_SIZE} 周")
    print()

    for k in TOP_K_LIST:
        pred = predictions[k][valid_mask]
        print(f"  ── TOP-{k} 预测 ──")
        print(f"  {'持仓天数':<10}", end="")
        for d in day_labels:
            print(f"{f'{d}日':>8}", end="")
        print(f"\n  {'-' * (10 + 8 * len(day_labels))}")

        for metric, label in [("corr", "Corr"), ("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")]:
            row = f"  {label:<10}"
            for d_idx in day_indices:
                m = compute_metrics(actual[:, d_idx], pred[:, d_idx])
                v = m[metric]
                if not np.isnan(v):
                    fmt = "8.4f" if metric in ("rmse", "mae") else "8.3f"
                    row += f"{v:{fmt}}"
                else:
                    row += f"{'':>8}"
            print(row)
        print()
    print("=" * 90)


def plot_results(predictions, actual_rets, valid_mask):
    if not HAS_MPL:
        return

    actual = actual_rets[valid_mask]
    labels = [1, 2, 3, 5, 10, 15, 20, 30]
    idx_map = {d: d - 1 for d in labels}
    colors = plt.cm.tab10(np.linspace(0, 1, len(TOP_K_LIST)))

    # ---- Figure 1: 散点图 预测 vs 实际 ----
    plot_days = [5, 10, 20, 30]
    n_groups = len(TOP_K_LIST)
    fig, axes = plt.subplots(n_groups, len(plot_days), figsize=(4 * len(plot_days), 4 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for gi, k in enumerate(TOP_K_LIST):
        pred = predictions[k][valid_mask]
        for di, d in enumerate(plot_days):
            ax = axes[gi, di]
            d_idx = idx_map[d]
            a, p = actual[:, d_idx] * 100, pred[:, d_idx] * 100
            mask = ~(np.isnan(a) | np.isnan(p))
            a_c, p_c = a[mask], p[mask]

            if len(a_c) > 0:
                ax.scatter(p_c, a_c, s=4, alpha=0.3, color='steelblue', edgecolors='none')
                if len(a_c) > 2:
                    coeffs = np.polyfit(p_c, a_c, 1)
                    xs = np.linspace(p_c.min(), p_c.max(), 50)
                    ax.plot(xs, np.polyval(coeffs, xs), color='#e74c3c', linewidth=1.5)
                m = compute_metrics(a_c / 100, p_c / 100)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
                lims = [min(a_c.min(), p_c.min()), max(a_c.max(), p_c.max())]
                ax.plot(lims, lims, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
                ax.set_title(f'TOP-{k} | {d}-Day (n={m["n"]})', fontsize=10)
                ax.set_xlabel('Predicted Return (%)')
                ax.set_ylabel('Actual Return (%)')
                ax.text(0.03, 0.97,
                        f'Corr={m["corr"]:.3f}  R²={m["r2"]:.3f}\nRMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}',
                        transform=ax.transAxes, fontsize=7, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "backtest_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"散点图: {path}")

    # ---- Figure 2: 指标曲线 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for mi, (metric, mname) in enumerate(
        [("corr", "Correlation"), ("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")]
    ):
        ax = axes[mi // 2, mi % 2]
        for gi, k in enumerate(TOP_K_LIST):
            pred = predictions[k][valid_mask]
            vals = []
            for d in labels:
                m = compute_metrics(actual[:, idx_map[d]], pred[:, idx_map[d]])
                vals.append(m[metric])
            ax.plot(labels, vals, 'o-', color=colors[gi], label=f'TOP-{k}', linewidth=1.5, markersize=5)
        ax.set_xlabel('Holding Days'); ax.set_ylabel(mname)
        ax.set_title(f'{mname} by Holding Period'); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_xticks(labels)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "backtest_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"指标曲线: {path}")

    # ---- Figure 3: 分位数分析 ----
    plot_days_q = [5, 10, 20, 30]
    n_q = 5
    fig, axes = plt.subplots(len(TOP_K_LIST), len(plot_days_q),
                             figsize=(4 * len(plot_days_q), 3.5 * len(TOP_K_LIST)))
    if len(TOP_K_LIST) == 1:
        axes = axes.reshape(1, -1)

    for gi, k in enumerate(TOP_K_LIST):
        pred = predictions[k][valid_mask]
        for di, d in enumerate(plot_days_q):
            ax = axes[gi, di]
            d_idx = idx_map[d]
            a, p = actual[:, d_idx], pred[:, d_idx]
            mask = ~(np.isnan(a) | np.isnan(p))
            a_c, p_c = a[mask], p[mask]

            if len(a_c) < n_q * 2:
                ax.text(0.5, 0.5, 'Insufficient', ha='center', va='center', transform=ax.transAxes)
                continue

            order = np.argsort(p_c)
            a_s, p_s = a_c[order] * 100, p_c[order] * 100
            n_each = len(a_s) // n_q
            pred_means, actual_means = [], []
            for q in range(n_q):
                s, e = q * n_each, (q + 1) * n_each if q < n_q - 1 else len(a_s)
                pred_means.append(np.mean(p_s[s:e]))
                actual_means.append(np.mean(a_s[s:e]))

            x = np.arange(n_q)
            w = 0.35
            ax.bar(x - w / 2, pred_means, w, label='Predicted', color='steelblue', alpha=0.8)
            ax.bar(x + w / 2, actual_means, w, label='Actual', color='#e74c3c', alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels([f'Q{i + 1}' for i in range(n_q)])
            ax.set_title(f'TOP-{k} | {d}-Day Quantiles', fontsize=10)
            ax.set_xlabel('Predicted Return (Low → High)'); ax.set_ylabel('Mean Return (%)')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')
            ax.axhline(y=0, color='gray', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "backtest_quantile.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"分位数分析: {path}")

    # ---- Figure 4: 误差分布 ----
    fig, axes = plt.subplots(len(TOP_K_LIST), len(plot_days_q),
                             figsize=(4 * len(plot_days_q), 3 * len(TOP_K_LIST)))
    if len(TOP_K_LIST) == 1:
        axes = axes.reshape(1, -1)

    for gi, k in enumerate(TOP_K_LIST):
        pred = predictions[k][valid_mask]
        for di, d in enumerate(plot_days_q):
            ax = axes[gi, di]; d_idx = idx_map[d]
            a, p = actual[:, d_idx], pred[:, d_idx]
            mask = ~(np.isnan(a) | np.isnan(p))
            errors = (p[mask] - a[mask]) * 100
            if len(errors) > 0:
                ax.hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
                ax.axvline(x=np.mean(errors), color='#e74c3c', linestyle='--', linewidth=1.2,
                           label=f'Bias: {np.mean(errors):.3f}%')
                ax.set_title(f'TOP-{k} | {d}-Day Error', fontsize=10)
                ax.set_xlabel('Error (Pred - Actual, %)'); ax.set_ylabel('Count')
                ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "backtest_errors.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"误差分布: {path}")

    # ---- Figure 5: 综合摘要 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for gi, k in enumerate(TOP_K_LIST):
        pred = predictions[k][valid_mask]
        r2v, dirav = [], []
        for d in labels:
            d_idx = idx_map[d]
            a, p = actual[:, d_idx], pred[:, d_idx]
            mask = ~(np.isnan(a) | np.isnan(p))
            m = compute_metrics(a[mask], p[mask])
            r2v.append(m["r2"] if not np.isnan(m["r2"]) else 0)
            dirav.append(((p[mask] > 0) == (a[mask] > 0)).mean() if mask.sum() > 0 else np.nan)
        ax1.plot(labels, r2v, 'o-', color=colors[gi], label=f'TOP-{k}', linewidth=1.5, markersize=5)
        ax2.plot(labels, [da * 100 for da in dirav], 'o-', color=colors[gi],
                 label=f'TOP-{k}', linewidth=1.5, markersize=5)

    ax1.set_xlabel('Holding Days'); ax1.set_ylabel('R²')
    ax1.set_title('Prediction R² by Holding Period'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_xticks(labels); ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    ax2.set_xlabel('Holding Days'); ax2.set_ylabel('Direction Accuracy (%)')
    ax2.set_title('Up/Down Direction Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xticks(labels); ax2.axhline(y=50, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "backtest_summary.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"综合摘要: {path}")


# ========== 主函数 ==========

def main():
    t_start = time.time()
    print(f"配置: 窗口={WINDOW_SIZE}周, 采样={N_SAMPLES}, TOP-K={TOP_K_LIST}, seed={SEED}\n")

    # 1. 加载预编码数据
    vecs, meta, sid_to_idx = load_precomputed()

    # 2. 加载 FAISS 索引
    print("加载 FAISS 索引...")
    engine = SimilarityEngine(window_size=WINDOW_SIZE)

    # 3. 构建查询表 + 筛选候选
    faiss_to_sid, sid_to_faiss, sid_to_rets, id_map_sids = build_id_lookups(engine)
    candidates_df = filter_candidates(meta, sid_to_idx, id_map_sids)

    if len(candidates_df) == 0:
        print("无可用候选窗口! 检查数据时间范围或未来收益列是否完整。")
        return

    # 4. 采样
    sampled = sample_candidates(candidates_df, N_SAMPLES, SEED)
    if not sampled:
        print("无可用样本!")
        return

    # 5. 回测
    predictions, actual_rets, valid_mask = run_backtest(
        sampled, vecs, engine, faiss_to_sid, sid_to_faiss, sid_to_rets)

    # 6. 打印摘要
    print_summary(predictions, actual_rets, valid_mask)

    # 7. 图表
    if HAS_MPL:
        print("\n生成可视化报告...")
        plot_results(predictions, actual_rets, valid_mask)

    print(f"\n总耗时: {(time.time() - t_start)/60:.1f} 分钟")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
