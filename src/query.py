"""
相似历史检索 + 收益分析
输入: 股票代码 + 持仓周期(周)
输出: TOP-N 相似历史片段及未来收益分布
"""
import os
import sys
import numpy as np
import pandas as pd
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DATA_DIR = os.path.join(PROJECT_DIR, "stock_daily_data")
VECTORS_DIR = os.path.join(PROJECT_DIR, "data", "vectors")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "data", "samples")


class SimilarityEngine:
    """股票历史走势相似度检索引擎"""

    def __init__(self, window_size=4):
        self.ws = window_size
        self.index = None
        self.id_map = None
        self.meta = None
        self._load()

    def _load(self):
        """加载索引和元数据"""
        index_dir = os.path.join(VECTORS_DIR, f"window_{self.ws}w")
        index_path = os.path.join(index_dir, "index.faiss")
        id_map_path = os.path.join(index_dir, "id_map.parquet")
        meta_path = os.path.join(SAMPLES_DIR, f"window_{self.ws}w", "meta.parquet")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引不存在: {index_path}\n请先运行 build_index.py")

        self.index = faiss.read_index(index_path)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 32

        self.id_map = pd.read_parquet(id_map_path)
        self.meta = pd.read_parquet(meta_path)

        print(f"已加载 {self.ws}周索引: {self.index.ntotal} 向量, dim={self.index.d}")

    def query(self, ts_code, top_k=100):
        """
        检索与指定股票当前走势最相似的历史片段
        返回: dict
        """
        # 1. 获取该股票最近一个 N 周窗口的向量
        from encoders.kronos_encoder import KronosEncoder
        from preprocess import load_and_clean, assign_iso_week, make_windows

        csv_path = os.path.join(STOCK_DATA_DIR, f"{ts_code}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"股票数据不存在: {csv_path}")

        encoder = KronosEncoder(device='cpu')
        df = load_and_clean(csv_path)
        df = assign_iso_week(df)

        windows = make_windows(df, self.ws)
        if not windows:
            raise ValueError(f"股票 {ts_code} 无足够的 {self.ws} 周窗口")

        # 取最近一个窗口
        _, anchor_date, start_idx, end_idx = windows[-1]
        # 适配缺失列
        from preprocess import ENCODER_COLS
        df_arr = df.copy()
        for c in ENCODER_COLS:
            if c not in df_arr.columns:
                df_arr[c] = np.nan
        arr = df_arr[ENCODER_COLS].to_numpy(dtype=np.float32)
        win_arr = arr[start_idx: end_idx + 1]
        query_vec = encoder.encode([win_arr]).reshape(1, -1).astype(np.float32)

        if np.isnan(query_vec).any():
            raise ValueError("查询向量含 NaN")

        # L2 归一化
        faiss.normalize_L2(query_vec)

        # 2. Faiss 检索
        distances, indices = self.index.search(query_vec, top_k)
        distances = distances[0]
        indices = indices[0]

        # 3. 获取匹配的元数据
        matches = []
        faiss_ids_used = set()
        for rank, (faiss_id, sim) in enumerate(zip(indices, distances)):
            if faiss_id < 0 or faiss_id in faiss_ids_used:
                continue
            faiss_ids_used.add(faiss_id)

            # 映射到 sample_id
            id_row = self.id_map[self.id_map["faiss_id"] == faiss_id]
            if id_row.empty:
                continue
            sample_id = id_row.iloc[0]["sample_id"]

            # 获取该样本的详细数据
            meta_row = self.meta[self.meta["sample_id"] == sample_id]
            if meta_row.empty:
                continue

            mr = meta_row.iloc[0]
            ret_cols = [f"ret_{d}d" for d in range(1, 31)]
            future_rets = {col: mr[col] for col in ret_cols if col in mr.index}

            matches.append({
                "rank": len(matches) + 1,
                "sample_id": sample_id,
                "ts_code": mr["ts_code"],
                "period": f"{mr['window_start']} ~ {mr['window_end']}",
                "cosine_sim": float(sim),
                "future_rets": future_rets,
            })

            if len(matches) >= top_k:
                break

        # 4. 汇总分析
        print(f"[DEBUG] 实际匹配数: {len(matches)}, 请求top_k: {top_k}, 索引总量: {self.index.ntotal}")
        summary = self._summarize(matches)

        return {
            "query": {
                "ts_code": ts_code,
                "anchor_date": anchor_date,
                "window_size": self.ws,
                "window_period": f"{df.iloc[start_idx]['trade_date'].strftime('%Y-%m-%d')} ~ {df.iloc[end_idx]['trade_date'].strftime('%Y-%m-%d')}",
            },
            "top_matches": matches,
            "summary": summary,
        }

    def _summarize(self, matches):
        """按 TOP-K 分组统计未来收益，只统计不超过实际匹配数的分组"""
        all_groups = {"top10": 10, "top20": 20, "top50": 50, "top100": 100}
        n = len(matches)
        groups = {name: k for name, k in all_groups.items() if k <= n}
        summary = {}

        for name, k in groups.items():
            subset = matches[:k]
            if not subset:
                continue

            stats = {}
            for d in [1, 2, 3, 5, 10, 15, 20, 30]:
                key = f"ret_{d}d"
                vals = [m["future_rets"].get(key, np.nan) for m in subset]
                vals = np.array([v for v in vals if not np.isnan(v)], dtype=np.float32)
                if len(vals) > 0:
                    stats[f"avg_ret_{d}d"] = float(np.mean(vals))
                    stats[f"win_rate_{d}d"] = float((vals > 0).mean())
                else:
                    stats[f"avg_ret_{d}d"] = np.nan
                    stats[f"win_rate_{d}d"] = np.nan
            summary[name] = stats

        return summary

    def print_report(self, result):
        """格式化打印分析报告"""
        q = result["query"]
        s = result["summary"]

        print("=" * 70)
        print(f"  股票走势相似度分析")
        print("=" * 70)
        print(f"  查询股票:  {q['ts_code']}")
        print(f"  持仓周期:  {q['window_size']} 周")
        print(f"  查询窗口:  {q['window_period']}")
        print(f"  数据库:    {self.index.ntotal:,} 个历史窗口")
        print("-" * 70)

        # 未来收益表
        days = [1, 2, 3, 5, 10, 15, 20, 30]
        header = f"  {'指标':<18}"
        for d in days:
            header += f"{f'{d}日':>8}"
        print(header)
        print("  " + "-" * (18 + 8 * len(days)))

        for group_name, stats in s.items():
            # 平均收益行
            row_avg = f"  {group_name+' 平均收益':<18}"
            for d in days:
                val = stats.get(f"avg_ret_{d}d", np.nan)
                if not np.isnan(val):
                    row_avg += f"{val:>8.2%}"
                else:
                    row_avg += f"{'':>8}"
            print(row_avg)

            # 胜率行
            row_wr = f"  {group_name+' 胜率':<18}"
            for d in days:
                val = stats.get(f"win_rate_{d}d", np.nan)
                if not np.isnan(val):
                    row_wr += f"{val:>8.1%}"
                else:
                    row_wr += f"{'':>8}"
            print(row_wr)
            print()

        print("-" * 70)
        print("  TOP-10 相似历史:")
        for m in result["top_matches"][:10]:
            print(f"    {m['rank']:>3}. {m['ts_code']:<12} {m['period']:<24}  sim={m['cosine_sim']:.4f}")
        print("=" * 70)

        # 生成可视化
        self._plot(result)

    def _plot(self, result):
        """生成收益分布可视化，保存到 output/ 目录"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            print("[WARN] matplotlib 未安装，跳过可视化")
            return

        # 尝试设置中文字体
        for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
            try:
                matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                break
            except Exception:
                continue

        q = result["query"]
        s = result["summary"]
        matches = result["top_matches"]
        groups = list(s.keys())
        if not groups or not matches:
            return

        save_dir = os.path.join(PROJECT_DIR, "output")
        os.makedirs(save_dir, exist_ok=True)

        days = [1, 2, 3, 5, 10, 15, 20, 30]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))

        # ---- Figure 1: 收益曲线 + 胜率曲线 ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for i, group_name in enumerate(groups):
            stats = s[group_name]
            avg_rets = [stats.get(f"avg_ret_{d}d", np.nan) for d in days]
            win_rates = [stats.get(f"win_rate_{d}d", np.nan) for d in days]

            ax1.plot(days, [r * 100 for r in avg_rets], 'o-', color=colors[i],
                     label=group_name, linewidth=1.5, markersize=5)
            ax2.plot(days, [w * 100 for w in win_rates], 'o-', color=colors[i],
                     label=group_name, linewidth=1.5, markersize=5)

        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_xlabel('Holding Days')
        ax1.set_ylabel('Avg Return (%)')
        ax1.set_title(f'Avg Return by Holding Period\n{q["ts_code"]} ({q["window_size"]}W)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(days)

        ax2.axhline(y=50, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Holding Days')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate by Holding Period')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(days)

        plt.tight_layout()
        curve_path = os.path.join(save_dir, f"{q['ts_code']}_{q['window_size']}w_curves.png")
        fig.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n收益曲线: {curve_path}")

        # ---- Figure 2: 收益分布直方图 ----
        largest_group = groups[-1]
        top_k = int(largest_group.replace('top', ''))
        subset = matches[:top_k]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plot_days = [5, 10, 20, 30]

        for idx, d in enumerate(plot_days):
            ax = axes[idx // 2, idx % 2]
            key = f"ret_{d}d"
            vals = np.array([m["future_rets"].get(key, np.nan) for m in subset])
            vals = vals[~np.isnan(vals)] * 100

            if len(vals) > 0:
                bins = max(8, min(30, len(vals) // 3))
                ax.hist(vals, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
                mean_val = np.mean(vals)
                ax.axvline(x=mean_val, color='#e74c3c', linestyle='--', linewidth=1.8,
                           label=f'Mean: {mean_val:.2f}%')
                ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
                ax.set_title(f'{d}-Day Return Distribution ({largest_group}, n={len(vals)})', fontsize=11)
                ax.set_xlabel('Return (%)')
                ax.set_ylabel('Count')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.2)

        plt.tight_layout()
        dist_path = os.path.join(save_dir, f"{q['ts_code']}_{q['window_size']}w_dist.png")
        fig.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"收益分布: {dist_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="股票相似度检索")
    parser.add_argument("ts_code", help="股票代码, 如 000001.SZ")
    parser.add_argument("-w", "--window", type=int, default=4, help="持仓周期(周), 默认4")
    parser.add_argument("-k", "--top_k", type=int, default=100, help="返回数量, 默认100")
    args = parser.parse_args()

    engine = SimilarityEngine(window_size=args.window)
    result = engine.query(args.ts_code, top_k=args.top_k)
    engine.print_report(result)


if __name__ == "__main__":
    main()
