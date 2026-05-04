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
        from encoders.handcraft import HandcraftEncoder
        from preprocess import load_and_clean, assign_iso_week, make_windows

        csv_path = os.path.join(STOCK_DATA_DIR, f"{ts_code}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"股票数据不存在: {csv_path}")

        encoder = HandcraftEncoder()
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
        query_vec = encoder.encode(win_arr).reshape(1, -1).astype(np.float32)

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
        """按 TOP-K 分组统计未来收益"""
        groups = {"top10": 10, "top20": 20, "top50": 50, "top100": 100}
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
