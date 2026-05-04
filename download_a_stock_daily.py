"""
A 股全量日线数据下载 —— 30 线程并发, 限流 <= 180 请求/分钟
直接调用 pro.daily(), 避免 pro_bar 内部多接口导致超限
"""
import os
import time
import datetime
import threading
import pandas as pd
import tushare as ts
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 配置 ==========
DATA_DIR = "stock_daily_data"
START_DATE = "19900101"
END_DATE = datetime.date.today().strftime("%Y%m%d")
MAX_WORKERS = 30            # 并发线程数
MAX_RPM = 500               # 每分钟最大请求数
MAX_RETRIES = 3
# ==========================

print_lock = threading.Lock()
write_lock = threading.Lock()


class RateLimiter:
    """滑动窗口限流器: 确保每秒均匀分布, 每分钟不超过 MAX_RPM 次"""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.last = time.time() - self.interval

    def acquire(self):
        with self.lock:
            now = time.time()
            wait = self.last + self.interval - now
            if wait > 0:
                time.sleep(wait)
                self.last = time.time()
            else:
                self.last = now


def load_api_key(filepath="api_key.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def download_one(ts_code, name, pro, limiter):
    """
    下载单只股票全量日线数据 (直接调用 pro.daily)
    返回 (ok: bool, ts_code: str, name: str, error: str|None)
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.acquire()
            df = pro.daily(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE,
            )

            if df is not None and not df.empty:
                df.sort_values("trade_date", inplace=True)
                filepath = os.path.join(DATA_DIR, f"{ts_code}.csv")
                with write_lock:
                    df.to_csv(filepath, index=False, encoding="utf-8-sig")

            return True, ts_code, name, None

        except Exception as e:
            msg = str(e)
            # 如果是频率超限, 多等一会
            if "频率超限" in msg:
                time.sleep(3 * attempt)
            elif attempt < MAX_RETRIES:
                time.sleep(1 * attempt)
            else:
                return False, ts_code, name, msg

    return False, ts_code, name, "unknown"


def main():
    token = load_api_key()
    ts.set_token(token)
    pro = ts.pro_api(token)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ——— 1. 获取全量股票列表 ———
    print("[1/3] 获取 A 股股票列表...")
    fields = "ts_code,symbol,name,area,industry,list_date,delist_date"
    stocks_l = pro.stock_basic(exchange="", list_status="L", fields=fields)
    stocks_d = pro.stock_basic(exchange="", list_status="D", fields=fields)
    stocks_p = pro.stock_basic(exchange="", list_status="P", fields=fields)

    all_stocks = pd.concat([stocks_l, stocks_d, stocks_p], ignore_index=True)
    all_stocks.drop_duplicates(subset="ts_code", keep="first", inplace=True)
    all_stocks.reset_index(drop=True, inplace=True)

    # ——— 2. 断点续传 ———
    already = set(
        f.replace(".csv", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
    )
    remaining = all_stocks[~all_stocks["ts_code"].isin(already)]
    remaining = remaining.reset_index(drop=True)

    total = len(remaining)
    print(f"  已下载 {len(already)} 只, 剩余 {total} 只")

    if total == 0:
        print("  已全部下载完成!")
        return

    # ——— 3. 30 线程 + 限流 180/分钟 ———
    print(f"[2/3] {MAX_WORKERS} 线程并发, 限流 {MAX_RPM} 次/分钟 (仅调 daily 接口)")
    limiter = RateLimiter(MAX_RPM)
    start = time.time()

    success = 0
    fail = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_one, row["ts_code"], row["name"], pro, limiter): row["ts_code"]
            for _, row in remaining.iterrows()
        }

        for future in as_completed(futures):
            ok, ts_code, name, err = future.result()
            completed += 1

            if ok:
                success += 1
            else:
                fail += 1

            if completed % 200 == 0 or completed == total:
                elapsed = time.time() - start
                rps = completed / elapsed if elapsed > 0 else 0
                eta_min = (total - completed) / rps / 60 if rps > 0 else 0
                with print_lock:
                    print(
                        f"  进度: {completed}/{total} "
                        f"(✅{success} ❌{fail}) "
                        f"| {rps:.1f} 只/秒 "
                        f"| 耗时 {elapsed/60:.1f}min "
                        f"| 剩余 ~{eta_min:.1f}min"
                    )

    elapsed = time.time() - start
    print(f"\n[3/3] 完成! 成功: {success}, 失败: {fail}, 耗时: {elapsed/60:.1f} 分钟")
    print(f"数据路径: {os.path.abspath(DATA_DIR)}/")
    if fail > 0 and success > 0:
        print("失败的可重新运行脚本自动重试 (断点续传)")


if __name__ == "__main__":
    main()
