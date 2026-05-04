"""
商品期货 + 期权日线数据下载 —— 30 线程并发, 限流 <= 500 请求/分钟
期货: CFFEX / DCE / CZCE / SHFE / INE (2008年起)
期权: 各交易所商品期权
"""
import os
import time
import datetime
import threading
import pandas as pd
import tushare as ts
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 配置 ==========
DATA_DIR = "futures_options_daily_data"
START_DATE = "20080101"
END_DATE = datetime.date.today().strftime("%Y%m%d")
MAX_WORKERS = 30
MAX_RPM = 500
MAX_RETRIES = 3

# 五大期货交易所
EXCHANGES = ["CFFEX", "DCE", "CZCE", "SHFE", "INE"]
# ==========================

print_lock = threading.Lock()
write_lock = threading.Lock()


class RateLimiter:
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


def download_futures_one(ts_code, name, pro, limiter):
    """下载单只期货日线"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.acquire()
            df = pro.fut_daily(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE,
            )
            if df is not None and not df.empty:
                df.sort_values("trade_date", inplace=True)
                filepath = os.path.join(DATA_DIR, "futures", f"{ts_code}.csv")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with write_lock:
                    df.to_csv(filepath, index=False, encoding="utf-8-sig")
            return True, ts_code, name, None

        except Exception as e:
            msg = str(e)
            if "频率超限" in msg:
                time.sleep(3 * attempt)
            elif attempt < MAX_RETRIES:
                time.sleep(1 * attempt)
            else:
                return False, ts_code, name, msg
    return False, ts_code, name, "unknown"


def download_opt_one(ts_code, name, pro, limiter):
    """下载单只期权日线"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.acquire()
            df = pro.opt_daily(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE,
            )
            if df is not None and not df.empty:
                df.sort_values("trade_date", inplace=True)
                filepath = os.path.join(DATA_DIR, "options", f"{ts_code}.csv")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with write_lock:
                    df.to_csv(filepath, index=False, encoding="utf-8-sig")
            return True, ts_code, name, None

        except Exception as e:
            msg = str(e)
            if "频率超限" in msg:
                time.sleep(3 * attempt)
            elif attempt < MAX_RETRIES:
                time.sleep(1 * attempt)
            else:
                return False, ts_code, name, msg
    return False, ts_code, name, "unknown"


def collect_contracts(pro):
    """收集所有商品期货 + 期权合约"""
    all_contracts = []  # [(ts_code, name, type), ...]

    # ---- 期货 ----
    print("  获取期货合约列表...")
    for ex in EXCHANGES:
        try:
            df = pro.fut_basic(exchange=ex, fut_type=1, fields="ts_code,name,exchange,list_date,delist_date")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    all_contracts.append((row["ts_code"], row["name"], "futures"))
        except Exception as e:
            print(f"    交易所 {ex} 期货列表获取失败: {e}")

    # ---- 期权 ----
    print("  获取期权合约列表...")
    for ex in EXCHANGES:
        try:
            df = pro.opt_basic(exchange=ex, fields="ts_code,name,exchange,list_date,delist_date")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    all_contracts.append((row["ts_code"], row["name"], "options"))
        except Exception as e:
            print(f"    交易所 {ex} 期权列表获取失败: {e}")

    return all_contracts


def main():
    token = load_api_key()
    ts.set_token(token)
    pro = ts.pro_api(token)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ——— 1. 收集所有合约 ———
    print("[1/3] 收集期货 + 期权合约列表...")
    contracts = collect_contracts(pro)
    print(f"  共 {len(contracts)} 只合约 (期货 + 期权)")

    # 按类型统计
    n_fut = sum(1 for c in contracts if c[2] == "futures")
    n_opt = sum(1 for c in contracts if c[2] == "options")
    print(f"    期货: {n_fut}  期权: {n_opt}")

    # ——— 2. 断点续传 ———
    already_fut = set()
    already_opt = set()
    fut_dir = os.path.join(DATA_DIR, "futures")
    opt_dir = os.path.join(DATA_DIR, "options")

    if os.path.exists(fut_dir):
        already_fut = set(f.replace(".csv", "") for f in os.listdir(fut_dir) if f.endswith(".csv"))
    if os.path.exists(opt_dir):
        already_opt = set(f.replace(".csv", "") for f in os.listdir(opt_dir) if f.endswith(".csv"))

    remaining = [
        c for c in contracts
        if (c[2] == "futures" and c[0] not in already_fut)
        or (c[2] == "options" and c[0] not in already_opt)
    ]
    total = len(remaining)
    n_fut_rem = sum(1 for c in remaining if c[2] == "futures")
    n_opt_rem = sum(1 for c in remaining if c[2] == "options")
    print(f"  已下载: 期货 {len(already_fut)} / 期权 {len(already_opt)}")
    print(f"  剩余:  期货 {n_fut_rem} / 期权 {n_opt_rem} / 共 {total}")

    if total == 0:
        print("  已全部下载完成!")
        return

    # ——— 3. 并发下载 ———
    print(f"[2/3] {MAX_WORKERS} 线程并发, 限流 {MAX_RPM} 次/分钟")
    limiter = RateLimiter(MAX_RPM)
    start = time.time()

    success = 0
    fail = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures_map = {}
        for ts_code, name, ctype in remaining:
            if ctype == "futures":
                fut = executor.submit(download_futures_one, ts_code, name, pro, limiter)
            else:
                fut = executor.submit(download_opt_one, ts_code, name, pro, limiter)
            futures_map[fut] = ts_code

        for future in as_completed(futures_map):
            ok, ts_code, name, err = future.result()
            completed += 1

            if ok:
                success += 1
            else:
                fail += 1

            if completed % 500 == 0 or completed == total:
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
    print(f"  期货: {os.path.join(DATA_DIR, 'futures')}/")
    print(f"  期权: {os.path.join(DATA_DIR, 'options')}/")
    if fail > 0 and success > 0:
        print("失败的可重新运行脚本自动重试 (断点续传)")


if __name__ == "__main__":
    main()
