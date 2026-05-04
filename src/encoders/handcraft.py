"""
手工特征编码器: 从 N 周日线 OHLCV 序列提取 81 维统计特征
"""
import numpy as np
from .base import BaseEncoder


class HandcraftEncoder(BaseEncoder):
    """
    输入: (T, F) numpy array, 列顺序为:
      0: open, 1: high, 2: low, 3: close, 4: pre_close,
      5: change, 6: pct_chg, 7: vol, 8: amount, 9: turnover_rate,
      10: volume_ratio, 11: ma5, 12: ma_v_5, 13: ma10, 14: ma_v_10,
      15: ma20, 16: ma_v_20, 17: ma60, 18: ma_v_60
    """

    @property
    def dim(self) -> int:
        return 81

    def encode(self, daily_data: np.ndarray) -> np.ndarray:
        if len(daily_data) < 5:
            return np.full(self.dim, np.nan, dtype=np.float32)

        o, h, l, c = [daily_data[:, i] for i in range(4)]
        vol = daily_data[:, 7]
        pct = daily_data[:, 6]
        turnover = daily_data[:, 9]

        # ---- 清理无效值 ----
        valid_mask = ~(np.isnan(o) | np.isnan(c))
        if valid_mask.sum() < 5:
            return np.full(self.dim, np.nan, dtype=np.float32)

        o, h, l, c = o[valid_mask], h[valid_mask], l[valid_mask], c[valid_mask]
        vol = vol[valid_mask]
        pct = pct[valid_mask]
        turnover = turnover[valid_mask]

        base = self._compute_features(o, h, l, c, vol, pct, turnover)

        # 多周期: 前半段 + 后半段
        mid = len(o) // 2
        if mid >= 3:
            first = self._compute_features(
                o[:mid], h[:mid], l[:mid], c[:mid],
                vol[:mid], pct[:mid], turnover[:mid]
            )
            second = self._compute_features(
                o[mid:], h[mid:], l[mid:], c[mid:],
                vol[mid:], pct[mid:], turnover[mid:]
            )
        else:
            first = np.full(27, np.nan, dtype=np.float32)
            second = np.full(27, np.nan, dtype=np.float32)

        vec = np.concatenate([base, first, second]).astype(np.float32)
        vec[np.isnan(vec)] = 0.0
        vec[np.isinf(vec)] = 0.0
        return vec

    def _compute_features(self, o, h, l, c, vol, pct, turnover):
        """计算单段 27 维特征"""
        feats = []

        # ——— 1. 价格趋势 (6) ———
        total_ret = (c[-1] - c[0]) / c[0] if c[0] != 0 else 0.0

        cummax = np.maximum.accumulate(c)
        drawdown = (cummax - c) / cummax
        max_dd = np.max(drawdown)

        daily_ret = np.diff(c) / c[:-1]
        mu, sigma = daily_ret.mean(), daily_ret.std()
        sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0

        pos_ratio = (daily_ret > 0).mean()
        avg_daily_ret = mu + 0.0  # copy

        feats.extend([total_ret, max_dd, sharpe, pos_ratio, avg_daily_ret, sigma])

        # ——— 2. 波动率 (5) ———
        tr = np.maximum(h[1:] - l[1:],
                        np.abs(h[1:] - c[:-1]))
        tr = np.maximum(tr, np.abs(l[1:] - c[:-1]))
        atr_pct = (tr / c[1:]).mean() if len(tr) > 0 else 0.0

        hl_spread = ((h - l) / c).mean()
        co_spread = (np.abs(c - o) / o).mean()
        amplitude = ((h - l) / np.roll(c, 1)).mean()
        vol_std = sigma  # same as daily return std

        feats.extend([vol_std, atr_pct, hl_spread, co_spread, amplitude])

        # ——— 3. 量价关系 (5) ———
        abs_ret = np.abs(daily_ret)
        vp_corr = np.corrcoef(vol[1:], abs_ret)[0, 1] if len(vol) > 2 else 0.0

        obv = np.cumsum(np.sign(daily_ret) * vol[1:])
        t = np.arange(len(obv))
        obv_slope = np.polyfit(t, obv, 1)[0] / (np.abs(obv).mean() + 1) if len(obv) > 2 else 0.0

        t2 = np.arange(len(vol))
        vol_slope = np.polyfit(t2, vol, 1)[0] / (vol.mean() + 1) if len(vol) > 2 else 0.0

        turnover_mean = turnover.mean()
        vol_ratio_mean = 0.0  # volume_ratio already relative

        feats.extend([vp_corr, obv_slope, vol_slope, turnover_mean, vol_ratio_mean])

        # ——— 4. 形态特征 (6) ———
        t3 = np.arange(len(c))
        slope, intercept = np.polyfit(t3, c, 1)
        slope_norm = slope / c.mean() if c.mean() != 0 else 0.0

        pred = slope * t3 + intercept
        ss_res = ((c - pred) ** 2).sum()
        ss_tot = ((c - c.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        rsi = self._compute_rsi(c, 14)
        rsi_mean_val = rsi.mean()
        rsi_last_val = rsi[-1] if len(rsi) > 0 else 50.0

        ma5 = np.convolve(c, np.ones(5) / 5, mode='valid')
        above_ma5 = (c[-len(ma5):] > ma5).mean()

        consec_up = 0
        max_consec = 0
        for r in daily_ret:
            if r > 0:
                consec_up += 1
                max_consec = max(max_consec, consec_up)
            else:
                consec_up = 0

        feats.extend([slope_norm, r2, rsi_mean_val, rsi_last_val, above_ma5, max_consec / (len(daily_ret) + 1)])

        # ——— 5. 分布特征 (5) ———
        skew = self._skewness(daily_ret)
        kurt = self._kurtosis(daily_ret)

        pos = daily_ret[daily_ret > 0]
        neg = daily_ret[daily_ret < 0]
        up_down = pos.mean() / abs(neg.mean()) if len(pos) > 0 and len(neg) > 0 else 1.0

        tail5 = np.percentile(daily_ret, 5)
        upside95 = np.percentile(daily_ret, 95)

        feats.extend([skew, kurt, up_down, tail5, upside95])

        # 确保 27 维
        result = np.array(feats, dtype=np.float32)
        if len(result) != 27:
            padded = np.zeros(27, dtype=np.float32)
            padded[:len(result)] = result[:27]
            return padded
        return result

    @staticmethod
    def _compute_rsi(close, period=14):
        if len(close) < period + 1:
            return np.array([50.0])
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
            rsi[np.isnan(rsi)] = 50.0
        return rsi

    @staticmethod
    def _skewness(x):
        n = len(x)
        if n < 3:
            return 0.0
        mu, sigma = x.mean(), x.std()
        if sigma == 0:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * np.sum(((x - mu) / sigma) ** 3)

    @staticmethod
    def _kurtosis(x):
        n = len(x)
        if n < 4:
            return 0.0
        mu, sigma = x.mean(), x.std()
        if sigma == 0:
            return 0.0
        return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((x - mu) / sigma) ** 4) \
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
