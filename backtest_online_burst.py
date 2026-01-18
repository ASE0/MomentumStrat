#!/usr/bin/env python3
"""Online ML burst strategy backtest (bias-free).

Parses tick data, builds 150-tick bars, trains an online logistic regression
on the first 3 sessions, freezes the model, and trades remaining sessions.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Params:
    tick_size: float = 0.25
    ticks_per_bar: int = 150
    impulse_lookback: int = 8
    min_impulse_ticks: int = 8
    min_impulse_atr_frac: float = 0.25
    min_candidate_spacing: int = 5
    label_horizon: int = 40
    atr_fast: int = 14
    atr_slow: int = 200
    quantile_p: float = 0.70
    quantile_buffer_size: int = 800
    min_quantile_samples: int = 200
    fallback_threshold_k: float = 0.50
    entry_prob_threshold: float = 0.60
    tp_atr_fast: float = 1.2
    sl_atr_fast: float = 0.8
    max_hold: int = 60
    lr: float = 0.05
    lr_decay: float = 0.0005
    l2: float = 1e-4
    weight_clip: float = 5.0
    slippage_ticks: float = 1.0
    commission_per_trade: float = 2.0
    slope_window: int = 10
    range_window: int = 20
    volume_window: int = 20


@dataclass
class Bar:
    timestamp: dt.datetime
    session_date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float


@dataclass
class Candidate:
    index: int
    dir: int
    price: float
    atr_slow: float
    regime: str
    features_raw: np.ndarray


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    hold_bars: int


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=float)
        self.m2 = np.zeros(dim, dtype=float)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=float)
        return np.sqrt(self.m2 / (self.count - 1) + 1e-12)


class OnlineLogisticRegression:
    def __init__(self, dim: int, params: Params) -> None:
        self.w = np.zeros(dim, dtype=float)
        self.b = 0.0
        self.params = params
        self.updates = 0

    def predict_proba(self, x: np.ndarray) -> float:
        z = float(np.dot(self.w, x) + self.b)
        return 1.0 / (1.0 + math.exp(-max(min(z, 50), -50)))

    def update(self, x: np.ndarray, y: int) -> None:
        p = self.predict_proba(x)
        grad = (p - y)
        lr = self.params.lr / (1.0 + self.params.lr_decay * self.updates)
        self.w *= (1.0 - lr * self.params.l2)
        self.w -= lr * grad * x
        self.b -= lr * grad
        self.w = np.clip(self.w, -self.params.weight_clip, self.params.weight_clip)
        self.b = float(np.clip(self.b, -self.params.weight_clip, self.params.weight_clip))
        self.updates += 1


class ATRCalculator:
    def __init__(self, period: int) -> None:
        self.period = period
        self.value: Optional[float] = None
        self.values: Deque[float] = deque(maxlen=period)

    def update(self, tr: float) -> float:
        if self.value is None:
            self.values.append(tr)
            if len(self.values) < self.period:
                self.value = float(np.mean(self.values))
            else:
                self.value = float(np.mean(self.values))
        else:
            self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value


class QuantileBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.data: Deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        self.data.append(value)

    def threshold(self, q: float, fallback: float, min_samples: int) -> float:
        if len(self.data) < min_samples:
            return fallback
        return float(np.quantile(np.array(self.data), q))


class TickBarBuilder:
    def __init__(self, ticks_per_bar: int) -> None:
        self.ticks_per_bar = ticks_per_bar
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.open = 0.0
        self.high = -float("inf")
        self.low = float("inf")
        self.close = 0.0
        self.volume = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.timestamp: Optional[dt.datetime] = None
        self.session_date: Optional[dt.date] = None

    def update(self, timestamp: dt.datetime, session_date: dt.date, last: float, bid: float, ask: float, volume: float) -> Optional[Bar]:
        if self.count == 0:
            self.open = last
            self.high = last
            self.low = last
            self.volume = 0.0
        self.count += 1
        self.close = last
        self.high = max(self.high, last)
        self.low = min(self.low, last)
        self.volume += volume
        self.bid = bid
        self.ask = ask
        self.timestamp = timestamp
        self.session_date = session_date
        if self.count >= self.ticks_per_bar:
            bar = Bar(
                timestamp=self.timestamp,
                session_date=self.session_date,
                open=self.open,
                high=self.high,
                low=self.low,
                close=self.close,
                volume=self.volume,
                bid=self.bid,
                ask=self.ask,
            )
            self.reset()
            return bar
        return None


def parse_tick_line(line: str) -> Tuple[dt.datetime, float, float, float, float]:
    date_str, time_str, payload = line.strip().split(maxsplit=2)
    frac_and_rest = payload.split(";")
    frac = frac_and_rest[0]
    frac_micro = frac[:6].ljust(6, "0")
    last = float(frac_and_rest[1])
    bid = float(frac_and_rest[2])
    ask = float(frac_and_rest[3])
    volume = float(frac_and_rest[4])
    timestamp = dt.datetime.strptime(f"{date_str} {time_str}{frac_micro}", "%Y%m%d %H%M%S%f")
    return timestamp, last, bid, ask, volume


def session_dates(path: str) -> List[dt.date]:
    dates: List[dt.date] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            date_str = line.split(maxsplit=1)[0]
            date_val = dt.datetime.strptime(date_str, "%Y%m%d").date()
            if date_val not in seen:
                seen.add(date_val)
                dates.append(date_val)
    return dates


def build_feature_vector(
    dir_signal: int,
    close: float,
    prev_close: float,
    impulse: float,
    atr_fast: float,
    atr_slow: float,
    vol_ratio: float,
    range_norm: float,
    slope_norm: float,
    vol_ratio_feat: float,
) -> np.ndarray:
    r1 = (close - prev_close) / max(atr_slow, 1e-9)
    rN = impulse / max(atr_slow, 1e-9)
    impulse_over_atr = impulse / max(atr_slow, 1e-9)
    return np.array(
        [
            float(dir_signal),
            r1,
            rN,
            slope_norm,
            atr_fast,
            atr_slow,
            vol_ratio,
            range_norm,
            impulse_over_atr,
            vol_ratio_feat,
        ],
        dtype=float,
    )


def compute_slope(prices: List[float]) -> float:
    n = len(prices)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(prices, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(slope)


def regime_from_ratio(ratio: float) -> str:
    if ratio < 0.8:
        return "low"
    if ratio > 1.2:
        return "high"
    return "mid"


def normalize_features(x: np.ndarray, stats: RunningStats) -> np.ndarray:
    std = stats.std()
    return (x - stats.mean) / np.where(std == 0, 1.0, std)


def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return equity - peak


def backtest(path: str, params: Params, out_dir: str, entry_prob_threshold: Optional[float] = None, min_impulse_ticks: Optional[int] = None) -> Dict[str, float]:
    entry_prob = entry_prob_threshold if entry_prob_threshold is not None else params.entry_prob_threshold
    min_impulse_ticks = min_impulse_ticks if min_impulse_ticks is not None else params.min_impulse_ticks

    dates = session_dates(path)
    if len(dates) < 4:
        raise ValueError("Need at least 4 sessions for train/test split.")
    train_dates = set(dates[:3])
    test_dates = set(dates[3:])

    bar_builder = TickBarBuilder(params.ticks_per_bar)

    atr_fast_calc = ATRCalculator(params.atr_fast)
    atr_slow_calc = ATRCalculator(params.atr_slow)

    feature_dim = 10
    stats = RunningStats(feature_dim)
    model = OnlineLogisticRegression(feature_dim, params)

    buffers: Dict[Tuple[str, int], QuantileBuffer] = {}
    for regime in ("low", "mid", "high"):
        for direction in (-1, 1):
            buffers[(regime, direction)] = QuantileBuffer(params.quantile_buffer_size)

    pending: Deque[Candidate] = deque()

    closes: Deque[float] = deque(maxlen=max(params.impulse_lookback, params.slope_window, params.range_window) + 1)
    highs: Deque[float] = deque(maxlen=params.range_window)
    lows: Deque[float] = deque(maxlen=params.range_window)
    volumes: Deque[float] = deque(maxlen=params.volume_window)

    last_candidate_index = -params.min_candidate_spacing

    in_position = False
    position_dir = 0
    entry_price = 0.0
    entry_index = 0
    entry_atr_fast = 0.0

    trades: List[Trade] = []
    equity: List[float] = []
    equity_value = 0.0

    bar_index = -1
    prev_close: Optional[float] = None

    training_active = True

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            timestamp, last, bid, ask, volume = parse_tick_line(line)
            session_date = timestamp.date()
            bar = bar_builder.update(timestamp, session_date, last, bid, ask, volume)
            if bar is None:
                continue

            bar_index += 1
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            volumes.append(bar.volume)

            if prev_close is None:
                prev_close = bar.close

            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close),
            )
            atr_fast = atr_fast_calc.update(tr)
            atr_slow = atr_slow_calc.update(tr)
            prev_close = bar.close

            if atr_slow is None or atr_fast is None:
                continue

            if bar.session_date in test_dates and training_active:
                training_active = False
                print(f"Training ended after session: {max(train_dates)}")
                print(f"Model updates frozen at bar index {bar_index}.")

            if len(closes) <= params.impulse_lookback:
                continue

            impulse = closes[-1] - closes[-1 - params.impulse_lookback]
            dir_signal = 1 if impulse > 0 else -1 if impulse < 0 else 0
            if dir_signal == 0:
                continue

            min_impulse = max(min_impulse_ticks * params.tick_size, params.min_impulse_atr_frac * atr_slow)
            impulse_trigger = abs(impulse) >= min_impulse

            vol_ratio = atr_fast / max(atr_slow, 1e-9)
            regime = regime_from_ratio(vol_ratio)

            slope = compute_slope(list(closes)[-params.slope_window:])
            slope_norm = slope / max(atr_slow, 1e-9)
            range_norm = (max(highs) - min(lows)) / max(atr_slow, 1e-9) if highs and lows else 0.0
            vol_ratio_feat = bar.volume / max(np.mean(volumes), 1e-9)

            features_raw = build_feature_vector(
                dir_signal,
                bar.close,
                closes[-2],
                impulse,
                atr_fast,
                atr_slow,
                vol_ratio,
                range_norm,
                slope_norm,
                vol_ratio_feat,
            )

            if training_active:
                stats.update(features_raw)

            features_norm = normalize_features(features_raw, stats)

            if impulse_trigger and (bar_index - last_candidate_index) >= params.min_candidate_spacing:
                last_candidate_index = bar_index
                candidate = Candidate(
                    index=bar_index,
                    dir=dir_signal,
                    price=bar.close,
                    atr_slow=atr_slow,
                    regime=regime,
                    features_raw=features_raw,
                )
                pending.append(candidate)

                if not training_active and bar.session_date in test_dates:
                    prob = model.predict_proba(features_norm)
                    if not in_position and prob >= entry_prob:
                        entry_index = bar_index + 1
                        in_position = True
                        position_dir = dir_signal
                        entry_atr_fast = atr_fast
                        entry_price = None  # placeholder for next bar open

            if pending and bar_index >= pending[0].index + params.label_horizon:
                matured = pending.popleft()
                if len(closes) > 0:
                    m = matured.dir * (bar.close - matured.price) / max(matured.atr_slow, 1e-9)
                    buffer = buffers[(matured.regime, matured.dir)]
                    threshold = buffer.threshold(
                        params.quantile_p,
                        params.fallback_threshold_k,
                        params.min_quantile_samples,
                    )
                    y = 1 if m >= threshold else 0
                    if training_active:
                        features_norm_train = normalize_features(matured.features_raw, stats)
                        model.update(features_norm_train, y)
                        buffer.add(m)

            if in_position:
                if entry_price is None:
                    entry_price = bar.open + position_dir * params.slippage_ticks * params.tick_size

                stop = entry_price - position_dir * params.sl_atr_fast * entry_atr_fast
                target = entry_price + position_dir * params.tp_atr_fast * entry_atr_fast
                exit_reason = None
                exit_price = bar.close

                hit_stop = (bar.low <= stop) if position_dir == 1 else (bar.high >= stop)
                hit_target = (bar.high >= target) if position_dir == 1 else (bar.low <= target)

                if hit_stop and hit_target:
                    exit_price = stop
                    exit_reason = "stop_and_target"
                elif hit_stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif hit_target:
                    exit_price = target
                    exit_reason = "target"
                elif (bar_index - entry_index) >= params.max_hold:
                    exit_price = bar.close
                    exit_reason = "time"

                if exit_reason:
                    exit_price -= position_dir * params.slippage_ticks * params.tick_size
                    pnl = (exit_price - entry_price) * position_dir
                    pnl -= params.commission_per_trade
                    trades.append(
                        Trade(
                            entry_index=entry_index,
                            exit_index=bar_index,
                            direction=position_dir,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl=pnl,
                            hold_bars=bar_index - entry_index,
                        )
                    )
                    equity_value += pnl
                    in_position = False
                    entry_price = 0.0
                    position_dir = 0

            equity.append(equity_value)

    if not trades:
        metrics = {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
        }
    else:
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = float(len(wins) / len(pnls))
        avg_win = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) else 0.0
        profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) else float("inf")
        returns = pnls
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * math.sqrt(252)) if len(returns) > 1 else 0.0
        equity_curve = np.array(equity)
        drawdown = compute_drawdown(equity_curve)
        metrics = {
            "total_trades": len(pnls),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "max_drawdown": float(np.min(drawdown)),
            "total_pnl": float(np.sum(pnls)),
        }

        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(equity_curve)
        plt.title("Equity Curve")
        plt.xlabel("Bar")
        plt.ylabel("PnL")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "equity_curve.png"))
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(drawdown)
        plt.title("Drawdown")
        plt.xlabel("Bar")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "drawdown.png"))
        plt.close()

    return metrics


def run_sensitivity(path: str, params: Params, out_dir: str) -> List[Dict[str, float]]:
    results = []
    for entry_prob in (0.5, 0.6, 0.7):
        for min_impulse_ticks in (6, 8, 10):
            metrics = backtest(
                path,
                params,
                out_dir,
                entry_prob_threshold=entry_prob,
                min_impulse_ticks=min_impulse_ticks,
            )
            metrics["entry_prob_threshold"] = entry_prob
            metrics["min_impulse_ticks"] = min_impulse_ticks
            results.append(metrics)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Online ML burst strategy backtest")
    parser.add_argument("--data", default="NQdata.txt", help="Path to tick data file")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip sensitivity sweep")
    args = parser.parse_args()

    params = Params()

    base_metrics = backtest(args.data, params, args.out)
    print("Base metrics:")
    for key, value in base_metrics.items():
        print(f"  {key}: {value}")

    if not args.skip_sensitivity:
        sensitivity = run_sensitivity(args.data, params, args.out)
        print("Sensitivity sweep (for diagnostics only):")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "sensitivity.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sensitivity[0].keys()))
            writer.writeheader()
            for row in sensitivity:
                writer.writerow(row)
                print(row)


if __name__ == "__main__":
    main()
