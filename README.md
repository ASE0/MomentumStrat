# MomentumStrat

Online-ML momentum “burst” strategy with a bias-free backtest on NQ tick data.

## Files

- `backtest_online_burst.py`: main script.
- `NQdata.txt`: tick data in `yyyyMMdd HHmmss fffffff;last;bid;ask;volume` format.

## How it works

### Parsing and session definition

- The script parses each line into a timestamp, last, bid, ask, and volume.
- **Sessions are defined by the date field** (`yyyyMMdd`). The first three unique dates are the training sessions. Sessions 4+ are the test sessions.
- The script prints the exact session boundary when training ends.

### Bar construction (150-tick bars)

To keep the event stream manageable and consistent with the repo’s original intent, the script aggregates ticks into fixed **150-tick bars**. Each bar uses only information from its own ticks and is emitted once the 150th tick arrives (no future ticks needed).

Fields per bar:
- Open/High/Low/Close of last price
- Sum of volume
- Last bid/ask in the bar

### No lookahead / leakage prevention

- Features at time *t* are computed using only data available up to bar *t*.
- Labels are created **only when their horizon matures**, and the model is updated at that time (online, delayed labels).
- **Training stops after session 3**. After that:
  - Model weights are frozen.
  - Normalization statistics are frozen.
  - Quantile buffers are frozen.
- Trading occurs **only** on sessions 4+.

### Candidate generation (impulse trigger)

At each bar *t*:

```
impulse = close(t) - close(t - N)
dir = sign(impulse)
```

A candidate is created if:

```
abs(impulse) >= max(MinImpulseTicks * tickSize, MinImpulseATRFrac * ATR_slow)
```

Candidates are spaced out by `min_candidate_spacing` bars.

### Volatility baseline and regimes

- `ATR_fast` and `ATR_slow` are computed using Wilder’s smoothing.
- `volRatio = ATR_fast / ATR_slow` and bucketed into `low / mid / high`.

### Labels (learnable burst)

For a candidate at time *t0* with direction `dir`:

```
H = label horizon
m = dir * (price(t0 + H) - price(t0)) / ATR_slow(t0)
```

A rolling quantile threshold is computed per `(regime, direction)` bucket using **training-only** samples.

```
y = 1 if m >= Quantile_p else 0
```

If the buffer has fewer than `MinQuantileSamples`, a fallback threshold `FallbackThresholdK` is used.

### Features

Feature vector:

- Direction (`dir`)
- `r1`: 1-bar return normalized by `ATR_slow`
- `rN`: N-bar return normalized by `ATR_slow`
- Linear regression slope (normalized by `ATR_slow`)
- `ATR_fast`
- `ATR_slow`
- `volRatio`
- Range over window normalized by `ATR_slow`
- `impulseOverAtrSlow`
- Volume ratio (`current_volume / SMA(volume)`)

Features are normalized with running mean/std computed **only during training**.

### Online model

Logistic regression with SGD updates on label arrival:

```
p = sigmoid(w·x + b)
```

- L2 regularization
- Learning rate decay
- Weight clipping for stability

### Trading logic (test only)

Entry (sessions 4+):

- Candidate fires
- `pBurst >= EntryProbThreshold`
- Direction follows impulse
- **Fill at next bar open** (conservative, forward-only)
- Slippage and commission applied

Exit:

- Stop loss = `SL_ATRfast * ATR_fast(entry)`
- Take profit = `TP_ATRfast * ATR_fast(entry)`
- Max hold time = `MaxHold` bars

If both stop and target are hit inside the same bar, the stop is taken first (conservative).

### Outputs

- Metrics: total trades, win rate, average win/loss, profit factor, Sharpe, max drawdown.
- `outputs/equity_curve.png` and `outputs/drawdown.png`.
- `outputs/sensitivity.csv` (small parameter sweep for diagnostics only).

## Running

```
python backtest_online_burst.py --data NQdata.txt --out outputs
```

## Parameter defaults

See `Params` in `backtest_online_burst.py` for defaults. Key values match the provided spec:

- `tick_size = 0.25`
- `impulse_lookback = 8`
- `min_impulse_ticks = 8`
- `min_impulse_atr_frac = 0.25`
- `label_horizon = 40`
- `atr_fast = 14`
- `atr_slow = 200`
- `quantile_p = 0.70`
- `quantile_buffer_size = 800`
- `min_quantile_samples = 200`
- `fallback_threshold_k = 0.50`
- `entry_prob_threshold = 0.60`
- `tp_atr_fast = 1.2`
- `sl_atr_fast = 0.8`
- `max_hold = 60`

## Notes

- The sensitivity sweep is **not** used for hyperparameter selection; it is printed for diagnostics only.
- The code avoids bid/ask size and uses only the provided fields.
- Model updates, normalization, and quantile buffers are frozen after session 3 to prevent leakage.
