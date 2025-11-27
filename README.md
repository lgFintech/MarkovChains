# MarkovChains

A Python library for Markov Chain analysis in trading applications.

## Overview

MarkovChains provides tools for building, analyzing, and applying discrete-time Markov chains to financial market data for algorithmic trading strategies. The library includes:

- **Core Markov Chain implementation** with transition matrix operations, steady-state computation, and simulation capabilities
- **Trading-specific model** for classifying market states (bull/bear/sideways) and generating trading signals

## Installation

```bash
# Clone the repository
git clone https://github.com/lgFintech/MarkovChains.git
cd MarkovChains

# Install with pip
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Markov Chain

```python
import numpy as np
from markov_chains import MarkovChain

# Create a Markov chain with defined states
mc = MarkovChain(["A", "B", "C"])

# Set transition probabilities manually
transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # From state A
    [0.3, 0.4, 0.3],  # From state B
    [0.2, 0.3, 0.5]   # From state C
])
mc.set_transition_matrix(transition_matrix)

# Or fit from observed data
mc.fit(["A", "A", "B", "C", "B", "A", "C", "C"])

# Predict next state
next_state, probability = mc.predict_next("A")
print(f"Most likely next state: {next_state} (p={probability:.2f})")

# Get steady-state distribution
steady_state = mc.steady_state_distribution()
print(f"Long-term probabilities: {steady_state}")

# Simulate future states
future_states = mc.simulate("A", n_steps=10)
print(f"Simulated sequence: {future_states}")
```

### Trading Model

```python
from markov_chains import TradingMarkovModel, MarketState

# Create a trading model
model = TradingMarkovModel(
    bull_threshold=0.02,   # +2% return = bull market
    bear_threshold=-0.02   # -2% return = bear market
)

# Fit on historical price data
prices = [100, 102, 101, 103, 105, 104, 102, 100, 98, 99, 101, 103]
model.fit(prices)

# Generate trading signal
signal = model.predict(current_return=0.015)
print(f"Action: {signal.action}")
print(f"Confidence: {signal.confidence:.2%}")
print(f"Current state: {signal.current_state.value}")
print(f"Predicted state: {signal.predicted_state.value}")

# Get regime probabilities
regime_probs = model.get_regime_probabilities()
print(f"Long-term regime probabilities: {regime_probs}")

# Expected duration of market regimes
bull_duration = model.get_expected_regime_duration(MarketState.BULL)
print(f"Expected bull market duration: {bull_duration:.1f} periods")

# Backtest the strategy
results = model.backtest(prices, initial_capital=10000)
print(f"Strategy return: {results['total_return']:.2%}")
print(f"Buy-and-hold return: {results['buy_hold_return']:.2%}")
```

## API Reference

### MarkovChain

Core class for discrete-time Markov chain operations.

**Methods:**
- `set_transition_matrix(matrix)` - Set the transition probability matrix
- `fit(sequence)` - Estimate probabilities from observed data
- `get_transition_probability(from_state, to_state)` - Get specific transition probability
- `predict_next(current_state)` - Predict most likely next state
- `sample_next(current_state)` - Sample next state from distribution
- `simulate(initial_state, n_steps)` - Generate a sequence of states
- `steady_state_distribution()` - Compute long-term distribution
- `n_step_transition_matrix(n)` - Get n-step transition probabilities
- `is_ergodic()` - Check if chain is ergodic
- `expected_return_time(state)` - Expected steps to return to a state

### TradingMarkovModel

Trading-specific model for market state analysis.

**Methods:**
- `fit(prices, period=1)` - Fit model on price data
- `fit_from_returns(returns)` - Fit model on return data
- `predict(current_return)` - Generate trading signal
- `classify_return(return_value)` - Classify return into market state
- `get_regime_probabilities()` - Get steady-state regime distribution
- `get_expected_regime_duration(state)` - Expected duration of a regime
- `backtest(prices, initial_capital, transaction_cost)` - Backtest strategy
- `simulate_market(initial_state, n_periods)` - Simulate future market states
- `get_transition_matrix()` - Get the transition matrix

### MarketState

Enum for market states:
- `MarketState.BULL` - Strong upward trend
- `MarketState.BEAR` - Strong downward trend  
- `MarketState.SIDEWAYS` - No clear trend

### TradingSignal

Dataclass containing trading signal information:
- `action` - "buy", "sell", or "hold"
- `confidence` - Probability of predicted state
- `current_state` - Current market state
- `predicted_state` - Predicted next state
- `state_probabilities` - Full probability distribution

## Mathematical Background

### Markov Chains

A Markov chain is a stochastic process with the *Markov property*: the probability of transitioning to any particular state depends solely on the current state, not on the sequence of events that preceded it.

For states $S = \{s_1, s_2, ..., s_n\}$, the transition matrix $P$ where $P_{ij}$ represents the probability of moving from state $s_i$ to state $s_j$:

$$P(X_{t+1} = s_j | X_t = s_i) = P_{ij}$$

The steady-state distribution $\pi$ satisfies:

$$\pi = \pi P$$

### Application to Trading

In trading, we classify market conditions into discrete states based on returns:
- **Bull**: Returns exceed a positive threshold
- **Bear**: Returns fall below a negative threshold  
- **Sideways**: Returns are within the threshold range

By analyzing historical transitions between these states, we can:
1. Estimate the probability of market regime changes
2. Predict the most likely future market state
3. Generate trading signals based on predictions
4. Calculate expected duration of market regimes

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/markov_chains

# Run specific test file
pytest tests/test_markov_chain.py -v
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
Do this once per ticker (META, AMZN, TGT) plus SPY/QQQ.

1. Price trend panel (per ticker)

On each ticker chart (META, AMZN, TGT):

Add indicators:

50-period EMA (on daily)

200-period EMA (on daily)

Regime cues:

Bull trend:

Price above 200 EMA

50 EMA above 200 EMA and both sloping up

Bear/stressed trend:

Price below 200 EMA

50 EMA below 200 EMA and sloping down

Chop/transition:

Price whipping around 50 EMA, 50 ≈ 200 (flat / crossing often)

In TradingView:
Indicators → Moving Average Exponential (EMA), add two, set lengths 50 and 200.

2. Volatility panel (per ticker)

Use:

ATR(14) → measures range/volatility

Historical Volatility (if you have it) or StDev indicator

In TradingView:

Indicators → Average True Range → length 14

Indicators → Standard Deviation → length 20, applied to Close

Regime cues (per ticker):

ATR(14) below its 6–12 month median → Calm

ATR(14) above its 70–80th percentile → Stressed

In between → Neutral

You can eyeball this at first; later you can codify as exact level marks.

3. Market-wide stress (SPY / QQQ + VIX)

On a separate layout:

Chart: SPY or QQQ (daily)

Add 50 & 200 EMAs again

Add VIX in a separate pane (symbol: VIX or TVC:VIX in TradingView)

Rules of thumb:

Calm / Risk-on:

SPY/QQQ above 200 EMA, trend up

VIX < ~18–20 and drifting sideways/down

Stressed / Risk-off:

SPY/QQQ breaks below 200 EMA

VIX spikes above ~22–25, especially with term-structure inversion (front VIX futures > back ones – you’ll see this in futures, but as a shortcut: VIX big spike + index breakdown is enough)

You can add Correlation Coefficient between your ticker and SPY:

Indicators → Correlation Coefficient → SPY as input

If correlation jumps toward 1.0 during a drop, that’s a systemic risk regime.

4. How to read it for your strategies

Good environment for PMCC / short calls:

Ticker:

Price above 200 EMA, 50 > 200, both up

Ticker ATR moderate (not spiking)

Market:

SPY/QQQ above 200 EMA

VIX calm / drifting down

Good for debit spreads / long gamma / upside convexity:

Ticker:

Strong breakout above resistance or 50 EMA with volume

ATR rising but not crazy

Market:

Risk-on or transitioning from calm to momentum

Good for hedges / reducing size:

Market:

SPY/QQQ losing 200 EMA

VIX spiking & staying > 22–25

Tickers:

Big expansion in ATR

Gaps, erratic days

B. Excel regime dashboard (DIY quant version)

Let’s say you have daily data (Date, Close) for:

SPY (or QQQ)

VIX

META, AMZN, TGT

You can create a simple sheet per symbol.

Example columns (for SPY tab)

Assume:

Column A: Date

Column B: Close

Add:

C – 20d Return StdDev (Realized Vol Proxy)

=IF(COUNT(B2:B21)<20,"",STDEV.S(B2:B21))


Drag down. (Adjust ranges so the last 20 closes are always used.)

D – 50d Moving Average

=IF(COUNT(B2:B51)<50,"",AVERAGE(B2:B51))


E – 200d Moving Average

=IF(COUNT(B2:B201)<200,"",AVERAGE(B2:B201))


F – Regime Flag (text)
Example logic:

=IF(OR(E2="",D2=""),"",
   IF(AND(B2>E2, D2>E2),
      "TREND_UP",
      IF(AND(B2<E2, D2<E2),
         "TREND_DOWN",
         "CHOP"
      )
   )
 )


G – Vol Regime
First compute a long-term average/percentile of C in a helper area (e.g., average and 70th percentile using AVERAGE() and PERCENTILE.INC() over the last N rows).
Then something like:

=IF(C2="","",
   IF(C2<$M$1,"CALM",
      IF(C2>$N$1,"STRESSED","MID")
   )
 )


where M1 = calm threshold, N1 = stressed threshold.

You can repeat this structure for META, AMZN, TGT, and then make a Summary tab with:

SPY Regime

META Trend + Vol Regime

AMZN Trend + Vol Regime

TGT Trend + Vol Regime

Then your brain does:

“SPY = TREND_UP & CALM; META = TREND_UP & MID → PMCC ok, write short calls”

“SPY = TREND_DOWN & STRESSED → reduce size, roll further OTM, maybe add hedges”

2️⃣ Regime-aware PMCC simulator (Python)

Now let’s upgrade your PMCC simulator so that:

It simulates fat-tailed paths

It classifies each day on each path into regimes based on:

20-day realized volatility (per path)

50-day moving average / trend (per path)

It returns:

PMCC P&L distribution

Regime labels for analysis

You can later plug in regime-based logic (e.g., “only write short calls in calm/trend regimes”).

