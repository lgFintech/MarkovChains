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