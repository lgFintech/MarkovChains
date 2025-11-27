#!/usr/bin/env python
"""
Example demonstrating the MarkovChains library for trading analysis.

This script shows how to:
1. Create and use a basic Markov chain
2. Fit a trading model on price data
3. Generate trading signals
4. Backtest a simple strategy
"""

import numpy as np
from markov_chains import MarkovChain, TradingMarkovModel, MarketState


def basic_markov_chain_example():
    """Demonstrate basic Markov chain functionality."""
    print("=" * 60)
    print("Basic Markov Chain Example")
    print("=" * 60)
    
    # Create a simple weather model
    states = ["Sunny", "Cloudy", "Rainy"]
    mc = MarkovChain(states)
    
    # Define transition probabilities
    # Sunny -> [0.7 sunny, 0.2 cloudy, 0.1 rainy]
    # Cloudy -> [0.3 sunny, 0.4 cloudy, 0.3 rainy]
    # Rainy -> [0.2 sunny, 0.4 cloudy, 0.4 rainy]
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.4, 0.4]
    ])
    mc.set_transition_matrix(transition_matrix)
    
    print("\nTransition Matrix:")
    for i, state in enumerate(states):
        probs = ", ".join(f"{s}: {p:.1%}" for s, p in zip(states, transition_matrix[i]))
        print(f"  {state} -> {probs}")
    
    # Predict next state
    current = "Sunny"
    next_state, prob = mc.predict_next(current)
    print(f"\nFrom '{current}', most likely next state: '{next_state}' (p={prob:.1%})")
    
    # Steady-state distribution
    steady = mc.steady_state_distribution()
    print("\nLong-term weather distribution:")
    for state, prob in steady.items():
        print(f"  {state}: {prob:.1%}")
    
    # Simulate a week of weather
    rng = np.random.default_rng(42)
    week = mc.simulate("Sunny", 7, rng)
    print(f"\nSimulated week of weather: {' -> '.join(week)}")


def trading_model_example():
    """Demonstrate the trading model with simulated data."""
    print("\n" + "=" * 60)
    print("Trading Model Example")
    print("=" * 60)
    
    # Generate synthetic price data with regimes
    np.random.seed(42)
    
    # Bull market: +1% average daily return
    bull_returns = np.random.normal(0.01, 0.02, 30)
    
    # Sideways market: 0% average daily return
    sideways_returns = np.random.normal(0.0, 0.015, 20)
    
    # Bear market: -0.8% average daily return
    bear_returns = np.random.normal(-0.008, 0.02, 25)
    
    # Another bull market
    bull_returns_2 = np.random.normal(0.008, 0.018, 25)
    
    # Combine returns
    all_returns = np.concatenate([bull_returns, sideways_returns, bear_returns, bull_returns_2])
    
    # Convert returns to prices starting at 100
    prices = [100.0]
    for r in all_returns:
        prices.append(prices[-1] * (1 + r))
    
    print(f"\nGenerated {len(prices)} price points")
    print(f"Start price: ${prices[0]:.2f}")
    print(f"End price: ${prices[-1]:.2f}")
    print(f"Total return: {(prices[-1] - prices[0]) / prices[0]:.1%}")
    
    # Fit trading model
    model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
    model.fit(prices)
    
    # Show transition matrix
    print("\nMarket State Transition Matrix:")
    matrix = model.get_transition_matrix()
    states = ["Bull", "Bear", "Sideways"]
    for i, state in enumerate(states):
        probs = ", ".join(f"{s}: {p:.1%}" for s, p in zip(states, matrix[i]))
        print(f"  {state} -> {probs}")
    
    # Regime probabilities
    regime_probs = model.get_regime_probabilities()
    print("\nLong-term Regime Probabilities:")
    for state, prob in regime_probs.items():
        print(f"  {state.capitalize()}: {prob:.1%}")
    
    # Expected regime durations
    print("\nExpected Regime Durations:")
    for state in MarketState:
        duration = model.get_expected_regime_duration(state)
        print(f"  {state.value.capitalize()}: {duration:.1f} periods")
    
    # Generate a signal
    current_return = 0.015  # Today's return was +1.5%
    signal = model.predict(current_return)
    print(f"\nTrading Signal (current return: {current_return:.1%}):")
    print(f"  Current State: {signal.current_state.value}")
    print(f"  Predicted State: {signal.predicted_state.value}")
    print(f"  Action: {signal.action.upper()}")
    print(f"  Confidence: {signal.confidence:.1%}")
    
    # Backtest
    print("\nBacktest Results:")
    results = model.backtest(prices, initial_capital=10000, transaction_cost=0.001)
    print(f"  Initial Capital: $10,000")
    print(f"  Final Capital: ${results['final_capital']:.2f}")
    print(f"  Strategy Return: {results['total_return']:.1%}")
    print(f"  Buy & Hold Return: {results['buy_hold_return']:.1%}")
    print(f"  Excess Return: {results['excess_return']:.1%}")
    print(f"  Number of Trades: {results['num_trades']}")


def fit_from_sequence_example():
    """Demonstrate fitting a Markov chain from observed data."""
    print("\n" + "=" * 60)
    print("Fitting from Sequence Example")
    print("=" * 60)
    
    # Observed market states over 30 days
    observed_states = [
        "bull", "bull", "bull", "sideways", "bear",
        "bear", "sideways", "sideways", "bull", "bull",
        "bull", "bull", "sideways", "sideways", "sideways",
        "bear", "bear", "bear", "sideways", "bull",
        "bull", "bull", "bull", "bull", "sideways",
        "sideways", "bear", "bear", "sideways", "bull"
    ]
    
    # Fit Markov chain
    mc = MarkovChain(["bull", "bear", "sideways"])
    mc.fit(observed_states)
    
    print("\nObserved sequence:")
    print(f"  {' -> '.join(observed_states[:10])}...")
    print(f"  Total observations: {len(observed_states)}")
    
    print("\nEstimated Transition Probabilities:")
    matrix = mc.transition_matrix
    states = mc.states
    for i, state in enumerate(states):
        probs = ", ".join(f"{s}: {p:.1%}" for s, p in zip(states, matrix[i]))
        print(f"  {state} -> {probs}")
    
    # Check if ergodic
    is_ergodic = mc.is_ergodic()
    print(f"\nChain is ergodic: {is_ergodic}")
    
    if is_ergodic:
        # Expected return times
        print("\nExpected Return Times:")
        for state in states:
            rt = mc.expected_return_time(state)
            print(f"  {state}: {rt:.1f} steps")


if __name__ == "__main__":
    basic_markov_chain_example()
    trading_model_example()
    fit_from_sequence_example()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
