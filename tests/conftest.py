"""Test configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def simple_transition_matrix():
    """A simple 2-state transition matrix."""
    return np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])


@pytest.fixture
def three_state_transition_matrix():
    """A 3-state transition matrix for market states."""
    return np.array([
        [0.6, 0.2, 0.2],  # Bull -> Bull, Bear, Sideways
        [0.3, 0.5, 0.2],  # Bear -> Bull, Bear, Sideways
        [0.3, 0.2, 0.5],  # Sideways -> Bull, Bear, Sideways
    ])


@pytest.fixture
def sample_prices():
    """Sample price data for testing."""
    # Generate some realistic-looking price data
    np.random.seed(42)
    base = 100.0
    returns = np.random.normal(0.0005, 0.02, 100)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


@pytest.fixture
def trending_prices():
    """Price data with a clear upward trend."""
    prices = [100 + 0.5 * i for i in range(50)]
    return prices


@pytest.fixture
def volatile_returns():
    """Return data with clear state changes."""
    # Bull -> Sideways -> Bear -> Bull
    returns = (
        [0.03, 0.02, 0.025] * 5 +  # Bull
        [0.005, -0.005, 0.002] * 5 +  # Sideways
        [-0.02, -0.03, -0.025] * 5 +  # Bear
        [0.02, 0.03, 0.02] * 5  # Bull
    )
    return returns
