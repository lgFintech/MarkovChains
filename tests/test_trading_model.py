"""Tests for the TradingMarkovModel class."""

import pytest
import numpy as np
from src.markov_chains import TradingMarkovModel, MarketState


class TestMarketStateClassification:
    """Tests for market state classification."""
    
    def test_classify_bull(self):
        """Test classification of bull market."""
        model = TradingMarkovModel(bull_threshold=0.02, bear_threshold=-0.02)
        assert model.classify_return(0.05) == MarketState.BULL
        assert model.classify_return(0.02) == MarketState.BULL
    
    def test_classify_bear(self):
        """Test classification of bear market."""
        model = TradingMarkovModel(bull_threshold=0.02, bear_threshold=-0.02)
        assert model.classify_return(-0.05) == MarketState.BEAR
        assert model.classify_return(-0.02) == MarketState.BEAR
    
    def test_classify_sideways(self):
        """Test classification of sideways market."""
        model = TradingMarkovModel(bull_threshold=0.02, bear_threshold=-0.02)
        assert model.classify_return(0.01) == MarketState.SIDEWAYS
        assert model.classify_return(-0.01) == MarketState.SIDEWAYS
        assert model.classify_return(0.0) == MarketState.SIDEWAYS
    
    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        model = TradingMarkovModel(bull_threshold=0.05, bear_threshold=-0.05)
        assert model.classify_return(0.03) == MarketState.SIDEWAYS
        assert model.classify_return(-0.03) == MarketState.SIDEWAYS


class TestFit:
    """Tests for model fitting."""
    
    def test_fit_from_prices(self):
        """Test fitting from price data."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        
        # Generate sample prices with trends
        prices = [100, 102, 104, 103, 101, 100, 102, 105, 107]
        model.fit(prices)
        
        assert model._fitted is True
        assert model.markov_chain.transition_matrix is not None
    
    def test_fit_from_returns(self):
        """Test fitting from return data."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        
        returns = [0.02, -0.015, 0.005, 0.03, -0.02, 0.01]
        model.fit_from_returns(returns)
        
        assert model._fitted is True
    
    def test_fit_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="at least"):
            model.fit([100])  # Only one price
    
    def test_fit_from_returns_insufficient_raises(self):
        """Test that insufficient returns raises ValueError."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="at least 2 returns"):
            model.fit_from_returns([0.01])


class TestPredict:
    """Tests for prediction functionality."""
    
    def test_predict_generates_signal(self):
        """Test that predict generates a valid signal."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        returns = [0.02, 0.03, 0.01, -0.02, -0.01, 0.02, 0.03]
        model.fit_from_returns(returns)
        
        signal = model.predict(0.02)
        
        assert signal.action in ["buy", "sell", "hold"]
        assert 0 <= signal.confidence <= 1
        assert signal.current_state == MarketState.BULL
        assert isinstance(signal.predicted_state, MarketState)
        assert len(signal.state_probabilities) == 3
    
    def test_predict_unfitted_raises(self):
        """Test that predicting without fitting raises ValueError."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(0.02)
    
    def test_predict_action_logic(self):
        """Test that action logic follows predicted state."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        
        # Create data that strongly favors bull -> bull transitions
        returns = [0.02] * 50 + [0.03] * 50
        model.fit_from_returns(returns)
        
        signal = model.predict(0.02)
        # Since bull -> bull is most likely, predicted should be bull -> action should be buy
        assert signal.predicted_state == MarketState.BULL
        assert signal.action == "buy"


class TestRegimeProbabilities:
    """Tests for regime probability calculations."""
    
    def test_get_regime_probabilities(self):
        """Test getting steady-state regime probabilities."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        returns = [0.02, -0.02, 0.005, 0.02, -0.02, 0.005] * 10
        model.fit_from_returns(returns)
        
        probs = model.get_regime_probabilities()
        
        assert set(probs.keys()) == {"bull", "bear", "sideways"}
        assert sum(probs.values()) == pytest.approx(1.0)
        for p in probs.values():
            assert 0 <= p <= 1
    
    def test_get_regime_probabilities_unfitted_raises(self):
        """Test that getting regime probabilities without fitting raises."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.get_regime_probabilities()


class TestExpectedRegimeDuration:
    """Tests for expected regime duration."""
    
    def test_get_expected_regime_duration(self):
        """Test calculating expected regime duration."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        
        # Create strongly persistent bull regime
        returns = [0.02, 0.03, 0.025, 0.02] * 25 + [-0.02] + [0.02] * 10
        model.fit_from_returns(returns)
        
        duration = model.get_expected_regime_duration(MarketState.BULL)
        
        assert duration > 1  # Should stay in state for multiple periods
        assert duration != float('inf')
    
    def test_duration_unfitted_raises(self):
        """Test that getting duration without fitting raises."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.get_expected_regime_duration(MarketState.BULL)


class TestBacktest:
    """Tests for backtesting functionality."""
    
    def test_backtest_returns_dict(self):
        """Test that backtest returns expected metrics."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        
        # First fit on training data
        train_prices = [100 + i * 0.5 for i in range(50)]
        model.fit(train_prices)
        
        # Then backtest on test data
        test_prices = [125 + i * 0.3 for i in range(30)]
        results = model.backtest(test_prices)
        
        assert "final_capital" in results
        assert "total_return" in results
        assert "buy_hold_return" in results
        assert "excess_return" in results
        assert "num_trades" in results
        assert "win_rate" in results
    
    def test_backtest_unfitted_raises(self):
        """Test that backtesting without fitting raises."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.backtest([100, 101, 102])
    
    def test_backtest_insufficient_data_raises(self):
        """Test that backtesting with insufficient data raises."""
        model = TradingMarkovModel()
        returns = [0.02, 0.01, 0.03]
        model.fit_from_returns(returns)
        
        with pytest.raises(ValueError, match="at least 2 prices"):
            model.backtest([100])


class TestTransitionMatrix:
    """Tests for transition matrix access."""
    
    def test_get_transition_matrix(self):
        """Test getting the transition matrix."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        returns = [0.02, -0.02, 0.005] * 20
        model.fit_from_returns(returns)
        
        matrix = model.get_transition_matrix()
        
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix.sum(axis=1), 1.0)
    
    def test_get_transition_matrix_unfitted_raises(self):
        """Test that getting matrix without fitting raises."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.get_transition_matrix()


class TestSimulation:
    """Tests for market simulation."""
    
    def test_simulate_market(self):
        """Test simulating market states."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        returns = [0.02, -0.02, 0.005] * 20
        model.fit_from_returns(returns)
        
        states = model.simulate_market(MarketState.BULL, 10)
        
        assert len(states) == 11  # Initial + 10 steps
        assert states[0] == MarketState.BULL
        assert all(isinstance(s, MarketState) for s in states)
    
    def test_simulate_reproducible(self):
        """Test that simulation with seed is reproducible."""
        model = TradingMarkovModel(bull_threshold=0.01, bear_threshold=-0.01)
        returns = [0.02, -0.02, 0.005] * 20
        model.fit_from_returns(returns)
        
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        states1 = model.simulate_market(MarketState.BULL, 10, rng1)
        states2 = model.simulate_market(MarketState.BULL, 10, rng2)
        
        assert states1 == states2
    
    def test_simulate_unfitted_raises(self):
        """Test that simulating without fitting raises."""
        model = TradingMarkovModel()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.simulate_market(MarketState.BULL, 10)


class TestRepr:
    """Tests for string representation."""
    
    def test_repr_unfitted(self):
        """Test repr for unfitted model."""
        model = TradingMarkovModel(bull_threshold=0.02, bear_threshold=-0.02)
        repr_str = repr(model)
        
        assert "TradingMarkovModel" in repr_str
        assert "bull_threshold=0.02" in repr_str
        assert "not fitted" in repr_str
    
    def test_repr_fitted(self):
        """Test repr for fitted model."""
        model = TradingMarkovModel()
        model.fit_from_returns([0.02, 0.01, -0.01, 0.02])
        repr_str = repr(model)
        
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str
