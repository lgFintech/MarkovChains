"""Tests for the MarkovChain class."""

import pytest
import numpy as np
from src.markov_chains import MarkovChain


class TestMarkovChainInit:
    """Tests for MarkovChain initialization."""
    
    def test_init_without_matrix(self):
        """Test initialization without transition matrix."""
        mc = MarkovChain(["A", "B", "C"])
        assert mc.states == ["A", "B", "C"]
        assert mc.n_states == 3
        assert mc.transition_matrix is None
    
    def test_init_with_matrix(self):
        """Test initialization with valid transition matrix."""
        matrix = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5]
        ])
        mc = MarkovChain(["A", "B", "C"], matrix)
        assert mc.transition_matrix is not None
        np.testing.assert_array_almost_equal(mc.transition_matrix, matrix)


class TestSetTransitionMatrix:
    """Tests for setting the transition matrix."""
    
    def test_valid_matrix(self):
        """Test setting a valid transition matrix."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.6, 0.4], [0.3, 0.7]])
        mc.set_transition_matrix(matrix)
        np.testing.assert_array_almost_equal(mc.transition_matrix, matrix)
    
    def test_wrong_shape_raises(self):
        """Test that wrong shape matrix raises ValueError."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.5, 0.5]])  # Wrong shape
        with pytest.raises(ValueError, match="must be 2x2"):
            mc.set_transition_matrix(matrix)
    
    def test_negative_values_raises(self):
        """Test that negative values raise ValueError."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[1.1, -0.1], [0.5, 0.5]])
        with pytest.raises(ValueError, match="cannot be negative"):
            mc.set_transition_matrix(matrix)
    
    def test_rows_not_summing_to_one_raises(self):
        """Test that rows not summing to 1 raise ValueError."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.5, 0.4], [0.5, 0.5]])  # First row sums to 0.9
        with pytest.raises(ValueError, match="must sum to 1"):
            mc.set_transition_matrix(matrix)


class TestFit:
    """Tests for fitting from sequence data."""
    
    def test_fit_simple_sequence(self):
        """Test fitting from a simple sequence."""
        mc = MarkovChain(["A", "B"])
        sequence = ["A", "A", "B", "A", "B", "B"]
        mc.fit(sequence)
        
        # A -> A: 1, A -> B: 2 => P(A->A) = 1/3, P(A->B) = 2/3
        # B -> A: 1, B -> B: 1 => P(B->A) = 1/2, P(B->B) = 1/2
        expected = np.array([
            [1/3, 2/3],
            [1/2, 1/2]
        ])
        np.testing.assert_array_almost_equal(mc.transition_matrix, expected)
    
    def test_fit_unknown_state_raises(self):
        """Test that unknown states raise ValueError."""
        mc = MarkovChain(["A", "B"])
        sequence = ["A", "B", "C"]  # C is unknown
        with pytest.raises(ValueError, match="Unknown states"):
            mc.fit(sequence)
    
    def test_fit_handles_unvisited_states(self):
        """Test that states not in sequence get uniform distribution."""
        mc = MarkovChain(["A", "B", "C"])
        sequence = ["A", "B", "A", "B"]  # C never appears
        mc.fit(sequence)
        
        # C row should be uniform
        np.testing.assert_array_almost_equal(
            mc.transition_matrix[2], [1/3, 1/3, 1/3]
        )


class TestTransitionProbability:
    """Tests for getting transition probabilities."""
    
    def test_get_transition_probability(self):
        """Test getting specific transition probability."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        assert mc.get_transition_probability("A", "B") == pytest.approx(0.3)
        assert mc.get_transition_probability("B", "A") == pytest.approx(0.4)
    
    def test_no_matrix_raises(self):
        """Test that missing matrix raises ValueError."""
        mc = MarkovChain(["A", "B"])
        with pytest.raises(ValueError, match="not set"):
            mc.get_transition_probability("A", "B")


class TestPrediction:
    """Tests for prediction methods."""
    
    def test_predict_next(self):
        """Test predicting the most likely next state."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        state, prob = mc.predict_next("A")
        assert state == "A"
        assert prob == pytest.approx(0.7)
        
        state, prob = mc.predict_next("B")
        assert state == "B"
        assert prob == pytest.approx(0.6)
    
    def test_sample_next_reproducible(self):
        """Test that sampling with seed is reproducible."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc.set_transition_matrix(matrix)
        
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        samples1 = [mc.sample_next("A", rng1) for _ in range(10)]
        samples2 = [mc.sample_next("A", rng2) for _ in range(10)]
        
        assert samples1 == samples2


class TestSimulation:
    """Tests for simulation methods."""
    
    def test_simulate_length(self):
        """Test that simulation returns correct length."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        sequence = mc.simulate("A", 10)
        assert len(sequence) == 11  # Initial + 10 steps
        assert sequence[0] == "A"
    
    def test_simulate_reproducible(self):
        """Test that simulation with seed is reproducible."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc.set_transition_matrix(matrix)
        
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        seq1 = mc.simulate("A", 10, rng1)
        seq2 = mc.simulate("A", 10, rng2)
        
        assert seq1 == seq2


class TestSteadyState:
    """Tests for steady-state distribution."""
    
    def test_steady_state_simple(self):
        """Test steady-state for a simple chain."""
        mc = MarkovChain(["A", "B"])
        # Symmetric transition matrix -> equal steady state
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc.set_transition_matrix(matrix)
        
        steady = mc.steady_state_distribution()
        assert steady["A"] == pytest.approx(0.5)
        assert steady["B"] == pytest.approx(0.5)
    
    def test_steady_state_asymmetric(self):
        """Test steady-state for an asymmetric chain."""
        mc = MarkovChain(["A", "B"])
        # P(A->B) = 0.3, P(B->A) = 0.7
        # Steady state: π_A = 0.7/(0.3+0.7) = 0.7, π_B = 0.3
        matrix = np.array([[0.7, 0.3], [0.7, 0.3]])
        mc.set_transition_matrix(matrix)
        
        steady = mc.steady_state_distribution()
        assert steady["A"] == pytest.approx(0.7, rel=1e-3)
        assert steady["B"] == pytest.approx(0.3, rel=1e-3)


class TestNStepTransition:
    """Tests for n-step transition matrix."""
    
    def test_zero_steps(self):
        """Test that 0 steps gives identity matrix."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        result = mc.n_step_transition_matrix(0)
        np.testing.assert_array_almost_equal(result, np.eye(2))
    
    def test_one_step(self):
        """Test that 1 step gives original matrix."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        result = mc.n_step_transition_matrix(1)
        np.testing.assert_array_almost_equal(result, matrix)
    
    def test_multiple_steps(self):
        """Test n-step transition for multiple steps."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        result = mc.n_step_transition_matrix(2)
        expected = matrix @ matrix
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_negative_raises(self):
        """Test that negative n raises ValueError."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        with pytest.raises(ValueError, match="non-negative"):
            mc.n_step_transition_matrix(-1)


class TestErgodic:
    """Tests for ergodicity check."""
    
    def test_ergodic_chain(self):
        """Test that a fully connected chain is ergodic."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc.set_transition_matrix(matrix)
        
        assert mc.is_ergodic() is True
    
    def test_absorbing_chain_not_ergodic(self):
        """Test that an absorbing chain is not ergodic."""
        mc = MarkovChain(["A", "B"])
        # State B is absorbing
        matrix = np.array([[0.5, 0.5], [0.0, 1.0]])
        mc.set_transition_matrix(matrix)
        
        assert mc.is_ergodic() is False


class TestExpectedReturnTime:
    """Tests for expected return time."""
    
    def test_expected_return_time(self):
        """Test expected return time calculation."""
        mc = MarkovChain(["A", "B"])
        matrix = np.array([[0.7, 0.3], [0.7, 0.3]])
        mc.set_transition_matrix(matrix)
        
        # Steady state is [0.7, 0.3]
        # Expected return time to A = 1/0.7 ≈ 1.43
        # Expected return time to B = 1/0.3 ≈ 3.33
        
        return_time_a = mc.expected_return_time("A")
        return_time_b = mc.expected_return_time("B")
        
        assert return_time_a == pytest.approx(1/0.7, rel=1e-2)
        assert return_time_b == pytest.approx(1/0.3, rel=1e-2)
