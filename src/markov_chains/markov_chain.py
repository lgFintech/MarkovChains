"""
Core Markov Chain implementation for mathematical operations.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple


class MarkovChain:
    """
    A discrete-time Markov chain implementation.
    
    A Markov chain is a stochastic process that transitions from one state
    to another based on transition probabilities that depend only on the
    current state (memoryless property).
    
    Attributes:
        states: List of state names
        transition_matrix: Square matrix of transition probabilities
        n_states: Number of states in the chain
    """
    
    def __init__(
        self,
        states: List[str],
        transition_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize a Markov chain.
        
        Args:
            states: List of state names
            transition_matrix: Optional transition probability matrix.
                              If None, must be set later using fit() or set_transition_matrix()
        """
        self.states = states
        self.n_states = len(states)
        self._state_to_index = {state: i for i, state in enumerate(states)}
        self._index_to_state = {i: state for i, state in enumerate(states)}
        
        if transition_matrix is not None:
            self.set_transition_matrix(transition_matrix)
        else:
            self.transition_matrix = None
    
    def set_transition_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the transition probability matrix.
        
        Args:
            matrix: Square matrix where entry [i,j] is P(state_j | state_i)
        
        Raises:
            ValueError: If matrix is not valid (wrong shape or rows don't sum to 1)
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        
        if matrix.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"Transition matrix must be {self.n_states}x{self.n_states}, "
                f"got {matrix.shape}"
            )
        
        # Check for non-negative values
        if np.any(matrix < 0):
            raise ValueError("Transition probabilities cannot be negative")
        
        # Check row sums (should be 1 within tolerance)
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10):
            raise ValueError(
                f"Each row must sum to 1. Got row sums: {row_sums}"
            )
        
        self.transition_matrix = matrix
    
    def fit(self, sequence: List[str]) -> "MarkovChain":
        """
        Estimate transition probabilities from a sequence of observations.
        
        Args:
            sequence: List of observed states
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If sequence contains unknown states
        """
        # Validate sequence
        unknown_states = set(sequence) - set(self.states)
        if unknown_states:
            raise ValueError(f"Unknown states in sequence: {unknown_states}")
        
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(sequence) - 1):
            from_state = self._state_to_index[sequence[i]]
            to_state = self._state_to_index[sequence[i + 1]]
            transition_counts[from_state, to_state] += 1
        
        # Convert counts to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        
        # Handle rows with zero counts (assign uniform probability)
        zero_rows = row_sums.flatten() == 0
        row_sums[zero_rows] = 1  # Prevent division by zero
        
        self.transition_matrix = transition_counts / row_sums
        
        # For states that never appeared, use uniform distribution
        self.transition_matrix[zero_rows] = 1.0 / self.n_states
        
        return self
    
    def get_transition_probability(self, from_state: str, to_state: str) -> float:
        """
        Get the probability of transitioning from one state to another.
        
        Args:
            from_state: Starting state
            to_state: Ending state
        
        Returns:
            Transition probability
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not set. Call fit() or set_transition_matrix() first.")
        
        from_idx = self._state_to_index[from_state]
        to_idx = self._state_to_index[to_state]
        return float(self.transition_matrix[from_idx, to_idx])
    
    def predict_next(self, current_state: str) -> Tuple[str, float]:
        """
        Predict the most likely next state.
        
        Args:
            current_state: Current state
        
        Returns:
            Tuple of (predicted_state, probability)
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not set. Call fit() or set_transition_matrix() first.")
        
        current_idx = self._state_to_index[current_state]
        probabilities = self.transition_matrix[current_idx]
        next_idx = np.argmax(probabilities)
        
        return self._index_to_state[next_idx], float(probabilities[next_idx])
    
    def sample_next(self, current_state: str, rng: Optional[np.random.Generator] = None) -> str:
        """
        Sample the next state according to transition probabilities.
        
        Args:
            current_state: Current state
            rng: Optional random number generator for reproducibility
        
        Returns:
            Sampled next state
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not set. Call fit() or set_transition_matrix() first.")
        
        if rng is None:
            rng = np.random.default_rng()
        
        current_idx = self._state_to_index[current_state]
        probabilities = self.transition_matrix[current_idx]
        next_idx = rng.choice(self.n_states, p=probabilities)
        
        return self._index_to_state[next_idx]
    
    def simulate(
        self,
        initial_state: str,
        n_steps: int,
        rng: Optional[np.random.Generator] = None
    ) -> List[str]:
        """
        Simulate a sequence of states.
        
        Args:
            initial_state: Starting state
            n_steps: Number of steps to simulate
            rng: Optional random number generator for reproducibility
        
        Returns:
            List of states including the initial state
        """
        if rng is None:
            rng = np.random.default_rng()
        
        sequence = [initial_state]
        current_state = initial_state
        
        for _ in range(n_steps):
            current_state = self.sample_next(current_state, rng)
            sequence.append(current_state)
        
        return sequence
    
    def steady_state_distribution(self, tol: float = 1e-10, max_iter: int = 1000) -> Dict[str, float]:
        """
        Compute the steady-state (stationary) distribution of the Markov chain.
        
        The steady-state distribution π satisfies: π = π * P
        where P is the transition matrix.
        
        Args:
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
        
        Returns:
            Dictionary mapping states to their steady-state probabilities
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not set. Call fit() or set_transition_matrix() first.")
        
        # Power iteration method
        # Start with uniform distribution
        pi = np.ones(self.n_states) / self.n_states
        
        for _ in range(max_iter):
            pi_new = pi @ self.transition_matrix
            if np.allclose(pi, pi_new, atol=tol):
                break
            pi = pi_new
        
        return {state: float(pi[i]) for i, state in enumerate(self.states)}
    
    def expected_return_time(self, state: str) -> float:
        """
        Compute the expected return time to a given state.
        
        The expected return time is the expected number of steps to return
        to a state, starting from that state. For ergodic chains, this equals
        1 / π(state) where π is the steady-state distribution.
        
        Args:
            state: The state to compute return time for
        
        Returns:
            Expected number of steps to return to the state
        """
        steady_state = self.steady_state_distribution()
        prob = steady_state[state]
        
        if prob == 0:
            return float('inf')
        
        return 1.0 / prob
    
    def n_step_transition_matrix(self, n: int) -> np.ndarray:
        """
        Compute the n-step transition probability matrix.
        
        P^n[i,j] = Probability of being in state j after n steps, starting from state i
        
        Args:
            n: Number of steps
        
        Returns:
            n-step transition probability matrix
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not set. Call fit() or set_transition_matrix() first.")
        
        if n < 0:
            raise ValueError("n must be non-negative")
        
        if n == 0:
            return np.eye(self.n_states)
        
        return np.linalg.matrix_power(self.transition_matrix, n)
    
    def is_ergodic(self, tol: float = 1e-10) -> bool:
        """
        Check if the Markov chain is ergodic (irreducible and aperiodic).
        
        An ergodic chain has a unique steady-state distribution that is
        reached from any starting state.
        
        Args:
            tol: Tolerance for comparing probabilities to zero
        
        Returns:
            True if the chain is ergodic
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not set. Call fit() or set_transition_matrix() first.")
        
        # Check if the chain is irreducible by checking if P^k is positive
        # for some k (use n^2 as upper bound for finite chains)
        power_matrix = np.eye(self.n_states)
        combined = np.zeros((self.n_states, self.n_states))
        
        for _ in range(self.n_states ** 2):
            power_matrix = power_matrix @ self.transition_matrix
            combined += power_matrix
            
            # Check if all entries are positive
            if np.all(combined > tol):
                return True
        
        return False
    
    def __repr__(self) -> str:
        """String representation of the Markov chain."""
        return f"MarkovChain(states={self.states}, n_states={self.n_states})"
