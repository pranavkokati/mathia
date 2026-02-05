"""
Markov Chain Model for Chess Move Prediction

Mathematical Foundation:
========================
A Markov chain is a stochastic process where the probability of transitioning
to any particular state depends only on the current state (Markov property).

For chess moves:
- State: A chess move in Standard Algebraic Notation (e.g., "e4", "Nf3")
- Transition: Moving from one state (current move) to another (next move)

The transition probability P(j|i) is defined as:
    P_ij = Count(move j follows move i) / Count(all moves following move i)

Transition Matrix:
    P = [P_ij] where P_ij = probability of transitioning from state i to state j

    Properties:
    - Each row sums to 1: Σ_j P_ij = 1
    - All entries are non-negative: P_ij ≥ 0

State Distribution Evolution:
    If π_n is the probability distribution over states at step n,
    then: π_(n+1) = π_n * P

    After k steps: π_(n+k) = π_n * P^k
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json


class ChessMarkovChain:
    """
    First-order Markov Chain model for chess move prediction.

    Attributes:
        states: List of unique states (moves)
        state_to_idx: Mapping from state name to index
        idx_to_state: Mapping from index to state name
        transition_counts: Raw counts of transitions
        transition_matrix: Probability matrix P
    """

    def __init__(self):
        self.states: List[str] = []
        self.state_to_idx: Dict[str, int] = {}
        self.idx_to_state: Dict[int, str] = {}
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_matrix: Optional[np.ndarray] = None
        self.state_counts: Dict[str, int] = defaultdict(int)
        self.total_transitions = 0

    def fit(self, transitions: List[Tuple[str, str]]) -> 'ChessMarkovChain':
        """
        Fit the Markov chain model on observed transitions.

        Mathematical Process:
        1. Count all transitions: n_ij = count of (state_i → state_j)
        2. Compute row sums: n_i = Σ_j n_ij
        3. Calculate probabilities: P_ij = n_ij / n_i

        Args:
            transitions: List of (current_state, next_state) tuples

        Returns:
            self for method chaining
        """
        print(f"\n{'='*60}")
        print("FITTING MARKOV CHAIN MODEL")
        print(f"{'='*60}")
        print(f"Total transitions observed: {len(transitions)}")

        # Step 1: Count all transitions
        for current, next_state in transitions:
            self.transition_counts[current][next_state] += 1
            self.state_counts[current] += 1
            self.total_transitions += 1

        # Step 2: Build state index
        all_states = set()
        for current, next_state in transitions:
            all_states.add(current)
            all_states.add(next_state)

        self.states = sorted(list(all_states))
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}

        print(f"Unique states (moves): {len(self.states)}")

        # Step 3: Build transition matrix
        n = len(self.states)
        self.transition_matrix = np.zeros((n, n))

        for i, state_i in enumerate(self.states):
            total_from_i = sum(self.transition_counts[state_i].values())
            if total_from_i > 0:
                for j, state_j in enumerate(self.states):
                    count = self.transition_counts[state_i].get(state_j, 0)
                    self.transition_matrix[i, j] = count / total_from_i

        # Verify stochastic property: each row sums to 1
        row_sums = self.transition_matrix.sum(axis=1)
        valid_rows = row_sums > 0

        print(f"\nTransition Matrix Shape: {self.transition_matrix.shape}")
        print(f"Non-zero entries: {np.count_nonzero(self.transition_matrix)}")
        print(f"Sparsity: {1 - np.count_nonzero(self.transition_matrix) / (n*n):.2%}")

        return self

    def get_transition_probability(self, from_state: str, to_state: str) -> float:
        """
        Get P(to_state | from_state).

        Mathematical notation: P_ij where i=from_state, j=to_state
        """
        if from_state not in self.state_to_idx or to_state not in self.state_to_idx:
            return 0.0

        i = self.state_to_idx[from_state]
        j = self.state_to_idx[to_state]
        return self.transition_matrix[i, j]

    def predict_next_move(self, current_move: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the most likely next moves given current move.

        Returns top-k moves with highest P(next | current).

        Args:
            current_move: The current state
            top_k: Number of top predictions to return

        Returns:
            List of (move, probability) tuples, sorted by probability descending
        """
        if current_move not in self.state_to_idx:
            return []

        i = self.state_to_idx[current_move]
        probs = self.transition_matrix[i, :]

        # Get top-k indices
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            if probs[idx] > 0:
                predictions.append((self.idx_to_state[idx], probs[idx]))

        return predictions

    def predict_best_move(self, current_move: str) -> Tuple[str, float]:
        """
        Predict the single most likely next move.

        Returns:
            Tuple of (predicted_move, probability)
        """
        predictions = self.predict_next_move(current_move, top_k=1)
        if predictions:
            return predictions[0]
        return ("", 0.0)

    def get_row_entropy(self, state: str) -> float:
        """
        Calculate the entropy of transitions from a given state.

        Entropy Formula:
            H(X|state) = -Σ P(x|state) * log₂(P(x|state))

        Properties:
        - H = 0 when all probability is on one state (deterministic)
        - H is maximized when all transitions are equally likely

        Args:
            state: The state to calculate entropy for

        Returns:
            Entropy value in bits
        """
        if state not in self.state_to_idx:
            return 0.0

        i = self.state_to_idx[state]
        probs = self.transition_matrix[i, :]

        # Filter out zero probabilities (log(0) is undefined)
        nonzero_probs = probs[probs > 0]

        if len(nonzero_probs) == 0:
            return 0.0

        # H = -Σ p * log₂(p)
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        return entropy

    def get_average_entropy(self) -> float:
        """
        Calculate the average entropy across all states.

        Weighted by state frequency:
            H_avg = Σ_i (count_i / total) * H_i
        """
        total_weight = 0
        weighted_entropy = 0

        for state in self.states:
            if state in self.state_counts:
                weight = self.state_counts[state]
                entropy = self.get_row_entropy(state)
                weighted_entropy += weight * entropy
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_entropy / total_weight

    def simulate_game(self, start_move: str, num_moves: int = 10) -> List[str]:
        """
        Simulate a sequence of moves using the Markov chain.

        Uses the probability distribution to randomly select next moves.
        This demonstrates: π_(n+1) = π_n * P

        Args:
            start_move: Initial state
            num_moves: Number of moves to simulate

        Returns:
            List of simulated moves
        """
        if start_move not in self.state_to_idx:
            return [start_move]

        sequence = [start_move]
        current = start_move

        for _ in range(num_moves - 1):
            i = self.state_to_idx[current]
            probs = self.transition_matrix[i, :]

            if probs.sum() == 0:
                break

            # Sample from the probability distribution
            next_idx = np.random.choice(len(self.states), p=probs)
            current = self.idx_to_state[next_idx]
            sequence.append(current)

        return sequence

    def get_stationary_distribution(self, max_iterations: int = 1000,
                                     tolerance: float = 1e-10) -> np.ndarray:
        """
        Calculate the stationary distribution π such that π = π * P.

        The stationary distribution represents the long-term probability
        of being in each state, regardless of starting state.

        Method: Power iteration
            π^(k+1) = π^(k) * P
            until ||π^(k+1) - π^(k)|| < tolerance

        Returns:
            Stationary distribution vector
        """
        n = len(self.states)
        if n == 0:
            return np.array([])

        # Start with uniform distribution
        pi = np.ones(n) / n

        for _ in range(max_iterations):
            pi_new = pi @ self.transition_matrix

            # Normalize (in case of numerical issues)
            if pi_new.sum() > 0:
                pi_new = pi_new / pi_new.sum()

            # Check convergence
            if np.linalg.norm(pi_new - pi) < tolerance:
                break

            pi = pi_new

        return pi

    def get_top_states(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get the n most common states."""
        sorted_states = sorted(self.state_counts.items(), key=lambda x: -x[1])
        return sorted_states[:n]

    def get_top_transitions(self, n: int = 20) -> List[Tuple[str, str, int, float]]:
        """
        Get the n most common transitions.

        Returns:
            List of (from_state, to_state, count, probability) tuples
        """
        transitions = []
        for from_state, to_dict in self.transition_counts.items():
            for to_state, count in to_dict.items():
                prob = self.get_transition_probability(from_state, to_state)
                transitions.append((from_state, to_state, count, prob))

        return sorted(transitions, key=lambda x: -x[2])[:n]

    def print_transition_matrix_sample(self, states_to_show: List[str] = None,
                                        n: int = 10):
        """Print a sample of the transition matrix."""
        if states_to_show is None:
            # Show most common states
            states_to_show = [s for s, _ in self.get_top_states(n)]

        print(f"\n{'='*60}")
        print("TRANSITION MATRIX SAMPLE")
        print(f"{'='*60}")
        print(f"Showing transitions between top {len(states_to_show)} states")
        print()

        # Header
        header = "From\\To   " + "  ".join(f"{s:>7}" for s in states_to_show)
        print(header)
        print("-" * len(header))

        for from_state in states_to_show:
            if from_state in self.state_to_idx:
                row = f"{from_state:>8}  "
                for to_state in states_to_show:
                    prob = self.get_transition_probability(from_state, to_state)
                    if prob > 0:
                        row += f"{prob:>7.3f}  "
                    else:
                        row += f"{'  -  ':>7}  "
                print(row)

    def export_model(self, filepath: str):
        """Export the model to JSON."""
        model_data = {
            'states': self.states,
            'transition_counts': {k: dict(v) for k, v in self.transition_counts.items()},
            'state_counts': dict(self.state_counts),
            'total_transitions': self.total_transitions
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model exported to {filepath}")


def create_markov_model(transitions: List[Tuple[str, str]]) -> ChessMarkovChain:
    """
    Create and fit a Markov chain model from transitions.

    Args:
        transitions: List of (current_move, next_move) tuples

    Returns:
        Fitted ChessMarkovChain model
    """
    model = ChessMarkovChain()
    model.fit(transitions)
    return model


if __name__ == "__main__":
    # Test with sample data
    sample_transitions = [
        ("e4", "e5"), ("e5", "Nf3"), ("Nf3", "Nc6"), ("Nc6", "Bb5"),
        ("e4", "c5"), ("c5", "Nf3"), ("Nf3", "d6"),
        ("e4", "e5"), ("e5", "Nf3"), ("Nf3", "Nc6"), ("Nc6", "Bc4"),
        ("d4", "d5"), ("d5", "c4"), ("c4", "e6"),
        ("d4", "Nf6"), ("Nf6", "c4"), ("c4", "e6"),
    ]

    model = create_markov_model(sample_transitions)
    model.print_transition_matrix_sample()

    print("\nPredictions after 'e4':")
    for move, prob in model.predict_next_move("e4"):
        print(f"  {move}: {prob:.3f}")
