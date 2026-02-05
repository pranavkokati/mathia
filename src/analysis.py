"""
Analysis Module for Chess Markov Chain Prediction

This module implements:
1. Prediction accuracy calculations
2. Entropy analysis for opening predictability
3. Comparative analysis between different openings
4. Statistical significance testing
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass
from scipy import stats

from .markov_chain import ChessMarkovChain, create_markov_model
from .pgn_parser import ChessGame


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics."""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    top3_accuracy: float  # Correct if true move is in top 3 predictions
    top5_accuracy: float  # Correct if true move is in top 5 predictions
    per_position_accuracy: Dict[int, float]  # Accuracy at each move position


@dataclass
class EntropyAnalysis:
    """Container for entropy analysis results."""
    average_entropy: float
    position_entropies: Dict[int, float]  # Entropy at each move position
    most_predictable_states: List[Tuple[str, float]]  # (state, entropy)
    least_predictable_states: List[Tuple[str, float]]
    entropy_by_move_number: List[float]


@dataclass
class OpeningComparison:
    """Comparison between two openings."""
    opening1_name: str
    opening2_name: str
    opening1_accuracy: float
    opening2_accuracy: float
    opening1_entropy: float
    opening2_entropy: float
    opening1_games: int
    opening2_games: int
    statistical_difference: float  # p-value


class ChessAnalyzer:
    """
    Comprehensive analyzer for chess Markov chain predictions.

    Mathematical Framework:
    =======================

    1. ACCURACY:
       Accuracy = (Correct Predictions) / (Total Predictions)

       For top-k accuracy:
       Top-k Accuracy = P(true move ∈ top k predictions)

    2. ENTROPY (Information Theory):
       H(X) = -Σ p(x) * log₂(p(x))

       Interpretation:
       - H = 0: Completely deterministic (one outcome certain)
       - H = log₂(n): Maximum uncertainty (uniform distribution)

       For chess openings:
       - Low entropy = Highly predictable sequence
       - High entropy = Many equally likely continuations

    3. CONDITIONAL ENTROPY:
       H(Y|X) = -Σ p(x,y) * log₂(p(y|x))

       For our Markov chain:
       H(next_move | current_move) = -Σ P(i,j) * log₂(P(j|i))
    """

    def __init__(self, games: List[ChessGame]):
        self.games = games
        self.e4_games = [g for g in games if g.first_move == 'e4']
        self.d4_games = [g for g in games if g.first_move == 'd4']

        print(f"\n{'='*60}")
        print("CHESS ANALYZER INITIALIZED")
        print(f"{'='*60}")
        print(f"Total games: {len(games)}")
        print(f"1.e4 games (King's Pawn): {len(self.e4_games)}")
        print(f"1.d4 games (Queen's Pawn): {len(self.d4_games)}")

    def calculate_accuracy(self, model: ChessMarkovChain,
                          test_transitions: List[Tuple[str, str]],
                          max_move: int = 20) -> AccuracyMetrics:
        """
        Calculate prediction accuracy metrics.

        Mathematical Definition:
        Accuracy = (1/n) * Σ I(predicted_move == actual_move)

        where I() is the indicator function.

        For top-k accuracy:
        Top-k Accuracy = (1/n) * Σ I(actual_move ∈ top_k_predictions)
        """
        total = 0
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0

        position_correct = defaultdict(int)
        position_total = defaultdict(int)

        for i, (current, actual_next) in enumerate(test_transitions):
            position = i % max_move  # Approximate position in game

            predictions = model.predict_next_move(current, top_k=5)

            if not predictions:
                continue

            total += 1
            position_total[position] += 1

            # Check top-1 accuracy
            if predictions[0][0] == actual_next:
                correct_top1 += 1
                position_correct[position] += 1

            # Check top-3 accuracy
            top3_moves = [p[0] for p in predictions[:3]]
            if actual_next in top3_moves:
                correct_top3 += 1

            # Check top-5 accuracy
            top5_moves = [p[0] for p in predictions[:5]]
            if actual_next in top5_moves:
                correct_top5 += 1

        # Calculate per-position accuracy
        per_position = {}
        for pos in position_total:
            if position_total[pos] > 0:
                per_position[pos] = position_correct[pos] / position_total[pos]

        return AccuracyMetrics(
            total_predictions=total,
            correct_predictions=correct_top1,
            accuracy=correct_top1 / total if total > 0 else 0,
            top3_accuracy=correct_top3 / total if total > 0 else 0,
            top5_accuracy=correct_top5 / total if total > 0 else 0,
            per_position_accuracy=per_position
        )

    def calculate_entropy_analysis(self, model: ChessMarkovChain,
                                   num_positions: int = 10) -> EntropyAnalysis:
        """
        Perform comprehensive entropy analysis.

        Entropy Calculation:
        H(state) = -Σ_j P(j|state) * log₂(P(j|state))

        Average Entropy (weighted by frequency):
        H_avg = Σ_i f(i) * H(i) / Σ_i f(i)

        where f(i) is the frequency of state i.
        """
        state_entropies = {}

        for state in model.states:
            entropy = model.get_row_entropy(state)
            state_entropies[state] = entropy

        # Sort by entropy
        sorted_by_entropy = sorted(state_entropies.items(), key=lambda x: x[1])

        # Get most and least predictable
        most_predictable = sorted_by_entropy[:10]
        least_predictable = sorted_by_entropy[-10:][::-1]

        # Calculate entropy at different move positions
        # (This requires game-level analysis)
        position_entropies = {}
        for pos in range(1, num_positions + 1):
            pos_transitions = []
            for game in self.games:
                if len(game.moves) > pos:
                    pos_transitions.append((game.moves[pos-1], game.moves[pos]))

            if pos_transitions:
                temp_model = create_markov_model(pos_transitions)
                position_entropies[pos] = temp_model.get_average_entropy()

        return EntropyAnalysis(
            average_entropy=model.get_average_entropy(),
            position_entropies=position_entropies,
            most_predictable_states=most_predictable,
            least_predictable_states=least_predictable,
            entropy_by_move_number=list(position_entropies.values())
        )

    def compare_openings(self, model_e4: ChessMarkovChain,
                        model_d4: ChessMarkovChain,
                        test_e4: List[Tuple[str, str]],
                        test_d4: List[Tuple[str, str]]) -> OpeningComparison:
        """
        Compare prediction accuracy and entropy between e4 and d4 openings.

        Statistical Test:
        We use a chi-square test to determine if the difference in
        accuracy between openings is statistically significant.

        Null hypothesis H₀: Accuracies are equal
        Alternative H₁: Accuracies are different
        """
        # Calculate accuracies
        acc_e4 = self.calculate_accuracy(model_e4, test_e4)
        acc_d4 = self.calculate_accuracy(model_d4, test_d4)

        # Calculate entropies
        entropy_e4 = model_e4.get_average_entropy()
        entropy_d4 = model_d4.get_average_entropy()

        # Statistical test (chi-square for proportions)
        # Contingency table: [[correct_e4, incorrect_e4], [correct_d4, incorrect_d4]]
        contingency = np.array([
            [acc_e4.correct_predictions, acc_e4.total_predictions - acc_e4.correct_predictions],
            [acc_d4.correct_predictions, acc_d4.total_predictions - acc_d4.correct_predictions]
        ])

        if contingency.min() > 0:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        else:
            p_value = 1.0

        return OpeningComparison(
            opening1_name="King's Pawn (1.e4)",
            opening2_name="Queen's Pawn (1.d4)",
            opening1_accuracy=acc_e4.accuracy,
            opening2_accuracy=acc_d4.accuracy,
            opening1_entropy=entropy_e4,
            opening2_entropy=entropy_d4,
            opening1_games=len(self.e4_games),
            opening2_games=len(self.d4_games),
            statistical_difference=p_value
        )

    def run_full_analysis(self, train_ratio: float = 0.8) -> Dict:
        """
        Run complete analysis pipeline.

        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n{'='*60}")
        print("RUNNING FULL ANALYSIS")
        print(f"{'='*60}")

        results = {}

        # =====================
        # 1. Build All-Games Model
        # =====================
        print("\n[1/5] Building overall Markov model...")
        all_transitions = []
        for game in self.games:
            for i in range(len(game.moves) - 1):
                all_transitions.append((game.moves[i], game.moves[i + 1]))

        # Split into train/test
        np.random.seed(42)
        indices = np.random.permutation(len(all_transitions))
        split_idx = int(len(indices) * train_ratio)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        train_transitions = [all_transitions[i] for i in train_idx]
        test_transitions = [all_transitions[i] for i in test_idx]

        overall_model = create_markov_model(train_transitions)
        results['overall_model'] = overall_model

        # =====================
        # 2. Calculate Overall Accuracy
        # =====================
        print("\n[2/5] Calculating prediction accuracy...")
        overall_accuracy = self.calculate_accuracy(overall_model, test_transitions)
        results['overall_accuracy'] = overall_accuracy

        print(f"\n  Overall Prediction Metrics:")
        print(f"  {'─'*40}")
        print(f"  Total test predictions: {overall_accuracy.total_predictions}")
        print(f"  Correct (top-1): {overall_accuracy.correct_predictions}")
        print(f"  Top-1 Accuracy: {overall_accuracy.accuracy:.2%}")
        print(f"  Top-3 Accuracy: {overall_accuracy.top3_accuracy:.2%}")
        print(f"  Top-5 Accuracy: {overall_accuracy.top5_accuracy:.2%}")

        # =====================
        # 3. Entropy Analysis
        # =====================
        print("\n[3/5] Performing entropy analysis...")
        entropy_analysis = self.calculate_entropy_analysis(overall_model)
        results['entropy_analysis'] = entropy_analysis

        print(f"\n  Entropy Metrics (bits):")
        print(f"  {'─'*40}")
        print(f"  Average Entropy: {entropy_analysis.average_entropy:.4f}")
        print(f"\n  Most Predictable Moves (lowest entropy):")
        for move, ent in entropy_analysis.most_predictable_states[:5]:
            print(f"    {move}: H = {ent:.4f} bits")
        print(f"\n  Least Predictable Moves (highest entropy):")
        for move, ent in entropy_analysis.least_predictable_states[:5]:
            print(f"    {move}: H = {ent:.4f} bits")

        # =====================
        # 4. Opening-Specific Models
        # =====================
        print("\n[4/5] Building opening-specific models...")

        # E4 Model
        e4_transitions = []
        for game in self.e4_games:
            for i in range(len(game.moves) - 1):
                e4_transitions.append((game.moves[i], game.moves[i + 1]))

        if e4_transitions:
            e4_split = int(len(e4_transitions) * train_ratio)
            e4_train = e4_transitions[:e4_split]
            e4_test = e4_transitions[e4_split:]
            model_e4 = create_markov_model(e4_train)
            results['model_e4'] = model_e4
            results['e4_test'] = e4_test
        else:
            model_e4 = None
            e4_test = []

        # D4 Model
        d4_transitions = []
        for game in self.d4_games:
            for i in range(len(game.moves) - 1):
                d4_transitions.append((game.moves[i], game.moves[i + 1]))

        if d4_transitions:
            d4_split = int(len(d4_transitions) * train_ratio)
            d4_train = d4_transitions[:d4_split]
            d4_test = d4_transitions[d4_split:]
            model_d4 = create_markov_model(d4_train)
            results['model_d4'] = model_d4
            results['d4_test'] = d4_test
        else:
            model_d4 = None
            d4_test = []

        # =====================
        # 5. Opening Comparison
        # =====================
        print("\n[5/5] Comparing openings...")

        if model_e4 and model_d4 and e4_test and d4_test:
            comparison = self.compare_openings(model_e4, model_d4, e4_test, d4_test)
            results['opening_comparison'] = comparison

            print(f"\n  Opening Comparison:")
            print(f"  {'─'*40}")
            print(f"  {comparison.opening1_name}:")
            print(f"    Games: {comparison.opening1_games}")
            print(f"    Accuracy: {comparison.opening1_accuracy:.2%}")
            print(f"    Entropy: {comparison.opening1_entropy:.4f} bits")
            print(f"\n  {comparison.opening2_name}:")
            print(f"    Games: {comparison.opening2_games}")
            print(f"    Accuracy: {comparison.opening2_accuracy:.2%}")
            print(f"    Entropy: {comparison.opening2_entropy:.4f} bits")
            print(f"\n  Statistical significance (p-value): {comparison.statistical_difference:.4f}")

            if comparison.opening1_entropy < comparison.opening2_entropy:
                more_predictable = comparison.opening1_name
            else:
                more_predictable = comparison.opening2_name
            print(f"  More predictable opening: {more_predictable}")
        else:
            print("  Insufficient data for opening comparison")

        return results

    def print_mathematical_summary(self, results: Dict):
        """Print a mathematical summary of the analysis."""
        print(f"\n{'='*60}")
        print("MATHEMATICAL SUMMARY")
        print(f"{'='*60}")

        print("""
┌─────────────────────────────────────────────────────────────────┐
│                    MARKOV CHAIN FORMULATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STATE SPACE:                                                   │
│    S = {s₁, s₂, ..., sₙ} where sᵢ represents a chess move     │
│                                                                 │
│  TRANSITION PROBABILITY:                                        │
│                    Count(sⱼ follows sᵢ)                         │
│    P(sⱼ|sᵢ) = ─────────────────────────────                    │
│                Total transitions from sᵢ                        │
│                                                                 │
│  TRANSITION MATRIX P:                                           │
│    P = [Pᵢⱼ] where Pᵢⱼ = P(sⱼ|sᵢ)                              │
│    Properties: Pᵢⱼ ≥ 0, Σⱼ Pᵢⱼ = 1                             │
│                                                                 │
│  STATE EVOLUTION:                                               │
│    π_{n+1} = πₙ · P                                            │
│    After k steps: π_{n+k} = πₙ · Pᵏ                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        """)

        model = results.get('overall_model')
        if model:
            print(f"\n  Model Statistics:")
            print(f"  ─────────────────")
            print(f"  Number of states |S|: {len(model.states)}")
            print(f"  Matrix dimensions: {model.transition_matrix.shape[0]} × {model.transition_matrix.shape[1]}")
            print(f"  Non-zero entries: {np.count_nonzero(model.transition_matrix)}")
            print(f"  Sparsity: {1 - np.count_nonzero(model.transition_matrix) / model.transition_matrix.size:.4f}")

        print("""
┌─────────────────────────────────────────────────────────────────┐
│                      ENTROPY CALCULATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SHANNON ENTROPY:                                               │
│    H(X) = -Σᵢ P(xᵢ) · log₂(P(xᵢ))                              │
│                                                                 │
│  CONDITIONAL ENTROPY (for state sᵢ):                            │
│    H(next|sᵢ) = -Σⱼ P(sⱼ|sᵢ) · log₂(P(sⱼ|sᵢ))                  │
│                                                                 │
│  INTERPRETATION:                                                 │
│    H = 0 bits:     Completely deterministic                     │
│    H = log₂(n):    Maximum uncertainty (uniform)                │
│                                                                 │
│  For chess openings:                                            │
│    Low entropy  → Highly predictable sequence                   │
│    High entropy → Many equally likely continuations             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        """)

        entropy = results.get('entropy_analysis')
        if entropy:
            print(f"\n  Entropy Results:")
            print(f"  ─────────────────")
            print(f"  Average Entropy: {entropy.average_entropy:.4f} bits")

            if entropy.position_entropies:
                print(f"\n  Entropy by Move Position:")
                for pos, ent in sorted(entropy.position_entropies.items()):
                    bar = "█" * int(ent * 5) if ent else ""
                    print(f"    Move {pos:2d}: {ent:.4f} bits  {bar}")

        print("""
┌─────────────────────────────────────────────────────────────────┐
│                     ACCURACY CALCULATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TOP-1 ACCURACY:                                                 │
│                  # Correct Predictions                          │
│    Accuracy = ──────────────────────────                        │
│                 Total Predictions                               │
│                                                                 │
│  TOP-K ACCURACY:                                                 │
│                      # (True move ∈ Top-k predictions)          │
│    Accuracy@k = ─────────────────────────────────────           │
│                         Total Predictions                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        """)

        acc = results.get('overall_accuracy')
        if acc:
            print(f"\n  Accuracy Results:")
            print(f"  ─────────────────")
            print(f"  Total Predictions: {acc.total_predictions}")
            print(f"  Top-1 Accuracy:    {acc.accuracy:.4f} ({acc.accuracy*100:.2f}%)")
            print(f"  Top-3 Accuracy:    {acc.top3_accuracy:.4f} ({acc.top3_accuracy*100:.2f}%)")
            print(f"  Top-5 Accuracy:    {acc.top5_accuracy:.4f} ({acc.top5_accuracy*100:.2f}%)")

        comparison = results.get('opening_comparison')
        if comparison:
            print("""
┌─────────────────────────────────────────────────────────────────┐
│                    OPENING COMPARISON                            │
├─────────────────────────────────────────────────────────────────┤
            """)
            print(f"│  {comparison.opening1_name:30} vs {comparison.opening2_name:15}  │")
            print(f"│                                                                 │")
            print(f"│  Games:    {comparison.opening1_games:>10}               {comparison.opening2_games:>10}           │")
            print(f"│  Accuracy: {comparison.opening1_accuracy:>10.4f}               {comparison.opening2_accuracy:>10.4f}           │")
            print(f"│  Entropy:  {comparison.opening1_entropy:>10.4f}               {comparison.opening2_entropy:>10.4f}           │")
            print(f"│                                                                 │")

            if comparison.opening1_entropy < comparison.opening2_entropy:
                winner = comparison.opening1_name
            else:
                winner = comparison.opening2_name
            print(f"│  More predictable: {winner:40}   │")
            print(f"│  Chi-square p-value: {comparison.statistical_difference:.6f}                            │")

            if comparison.statistical_difference < 0.05:
                print(f"│  Result: Statistically significant difference (p < 0.05)       │")
            else:
                print(f"│  Result: No significant difference (p ≥ 0.05)                  │")

            print("└─────────────────────────────────────────────────────────────────┘")


def run_analysis(games: List[ChessGame]) -> Dict:
    """Run the complete analysis pipeline."""
    analyzer = ChessAnalyzer(games)
    results = analyzer.run_full_analysis()
    analyzer.print_mathematical_summary(results)
    return results


if __name__ == "__main__":
    print("Analysis module - run from main.py")
