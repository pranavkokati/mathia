"""
Visualization Module for Chess Markov Chain Analysis

Creates comprehensive visualizations including:
1. Transition matrix heatmaps
2. State distribution charts
3. Entropy analysis plots
4. Accuracy comparison graphs
5. Opening comparison visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.patches as mpatches

from .markov_chain import ChessMarkovChain


class ChessVisualizer:
    """Creates visualizations for Markov chain chess analysis."""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def plot_transition_matrix(self, model: ChessMarkovChain,
                               top_n: int = 15,
                               title: str = "Chess Move Transition Matrix",
                               filename: str = "transition_matrix.png") -> str:
        """
        Create a heatmap of the transition matrix for top states.

        Mathematical visualization of P = [P_ij]
        """
        # Get top states by frequency
        top_states = [s for s, _ in model.get_top_states(top_n)]

        # Build subset matrix
        n = len(top_states)
        subset_matrix = np.zeros((n, n))

        for i, state_i in enumerate(top_states):
            for j, state_j in enumerate(top_states):
                prob = model.get_transition_probability(state_i, state_j)
                subset_matrix[i, j] = prob

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        im = ax.imshow(subset_matrix, cmap='YlOrRd', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Transition Probability P(j|i)', fontsize=12)

        # Set ticks and labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(top_states, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(top_states, fontsize=10)

        # Labels
        ax.set_xlabel('Next Move (j)', fontsize=12)
        ax.set_ylabel('Current Move (i)', fontsize=12)
        ax.set_title(f'{title}\n(Top {top_n} most frequent moves)', fontsize=14)

        # Add probability values to cells
        for i in range(n):
            for j in range(n):
                if subset_matrix[i, j] > 0.01:
                    text_color = 'white' if subset_matrix[i, j] > 0.3 else 'black'
                    ax.text(j, i, f'{subset_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8, color=text_color)

        # Add mathematical annotation
        ax.text(0.02, -0.15,
                r'$P_{ij} = P(\text{move } j | \text{move } i) = \frac{\text{Count}(j \text{ follows } i)}{\text{Total from } i}$',
                transform=ax.transAxes, fontsize=10, style='italic')

        plt.tight_layout()

        # Save
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")
        return str(filepath)

    def plot_state_distribution(self, model: ChessMarkovChain,
                                top_n: int = 20,
                                filename: str = "state_distribution.png") -> str:
        """Plot the distribution of move frequencies."""
        top_states = model.get_top_states(top_n)
        moves = [s for s, _ in top_states]
        counts = [c for _, c in top_states]
        total = sum(model.state_counts.values())
        probs = [c / total for c in counts]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart of counts
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(moves)))
        bars = ax1.barh(range(len(moves)), counts, color=colors)
        ax1.set_yticks(range(len(moves)))
        ax1.set_yticklabels(moves)
        ax1.invert_yaxis()
        ax1.set_xlabel('Frequency Count', fontsize=12)
        ax1.set_title('Move Frequency Distribution', fontsize=14)

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(count + max(counts)*0.01, i, f'{count:,}',
                    va='center', fontsize=9)

        # Probability distribution
        ax2.barh(range(len(moves)), probs, color=colors)
        ax2.set_yticks(range(len(moves)))
        ax2.set_yticklabels(moves)
        ax2.invert_yaxis()
        ax2.set_xlabel('Probability P(move)', fontsize=12)
        ax2.set_title('Move Probability Distribution', fontsize=14)

        # Add probability labels
        for i, prob in enumerate(probs):
            ax2.text(prob + max(probs)*0.01, i, f'{prob:.3f}',
                    va='center', fontsize=9)

        # Add mathematical annotation
        fig.text(0.5, 0.02,
                r'$P(\text{move}_i) = \frac{\text{Count}(\text{move}_i)}{\sum_j \text{Count}(\text{move}_j)}$',
                ha='center', fontsize=11, style='italic')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")
        return str(filepath)

    def plot_entropy_analysis(self, entropy_data: Dict,
                              filename: str = "entropy_analysis.png") -> str:
        """
        Visualize entropy analysis results.

        H(X) = -Σ P(x) log₂(P(x))
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Entropy by move position
        ax1 = axes[0, 0]
        if entropy_data.get('position_entropies'):
            positions = sorted(entropy_data['position_entropies'].keys())
            entropies = [entropy_data['position_entropies'][p] for p in positions]

            ax1.plot(positions, entropies, 'o-', linewidth=2, markersize=8, color='#2ecc71')
            ax1.fill_between(positions, entropies, alpha=0.3, color='#2ecc71')
            ax1.set_xlabel('Move Number', fontsize=12)
            ax1.set_ylabel('Entropy (bits)', fontsize=12)
            ax1.set_title('Entropy by Move Position in Game', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Add average line
            avg = np.mean(entropies)
            ax1.axhline(y=avg, color='red', linestyle='--', label=f'Average: {avg:.3f}')
            ax1.legend()

        # 2. Most vs Least Predictable Moves
        ax2 = axes[0, 1]
        most_pred = entropy_data.get('most_predictable_states', [])[:8]
        least_pred = entropy_data.get('least_predictable_states', [])[:8]

        if most_pred and least_pred:
            all_moves = [m for m, _ in most_pred] + [m for m, _ in least_pred]
            all_entropies = [e for _, e in most_pred] + [e for _, e in least_pred]
            colors = ['#27ae60'] * len(most_pred) + ['#e74c3c'] * len(least_pred)

            y_pos = range(len(all_moves))
            ax2.barh(y_pos, all_entropies, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(all_moves)
            ax2.set_xlabel('Entropy (bits)', fontsize=12)
            ax2.set_title('Most vs Least Predictable Moves', fontsize=14)

            # Add legend
            green_patch = mpatches.Patch(color='#27ae60', label='Most Predictable (Low H)')
            red_patch = mpatches.Patch(color='#e74c3c', label='Least Predictable (High H)')
            ax2.legend(handles=[green_patch, red_patch], loc='lower right')

        # 3. Entropy Formula Explanation
        ax3 = axes[1, 0]
        ax3.axis('off')
        explanation = """
        ENTROPY - MEASURE OF UNCERTAINTY

        Shannon Entropy Formula:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        H(X) = -Σᵢ P(xᵢ) · log₂(P(xᵢ))

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Interpretation for Chess:
        ┌────────────────────────────────────────┐
        │ H = 0 bits                             │
        │   → Only one possible next move        │
        │   → Completely deterministic           │
        │                                        │
        │ H = log₂(n) bits                       │
        │   → All n moves equally likely         │
        │   → Maximum uncertainty                │
        └────────────────────────────────────────┘

        Lower entropy = More predictable opening
        Higher entropy = More complex/varied responses
        """
        ax3.text(0.1, 0.9, explanation, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. Average Entropy Gauge
        ax4 = axes[1, 1]
        avg_entropy = entropy_data.get('average_entropy', 0)

        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background arc
        for i, color in enumerate(['#27ae60', '#f1c40f', '#e74c3c']):
            start = i * np.pi / 3
            end = (i + 1) * np.pi / 3
            theta_section = np.linspace(start, end, 30)
            ax4.fill_between(theta_section, 0.7, 1.0,
                           alpha=0.3, color=color)

        # Current entropy indicator
        # Assume entropy range is 0-6 bits
        max_entropy = 6
        angle = np.pi * (1 - avg_entropy / max_entropy)
        ax4.annotate('', xy=(np.cos(angle) * 0.85, np.sin(angle) * 0.85),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=3))

        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-0.2, 1.2)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title(f'Average Entropy: {avg_entropy:.4f} bits', fontsize=14, pad=20)

        # Add labels
        ax4.text(-1, -0.1, 'Predictable', fontsize=10, ha='center', color='#27ae60')
        ax4.text(1, -0.1, 'Unpredictable', fontsize=10, ha='center', color='#e74c3c')
        ax4.text(0, -0.1, 'Moderate', fontsize=10, ha='center', color='#f1c40f')

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")
        return str(filepath)

    def plot_accuracy_results(self, accuracy_data: Dict,
                              filename: str = "accuracy_results.png") -> str:
        """Visualize prediction accuracy metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Top-k Accuracy Comparison
        ax1 = axes[0, 0]
        accuracies = [
            accuracy_data.get('accuracy', 0),
            accuracy_data.get('top3_accuracy', 0),
            accuracy_data.get('top5_accuracy', 0)
        ]
        labels = ['Top-1', 'Top-3', 'Top-5']
        colors = ['#3498db', '#2ecc71', '#9b59b6']

        bars = ax1.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Prediction Accuracy (Top-k)', fontsize=14)
        ax1.set_ylim(0, 1)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2%}', ha='center', fontsize=12, fontweight='bold')

        # Add grid
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)

        # 2. Accuracy by Move Position
        ax2 = axes[0, 1]
        per_position = accuracy_data.get('per_position_accuracy', {})
        if per_position:
            positions = sorted(per_position.keys())[:15]  # First 15 positions
            accs = [per_position[p] for p in positions]

            ax2.plot(positions, accs, 'o-', linewidth=2, markersize=8, color='#e74c3c')
            ax2.fill_between(positions, accs, alpha=0.3, color='#e74c3c')
            ax2.set_xlabel('Move Position in Game', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('Prediction Accuracy by Move Position', fontsize=14)
            ax2.set_ylim(0, max(accs) * 1.2 if accs else 1)
            ax2.grid(True, alpha=0.3)

        # 3. Accuracy Formula
        ax3 = axes[1, 0]
        ax3.axis('off')
        formula_text = """
        ACCURACY METRICS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Top-1 Accuracy:
        ┌─────────────────────────────────────────────┐
        │         # Correct Predictions               │
        │ Acc = ──────────────────────────            │
        │         Total Predictions                   │
        └─────────────────────────────────────────────┘

        Top-k Accuracy:
        ┌─────────────────────────────────────────────┐
        │       # (True move ∈ Top-k predictions)     │
        │ Acc@k = ────────────────────────────────    │
        │              Total Predictions              │
        └─────────────────────────────────────────────┘

        Current Results:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        formula_text += f"""
        Total Predictions: {accuracy_data.get('total_predictions', 0):,}
        Correct (Top-1):   {accuracy_data.get('correct_predictions', 0):,}

        Top-1 Accuracy:    {accuracy_data.get('accuracy', 0):.4f}
        Top-3 Accuracy:    {accuracy_data.get('top3_accuracy', 0):.4f}
        Top-5 Accuracy:    {accuracy_data.get('top5_accuracy', 0):.4f}
        """

        ax3.text(0.1, 0.95, formula_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 4. Prediction Breakdown Pie Chart
        ax4 = axes[1, 1]
        correct = accuracy_data.get('correct_predictions', 0)
        total = accuracy_data.get('total_predictions', 1)
        incorrect = total - correct

        sizes = [correct, incorrect]
        labels_pie = [f'Correct\n({correct:,})', f'Incorrect\n({incorrect:,})']
        colors_pie = ['#27ae60', '#e74c3c']
        explode = (0.05, 0)

        ax4.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90,
               textprops={'fontsize': 11})
        ax4.set_title('Prediction Outcome Distribution', fontsize=14)

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")
        return str(filepath)

    def plot_opening_comparison(self, comparison_data: Dict,
                                filename: str = "opening_comparison.png") -> str:
        """Compare e4 vs d4 openings."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        opening1 = comparison_data.get('opening1_name', 'e4')
        opening2 = comparison_data.get('opening2_name', 'd4')

        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        accuracies = [
            comparison_data.get('opening1_accuracy', 0),
            comparison_data.get('opening2_accuracy', 0)
        ]
        colors = ['#3498db', '#e74c3c']

        bars = ax1.bar([opening1, opening2], accuracies, color=colors,
                      edgecolor='black', linewidth=2, width=0.6)
        ax1.set_ylabel('Prediction Accuracy', fontsize=12)
        ax1.set_title('Accuracy Comparison by Opening', fontsize=14)
        ax1.set_ylim(0, max(accuracies) * 1.3 if accuracies else 1)

        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2%}', ha='center', fontsize=14, fontweight='bold')

        # 2. Entropy Comparison
        ax2 = axes[0, 1]
        entropies = [
            comparison_data.get('opening1_entropy', 0),
            comparison_data.get('opening2_entropy', 0)
        ]

        bars = ax2.bar([opening1, opening2], entropies, color=colors,
                      edgecolor='black', linewidth=2, width=0.6)
        ax2.set_ylabel('Average Entropy (bits)', fontsize=12)
        ax2.set_title('Entropy Comparison (Lower = More Predictable)', fontsize=14)

        for bar, ent in zip(bars, entropies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ent:.3f}', ha='center', fontsize=14, fontweight='bold')

        # 3. Game Count Comparison
        ax3 = axes[1, 0]
        games = [
            comparison_data.get('opening1_games', 0),
            comparison_data.get('opening2_games', 0)
        ]

        bars = ax3.bar([opening1, opening2], games, color=colors,
                      edgecolor='black', linewidth=2, width=0.6)
        ax3.set_ylabel('Number of Games', fontsize=12)
        ax3.set_title('Sample Size by Opening', fontsize=14)

        for bar, count in zip(bars, games):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(games)*0.02,
                    f'{count:,}', ha='center', fontsize=14, fontweight='bold')

        # 4. Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        p_value = comparison_data.get('statistical_difference', 1.0)
        significance = "YES (p < 0.05)" if p_value < 0.05 else "NO (p ≥ 0.05)"

        # Determine more predictable opening
        if entropies[0] < entropies[1]:
            more_pred = opening1
            pred_entropy = entropies[0]
        else:
            more_pred = opening2
            pred_entropy = entropies[1]

        # Determine more accurate
        if accuracies[0] > accuracies[1]:
            more_acc = opening1
            acc_val = accuracies[0]
        else:
            more_acc = opening2
            acc_val = accuracies[1]

        summary = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          OPENING COMPARISON SUMMARY                ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                    ┃
┃  {opening1:20} │ {opening2:20}    ┃
┃  ─────────────────────┼────────────────────────    ┃
┃  Games: {games[0]:>10,}    │ Games: {games[1]:>10,}       ┃
┃  Accuracy: {accuracies[0]:>8.2%}   │ Accuracy: {accuracies[1]:>8.2%}      ┃
┃  Entropy: {entropies[0]:>8.4f}    │ Entropy: {entropies[1]:>8.4f}       ┃
┃                                                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  CONCLUSIONS:                                      ┃
┃  ─────────────                                     ┃
┃  More Predictable: {more_pred:25}      ┃
┃    (Entropy: {pred_entropy:.4f} bits)                       ┃
┃                                                    ┃
┃  Higher Accuracy: {more_acc:26}      ┃
┃    (Accuracy: {acc_val:.2%})                            ┃
┃                                                    ┃
┃  Statistically Significant: {significance:15}      ┃
┃    (Chi-square p-value: {p_value:.6f})               ┃
┃                                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        """

        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")
        return str(filepath)

    def plot_simulated_game(self, model: ChessMarkovChain,
                           start_move: str = "e4",
                           num_moves: int = 10,
                           filename: str = "simulated_game.png") -> str:
        """Visualize a simulated game sequence."""
        # Simulate multiple games
        num_simulations = 5
        simulations = []

        for _ in range(num_simulations):
            sim = model.simulate_game(start_move, num_moves)
            simulations.append(sim)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot each simulation as a timeline
        colors = plt.cm.Set2(np.linspace(0, 1, num_simulations))

        for i, (sim, color) in enumerate(zip(simulations, colors)):
            y = num_simulations - i - 1
            for j, move in enumerate(sim):
                ax.text(j, y, move, ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

                if j < len(sim) - 1:
                    # Draw arrow
                    ax.annotate('', xy=(j + 0.4, y), xytext=(j + 0.6, y),
                              arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

        ax.set_xlim(-0.5, num_moves + 0.5)
        ax.set_ylim(-0.5, num_simulations + 0.5)
        ax.set_xlabel('Move Number', fontsize=12)
        ax.set_ylabel('Simulation', fontsize=12)
        ax.set_title(f'Simulated Chess Games (Markov Chain)\nStarting with {start_move}', fontsize=14)

        # Add move numbers
        for j in range(num_moves):
            ax.text(j, -0.3, str(j + 1), ha='center', fontsize=10, color='gray')

        ax.axis('off')

        # Add mathematical note
        fig.text(0.5, 0.02,
                r'Generated using: $\pi_{n+1} = \pi_n \cdot P$ where $P$ is the transition matrix',
                ha='center', fontsize=11, style='italic')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")
        return str(filepath)

    def create_all_visualizations(self, results: Dict) -> List[str]:
        """Create all visualizations from analysis results."""
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")

        created_files = []

        # 1. Transition Matrix
        if 'overall_model' in results:
            model = results['overall_model']
            created_files.append(
                self.plot_transition_matrix(model, top_n=15)
            )
            created_files.append(
                self.plot_state_distribution(model, top_n=20)
            )
            created_files.append(
                self.plot_simulated_game(model)
            )

        # 2. Entropy Analysis
        if 'entropy_analysis' in results:
            entropy_data = results['entropy_analysis']
            entropy_dict = {
                'average_entropy': entropy_data.average_entropy,
                'position_entropies': entropy_data.position_entropies,
                'most_predictable_states': entropy_data.most_predictable_states,
                'least_predictable_states': entropy_data.least_predictable_states
            }
            created_files.append(
                self.plot_entropy_analysis(entropy_dict)
            )

        # 3. Accuracy Results
        if 'overall_accuracy' in results:
            acc = results['overall_accuracy']
            acc_dict = {
                'accuracy': acc.accuracy,
                'top3_accuracy': acc.top3_accuracy,
                'top5_accuracy': acc.top5_accuracy,
                'total_predictions': acc.total_predictions,
                'correct_predictions': acc.correct_predictions,
                'per_position_accuracy': acc.per_position_accuracy
            }
            created_files.append(
                self.plot_accuracy_results(acc_dict)
            )

        # 4. Opening Comparison
        if 'opening_comparison' in results:
            comp = results['opening_comparison']
            comp_dict = {
                'opening1_name': comp.opening1_name,
                'opening2_name': comp.opening2_name,
                'opening1_accuracy': comp.opening1_accuracy,
                'opening2_accuracy': comp.opening2_accuracy,
                'opening1_entropy': comp.opening1_entropy,
                'opening2_entropy': comp.opening2_entropy,
                'opening1_games': comp.opening1_games,
                'opening2_games': comp.opening2_games,
                'statistical_difference': comp.statistical_difference
            }
            created_files.append(
                self.plot_opening_comparison(comp_dict)
            )

        # 5. Opening-specific transition matrices
        if 'model_e4' in results:
            created_files.append(
                self.plot_transition_matrix(
                    results['model_e4'],
                    top_n=12,
                    title="Transition Matrix - King's Pawn (1.e4)",
                    filename="transition_matrix_e4.png"
                )
            )

        if 'model_d4' in results:
            created_files.append(
                self.plot_transition_matrix(
                    results['model_d4'],
                    top_n=12,
                    title="Transition Matrix - Queen's Pawn (1.d4)",
                    filename="transition_matrix_d4.png"
                )
            )

        print(f"\nCreated {len(created_files)} visualizations")
        return created_files


def create_visualizations(results: Dict, output_dir: str = "visualizations") -> List[str]:
    """Main function to create all visualizations."""
    visualizer = ChessVisualizer(output_dir)
    return visualizer.create_all_visualizations(results)


if __name__ == "__main__":
    print("Visualization module - run from main.py")
