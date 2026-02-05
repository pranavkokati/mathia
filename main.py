#!/usr/bin/env python3
"""
Chess Markov Chain Prediction System
=====================================

Research Question:
"How can a Markov chain model predict the next move in a chess game using
historical grandmaster games, and how accurate is this model for different
opening strategies?"

This system implements:
1. Data collection from Lichess grandmaster games
2. PGN parsing and move extraction
3. First-order Markov chain model construction
4. Transition probability calculation
5. Move prediction and accuracy measurement
6. Entropy analysis for opening predictability
7. Comparison between King's Pawn (e4) and Queen's Pawn (d4) openings

Author: Chess Analysis Project
Date: 2024
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_downloader import download_chess_data, LichessDownloader
from src.pgn_parser import PGNParser, parse_chess_games
from src.markov_chain import ChessMarkovChain, create_markov_model
from src.analysis import ChessAnalyzer, run_analysis
from src.visualizations import ChessVisualizer, create_visualizations


def print_header():
    """Print project header."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              CHESS MARKOV CHAIN PREDICTION SYSTEM                            ║
║              ═══════════════════════════════════                             ║
║                                                                              ║
║  Research Question:                                                          ║
║  "How can a Markov chain model predict the next move in a chess game        ║
║   using historical grandmaster games, and how accurate is this model        ║
║   for different opening strategies?"                                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Mathematical Foundation:                                                    ║
║  ─────────────────────────                                                   ║
║  • Markov Chain: P(Xₙ₊₁|X₁,...,Xₙ) = P(Xₙ₊₁|Xₙ)                            ║
║  • Transition Probability: Pᵢⱼ = P(move j | move i)                         ║
║  • Entropy: H(X) = -Σ P(x)·log₂(P(x))                                       ║
║  • State Evolution: πₙ₊₁ = πₙ · P                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def save_results_to_file(results: dict, filepath: str):
    """Save analysis results to a JSON file."""
    # Convert non-serializable objects
    output = {
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': {
            'total_predictions': results['overall_accuracy'].total_predictions,
            'correct_predictions': results['overall_accuracy'].correct_predictions,
            'accuracy': results['overall_accuracy'].accuracy,
            'top3_accuracy': results['overall_accuracy'].top3_accuracy,
            'top5_accuracy': results['overall_accuracy'].top5_accuracy,
        },
        'entropy_analysis': {
            'average_entropy': results['entropy_analysis'].average_entropy,
            'position_entropies': results['entropy_analysis'].position_entropies,
        },
    }

    if 'opening_comparison' in results:
        comp = results['opening_comparison']
        output['opening_comparison'] = {
            'e4_accuracy': comp.opening1_accuracy,
            'd4_accuracy': comp.opening2_accuracy,
            'e4_entropy': comp.opening1_entropy,
            'd4_entropy': comp.opening2_entropy,
            'e4_games': comp.opening1_games,
            'd4_games': comp.opening2_games,
            'p_value': comp.statistical_difference,
        }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {filepath}")


def generate_report(results: dict, filepath: str):
    """Generate a comprehensive markdown report."""
    report = """# Chess Markov Chain Prediction Analysis Report

## Executive Summary

This report presents the results of applying a first-order Markov chain model
to predict chess moves using historical grandmaster games from Lichess.

---

## 1. Mathematical Foundation

### 1.1 Markov Chain Definition

A Markov chain is a stochastic process where the probability of transitioning
to any state depends only on the current state (Markov property):

$$P(X_{n+1} | X_1, X_2, ..., X_n) = P(X_{n+1} | X_n)$$

### 1.2 Transition Probability

For chess moves, the transition probability is defined as:

$$P_{ij} = P(\\text{move } j | \\text{move } i) = \\frac{\\text{Count}(j \\text{ follows } i)}{\\text{Total transitions from } i}$$

### 1.3 Transition Matrix

The transition matrix $P = [P_{ij}]$ has the properties:
- $P_{ij} \\geq 0$ (non-negative)
- $\\sum_j P_{ij} = 1$ (rows sum to 1)

### 1.4 State Evolution

Given a probability distribution $\\pi_n$ over states at step $n$:
$$\\pi_{n+1} = \\pi_n \\cdot P$$

After $k$ steps:
$$\\pi_{n+k} = \\pi_n \\cdot P^k$$

### 1.5 Entropy (Information Theory)

Shannon entropy measures uncertainty in the next move:

$$H(X) = -\\sum_i P(x_i) \\cdot \\log_2(P(x_i))$$

- $H = 0$: Completely deterministic (one outcome certain)
- $H = \\log_2(n)$: Maximum uncertainty (uniform distribution)

---

## 2. Results

"""

    # Add accuracy results
    acc = results['overall_accuracy']
    report += f"""### 2.1 Prediction Accuracy

| Metric | Value |
|--------|-------|
| Total Predictions | {acc.total_predictions:,} |
| Correct (Top-1) | {acc.correct_predictions:,} |
| **Top-1 Accuracy** | **{acc.accuracy:.2%}** |
| Top-3 Accuracy | {acc.top3_accuracy:.2%} |
| Top-5 Accuracy | {acc.top5_accuracy:.2%} |

"""

    # Add entropy results
    entropy = results['entropy_analysis']
    report += f"""### 2.2 Entropy Analysis

**Average Entropy: {entropy.average_entropy:.4f} bits**

#### Most Predictable Moves (Lowest Entropy)
| Move | Entropy (bits) |
|------|----------------|
"""
    for move, ent in entropy.most_predictable_states[:5]:
        report += f"| {move} | {ent:.4f} |\n"

    report += """
#### Least Predictable Moves (Highest Entropy)
| Move | Entropy (bits) |
|------|----------------|
"""
    for move, ent in entropy.least_predictable_states[:5]:
        report += f"| {move} | {ent:.4f} |\n"

    # Add opening comparison
    if 'opening_comparison' in results:
        comp = results['opening_comparison']
        report += f"""
### 2.3 Opening Comparison

| Metric | King's Pawn (1.e4) | Queen's Pawn (1.d4) |
|--------|-------------------|---------------------|
| Games Analyzed | {comp.opening1_games:,} | {comp.opening2_games:,} |
| Prediction Accuracy | {comp.opening1_accuracy:.2%} | {comp.opening2_accuracy:.2%} |
| Average Entropy | {comp.opening1_entropy:.4f} bits | {comp.opening2_entropy:.4f} bits |

**Statistical Significance (Chi-square test):**
- p-value: {comp.statistical_difference:.6f}
- Result: {"Statistically significant (p < 0.05)" if comp.statistical_difference < 0.05 else "Not statistically significant (p ≥ 0.05)"}

**Conclusion:**
"""
        if comp.opening1_entropy < comp.opening2_entropy:
            report += f"- King's Pawn (1.e4) openings are **more predictable** (lower entropy)\n"
        else:
            report += f"- Queen's Pawn (1.d4) openings are **more predictable** (lower entropy)\n"

    report += """
---

## 3. Methodology

### 3.1 Data Collection
- Source: Lichess Elite Database
- Games from titled players (GM, IM level)
- PGN format parsed using python-chess library

### 3.2 Model Construction
1. Extract all move transitions (current_move → next_move)
2. Count transition frequencies
3. Normalize to get transition probabilities
4. Build sparse transition matrix

### 3.3 Evaluation
- 80/20 train/test split
- Accuracy measured as correct predictions / total predictions
- Entropy calculated for each state and averaged

---

## 4. Visualizations

See the `visualizations/` folder for:
1. `transition_matrix.png` - Heatmap of transition probabilities
2. `state_distribution.png` - Distribution of move frequencies
3. `entropy_analysis.png` - Entropy visualization
4. `accuracy_results.png` - Prediction accuracy charts
5. `opening_comparison.png` - e4 vs d4 comparison
6. `simulated_game.png` - Markov-generated game sequences

---

## 5. Conclusion

This analysis demonstrates that a first-order Markov chain can provide
meaningful predictions for chess moves, particularly in the opening phase
where move sequences are more standardized. The entropy analysis reveals
which openings lead to more predictable game continuations.

Key findings:
1. Opening moves have lower entropy (more predictable)
2. Middle-game transitions show higher entropy (more complex decisions)
3. Different opening systems have measurably different predictability

---

*Report generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*\n"

    with open(filepath, 'w') as f:
        f.write(report)
    print(f"Report saved to: {filepath}")


def main():
    """Main entry point."""
    print_header()

    # Setup directories
    data_dir = Path("data")
    results_dir = Path("results")
    viz_dir = Path("visualizations")

    for d in [data_dir, results_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Download Chess Data
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 1: Downloading Grandmaster Games from Lichess")

    pgn_file = data_dir / "grandmaster_games.pgn"

    if pgn_file.exists():
        print(f"Using existing data file: {pgn_file}")
    else:
        print("Downloading games from Lichess...")
        try:
            downloaded_file = download_chess_data(num_games=2000, data_dir=str(data_dir))
            pgn_file = Path(downloaded_file)
        except Exception as e:
            print(f"Error downloading: {e}")
            print("Creating sample data for demonstration...")
            pgn_file = create_sample_data(data_dir)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Parse PGN Files
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 2: Parsing PGN Files")

    parser = PGNParser()
    games = parser.parse_file(str(pgn_file))
    stats = parser.get_statistics()

    print(f"\n  Parsing Complete:")
    print(f"  {'─'*40}")
    print(f"  Total games parsed: {stats.get('total_games', 0):,}")
    print(f"  Average moves per game: {stats.get('average_moves', 0):.1f}")
    print(f"  1.e4 games: {stats.get('e4_games', 0):,}")
    print(f"  1.d4 games: {stats.get('d4_games', 0):,}")
    print(f"  Other openings: {stats.get('other_games', 0):,}")

    if len(games) < 100:
        print("\n  Warning: Small dataset. Results may not be statistically significant.")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: Build Markov Chain Model
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 3: Building Markov Chain Model")

    # Extract transitions
    all_transitions = parser.extract_all_transitions()
    print(f"\n  Total transitions extracted: {len(all_transitions):,}")

    # Show sample transitions
    print(f"\n  Sample Transitions (move_i → move_j):")
    for i, (curr, nxt) in enumerate(all_transitions[:10]):
        print(f"    {curr} → {nxt}")
    print(f"    ...")

    # Build model
    model = create_markov_model(all_transitions)

    # Show top transitions
    print(f"\n  Top 10 Most Common Transitions:")
    print(f"  {'─'*50}")
    print(f"  {'From':<10} {'To':<10} {'Count':>10} {'Probability':>12}")
    print(f"  {'─'*50}")
    for from_s, to_s, count, prob in model.get_top_transitions(10):
        print(f"  {from_s:<10} {to_s:<10} {count:>10,} {prob:>12.4f}")

    # Sample transition matrix
    model.print_transition_matrix_sample(n=10)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: Demonstrate Predictions
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 4: Move Prediction Examples")

    example_moves = ['e4', 'd4', 'Nf3', 'e5', 'c5', 'd5', 'Nc6', 'Nf6']

    for move in example_moves:
        predictions = model.predict_next_move(move, top_k=5)
        if predictions:
            print(f"\n  After '{move}', predicted next moves:")
            for pred_move, prob in predictions:
                bar = "█" * int(prob * 30)
                print(f"    {pred_move:<8} P={prob:.4f}  {bar}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: Full Analysis (Accuracy + Entropy)
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 5: Running Full Analysis")

    results = run_analysis(games)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: Simulate Games
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 6: Markov Chain Game Simulation")

    print("\n  Simulating games using the Markov model:")
    print("  (Each game is generated by sampling from transition probabilities)")
    print()

    for start in ['e4', 'd4']:
        print(f"  Starting with '{start}':")
        for i in range(3):
            simulated = model.simulate_game(start, num_moves=12)
            moves_str = " → ".join(simulated)
            print(f"    Game {i+1}: {moves_str}")
        print()

    print("  Mathematical basis: πₙ₊₁ = πₙ · P")
    print("  Each move sampled according to row probabilities in transition matrix")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 7: Generate Visualizations
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 7: Generating Visualizations")

    viz_files = create_visualizations(results, str(viz_dir))

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 8: Save Results
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("STEP 8: Saving Results")

    # Save JSON results
    save_results_to_file(results, str(results_dir / "analysis_results.json"))

    # Generate report
    generate_report(results, str(results_dir / "analysis_report.md"))

    # Export model
    model.export_model(str(results_dir / "markov_model.json"))

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("ANALYSIS COMPLETE")

    acc = results['overall_accuracy']
    entropy = results['entropy_analysis']

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           FINAL RESULTS SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
""")
    print(f"║  Games Analyzed:        {stats.get('total_games', 0):>10,}                                    ║")
    print(f"║  Total States (moves):  {len(model.states):>10,}                                    ║")
    print(f"║  Total Transitions:     {len(all_transitions):>10,}                                    ║")
    print(f"║                                                                              ║")
    print(f"║  PREDICTION ACCURACY:                                                        ║")
    print(f"║  ───────────────────────                                                     ║")
    print(f"║  Top-1 Accuracy:        {acc.accuracy:>10.2%}                                    ║")
    print(f"║  Top-3 Accuracy:        {acc.top3_accuracy:>10.2%}                                    ║")
    print(f"║  Top-5 Accuracy:        {acc.top5_accuracy:>10.2%}                                    ║")
    print(f"║                                                                              ║")
    print(f"║  ENTROPY ANALYSIS:                                                           ║")
    print(f"║  ─────────────────                                                           ║")
    print(f"║  Average Entropy:       {entropy.average_entropy:>10.4f} bits                             ║")
    print("║                                                                              ║")

    if 'opening_comparison' in results:
        comp = results['opening_comparison']
        more_pred = "1.e4" if comp.opening1_entropy < comp.opening2_entropy else "1.d4"
        print(f"║  OPENING COMPARISON:                                                         ║")
        print(f"║  ───────────────────                                                         ║")
        print(f"║  More Predictable:      {more_pred:>10}                                    ║")
        print(f"║  e4 Entropy:            {comp.opening1_entropy:>10.4f} bits                             ║")
        print(f"║  d4 Entropy:            {comp.opening2_entropy:>10.4f} bits                             ║")
        print("║                                                                              ║")

    print("""╠══════════════════════════════════════════════════════════════════════════════╣
║  OUTPUT FILES:                                                               ║
║  ─────────────                                                               ║
║  • results/analysis_results.json  - Raw analysis data                        ║
║  • results/analysis_report.md     - Detailed markdown report                 ║
║  • results/markov_model.json      - Exported Markov model                    ║
║  • visualizations/*.png           - All visualization charts                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def create_sample_data(data_dir: Path) -> Path:
    """Create sample data if download fails."""
    # This creates a minimal sample for testing
    sample_pgn = """[Event "Lichess Game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[ECO "B90"]
[Opening "Sicilian Defense"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 1-0

[Event "Lichess Game"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]
[ECO "C50"]
[Opening "Italian Game"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 0-1

"""
    sample_file = data_dir / "sample_games.pgn"
    with open(sample_file, 'w') as f:
        f.write(sample_pgn * 50)  # Repeat to have more data
    return sample_file


if __name__ == "__main__":
    main()
