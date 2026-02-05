#!/usr/bin/env python3
"""
IB Math IA: Markov Chain Chess Move Prediction
===============================================

Research Question:
"How can a Markov chain model predict the next move in a chess game using
historical grandmaster games, and how accurate is this model for different
opening strategies?"

Subquestions:
1. How can chess moves be represented as states in a Markov chain?
2. What is the transition matrix of the first 2 moves in e4 vs d4 openings?
3. How accurate is the Markov model at predicting next moves?
4. Which opening is more predictable according to entropy?
"""

import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load and Parse Data
# ═══════════════════════════════════════════════════════════════════════════════

def parse_pgn_simple(filepath):
    """Parse PGN file and extract first 5 moves from each game."""
    import re

    games = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by game (look for [Event tags)
    game_texts = re.split(r'\n\[Event ', content)

    for game_text in game_texts:
        # Find the moves section (after the headers)
        lines = game_text.split('\n')
        move_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith('['):
                move_lines.append(line)

        move_text = ' '.join(move_lines)

        # Remove comments and variations
        move_text = re.sub(r'\{[^}]*\}', '', move_text)
        move_text = re.sub(r'\([^)]*\)', '', move_text)
        move_text = re.sub(r'\[%[^\]]*\]', '', move_text)

        # Remove result
        move_text = re.sub(r'(1-0|0-1|1/2-1/2|\*)', '', move_text)

        # Remove move numbers
        move_text = re.sub(r'\d+\.+', ' ', move_text)

        # Extract moves
        tokens = move_text.split()
        moves = []

        for token in tokens:
            token = token.strip().replace('!', '').replace('?', '')
            # Check if valid move
            if len(token) >= 2 and (
                re.match(r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]', token) or
                token in ['O-O', 'O-O-O', '0-0', '0-0-0']
            ):
                moves.append(token)

        if len(moves) >= 4:  # Need at least 2 full moves
            games.append(moves[:10])  # Keep first 10 half-moves (5 full moves)

    return games


# ═══════════════════════════════════════════════════════════════════════════════
# SUBQUESTION 1: How can chess moves be represented as states?
# ═══════════════════════════════════════════════════════════════════════════════

def print_subquestion_1():
    """Explain Markov chain representation of chess moves."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SUBQUESTION 1: How can chess moves be represented as states in a            ║
║                 Markov chain?                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINITION OF STATES
────────────────────
In our Markov chain model:
  • Each STATE is a chess move in Standard Algebraic Notation
  • Examples of states: e4, d4, Nf6, e5, c5, Nc6, Bb5, etc.

STATE SPACE
───────────
  S = {s₁, s₂, s₃, ...} where each sᵢ is a chess move

  For the opening phase, common states include:
  • e4  (King's Pawn opening)
  • d4  (Queen's Pawn opening)
  • e5  (response to e4)
  • Nf6 (Knight to f6)
  • c5  (Sicilian Defense)

TRANSITION PROBABILITY
──────────────────────
The probability of moving from state i to state j is:

                    Count(move j follows move i)
    P(j|i) = Pᵢⱼ = ─────────────────────────────────
                    Total moves following move i

MARKOV PROPERTY
───────────────
The key assumption: the next move depends ONLY on the current move,
not on any previous moves.

    P(Xₙ₊₁ = j | X₁, X₂, ..., Xₙ) = P(Xₙ₊₁ = j | Xₙ)

This is a FIRST-ORDER Markov chain.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SUBQUESTION 2: Build Transition Matrices for e4 and d4 Openings
# ═══════════════════════════════════════════════════════════════════════════════

def build_opening_data(games):
    """Separate games by opening and extract transitions."""
    e4_games = [g for g in games if g and g[0] == 'e4']
    d4_games = [g for g in games if g and g[0] == 'd4']

    return e4_games, d4_games


def count_transitions(games, max_moves=4):
    """
    Count transitions between moves.

    Returns:
        counts: dict of {from_move: {to_move: count}}
        totals: dict of {from_move: total_count}
    """
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for game in games:
        for i in range(min(len(game) - 1, max_moves)):
            from_move = game[i]
            to_move = game[i + 1]
            counts[from_move][to_move] += 1
            totals[from_move] += 1

    return dict(counts), dict(totals)


def build_transition_matrix(counts, totals, states):
    """
    Build transition matrix from counts.

    Mathematical formula:
        Pᵢⱼ = Count(j follows i) / Total from i
    """
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    P = np.zeros((n, n))

    for i, state_i in enumerate(states):
        if state_i in totals and totals[state_i] > 0:
            for j, state_j in enumerate(states):
                count = counts.get(state_i, {}).get(state_j, 0)
                P[i, j] = count / totals[state_i]

    return P, state_to_idx


def print_subquestion_2(e4_games, d4_games):
    """Build and display transition matrices for first 2 moves."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SUBQUESTION 2: What is the transition matrix of the first 2 moves           ║
║                 in selected openings (e4 vs d4)?                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # ─────────────────────────────────────────────────────────────────────────
    # E4 OPENING ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"KING'S PAWN OPENING (1.e4)")
    print(f"═══════════════════════════")
    print(f"Total games analyzed: {len(e4_games)}")

    # Count transitions for e4 games (first 4 half-moves = 2 full moves)
    e4_counts, e4_totals = count_transitions(e4_games, max_moves=4)

    # Get the most common responses to e4
    e4_responses = e4_counts.get('e4', {})
    print(f"\nStep 1: Count frequencies after 1.e4")
    print(f"─────────────────────────────────────")

    sorted_e4_responses = sorted(e4_responses.items(), key=lambda x: -x[1])[:8]
    total_after_e4 = e4_totals.get('e4', 1)

    print(f"{'Black Response':<15} {'Count':>10} {'Probability':>15}")
    print(f"{'─'*15} {'─'*10} {'─'*15}")
    for move, count in sorted_e4_responses:
        prob = count / total_after_e4
        print(f"{move:<15} {count:>10} {prob:>15.4f}")
    print(f"{'─'*15} {'─'*10} {'─'*15}")
    print(f"{'Total':<15} {total_after_e4:>10} {'1.0000':>15}")

    # Build small transition matrix for e4 opening (top responses)
    e4_states = ['e4'] + [m for m, _ in sorted_e4_responses[:5]]

    # Get second-move transitions
    print(f"\nStep 2: Build Transition Matrix")
    print(f"────────────────────────────────")
    print(f"\nTransition matrix P for e4 opening (first 2 moves):")
    print(f"States: {e4_states}")

    e4_matrix, e4_idx = build_transition_matrix(e4_counts, e4_totals, e4_states)

    # Print matrix
    print(f"\n{'':>8}", end='')
    for s in e4_states:
        print(f"{s:>8}", end='')
    print()

    for i, s_from in enumerate(e4_states):
        print(f"{s_from:>8}", end='')
        for j in range(len(e4_states)):
            if e4_matrix[i, j] > 0:
                print(f"{e4_matrix[i, j]:>8.3f}", end='')
            else:
                print(f"{'─':>8}", end='')
        print()

    # Show calculation example
    if sorted_e4_responses:
        top_response, top_count = sorted_e4_responses[0]
        print(f"\n📐 Example Calculation:")
        print(f"   P('{top_response}' | 'e4') = {top_count} / {total_after_e4} = {top_count/total_after_e4:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # D4 OPENING ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\nQUEEN'S PAWN OPENING (1.d4)")
    print(f"════════════════════════════")
    print(f"Total games analyzed: {len(d4_games)}")

    d4_counts, d4_totals = count_transitions(d4_games, max_moves=4)

    d4_responses = d4_counts.get('d4', {})
    print(f"\nStep 1: Count frequencies after 1.d4")
    print(f"─────────────────────────────────────")

    sorted_d4_responses = sorted(d4_responses.items(), key=lambda x: -x[1])[:8]
    total_after_d4 = d4_totals.get('d4', 1)

    print(f"{'Black Response':<15} {'Count':>10} {'Probability':>15}")
    print(f"{'─'*15} {'─'*10} {'─'*15}")
    for move, count in sorted_d4_responses:
        prob = count / total_after_d4
        print(f"{move:<15} {count:>10} {prob:>15.4f}")
    print(f"{'─'*15} {'─'*10} {'─'*15}")
    print(f"{'Total':<15} {total_after_d4:>10} {'1.0000':>15}")

    # Build small transition matrix for d4 opening
    d4_states = ['d4'] + [m for m, _ in sorted_d4_responses[:5]]

    print(f"\nStep 2: Build Transition Matrix")
    print(f"────────────────────────────────")
    print(f"\nTransition matrix P for d4 opening (first 2 moves):")
    print(f"States: {d4_states}")

    d4_matrix, d4_idx = build_transition_matrix(d4_counts, d4_totals, d4_states)

    # Print matrix
    print(f"\n{'':>8}", end='')
    for s in d4_states:
        print(f"{s:>8}", end='')
    print()

    for i, s_from in enumerate(d4_states):
        print(f"{s_from:>8}", end='')
        for j in range(len(d4_states)):
            if d4_matrix[i, j] > 0:
                print(f"{d4_matrix[i, j]:>8.3f}", end='')
            else:
                print(f"{'─':>8}", end='')
        print()

    if sorted_d4_responses:
        top_response, top_count = sorted_d4_responses[0]
        print(f"\n📐 Example Calculation:")
        print(f"   P('{top_response}' | 'd4') = {top_count} / {total_after_d4} = {top_count/total_after_d4:.4f}")

    return (e4_counts, e4_totals, e4_states, e4_matrix,
            d4_counts, d4_totals, d4_states, d4_matrix,
            sorted_e4_responses, sorted_d4_responses)


# ═══════════════════════════════════════════════════════════════════════════════
# SUBQUESTION 3: Prediction Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_accuracy(games, counts, totals):
    """
    Calculate prediction accuracy.

    For each transition in the data:
    1. Predict the most probable next move
    2. Compare with actual move
    3. Accuracy = correct / total
    """
    correct = 0
    total = 0

    predictions = []

    for game in games:
        for i in range(min(len(game) - 1, 4)):  # First 4 transitions
            current_move = game[i]
            actual_next = game[i + 1]

            # Get most probable next move from our model
            if current_move in counts and counts[current_move]:
                predicted = max(counts[current_move].items(), key=lambda x: x[1])[0]
                prob = counts[current_move][predicted] / totals[current_move]

                is_correct = (predicted == actual_next)
                if is_correct:
                    correct += 1
                total += 1

                predictions.append({
                    'current': current_move,
                    'predicted': predicted,
                    'actual': actual_next,
                    'probability': prob,
                    'correct': is_correct
                })

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, predictions


def print_subquestion_3(e4_games, d4_games, e4_counts, e4_totals, d4_counts, d4_totals):
    """Calculate and display prediction accuracy."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SUBQUESTION 3: How accurate is the Markov model at predicting               ║
║                 next moves for each opening?                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

ACCURACY FORMULA
────────────────
                    Number of Correct Predictions
    Accuracy = ─────────────────────────────────────
                    Total Number of Predictions

A prediction is CORRECT when the most probable next move (according to
our transition matrix) matches the actual move played in the game.
""")

    # E4 Opening Accuracy
    print(f"\nKING'S PAWN OPENING (1.e4) - Prediction Accuracy")
    print(f"═" * 50)

    e4_acc, e4_correct, e4_total, e4_preds = calculate_accuracy(
        e4_games, e4_counts, e4_totals
    )

    # Show sample predictions
    print(f"\nSample Predictions (first 10):")
    print(f"{'Current':<10} {'Predicted':<12} {'Actual':<12} {'P(pred)':<10} {'Correct':<8}")
    print(f"{'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

    for pred in e4_preds[:10]:
        check = "✓" if pred['correct'] else "✗"
        print(f"{pred['current']:<10} {pred['predicted']:<12} {pred['actual']:<12} {pred['probability']:<10.4f} {check:<8}")

    print(f"\n📐 Accuracy Calculation for e4 opening:")
    print(f"   Correct predictions: {e4_correct}")
    print(f"   Total predictions:   {e4_total}")
    print(f"   Accuracy = {e4_correct} / {e4_total} = {e4_acc:.4f} = {e4_acc*100:.2f}%")

    # D4 Opening Accuracy
    print(f"\n\nQUEEN'S PAWN OPENING (1.d4) - Prediction Accuracy")
    print(f"═" * 50)

    d4_acc, d4_correct, d4_total, d4_preds = calculate_accuracy(
        d4_games, d4_counts, d4_totals
    )

    print(f"\nSample Predictions (first 10):")
    print(f"{'Current':<10} {'Predicted':<12} {'Actual':<12} {'P(pred)':<10} {'Correct':<8}")
    print(f"{'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

    for pred in d4_preds[:10]:
        check = "✓" if pred['correct'] else "✗"
        print(f"{pred['current']:<10} {pred['predicted']:<12} {pred['actual']:<12} {pred['probability']:<10.4f} {check:<8}")

    print(f"\n📐 Accuracy Calculation for d4 opening:")
    print(f"   Correct predictions: {d4_correct}")
    print(f"   Total predictions:   {d4_total}")
    print(f"   Accuracy = {d4_correct} / {d4_total} = {d4_acc:.4f} = {d4_acc*100:.2f}%")

    # Comparison
    print(f"\n\nACCURACY COMPARISON")
    print(f"═" * 50)
    print(f"┌────────────────────┬────────────────┬────────────────┐")
    print(f"│ Opening            │ Accuracy       │ Correct/Total  │")
    print(f"├────────────────────┼────────────────┼────────────────┤")
    print(f"│ King's Pawn (e4)   │ {e4_acc*100:>10.2f}%   │ {e4_correct:>5}/{e4_total:<5}     │")
    print(f"│ Queen's Pawn (d4)  │ {d4_acc*100:>10.2f}%   │ {d4_correct:>5}/{d4_total:<5}     │")
    print(f"└────────────────────┴────────────────┴────────────────┘")

    return e4_acc, d4_acc


# ═══════════════════════════════════════════════════════════════════════════════
# SUBQUESTION 4: Entropy Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_entropy(probabilities):
    """
    Calculate Shannon entropy.

    H = -Σ Pⱼ × log₂(Pⱼ)

    Only sum over non-zero probabilities (0 × log(0) is defined as 0).
    """
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def print_subquestion_4(e4_counts, e4_totals, d4_counts, d4_totals,
                        sorted_e4_responses, sorted_d4_responses):
    """Calculate and compare entropy for each opening."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SUBQUESTION 4: Which opening is more predictable according to               ║
║                 the Markov model?                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

ENTROPY FORMULA (Shannon Entropy)
─────────────────────────────────
    H = -Σ Pⱼ × log₂(Pⱼ)

where Pⱼ is the probability of each possible next move.

INTERPRETATION:
  • H = 0 bits:     Only one possible next move (completely predictable)
  • Higher H:       More uncertainty (less predictable)
  • H = log₂(n):    All n moves equally likely (maximum uncertainty)

LOWER ENTROPY = MORE PREDICTABLE
""")

    # Calculate entropy for e4 opening (responses to e4)
    print(f"\nKING'S PAWN OPENING (1.e4) - Entropy Calculation")
    print(f"═" * 50)

    total_e4 = e4_totals.get('e4', 1)
    e4_probs = [(move, count/total_e4) for move, count in sorted_e4_responses]

    print(f"\nProbability distribution after 1.e4:")
    print(f"{'Move':<10} {'Count':>8} {'Probability':>12} {'-P×log₂(P)':>14}")
    print(f"{'─'*10} {'─'*8} {'─'*12} {'─'*14}")

    e4_entropy_terms = []
    for move, prob in e4_probs:
        count = int(prob * total_e4)
        term = -prob * np.log2(prob) if prob > 0 else 0
        e4_entropy_terms.append(term)
        print(f"{move:<10} {count:>8} {prob:>12.4f} {term:>14.4f}")

    H_e4 = sum(e4_entropy_terms)
    print(f"{'─'*10} {'─'*8} {'─'*12} {'─'*14}")
    print(f"{'Sum':>32} {H_e4:>14.4f}")

    print(f"\n📐 Entropy Calculation for e4:")
    print(f"   H(e4) = -Σ Pⱼ × log₂(Pⱼ)")
    calc_str = " + ".join([f"{t:.4f}" for t in e4_entropy_terms[:4]]) + " + ..."
    print(f"   H(e4) = {calc_str}")
    print(f"   H(e4) = {H_e4:.4f} bits")

    # Calculate entropy for d4 opening
    print(f"\n\nQUEEN'S PAWN OPENING (1.d4) - Entropy Calculation")
    print(f"═" * 50)

    total_d4 = d4_totals.get('d4', 1)
    d4_probs = [(move, count/total_d4) for move, count in sorted_d4_responses]

    print(f"\nProbability distribution after 1.d4:")
    print(f"{'Move':<10} {'Count':>8} {'Probability':>12} {'-P×log₂(P)':>14}")
    print(f"{'─'*10} {'─'*8} {'─'*12} {'─'*14}")

    d4_entropy_terms = []
    for move, prob in d4_probs:
        count = int(prob * total_d4)
        term = -prob * np.log2(prob) if prob > 0 else 0
        d4_entropy_terms.append(term)
        print(f"{move:<10} {count:>8} {prob:>12.4f} {term:>14.4f}")

    H_d4 = sum(d4_entropy_terms)
    print(f"{'─'*10} {'─'*8} {'─'*12} {'─'*14}")
    print(f"{'Sum':>32} {H_d4:>14.4f}")

    print(f"\n📐 Entropy Calculation for d4:")
    print(f"   H(d4) = -Σ Pⱼ × log₂(Pⱼ)")
    calc_str = " + ".join([f"{t:.4f}" for t in d4_entropy_terms[:4]]) + " + ..."
    print(f"   H(d4) = {calc_str}")
    print(f"   H(d4) = {H_d4:.4f} bits")

    # Final Comparison
    print(f"\n\n" + "═" * 70)
    print(f"ENTROPY COMPARISON - WHICH OPENING IS MORE PREDICTABLE?")
    print(f"═" * 70)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTROPY RESULTS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   King's Pawn (1.e4):     H = {H_e4:.4f} bits                        │
│   Queen's Pawn (1.d4):    H = {H_d4:.4f} bits                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   CONCLUSION:                                                       │
│   ───────────                                                       │""")

    if H_e4 < H_d4:
        diff = H_d4 - H_e4
        print(f"""│   The KING'S PAWN (1.e4) opening is MORE PREDICTABLE              │
│   because it has LOWER entropy.                                     │
│                                                                     │
│   Difference: {H_d4:.4f} - {H_e4:.4f} = {diff:.4f} bits                     │""")
        more_pred = "e4"
    else:
        diff = H_e4 - H_d4
        print(f"""│   The QUEEN'S PAWN (1.d4) opening is MORE PREDICTABLE             │
│   because it has LOWER entropy.                                     │
│                                                                     │
│   Difference: {H_e4:.4f} - {H_d4:.4f} = {diff:.4f} bits                     │""")
        more_pred = "d4"

    print(f"""│                                                                     │
│   Lower entropy means fewer "surprising" moves - the responses      │
│   are more concentrated on a few common choices.                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

    return H_e4, H_d4, more_pred


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_visualizations(sorted_e4_responses, sorted_d4_responses,
                          e4_totals, d4_totals, H_e4, H_d4, e4_acc, d4_acc):
    """Create clear visualizations for the IA."""

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # ─────────────────────────────────────────────────────────────────────
    # Figure 1: Response Distributions (Side by Side)
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # E4 responses
    ax1 = axes[0]
    e4_moves = [m for m, _ in sorted_e4_responses[:6]]
    e4_probs = [c/e4_totals['e4'] for _, c in sorted_e4_responses[:6]]
    colors_e4 = plt.cm.Blues(np.linspace(0.4, 0.8, len(e4_moves)))

    bars1 = ax1.bar(e4_moves, e4_probs, color=colors_e4, edgecolor='black')
    ax1.set_xlabel('Black\'s Response', fontsize=12)
    ax1.set_ylabel('Probability P(response | e4)', fontsize=12)
    ax1.set_title('Response Distribution after 1.e4\n(King\'s Pawn Opening)', fontsize=14)
    ax1.set_ylim(0, max(e4_probs) * 1.3)

    for bar, prob in zip(bars1, e4_probs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', fontsize=10)

    # D4 responses
    ax2 = axes[1]
    d4_moves = [m for m, _ in sorted_d4_responses[:6]]
    d4_probs = [c/d4_totals['d4'] for _, c in sorted_d4_responses[:6]]
    colors_d4 = plt.cm.Oranges(np.linspace(0.4, 0.8, len(d4_moves)))

    bars2 = ax2.bar(d4_moves, d4_probs, color=colors_d4, edgecolor='black')
    ax2.set_xlabel('Black\'s Response', fontsize=12)
    ax2.set_ylabel('Probability P(response | d4)', fontsize=12)
    ax2.set_title('Response Distribution after 1.d4\n(Queen\'s Pawn Opening)', fontsize=14)
    ax2.set_ylim(0, max(d4_probs) * 1.3)

    for bar, prob in zip(bars2, d4_probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'response_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: visualizations/response_distributions.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 2: Entropy Comparison
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    openings = ['King\'s Pawn\n(1.e4)', 'Queen\'s Pawn\n(1.d4)']
    entropies = [H_e4, H_d4]
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(openings, entropies, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Entropy H (bits)', fontsize=14)
    ax.set_title('Entropy Comparison: Which Opening is More Predictable?\n(Lower Entropy = More Predictable)', fontsize=14)

    for bar, ent in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'H = {ent:.4f}', ha='center', fontsize=14, fontweight='bold')

    # Add annotation for winner
    min_idx = np.argmin(entropies)
    ax.annotate('More\nPredictable',
                xy=(min_idx, entropies[min_idx]),
                xytext=(min_idx, entropies[min_idx] + 0.5),
                ha='center', fontsize=12,
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_ylim(0, max(entropies) * 1.4)

    # Add formula
    ax.text(0.5, -0.15, r'$H = -\sum P_j \times \log_2(P_j)$',
            transform=ax.transAxes, ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir / 'entropy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: visualizations/entropy_comparison.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 3: Accuracy Comparison
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = [e4_acc * 100, d4_acc * 100]

    bars = ax.bar(openings, accuracies, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=14)
    ax.set_title('Prediction Accuracy Comparison\nAccuracy = Correct Predictions / Total Predictions', fontsize=14)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', fontsize=14, fontweight='bold')

    ax.set_ylim(0, max(accuracies) * 1.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: visualizations/accuracy_comparison.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 4: Combined Summary
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: e4 distribution
    ax1 = axes[0, 0]
    bars1 = ax1.bar(e4_moves, e4_probs, color='#3498db', edgecolor='black')
    ax1.set_title('P(response | e4)', fontsize=12)
    ax1.set_ylabel('Probability')
    for bar, prob in zip(bars1, e4_probs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{prob:.3f}', ha='center', fontsize=9)

    # Top right: d4 distribution
    ax2 = axes[0, 1]
    bars2 = ax2.bar(d4_moves, d4_probs, color='#e74c3c', edgecolor='black')
    ax2.set_title('P(response | d4)', fontsize=12)
    ax2.set_ylabel('Probability')
    for bar, prob in zip(bars2, d4_probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{prob:.3f}', ha='center', fontsize=9)

    # Bottom left: Entropy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(['e4', 'd4'], entropies, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax3.set_title('Entropy Comparison', fontsize=12)
    ax3.set_ylabel('Entropy (bits)')
    for bar, ent in zip(bars3, entropies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ent:.4f}', ha='center', fontsize=11, fontweight='bold')

    # Bottom right: Accuracy
    ax4 = axes[1, 1]
    bars4 = ax4.bar(['e4', 'd4'], accuracies, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax4.set_title('Prediction Accuracy', fontsize=12)
    ax4.set_ylabel('Accuracy (%)')
    for bar, acc in zip(bars4, accuracies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('IB Math IA: Markov Chain Chess Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_dir / 'ia_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: visualizations/ia_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the complete IB Math IA analysis."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           IB MATHEMATICS IA: MARKOV CHAIN CHESS ANALYSIS                     ║
║           ══════════════════════════════════════════════════                 ║
║                                                                              ║
║  Research Question:                                                          ║
║  "How can a Markov chain model predict the next move in a chess game        ║
║   using historical grandmaster games, and how accurate is this model        ║
║   for different opening strategies?"                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Load data
    print("Loading chess games from data/grandmaster_games.pgn...")
    data_file = Path("data/grandmaster_games.pgn")

    if not data_file.exists():
        print("ERROR: Data file not found. Please run main.py first to download data.")
        return

    games = parse_pgn_simple(str(data_file))
    print(f"Loaded {len(games)} games\n")

    # Separate by opening
    e4_games, d4_games = build_opening_data(games)
    print(f"Games starting with 1.e4: {len(e4_games)}")
    print(f"Games starting with 1.d4: {len(d4_games)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Run each subquestion
    # ═══════════════════════════════════════════════════════════════════════

    # Subquestion 1
    print_subquestion_1()

    # Subquestion 2
    (e4_counts, e4_totals, e4_states, e4_matrix,
     d4_counts, d4_totals, d4_states, d4_matrix,
     sorted_e4_responses, sorted_d4_responses) = print_subquestion_2(e4_games, d4_games)

    # Subquestion 3
    e4_acc, d4_acc = print_subquestion_3(
        e4_games, d4_games, e4_counts, e4_totals, d4_counts, d4_totals
    )

    # Subquestion 4
    H_e4, H_d4, more_pred = print_subquestion_4(
        e4_counts, e4_totals, d4_counts, d4_totals,
        sorted_e4_responses, sorted_d4_responses
    )

    # Create visualizations
    print("\n" + "═" * 70)
    print("CREATING VISUALIZATIONS")
    print("═" * 70)
    create_visualizations(
        sorted_e4_responses, sorted_d4_responses,
        e4_totals, d4_totals, H_e4, H_d4, e4_acc, d4_acc
    )

    # Final Summary
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           FINAL SUMMARY                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DATA:                                                                       ║
║    • Total games analyzed: {len(games):>6}                                        ║
║    • King's Pawn (e4) games: {len(e4_games):>5}                                        ║
║    • Queen's Pawn (d4) games: {len(d4_games):>4}                                        ║
║                                                                              ║
║  PREDICTION ACCURACY:                                                        ║
║    • 1.e4 opening: {e4_acc*100:>6.2f}%                                              ║
║    • 1.d4 opening: {d4_acc*100:>6.2f}%                                              ║
║                                                                              ║
║  ENTROPY (Predictability):                                                   ║
║    • 1.e4 opening: H = {H_e4:.4f} bits                                       ║
║    • 1.d4 opening: H = {H_d4:.4f} bits                                       ║
║                                                                              ║
║  CONCLUSION:                                                                 ║
║    The {"1.e4" if more_pred == "e4" else "1.d4"} opening is MORE PREDICTABLE (lower entropy)              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
