#!/usr/bin/env python3
"""
IB Mathematics HL IA: Elo Rating Analysis
==========================================

Research Question:
"How does the Elo rating difference between two chess players affect the
probability of winning, and at what rating difference does the advantage
plateau such that further differences no longer significantly impact outcomes?"

This follows the EXACT mathematical steps outlined in the IA plan.
"""

import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("STEP 1: DATA PREPARATION")
print("="*70)

# Parse PGN file to extract White Elo, Black Elo, and Game Result
def parse_pgn(filepath):
    """Extract Elo ratings and results from PGN file."""
    games = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    header_pattern = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
    game_blocks = re.split(r'\n\n(?=\[Event )', content)

    for block in game_blocks:
        headers = {}
        for match in header_pattern.finditer(block):
            key, value = match.groups()
            headers[key] = value

        white_elo = headers.get('WhiteElo', '')
        black_elo = headers.get('BlackElo', '')
        result = headers.get('Result', '')

        if white_elo and black_elo and result in ['1-0', '0-1', '1/2-1/2']:
            try:
                white_elo = int(white_elo)
                black_elo = int(black_elo)
                if 1000 <= white_elo <= 3500 and 1000 <= black_elo <= 3500:
                    games.append({
                        'WhiteElo': white_elo,
                        'BlackElo': black_elo,
                        'Result': result
                    })
            except ValueError:
                continue

    return pd.DataFrame(games)

# Load data
print("\nParsing data/grandmaster_games.pgn...")
df = parse_pgn("data/grandmaster_games.pgn")
print(f"Total games extracted: {len(df)}")

# Convert results to numeric outcome: Win = 1, Draw = 0.5, Loss = 0
def result_to_score(result):
    if result == '1-0':
        return 1.0    # White wins
    elif result == '0-1':
        return 0.0    # White loses
    else:
        return 0.5    # Draw

df['Score'] = df['Result'].apply(result_to_score)

# Compute rating difference: ΔR = White Elo - Black Elo
df['DeltaR'] = df['WhiteElo'] - df['BlackElo']

print(f"\nResult distribution:")
print(f"  White wins (1-0):   {(df['Result']=='1-0').sum()} games")
print(f"  Black wins (0-1):   {(df['Result']=='0-1').sum()} games")
print(f"  Draws (1/2-1/2):    {(df['Result']=='1/2-1/2').sum()} games")
print(f"\nRating difference range: {df['DeltaR'].min()} to {df['DeltaR'].max()}")

# Bin ΔR in intervals of 50 points (using 50 instead of 10 for statistical significance)
BIN_SIZE = 50
print(f"\nBinning ΔR in intervals of {BIN_SIZE} points...")

min_bin = (df['DeltaR'].min() // BIN_SIZE) * BIN_SIZE
max_bin = ((df['DeltaR'].max() // BIN_SIZE) + 1) * BIN_SIZE
bins = np.arange(min_bin, max_bin + BIN_SIZE, BIN_SIZE)

df['Bin'] = pd.cut(df['DeltaR'], bins=bins, labels=bins[:-1] + BIN_SIZE/2)

# Compute empirical probability for each bin
# P_win(ΔR) = (Number of Wins + 0.5 * Number of Draws) / Total Games in Bin
binned_data = []

for bin_center in sorted(df['Bin'].dropna().unique()):
    bin_df = df[df['Bin'] == bin_center]
    n_games = len(bin_df)
    n_wins = (bin_df['Result'] == '1-0').sum()
    n_draws = (bin_df['Result'] == '1/2-1/2').sum()
    n_losses = (bin_df['Result'] == '0-1').sum()

    # P_win = (Wins + 0.5*Draws) / Total
    p_win = (n_wins + 0.5 * n_draws) / n_games if n_games > 0 else 0

    binned_data.append({
        'DeltaR': float(bin_center),
        'N_Games': n_games,
        'N_Wins': n_wins,
        'N_Draws': n_draws,
        'N_Losses': n_losses,
        'P_win': p_win
    })

binned_df = pd.DataFrame(binned_data)

# Filter bins with at least 5 games for statistical significance
binned_df = binned_df[binned_df['N_Games'] >= 5].reset_index(drop=True)

print(f"\nEmpirical Probabilities (P_win = (Wins + 0.5×Draws) / Total):")
print(f"\n{'ΔR':>8} {'Games':>8} {'Wins':>6} {'Draws':>6} {'Losses':>6} {'P_win':>10}")
print("-"*55)
for _, row in binned_df.iterrows():
    print(f"{row['DeltaR']:>8.0f} {row['N_Games']:>8} {row['N_Wins']:>6} {row['N_Draws']:>6} {row['N_Losses']:>6} {row['P_win']:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: MATHEMATICAL MODELING - Fit Logistic Function
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 2: MATHEMATICAL MODELING - Logistic Function Fitting")
print("="*70)

print("""
Model: P_win(ΔR) = 1 / (1 + e^(-k(ΔR - x₀)))

where:
  k  = slope parameter
  x₀ = ΔR where probability = 0.5 (midpoint)
""")

# Define logistic function
def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

# Prepare data for fitting
xdata = binned_df['DeltaR'].values
ydata = binned_df['P_win'].values

# Fit the curve
params, covariance = curve_fit(logistic, xdata, ydata, p0=[0.004, 0], maxfev=10000)
k, x0 = params

print(f"FITTED PARAMETERS:")
print(f"  k  = {k:.6f}")
print(f"  x₀ = {x0:.4f}")

# Calculate R² (goodness of fit)
y_pred = logistic(xdata, k, x0)
ss_res = np.sum((ydata - y_pred) ** 2)
ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"  R² = {r_squared:.4f}")

print(f"\nFitted Model:")
print(f"  P_win(ΔR) = 1 / (1 + e^(-{k:.6f}(ΔR - {x0:.2f})))")

# Sample predictions
print(f"\nPredictions at key points:")
for dr in [-400, -200, -100, 0, 100, 200, 400]:
    p = logistic(dr, k, x0)
    print(f"  P_win({dr:>4}) = {p:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: CALCULUS ANALYSIS - Derivatives
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 3: CALCULUS ANALYSIS - Derivatives")
print("="*70)

print("""
First Derivative (sensitivity of probability to rating difference):

  dP/dΔR = k × e^(-k(ΔR - x₀)) / (1 + e^(-k(ΔR - x₀)))²

Second Derivative (rate of change of sensitivity):

  d²P/dΔR² = -k² × e^(-k(ΔR - x₀)) × (1 - e^(-k(ΔR - x₀))) / (1 + e^(-k(ΔR - x₀)))³
""")

# Define derivatives
def dP_dDeltaR(x, k, x0):
    """First derivative of logistic function."""
    exp_term = np.exp(-k * (x - x0))
    return k * exp_term / ((1 + exp_term) ** 2)

def d2P_dDeltaR2(x, k, x0):
    """Second derivative of logistic function."""
    exp_term = np.exp(-k * (x - x0))
    return -k**2 * exp_term * (1 - exp_term) / ((1 + exp_term) ** 3)

# Calculate derivatives at key points
print(f"{'ΔR':>8} {'P(ΔR)':>10} {'dP/dΔR':>12} {'d²P/dΔR²':>14}")
print("-"*48)
for dr in [-400, -300, -200, -100, 0, 100, 200, 300, 400]:
    p = logistic(dr, k, x0)
    dp = dP_dDeltaR(dr, k, x0)
    d2p = d2P_dDeltaR2(dr, k, x0)
    print(f"{dr:>8} {p:>10.4f} {dp:>12.6f} {d2p:>14.8f}")

# Find plateau threshold ΔR* where dP/dΔR < ε
print(f"\nPLATEAU THRESHOLD ANALYSIS:")
print(f"Find ΔR* such that dP/dΔR < ε")
print()

epsilon_values = [0.001, 0.0005, 0.0001]

for epsilon in epsilon_values:
    # Find ΔR* by searching
    for delta_r in range(0, 1000):
        if dP_dDeltaR(x0 + delta_r, k, x0) < epsilon:
            delta_r_star = delta_r
            break
    else:
        delta_r_star = 999

    p_at_star = logistic(x0 + delta_r_star, k, x0)
    print(f"  ε = {epsilon}: ΔR* = ±{delta_r_star} (P_win at +ΔR* = {p_at_star:.4f})")

# Use ε = 0.0005 as primary threshold
EPSILON = 0.0005
for delta_r in range(0, 1000):
    if dP_dDeltaR(x0 + delta_r, k, x0) < EPSILON:
        DELTA_R_STAR = delta_r
        break

print(f"\nUsing ε = {EPSILON}:")
print(f"  Plateau threshold ΔR* = ±{DELTA_R_STAR}")
print(f"  At ΔR = +{DELTA_R_STAR}: P_win = {logistic(x0 + DELTA_R_STAR, k, x0):.4f}")
print(f"  At ΔR = -{DELTA_R_STAR}: P_win = {logistic(x0 - DELTA_R_STAR, k, x0):.4f}")
print(f"\n  Beyond ±{DELTA_R_STAR} rating points, further differences")
print(f"  have negligible impact on win probability.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: INTEGRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 4: INTEGRATION ANALYSIS")
print("="*70)

print("""
Average winning probability across rating spectrum:

  Average P_win = (1/(b-a)) × ∫[a,b] P_win(ΔR) dΔR
""")

# Define integration bounds
a, b = -400, 400

# Numerical integration
def p_win_func(x):
    return logistic(x, k, x0)

integral, error = quad(p_win_func, a, b)
average_p = integral / (b - a)

print(f"Integration from ΔR = {a} to ΔR = {b}:")
print(f"  ∫ P_win(ΔR) dΔR = {integral:.4f}")
print(f"  Range width = {b - a}")
print(f"  Average P_win = {integral:.4f} / {b - a} = {average_p:.4f}")

# Segment breakdown
print(f"\nBreakdown by segments:")
print(f"{'Segment':<20} {'∫P dΔR':>12} {'Avg P':>10}")
print("-"*44)

segments = [(-400, -200), (-200, 0), (0, 200), (200, 400)]
for seg_a, seg_b in segments:
    seg_integral, _ = quad(p_win_func, seg_a, seg_b)
    seg_avg = seg_integral / (seg_b - seg_a)
    print(f"[{seg_a}, {seg_b}]".ljust(20) + f"{seg_integral:>12.4f} {seg_avg:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: CREATE EXCEL FILE WITH ALL DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 5: CREATING EXCEL FILE")
print("="*70)

Path("results").mkdir(exist_ok=True)

with pd.ExcelWriter("results/elo_analysis.xlsx", engine='openpyxl') as writer:

    # Sheet 1: Raw Data (sample)
    df.head(500).to_excel(writer, sheet_name='1_RawData', index=False)

    # Sheet 2: Binned Empirical Probabilities
    binned_df.to_excel(writer, sheet_name='2_EmpiricalProbabilities', index=False)

    # Sheet 3: Model Parameters
    params_df = pd.DataFrame({
        'Parameter': ['k (slope)', 'x0 (midpoint)', 'R² (fit)', 'Plateau ΔR* (ε=0.0005)'],
        'Value': [k, x0, r_squared, DELTA_R_STAR]
    })
    params_df.to_excel(writer, sheet_name='3_ModelParameters', index=False)

    # Sheet 4: Predictions
    pred_data = []
    for dr in range(-500, 501, 25):
        pred_data.append({
            'DeltaR': dr,
            'P_win_predicted': logistic(dr, k, x0),
            'dP_dDeltaR': dP_dDeltaR(dr, k, x0),
            'd2P_dDeltaR2': d2P_dDeltaR2(dr, k, x0)
        })
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_excel(writer, sheet_name='4_Predictions', index=False)

    # Sheet 5: Integration Results
    int_data = []
    for seg_a, seg_b in [(-400, -200), (-200, 0), (0, 200), (200, 400), (-400, 400)]:
        seg_int, _ = quad(p_win_func, seg_a, seg_b)
        int_data.append({
            'Segment': f'[{seg_a}, {seg_b}]',
            'Integral': seg_int,
            'Width': seg_b - seg_a,
            'Average_P': seg_int / (seg_b - seg_a)
        })
    int_df = pd.DataFrame(int_data)
    int_df.to_excel(writer, sheet_name='5_Integration', index=False)

    # Sheet 6: Comparison (Empirical vs Model)
    compare_df = binned_df.copy()
    compare_df['P_predicted'] = logistic(compare_df['DeltaR'].values, k, x0)
    compare_df['Residual'] = compare_df['P_win'] - compare_df['P_predicted']
    compare_df.to_excel(writer, sheet_name='6_Comparison', index=False)

print("Excel file saved to: results/elo_analysis.xlsx")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 6: VISUALIZATIONS")
print("="*70)

Path("visualizations").mkdir(exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Empirical probability vs ΔR with logistic fit overlay
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot of empirical data
ax.scatter(binned_df['DeltaR'], binned_df['P_win'],
           s=binned_df['N_Games']*2, alpha=0.6, c='blue',
           label='Empirical Data (size ∝ sample size)', edgecolors='darkblue')

# Logistic fit curve
x_curve = np.linspace(-600, 600, 1000)
y_curve = logistic(x_curve, k, x0)
ax.plot(x_curve, y_curve, 'r-', linewidth=2.5,
        label=f'Logistic Fit: P = 1/(1+e^({-k:.5f}(ΔR-{x0:.1f})))')

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('Rating Difference (ΔR = White Elo - Black Elo)', fontsize=12)
ax.set_ylabel('P(White wins)', fontsize=12)
ax.set_title(f'Elo Rating Difference vs Win Probability\nR² = {r_squared:.4f}', fontsize=14)
ax.legend(loc='lower right')
ax.set_xlim(-600, 600)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/fig1_logistic_fit.png', dpi=150)
plt.close()
print("Saved: visualizations/fig1_logistic_fit.png")

# Figure 2: Derivative curve with plateau threshold marked
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# First derivative
y_deriv = dP_dDeltaR(x_curve, k, x0)
ax1.plot(x_curve, y_deriv, 'g-', linewidth=2, label='dP/dΔR')
ax1.axhline(y=EPSILON, color='red', linestyle='--', label=f'ε = {EPSILON}')
ax1.axvline(x=x0 + DELTA_R_STAR, color='purple', linestyle=':', label=f'ΔR* = ±{DELTA_R_STAR}')
ax1.axvline(x=x0 - DELTA_R_STAR, color='purple', linestyle=':')
ax1.fill_between(x_curve, y_deriv, where=(np.abs(x_curve - x0) > DELTA_R_STAR),
                  alpha=0.2, color='yellow', label='Plateau region')

ax1.set_xlabel('Rating Difference (ΔR)', fontsize=11)
ax1.set_ylabel('dP/dΔR', fontsize=11)
ax1.set_title('First Derivative: Sensitivity of Win Probability', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Second derivative
y_deriv2 = d2P_dDeltaR2(x_curve, k, x0)
ax2.plot(x_curve, y_deriv2, 'b-', linewidth=2, label='d²P/dΔR²')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax2.axvline(x=x0, color='red', linestyle='--', label=f'Inflection point: ΔR = {x0:.1f}')

ax2.set_xlabel('Rating Difference (ΔR)', fontsize=11)
ax2.set_ylabel('d²P/dΔR²', fontsize=11)
ax2.set_title('Second Derivative: Rate of Change of Sensitivity', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/fig2_derivatives.png', dpi=150)
plt.close()
print("Saved: visualizations/fig2_derivatives.png")

# Figure 3: Integration - area under curve
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(x_curve, y_curve, 'b-', linewidth=2, label='P_win(ΔR)')
ax.fill_between(x_curve[(x_curve >= -400) & (x_curve <= 400)],
                y_curve[(x_curve >= -400) & (x_curve <= 400)],
                alpha=0.3, color='blue', label=f'∫P dΔR = {integral:.2f}')

ax.axhline(y=average_p, color='red', linestyle='--',
           label=f'Average P_win = {average_p:.4f}')
ax.axvline(x=-400, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=400, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Rating Difference (ΔR)', fontsize=12)
ax.set_ylabel('P(White wins)', fontsize=12)
ax.set_title('Integration: Area Under Win Probability Curve\n' +
             f'Average P_win from ΔR=-400 to ΔR=400 = {average_p:.4f}', fontsize=14)
ax.legend()
ax.set_xlim(-600, 600)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/fig3_integration.png', dpi=150)
plt.close()
print("Saved: visualizations/fig3_integration.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 7: FINAL SUMMARY")
print("="*70)

print(f"""
DATA:
  Total games analyzed: {len(df)}
  Rating difference range: {df['DeltaR'].min()} to {df['DeltaR'].max()}

LOGISTIC MODEL:
  P_win(ΔR) = 1 / (1 + e^(-{k:.6f}(ΔR - {x0:.2f})))

  R² = {r_squared:.4f}

KEY PREDICTIONS:
  P_win(ΔR = 0)   = {logistic(0, k, x0):.4f}  (equal ratings)
  P_win(ΔR = 100) = {logistic(100, k, x0):.4f}
  P_win(ΔR = 200) = {logistic(200, k, x0):.4f}
  P_win(ΔR = 400) = {logistic(400, k, x0):.4f}

DERIVATIVE ANALYSIS:
  Maximum sensitivity (at ΔR ≈ {x0:.0f}): dP/dΔR = {k/4:.6f}
  Plateau threshold (ε = {EPSILON}): ΔR* = ±{DELTA_R_STAR}

  Beyond ±{DELTA_R_STAR} rating points, rating differences
  have negligible additional impact on win probability.

INTEGRATION:
  ∫[-400,400] P_win(ΔR) dΔR = {integral:.4f}
  Average P_win = {average_p:.4f}

OUTPUT FILES:
  • results/elo_analysis.xlsx (6 sheets with all data)
  • visualizations/fig1_logistic_fit.png
  • visualizations/fig2_derivatives.png
  • visualizations/fig3_integration.png
""")
