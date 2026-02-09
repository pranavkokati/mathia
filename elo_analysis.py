#!/usr/bin/env python3
"""
IB Mathematics HL IA: Elo Rating Analysis
==========================================

Research Question:
"How does the Elo rating difference between two chess players affect the
probability of winning, and at what rating difference does the advantage
plateau such that further differences no longer significantly impact outcomes?"

This script:
1. Parses real Lichess grandmaster games
2. Extracts Elo ratings and game outcomes
3. Calculates empirical win probabilities
4. Fits a logistic function
5. Computes derivatives to find plateau threshold
6. Performs integration analysis
7. Exports everything to Excel with full calculations shown
"""

import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA EXTRACTION FROM PGN
# ═══════════════════════════════════════════════════════════════════════════════

def extract_elo_data(pgn_path):
    """
    Extract Elo ratings and game results from PGN file.

    Returns:
        DataFrame with columns: WhiteElo, BlackElo, Result, DeltaR, Outcome
    """
    print("=" * 70)
    print("STEP 1: DATA EXTRACTION")
    print("=" * 70)
    print(f"\nParsing PGN file: {pgn_path}")

    games = []

    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Pattern to match game headers
    header_pattern = re.compile(r'\[(\w+)\s+"([^"]*)"\]')

    # Split by games (look for [Event tags)
    game_blocks = re.split(r'\n\n(?=\[Event )', content)

    for block in game_blocks:
        headers = {}
        for match in header_pattern.finditer(block):
            key, value = match.groups()
            headers[key] = value

        # Extract required fields
        white_elo = headers.get('WhiteElo', '')
        black_elo = headers.get('BlackElo', '')
        result = headers.get('Result', '')

        # Validate data
        if white_elo and black_elo and result in ['1-0', '0-1', '1/2-1/2']:
            try:
                white_elo = int(white_elo)
                black_elo = int(black_elo)

                # Only include reasonable Elo ranges (exclude provisional ratings)
                if 1000 <= white_elo <= 3500 and 1000 <= black_elo <= 3500:
                    games.append({
                        'WhiteElo': white_elo,
                        'BlackElo': black_elo,
                        'Result': result
                    })
            except ValueError:
                continue

    # Create DataFrame
    df = pd.DataFrame(games)

    if len(df) == 0:
        print("ERROR: No valid games found with Elo ratings")
        return None

    # Calculate rating difference: ΔR = White Elo - Black Elo
    df['DeltaR'] = df['WhiteElo'] - df['BlackElo']

    # Convert result to numeric outcome (from White's perspective)
    # Win = 1, Draw = 0.5, Loss = 0
    def result_to_outcome(result):
        if result == '1-0':
            return 1.0  # White wins
        elif result == '0-1':
            return 0.0  # White loses
        else:
            return 0.5  # Draw

    df['Outcome'] = df['Result'].apply(result_to_outcome)

    print(f"\nData Extraction Complete:")
    print(f"  Total games extracted: {len(df):,}")
    print(f"  White Elo range: {df['WhiteElo'].min()} to {df['WhiteElo'].max()}")
    print(f"  Black Elo range: {df['BlackElo'].min()} to {df['BlackElo'].max()}")
    print(f"  Rating difference range: {df['DeltaR'].min()} to {df['DeltaR'].max()}")
    print(f"\n  Result Distribution:")
    print(f"    White wins (1-0):  {(df['Result'] == '1-0').sum():>6} ({(df['Result'] == '1-0').mean()*100:.1f}%)")
    print(f"    Black wins (0-1):  {(df['Result'] == '0-1').sum():>6} ({(df['Result'] == '0-1').mean()*100:.1f}%)")
    print(f"    Draws (1/2-1/2):   {(df['Result'] == '1/2-1/2').sum():>6} ({(df['Result'] == '1/2-1/2').mean()*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: BIN DATA AND CALCULATE EMPIRICAL PROBABILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_empirical_probabilities(df, bin_size=50):
    """
    Bin rating differences and calculate empirical win probabilities.

    Formula:
        P_win(ΔR) = (Wins + 0.5 × Draws) / Total Games in Bin

    Args:
        df: DataFrame with DeltaR and Outcome columns
        bin_size: Width of each bin (default 50 points)

    Returns:
        DataFrame with binned probabilities
    """
    print("\n" + "=" * 70)
    print("STEP 2: EMPIRICAL PROBABILITY CALCULATION")
    print("=" * 70)

    print(f"\nBinning rating differences with bin size = {bin_size} points")
    print(f"\nFormula: P_win(ΔR) = (Wins + 0.5 × Draws) / Total Games")

    # Create bins
    min_delta = (df['DeltaR'].min() // bin_size) * bin_size
    max_delta = ((df['DeltaR'].max() // bin_size) + 1) * bin_size
    bins = np.arange(min_delta, max_delta + bin_size, bin_size)

    # Assign each game to a bin
    df['Bin'] = pd.cut(df['DeltaR'], bins=bins, labels=bins[:-1] + bin_size/2)

    # Calculate statistics for each bin
    binned_data = []

    for bin_center in sorted(df['Bin'].dropna().unique()):
        bin_df = df[df['Bin'] == bin_center]

        n_games = len(bin_df)
        n_wins = (bin_df['Result'] == '1-0').sum()
        n_losses = (bin_df['Result'] == '0-1').sum()
        n_draws = (bin_df['Result'] == '1/2-1/2').sum()

        # Empirical probability
        p_win = bin_df['Outcome'].mean()

        binned_data.append({
            'DeltaR': float(bin_center),
            'N_Games': n_games,
            'N_Wins': n_wins,
            'N_Losses': n_losses,
            'N_Draws': n_draws,
            'P_win_empirical': p_win
        })

    binned_df = pd.DataFrame(binned_data)

    # Filter bins with at least 10 games for statistical significance
    min_games = 10
    binned_df = binned_df[binned_df['N_Games'] >= min_games].reset_index(drop=True)

    print(f"\nEmpirical Probability Calculation:")
    print(f"  Total bins created: {len(binned_df)}")
    print(f"  Minimum games per bin: {min_games}")
    print(f"  ΔR range used: {binned_df['DeltaR'].min():.0f} to {binned_df['DeltaR'].max():.0f}")

    print(f"\n  Sample Calculations:")
    print(f"  {'ΔR':>8} {'Games':>8} {'Wins':>6} {'Draws':>6} {'Losses':>6} {'P_win':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")

    # Show a few sample rows
    sample_indices = [0, len(binned_df)//4, len(binned_df)//2, 3*len(binned_df)//4, len(binned_df)-1]
    for idx in sample_indices:
        if idx < len(binned_df):
            row = binned_df.iloc[idx]
            print(f"  {row['DeltaR']:>8.0f} {row['N_Games']:>8} {row['N_Wins']:>6} "
                  f"{row['N_Draws']:>6} {row['N_Losses']:>6} {row['P_win_empirical']:>10.4f}")

    return binned_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: LOGISTIC FUNCTION FITTING
# ═══════════════════════════════════════════════════════════════════════════════

def logistic_function(x, k, x0):
    """
    Logistic function for modeling win probability.

    P(ΔR) = 1 / (1 + e^(-k(ΔR - x₀)))

    Parameters:
        x: Rating difference (ΔR)
        k: Slope parameter (determines steepness)
        x0: Midpoint (ΔR where P = 0.5)

    Returns:
        Probability of winning
    """
    return 1 / (1 + np.exp(-k * (x - x0)))


def fit_logistic_model(binned_df):
    """
    Fit logistic function to empirical data using least squares.

    Returns:
        k, x0: Fitted parameters
        r_squared: Goodness of fit
        residuals: Difference between predicted and actual
    """
    print("\n" + "=" * 70)
    print("STEP 3: LOGISTIC FUNCTION FITTING")
    print("=" * 70)

    print(f"\nModel: P(ΔR) = 1 / (1 + e^(-k(ΔR - x₀)))")
    print(f"\nwhere:")
    print(f"  k  = slope parameter (steepness of curve)")
    print(f"  x₀ = midpoint (ΔR where probability = 0.5)")

    # Prepare data
    x_data = binned_df['DeltaR'].values
    y_data = binned_df['P_win_empirical'].values
    weights = np.sqrt(binned_df['N_Games'].values)  # Weight by sample size

    # Initial guess
    p0 = [0.004, 0]  # k around 1/250, x0 around 0

    # Fit curve
    try:
        params, covariance = curve_fit(
            logistic_function, x_data, y_data,
            p0=p0,
            sigma=1/weights,
            maxfev=10000
        )
        k, x0 = params

        # Calculate predictions
        y_pred = logistic_function(x_data, k, x0)

        # Calculate R²
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Residuals
        residuals = y_data - y_pred

        print(f"\n┌{'─'*50}┐")
        print(f"│{'FITTED PARAMETERS':^50}│")
        print(f"├{'─'*50}┤")
        print(f"│  k (slope)     = {k:>20.6f}             │")
        print(f"│  x₀ (midpoint) = {x0:>20.4f}             │")
        print(f"│  R² (fit)      = {r_squared:>20.4f}             │")
        print(f"└{'─'*50}┘")

        print(f"\nInterpretation:")
        print(f"  • The fitted logistic function is:")
        print(f"    P(ΔR) = 1 / (1 + e^({-k:.6f} × (ΔR - {x0:.2f})))")
        print(f"  • At ΔR = 0 (equal ratings), P(win) = {logistic_function(0, k, x0):.4f}")
        print(f"  • At ΔR = 100, P(win) = {logistic_function(100, k, x0):.4f}")
        print(f"  • At ΔR = 200, P(win) = {logistic_function(200, k, x0):.4f}")
        print(f"  • At ΔR = 400, P(win) = {logistic_function(400, k, x0):.4f}")

        return k, x0, r_squared, residuals, y_pred

    except Exception as e:
        print(f"Error fitting logistic function: {e}")
        return None, None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: DERIVATIVE ANALYSIS (CALCULUS)
# ═══════════════════════════════════════════════════════════════════════════════

def logistic_derivative(x, k, x0):
    """
    First derivative of logistic function.

    dP/dΔR = k × e^(-k(ΔR - x₀)) / (1 + e^(-k(ΔR - x₀)))²

    This represents the SENSITIVITY of probability to rating changes.
    Higher derivative = more sensitive to rating difference.
    """
    exp_term = np.exp(-k * (x - x0))
    return k * exp_term / ((1 + exp_term) ** 2)


def logistic_second_derivative(x, k, x0):
    """
    Second derivative of logistic function.

    d²P/dΔR² = -k² × e^(-k(ΔR - x₀)) × (1 - e^(-k(ΔR - x₀))) / (1 + e^(-k(ΔR - x₀)))³

    Shows the inflection point where probability sensitivity changes fastest.
    """
    exp_term = np.exp(-k * (x - x0))
    numerator = -k**2 * exp_term * (1 - exp_term)
    denominator = (1 + exp_term) ** 3
    return numerator / denominator


def find_plateau_threshold(k, x0, epsilon=0.0001):
    """
    Find the rating difference beyond which further differences don't matter.

    Solve for ΔR* where: dP/dΔR < ε

    At the plateau, the derivative approaches zero, meaning changes in
    rating difference have negligible effect on win probability.

    Args:
        k, x0: Logistic parameters
        epsilon: Threshold for "negligible" derivative

    Returns:
        delta_r_star: Plateau threshold (both positive and negative)
    """
    # Maximum derivative occurs at x0
    max_derivative = k / 4  # This is the value at x = x0

    # Find where derivative drops below epsilon
    # Using numerical search
    for delta_r in range(0, 1000, 1):
        deriv = logistic_derivative(x0 + delta_r, k, x0)
        if deriv < epsilon:
            return delta_r

    return 500  # Default if not found


def derivative_analysis(k, x0, binned_df):
    """
    Perform derivative analysis to find plateau threshold.
    """
    print("\n" + "=" * 70)
    print("STEP 4: DERIVATIVE ANALYSIS (CALCULUS)")
    print("=" * 70)

    print(f"\nFirst Derivative Formula:")
    print(f"  dP/dΔR = k × e^(-k(ΔR - x₀)) / (1 + e^(-k(ΔR - x₀)))²")
    print(f"\nThis represents the SENSITIVITY of win probability to rating changes.")

    # Calculate derivatives at key points
    delta_r_values = np.arange(-400, 401, 50)

    print(f"\n  {'ΔR':>8} {'P(ΔR)':>10} {'dP/dΔR':>12} {'d²P/dΔR²':>14}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*14}")

    derivative_data = []

    for delta_r in delta_r_values:
        p = logistic_function(delta_r, k, x0)
        dp = logistic_derivative(delta_r, k, x0)
        d2p = logistic_second_derivative(delta_r, k, x0)

        derivative_data.append({
            'DeltaR': delta_r,
            'P_win': p,
            'dP_dDeltaR': dp,
            'd2P_dDeltaR2': d2p
        })

        if delta_r in [-400, -200, -100, 0, 100, 200, 400]:
            print(f"  {delta_r:>8} {p:>10.4f} {dp:>12.6f} {d2p:>14.8f}")

    derivative_df = pd.DataFrame(derivative_data)

    # Find plateau threshold
    epsilon_values = [0.001, 0.0005, 0.0001]

    print(f"\n┌{'─'*60}┐")
    print(f"│{'PLATEAU THRESHOLD ANALYSIS':^60}│")
    print(f"├{'─'*60}┤")
    print(f"│ The plateau is where dP/dΔR becomes negligibly small,        │")
    print(f"│ meaning further rating differences don't significantly       │")
    print(f"│ impact the probability of winning.                           │")
    print(f"├{'─'*60}┤")

    plateau_thresholds = {}
    for eps in epsilon_values:
        threshold = find_plateau_threshold(k, x0, eps)
        plateau_thresholds[eps] = threshold
        prob_at_threshold = logistic_function(x0 + threshold, k, x0)
        print(f"│ ε = {eps:<8} → ΔR* = {threshold:>4} (P_win = {prob_at_threshold:.4f})            │")

    print(f"└{'─'*60}┘")

    # Use middle epsilon as primary threshold
    delta_r_star = plateau_thresholds[0.0005]

    print(f"\n📐 Mathematical Interpretation:")
    print(f"   At ΔR* = ±{delta_r_star} rating points:")
    print(f"   • The derivative dP/dΔR ≈ 0.0005")
    print(f"   • Win probability has essentially plateaued")
    print(f"   • P(ΔR = +{delta_r_star}) = {logistic_function(x0 + delta_r_star, k, x0):.4f}")
    print(f"   • P(ΔR = -{delta_r_star}) = {logistic_function(x0 - delta_r_star, k, x0):.4f}")
    print(f"\n   Beyond this threshold, rating differences have")
    print(f"   diminishing returns on expected outcomes.")

    return derivative_df, delta_r_star


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: INTEGRATION ANALYSIS (HL MATH)
# ═══════════════════════════════════════════════════════════════════════════════

def integration_analysis(k, x0, delta_range=(-400, 400)):
    """
    Compute average winning probability across rating spectrum using integration.

    Average P_win = (1/ΔR_range) × ∫ P(ΔR) dΔR

    This shows the overall impact of rating differences.
    """
    print("\n" + "=" * 70)
    print("STEP 5: INTEGRATION ANALYSIS (HL CALCULUS)")
    print("=" * 70)

    print(f"\nComputing average win probability across rating spectrum:")
    print(f"  Range: ΔR ∈ [{delta_range[0]}, {delta_range[1]}]")
    print(f"\n  Formula: Average P_win = (1/(b-a)) × ∫[a,b] P(ΔR) dΔR")

    # Define function to integrate
    def p_win(x):
        return logistic_function(x, k, x0)

    # Numerical integration
    integral, error = quad(p_win, delta_range[0], delta_range[1])

    # Average probability
    range_width = delta_range[1] - delta_range[0]
    average_p = integral / range_width

    print(f"\n┌{'─'*60}┐")
    print(f"│{'INTEGRATION RESULTS':^60}│")
    print(f"├{'─'*60}┤")
    print(f"│ ∫ P(ΔR) dΔR from {delta_range[0]} to {delta_range[1]}                           │")
    print(f"│                                                            │")
    print(f"│ Integral value = {integral:>10.4f}                               │")
    print(f"│ Range width    = {range_width:>10.0f}                               │")
    print(f"│ Average P_win  = {average_p:>10.4f}                               │")
    print(f"└{'─'*60}┘")

    # Calculate area by segments
    print(f"\n  Breakdown by ΔR segments:")
    print(f"  {'Segment':>20} {'∫P dΔR':>12} {'Avg P':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10}")

    segments = [(-400, -200), (-200, 0), (0, 200), (200, 400)]
    segment_data = []

    for a, b in segments:
        seg_integral, _ = quad(p_win, a, b)
        seg_avg = seg_integral / (b - a)
        segment_data.append({
            'Segment': f"[{a}, {b}]",
            'Integral': seg_integral,
            'Average_P': seg_avg
        })
        print(f"  {f'[{a}, {b}]':>20} {seg_integral:>12.4f} {seg_avg:>10.4f}")

    segment_df = pd.DataFrame(segment_data)

    print(f"\n📐 Interpretation:")
    print(f"   The integral represents the 'total accumulated probability'")
    print(f"   across the rating difference spectrum.")
    print(f"   Average P_win = {average_p:.4f} ≈ {average_p*100:.1f}% across all ΔR values")

    # Analytical antiderivative (for reference)
    print(f"\n   Analytical Antiderivative:")
    print(f"   ∫ 1/(1+e^(-k(x-x₀))) dx = x + (1/k)ln(1+e^(-k(x-x₀))) + C")

    return integral, average_p, segment_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: CREATE EXCEL SPREADSHEET
# ═══════════════════════════════════════════════════════════════════════════════

def create_excel_workbook(df, binned_df, k, x0, r_squared, derivative_df,
                          delta_r_star, integral, average_p, segment_df):
    """
    Create comprehensive Excel workbook with all analysis.
    """
    print("\n" + "=" * 70)
    print("STEP 6: CREATING EXCEL SPREADSHEET")
    print("=" * 70)

    output_path = Path("results/elo_analysis.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formats
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#4472C4', 'font_color': 'white',
            'border': 1, 'align': 'center'
        })
        number_format = workbook.add_format({'num_format': '0.0000', 'border': 1})
        int_format = workbook.add_format({'num_format': '0', 'border': 1})
        percent_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
        title_format = workbook.add_format({
            'bold': True, 'font_size': 14, 'bg_color': '#2E75B6', 'font_color': 'white'
        })
        formula_format = workbook.add_format({
            'italic': True, 'font_color': '#7030A0', 'font_size': 11
        })

        # ═══════════════════════════════════════════════════════════════════
        # SHEET 1: RAW DATA
        # ═══════════════════════════════════════════════════════════════════
        raw_sample = df.head(500)  # First 500 games
        raw_sample.to_excel(writer, sheet_name='1_Raw_Data', index=False, startrow=2)

        ws1 = writer.sheets['1_Raw_Data']
        ws1.write('A1', 'RAW GAME DATA (Sample of first 500 games)', title_format)
        ws1.write('A2', f'Total games in dataset: {len(df)}', formula_format)

        # ═══════════════════════════════════════════════════════════════════
        # SHEET 2: BINNED PROBABILITIES
        # ═══════════════════════════════════════════════════════════════════
        binned_df.to_excel(writer, sheet_name='2_Empirical_Probabilities', index=False, startrow=4)

        ws2 = writer.sheets['2_Empirical_Probabilities']
        ws2.write('A1', 'EMPIRICAL PROBABILITY CALCULATION', title_format)
        ws2.write('A2', 'Formula: P_win(ΔR) = (Wins + 0.5 × Draws) / Total Games', formula_format)
        ws2.write('A3', f'Bin size: 50 rating points | Minimum games per bin: 10', formula_format)

        # ═══════════════════════════════════════════════════════════════════
        # SHEET 3: LOGISTIC FIT
        # ═══════════════════════════════════════════════════════════════════
        ws3 = workbook.add_worksheet('3_Logistic_Model')

        ws3.write('A1', 'LOGISTIC FUNCTION MODEL', title_format)
        ws3.merge_range('A1:F1', 'LOGISTIC FUNCTION MODEL', title_format)

        ws3.write('A3', 'Model Formula:', header_format)
        ws3.merge_range('B3:F3', 'P(ΔR) = 1 / (1 + e^(-k(ΔR - x₀)))', formula_format)

        ws3.write('A5', 'FITTED PARAMETERS', header_format)
        ws3.merge_range('A5:B5', 'FITTED PARAMETERS', header_format)

        ws3.write('A6', 'k (slope)')
        ws3.write('B6', k, number_format)
        ws3.write('C6', '← Determines steepness of curve')

        ws3.write('A7', 'x₀ (midpoint)')
        ws3.write('B7', x0, number_format)
        ws3.write('C7', '← ΔR where P = 0.5')

        ws3.write('A8', 'R² (goodness of fit)')
        ws3.write('B8', r_squared, number_format)
        ws3.write('C8', '← Higher is better (max 1.0)')

        # Predictions table
        ws3.write('A11', 'PREDICTIONS FROM MODEL', header_format)
        ws3.merge_range('A11:D11', 'PREDICTIONS FROM MODEL', header_format)

        ws3.write('A12', 'ΔR', header_format)
        ws3.write('B12', 'P(win) Predicted', header_format)
        ws3.write('C12', 'P(win) Empirical', header_format)
        ws3.write('D12', 'Residual', header_format)

        for i, (_, row) in enumerate(binned_df.iterrows()):
            ws3.write(12 + i + 1, 0, row['DeltaR'], int_format)
            ws3.write(12 + i + 1, 1, logistic_function(row['DeltaR'], k, x0), number_format)
            ws3.write(12 + i + 1, 2, row['P_win_empirical'], number_format)
            ws3.write(12 + i + 1, 3, row['P_win_empirical'] - logistic_function(row['DeltaR'], k, x0), number_format)

        ws3.set_column('A:A', 15)
        ws3.set_column('B:D', 18)

        # ═══════════════════════════════════════════════════════════════════
        # SHEET 4: DERIVATIVE ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        ws4 = workbook.add_worksheet('4_Derivative_Analysis')

        ws4.write('A1', 'DERIVATIVE ANALYSIS (CALCULUS)', title_format)
        ws4.merge_range('A1:E1', 'DERIVATIVE ANALYSIS (CALCULUS)', title_format)

        ws4.write('A3', 'First Derivative:', header_format)
        ws4.merge_range('B3:E3', 'dP/dΔR = k × e^(-k(ΔR - x₀)) / (1 + e^(-k(ΔR - x₀)))²', formula_format)

        ws4.write('A4', 'Interpretation:', header_format)
        ws4.merge_range('B4:E4', 'Sensitivity of win probability to rating changes', formula_format)

        ws4.write('A6', 'PLATEAU THRESHOLD', header_format)
        ws4.merge_range('A6:B6', 'PLATEAU THRESHOLD', header_format)

        ws4.write('A7', 'ΔR* (ε=0.0005)')
        ws4.write('B7', delta_r_star, int_format)
        ws4.write('C7', f'Rating difference beyond which outcomes plateau')

        ws4.write('A8', 'P(win) at +ΔR*')
        ws4.write('B8', logistic_function(x0 + delta_r_star, k, x0), number_format)

        ws4.write('A9', 'P(win) at -ΔR*')
        ws4.write('B9', logistic_function(x0 - delta_r_star, k, x0), number_format)

        # Derivative table
        derivative_df.to_excel(writer, sheet_name='4_Derivative_Analysis', index=False, startrow=11)

        ws4.set_column('A:A', 15)
        ws4.set_column('B:D', 18)

        # ═══════════════════════════════════════════════════════════════════
        # SHEET 5: INTEGRATION ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        ws5 = workbook.add_worksheet('5_Integration_Analysis')

        ws5.write('A1', 'INTEGRATION ANALYSIS (HL CALCULUS)', title_format)
        ws5.merge_range('A1:E1', 'INTEGRATION ANALYSIS (HL CALCULUS)', title_format)

        ws5.write('A3', 'Formula:', header_format)
        ws5.merge_range('B3:E3', 'Average P_win = (1/(b-a)) × ∫[a,b] P(ΔR) dΔR', formula_format)

        ws5.write('A5', 'INTEGRATION RESULTS', header_format)
        ws5.merge_range('A5:B5', 'INTEGRATION RESULTS', header_format)

        ws5.write('A6', 'Integration Range')
        ws5.write('B6', '[-400, 400]')

        ws5.write('A7', '∫ P(ΔR) dΔR')
        ws5.write('B7', integral, number_format)

        ws5.write('A8', 'Range Width')
        ws5.write('B8', 800, int_format)

        ws5.write('A9', 'Average P_win')
        ws5.write('B9', average_p, number_format)

        ws5.write('A11', 'Segment Breakdown:', header_format)
        segment_df.to_excel(writer, sheet_name='5_Integration_Analysis', index=False, startrow=12)

        ws5.set_column('A:A', 20)
        ws5.set_column('B:C', 15)

        # ═══════════════════════════════════════════════════════════════════
        # SHEET 6: SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        ws6 = workbook.add_worksheet('6_Summary')

        ws6.merge_range('A1:F1', 'IB MATHEMATICS HL IA: ELO RATING ANALYSIS SUMMARY', title_format)

        ws6.write('A3', 'Research Question:', header_format)
        ws6.merge_range('B3:F3', '"How does the Elo rating difference between two chess players affect '
                       'the probability of winning, and at what rating difference does the advantage '
                       'plateau?"', formula_format)

        ws6.write('A5', 'DATA SUMMARY', header_format)
        ws6.write('A6', 'Total Games Analyzed')
        ws6.write('B6', len(df), int_format)

        ws6.write('A7', 'White Win Rate')
        ws6.write('B7', (df['Result'] == '1-0').mean(), percent_format)

        ws6.write('A8', 'Black Win Rate')
        ws6.write('B8', (df['Result'] == '0-1').mean(), percent_format)

        ws6.write('A9', 'Draw Rate')
        ws6.write('B9', (df['Result'] == '1/2-1/2').mean(), percent_format)

        ws6.write('A11', 'KEY FINDINGS', header_format)

        ws6.write('A12', '1. Logistic Model')
        ws6.write('B12', f'P(ΔR) = 1 / (1 + e^({-k:.6f}(ΔR - {x0:.2f})))')

        ws6.write('A13', '2. Model Fit (R²)')
        ws6.write('B13', r_squared, number_format)

        ws6.write('A14', '3. Plateau Threshold')
        ws6.write('B14', f'±{delta_r_star} rating points')

        ws6.write('A15', '4. At Equal Ratings')
        ws6.write('B15', f'P(win) = {logistic_function(0, k, x0):.4f}')

        ws6.write('A16', '5. Average P_win')
        ws6.write('B16', average_p, number_format)

        ws6.write('A18', 'MATHEMATICAL CONCEPTS USED', header_format)
        ws6.write('A19', '• Probability (empirical vs theoretical)')
        ws6.write('A20', '• Curve fitting (logistic regression)')
        ws6.write('A21', '• Calculus: First and second derivatives')
        ws6.write('A22', '• Calculus: Definite integration')
        ws6.write('A23', '• Sensitivity analysis and thresholds')

        ws6.set_column('A:A', 25)
        ws6.set_column('B:F', 15)

    print(f"\n  Excel workbook saved to: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: CREATE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_visualizations(binned_df, k, x0, derivative_df, delta_r_star, r_squared):
    """Create publication-quality visualizations."""
    print("\n" + "=" * 70)
    print("STEP 7: CREATING VISUALIZATIONS")
    print("=" * 70)

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # ─────────────────────────────────────────────────────────────────────
    # Figure 1: Logistic Fit with Empirical Data
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot empirical data
    scatter = ax.scatter(binned_df['DeltaR'], binned_df['P_win_empirical'],
                        s=binned_df['N_Games']/2, alpha=0.6, c='#3498db',
                        label='Empirical Data (size ∝ sample size)', edgecolors='navy')

    # Plot fitted logistic curve
    x_smooth = np.linspace(-500, 500, 1000)
    y_smooth = logistic_function(x_smooth, k, x0)
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2.5,
            label=f'Logistic Fit: P = 1/(1+e^({-k:.5f}(ΔR-{x0:.1f})))')

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='P = 0.5')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='ΔR = 0')

    ax.set_xlabel('Rating Difference (ΔR = White Elo - Black Elo)', fontsize=12)
    ax.set_ylabel('Probability White Wins', fontsize=12)
    ax.set_title('Elo Rating Difference vs Win Probability\nLogistic Model Fit', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-500, 500)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'elo_logistic_fit.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: visualizations/elo_logistic_fit.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 2: Derivative Analysis
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # First derivative
    ax1 = axes[0]
    x_range = np.linspace(-500, 500, 1000)
    y_deriv = logistic_derivative(x_range, k, x0)

    ax1.plot(x_range, y_deriv, 'g-', linewidth=2, label='dP/dΔR')
    ax1.axhline(y=0.0005, color='red', linestyle='--', alpha=0.7, label='Threshold ε = 0.0005')
    ax1.axvline(x=delta_r_star, color='purple', linestyle=':', alpha=0.7,
                label=f'Plateau: ΔR* = {delta_r_star}')
    ax1.axvline(x=-delta_r_star, color='purple', linestyle=':', alpha=0.7)

    ax1.fill_between(x_range, y_deriv, where=(np.abs(x_range) > delta_r_star),
                     alpha=0.3, color='yellow', label='Plateau region')

    ax1.set_xlabel('Rating Difference (ΔR)', fontsize=11)
    ax1.set_ylabel('dP/dΔR', fontsize=11)
    ax1.set_title('First Derivative: Sensitivity of Win Probability to Rating Difference', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Second derivative
    ax2 = axes[1]
    y_deriv2 = logistic_second_derivative(x_range, k, x0)

    ax2.plot(x_range, y_deriv2, 'b-', linewidth=2, label='d²P/dΔR²')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(x=x0, color='red', linestyle='--', alpha=0.7, label=f'Inflection point: ΔR = {x0:.1f}')

    ax2.set_xlabel('Rating Difference (ΔR)', fontsize=11)
    ax2.set_ylabel('d²P/dΔR²', fontsize=11)
    ax2.set_title('Second Derivative: Rate of Change of Sensitivity', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'elo_derivative_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: visualizations/elo_derivative_analysis.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 3: Integration Visualization
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    x_range = np.linspace(-400, 400, 1000)
    y_prob = logistic_function(x_range, k, x0)

    ax.plot(x_range, y_prob, 'b-', linewidth=2, label='P(ΔR)')
    ax.fill_between(x_range, y_prob, alpha=0.3, color='blue',
                    label='∫P(ΔR)dΔR (area under curve)')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Rating Difference (ΔR)', fontsize=12)
    ax.set_ylabel('P(win)', fontsize=12)
    ax.set_title('Integration: Area Under the Win Probability Curve\n'
                 '∫P(ΔR)dΔR represents total accumulated probability', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'elo_integration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: visualizations/elo_integration.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 4: Summary Dashboard
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: Logistic fit
    ax1 = axes[0, 0]
    ax1.scatter(binned_df['DeltaR'], binned_df['P_win_empirical'],
               s=30, alpha=0.6, c='#3498db', edgecolors='navy')
    x_smooth = np.linspace(-500, 500, 500)
    ax1.plot(x_smooth, logistic_function(x_smooth, k, x0), 'r-', linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('ΔR')
    ax1.set_ylabel('P(win)')
    ax1.set_title(f'Logistic Model Fit (R² = {r_squared:.4f})')
    ax1.set_xlim(-500, 500)
    ax1.grid(True, alpha=0.3)

    # Top right: Derivative
    ax2 = axes[0, 1]
    ax2.plot(x_smooth, logistic_derivative(x_smooth, k, x0), 'g-', linewidth=2)
    ax2.axhline(y=0.0005, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=delta_r_star, color='purple', linestyle=':', alpha=0.7)
    ax2.axvline(x=-delta_r_star, color='purple', linestyle=':', alpha=0.7)
    ax2.set_xlabel('ΔR')
    ax2.set_ylabel('dP/dΔR')
    ax2.set_title(f'First Derivative (Plateau at ΔR* = ±{delta_r_star})')
    ax2.grid(True, alpha=0.3)

    # Bottom left: Key values table
    ax3 = axes[1, 0]
    ax3.axis('off')

    table_data = [
        ['Parameter', 'Value'],
        ['k (slope)', f'{k:.6f}'],
        ['x₀ (midpoint)', f'{x0:.2f}'],
        ['R²', f'{r_squared:.4f}'],
        ['Plateau ΔR*', f'±{delta_r_star}'],
        ['P(ΔR=0)', f'{logistic_function(0, k, x0):.4f}'],
        ['P(ΔR=100)', f'{logistic_function(100, k, x0):.4f}'],
        ['P(ΔR=200)', f'{logistic_function(200, k, x0):.4f}'],
        ['P(ΔR=400)', f'{logistic_function(400, k, x0):.4f}'],
    ]

    table = ax3.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Header formatting
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax3.set_title('Key Results', fontsize=12, pad=20)

    # Bottom right: Probability at key points
    ax4 = axes[1, 1]
    delta_r_points = [-400, -200, -100, 0, 100, 200, 400]
    probs = [logistic_function(dr, k, x0) for dr in delta_r_points]
    colors = ['#e74c3c' if p < 0.5 else '#27ae60' for p in probs]

    bars = ax4.bar([str(dr) for dr in delta_r_points], probs, color=colors, edgecolor='black')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Rating Difference (ΔR)')
    ax4.set_ylabel('P(White wins)')
    ax4.set_title('Win Probability at Key Rating Differences')

    for bar, prob in zip(bars, probs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', fontsize=9)

    ax4.set_ylim(0, 1.1)

    plt.suptitle('IB Math HL IA: Elo Rating Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_dir / 'elo_summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: visualizations/elo_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the complete Elo rating analysis."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          IB MATHEMATICS HL IA: ELO RATING ANALYSIS                           ║
║          ═════════════════════════════════════════════                       ║
║                                                                              ║
║  Research Question:                                                          ║
║  "How does the Elo rating difference between two chess players affect       ║
║   the probability of winning, and at what rating difference does the        ║
║   advantage plateau such that further differences no longer significantly   ║
║   impact outcomes?"                                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check for data file
    data_file = Path("data/grandmaster_games.pgn")
    if not data_file.exists():
        print("ERROR: Data file not found at data/grandmaster_games.pgn")
        print("Please run main.py first to download the data.")
        return

    # STEP 1: Extract Elo data
    df = extract_elo_data(str(data_file))
    if df is None or len(df) == 0:
        return

    # STEP 2: Calculate empirical probabilities
    binned_df = calculate_empirical_probabilities(df, bin_size=50)

    # STEP 3: Fit logistic model
    k, x0, r_squared, residuals, y_pred = fit_logistic_model(binned_df)
    if k is None:
        return

    # STEP 4: Derivative analysis
    derivative_df, delta_r_star = derivative_analysis(k, x0, binned_df)

    # STEP 5: Integration analysis
    integral, average_p, segment_df = integration_analysis(k, x0)

    # STEP 6: Create Excel workbook
    excel_path = create_excel_workbook(
        df, binned_df, k, x0, r_squared, derivative_df,
        delta_r_star, integral, average_p, segment_df
    )

    # STEP 7: Create visualizations
    create_visualizations(binned_df, k, x0, derivative_df, delta_r_star, r_squared)

    # Final Summary
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ANALYSIS COMPLETE                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DATA SUMMARY:                                                               ║
║    • Total games analyzed: {len(df):>7,}                                       ║
║    • Elo range: {df['WhiteElo'].min()}-{df['WhiteElo'].max()}                                           ║
║                                                                              ║
║  LOGISTIC MODEL:                                                             ║
║    • P(ΔR) = 1 / (1 + e^({-k:.6f}(ΔR - {x0:.1f})))                        ║
║    • R² = {r_squared:.4f} (goodness of fit)                                    ║
║                                                                              ║
║  KEY FINDINGS:                                                               ║
║    • At equal ratings (ΔR=0): P(win) = {logistic_function(0, k, x0):.4f}                     ║
║    • At ΔR=+100: P(win) = {logistic_function(100, k, x0):.4f}                                ║
║    • At ΔR=+200: P(win) = {logistic_function(200, k, x0):.4f}                                ║
║    • Plateau threshold: ΔR* = ±{delta_r_star} rating points                    ║
║                                                                              ║
║  CALCULUS RESULTS:                                                           ║
║    • Derivative analysis shows sensitivity drops below ε at ΔR*             ║
║    • Integration gives average P_win = {average_p:.4f} across spectrum          ║
║                                                                              ║
║  OUTPUT FILES:                                                               ║
║    • results/elo_analysis.xlsx - Complete Excel workbook                     ║
║    • visualizations/elo_logistic_fit.png                                     ║
║    • visualizations/elo_derivative_analysis.png                              ║
║    • visualizations/elo_integration.png                                      ║
║    • visualizations/elo_summary_dashboard.png                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
