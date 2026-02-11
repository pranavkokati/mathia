#!/usr/bin/env python3
"""
IB Mathematics HL Internal Assessment: Elo Rating Analysis

Research Question:
"How does the Elo rating difference between two chess players affect the
probability of winning, and at what rating difference does the advantage
plateau such that further differences no longer significantly impact outcomes?"

This script performs a complete mathematical analysis following these steps:
1. Data Preparation - Parse PGN data, compute rating differences, bin data
2. Mathematical Modeling - Fit logistic function using curve fitting
3. Derivative Analysis - Find plateau threshold using calculus
4. Integration Analysis - Compute average win probability
5. Excel Output - Export all analysis to spreadsheet
6. Visualizations - Create publication-quality figures
7. Summary Report - Print comprehensive results

Author: IB Mathematics HL IA Project
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re
import urllib.request
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"
RESULTS_DIR = "results"
VIZ_DIR = "visualizations"
PGN_FILE = os.path.join(DATA_DIR, "grandmaster_games.pgn")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# ============================================================================
# STEP 0: DATA ACQUISITION
# ============================================================================

def download_lichess_games():
    """
    Download real grandmaster games from Lichess API.
    Uses the Lichess games export API to get rated classical games.
    """
    print("=" * 60)
    print("STEP 0: DATA ACQUISITION")
    print("=" * 60)

    if os.path.exists(PGN_FILE):
        with open(PGN_FILE, 'r') as f:
            content = f.read()
            if len(content) > 1000:
                print(f"Using existing data file: {PGN_FILE}")
                return True

    print("Downloading real grandmaster games from Lichess...")

    # List of strong players to download games from
    players = ["DrNykterstein", "Hikaru", "GMWSO", "nihalsarin", "Firouzja2003",
               "LyonBeast", "FairChess_on_YouTube", "PinIern", "Jospem", "Polish_fighter3000"]

    all_games = []

    for player in players:
        try:
            url = f"https://lichess.org/api/games/user/{player}?max=200&rated=true&perfType=classical,rapid&clocks=false&evals=false&opening=false"

            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/x-chess-pgn')

            print(f"  Fetching games for {player}...")

            with urllib.request.urlopen(req, timeout=30) as response:
                games = response.read().decode('utf-8')
                all_games.append(games)
                time.sleep(1.5)  # Rate limiting

        except Exception as e:
            print(f"  Could not fetch games for {player}: {e}")
            continue

    if all_games:
        with open(PGN_FILE, 'w') as f:
            f.write('\n\n'.join(all_games))
        print(f"Downloaded and saved games to {PGN_FILE}")
        return True
    else:
        print("Could not download games. Generating synthetic data based on Elo theory...")
        return False


def generate_theoretical_data():
    """
    Generate data based on the theoretical Elo formula.
    P(win) = 1 / (1 + 10^(-delta/400))

    This is used as fallback if real data cannot be downloaded.
    """
    print("Generating data based on Elo rating theory...")

    np.random.seed(42)

    games = []
    n_games = 2000

    for i in range(n_games):
        # Generate realistic Elo ratings (2000-2800 range for strong players)
        white_elo = np.random.normal(2400, 200)
        white_elo = np.clip(white_elo, 2000, 2850)

        black_elo = np.random.normal(2400, 200)
        black_elo = np.clip(black_elo, 2000, 2850)

        delta_r = white_elo - black_elo

        # Theoretical win probability
        expected = 1 / (1 + 10 ** (-delta_r / 400))

        # Add some noise to make it realistic
        # Also account for draw probability (higher at top level)
        draw_prob = 0.3 + 0.1 * np.exp(-abs(delta_r) / 200)

        rand = np.random.random()
        if rand < draw_prob:
            result = "1/2-1/2"
        elif rand < draw_prob + expected * (1 - draw_prob):
            result = "1-0"
        else:
            result = "0-1"

        games.append({
            'white_elo': int(white_elo),
            'black_elo': int(black_elo),
            'result': result
        })

    # Create PGN-like format
    pgn_content = []
    for g in games:
        pgn_content.append(f'[WhiteElo "{g["white_elo"]}"]\n[BlackElo "{g["black_elo"]}"]\n[Result "{g["result"]}"]\n\n1. e4 e5 {g["result"]}\n')

    with open(PGN_FILE, 'w') as f:
        f.write('\n'.join(pgn_content))

    print(f"Generated {n_games} games based on Elo theory")
    return True


# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================

def parse_pgn_file(filepath):
    """
    Parse PGN file to extract game data.

    Extracts:
    - White Elo rating
    - Black Elo rating
    - Game result (1-0, 0-1, 1/2-1/2)

    Returns DataFrame with columns: white_elo, black_elo, result
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    print(f"\nParsing PGN file: {filepath}")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Regular expressions for extracting data
    white_elo_pattern = r'\[WhiteElo\s+"(\d+)"\]'
    black_elo_pattern = r'\[BlackElo\s+"(\d+)"\]'
    result_pattern = r'\[Result\s+"([^"]+)"\]'

    # Split into individual games
    games = re.split(r'\n\n(?=\[Event|\[White)', content)

    data = []
    for game in games:
        white_match = re.search(white_elo_pattern, game)
        black_match = re.search(black_elo_pattern, game)
        result_match = re.search(result_pattern, game)

        if white_match and black_match and result_match:
            white_elo = int(white_match.group(1))
            black_elo = int(black_match.group(1))
            result = result_match.group(1)

            # Filter valid ratings and results
            if 1000 <= white_elo <= 3500 and 1000 <= black_elo <= 3500:
                if result in ['1-0', '0-1', '1/2-1/2']:
                    data.append({
                        'white_elo': white_elo,
                        'black_elo': black_elo,
                        'result': result
                    })

    df = pd.DataFrame(data)
    print(f"Successfully parsed {len(df)} games")

    return df


def prepare_data(df):
    """
    Prepare data for analysis.

    Computes:
    - Rating difference (ΔR = White Elo - Black Elo)
    - Numerical result (1 for White win, 0.5 for draw, 0 for Black win)
    - Bins data into intervals
    - Calculates empirical win probability per bin

    Formula for empirical probability:
    P_win = (Number of White Wins + 0.5 × Number of Draws) / Total Games
    """
    print("\nComputing rating differences and results...")

    # Compute rating difference
    df['delta_r'] = df['white_elo'] - df['black_elo']

    # Convert result to numerical score for White
    result_map = {'1-0': 1.0, '1/2-1/2': 0.5, '0-1': 0.0}
    df['score'] = df['result'].map(result_map)

    print(f"\nRating difference range: [{df['delta_r'].min()}, {df['delta_r'].max()}]")
    print(f"Mean rating difference: {df['delta_r'].mean():.1f}")
    print(f"Results distribution:")
    print(f"  White wins: {(df['result'] == '1-0').sum()} ({100*(df['result'] == '1-0').mean():.1f}%)")
    print(f"  Draws:      {(df['result'] == '1/2-1/2').sum()} ({100*(df['result'] == '1/2-1/2').mean():.1f}%)")
    print(f"  Black wins: {(df['result'] == '0-1').sum()} ({100*(df['result'] == '0-1').mean():.1f}%)")

    # Bin data
    print("\nBinning data into 50-point intervals...")
    bin_width = 50
    df['bin'] = (df['delta_r'] / bin_width).round() * bin_width

    # Calculate empirical probabilities per bin
    binned = df.groupby('bin').agg({
        'score': ['mean', 'count', 'std'],
        'delta_r': 'mean'
    }).reset_index()

    binned.columns = ['bin', 'p_win', 'n_games', 'std', 'mean_delta_r']

    # Filter bins with enough games for statistical significance
    binned = binned[binned['n_games'] >= 5]

    # Calculate standard error
    binned['se'] = binned['std'] / np.sqrt(binned['n_games'])
    binned['se'] = binned['se'].fillna(0.1)

    print(f"Created {len(binned)} bins with sufficient data")

    return df, binned


# ============================================================================
# STEP 2: MATHEMATICAL MODELING - LOGISTIC FUNCTION FIT
# ============================================================================

def logistic(x, k, x0):
    """
    Logistic function for modeling win probability.

    P(ΔR) = 1 / (1 + e^(-k(ΔR - x₀)))

    Parameters:
    - k: Slope parameter (steepness of curve)
    - x₀: Midpoint (rating difference where P = 0.5)

    The standard Elo formula uses k = ln(10)/400 ≈ 0.00576
    """
    return 1 / (1 + np.exp(-k * (x - x0)))


def fit_logistic_model(binned):
    """
    Fit logistic model to empirical data using nonlinear least squares.

    Uses scipy.optimize.curve_fit with weighted regression.
    Weights are proportional to sample size in each bin.

    Returns fitted parameters (k, x₀) and R² value.
    """
    print("\n" + "=" * 60)
    print("STEP 2: MATHEMATICAL MODELING")
    print("=" * 60)

    print("\nFitting logistic function: P(ΔR) = 1 / (1 + e^(-k(ΔR - x₀)))")

    x_data = binned['bin'].values
    y_data = binned['p_win'].values
    weights = np.sqrt(binned['n_games'].values)  # Weight by sqrt(n)

    # Initial guess based on Elo theory: k ≈ ln(10)/400
    p0 = [0.005, 0]

    # Fit the model
    popt, pcov = curve_fit(
        logistic, x_data, y_data,
        p0=p0,
        sigma=1/weights,
        absolute_sigma=False,
        maxfev=5000
    )

    k, x0 = popt

    # Calculate R² (coefficient of determination)
    y_pred = logistic(x_data, k, x0)
    ss_res = np.sum(weights * (y_data - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_data - np.average(y_data, weights=weights)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Standard errors from covariance matrix
    perr = np.sqrt(np.diag(pcov))

    print(f"\nFitted Parameters:")
    print(f"  k  = {k:.6f} ± {perr[0]:.6f}")
    print(f"  x₀ = {x0:.2f} ± {perr[1]:.2f}")
    print(f"\nGoodness of Fit:")
    print(f"  R² = {r_squared:.4f}")

    # Compare to theoretical Elo
    k_theoretical = np.log(10) / 400
    print(f"\nComparison to Elo Theory:")
    print(f"  Theoretical k = ln(10)/400 = {k_theoretical:.6f}")
    print(f"  Fitted k = {k:.6f}")
    print(f"  Ratio: {k/k_theoretical:.3f}")

    return k, x0, r_squared


# ============================================================================
# STEP 3: DERIVATIVE ANALYSIS
# ============================================================================

def dP_dDeltaR(x, k, x0):
    """
    First derivative of logistic function.

    dP/dΔR = k × e^(-k(ΔR - x₀)) / (1 + e^(-k(ΔR - x₀)))²

    This represents the sensitivity of win probability to rating difference.
    Maximum sensitivity occurs at ΔR = x₀.
    """
    exp_term = np.exp(-k * (x - x0))
    return k * exp_term / ((1 + exp_term) ** 2)


def d2P_dDeltaR2(x, k, x0):
    """
    Second derivative of logistic function.

    d²P/dΔR² = -k² × e^(-k(ΔR-x₀)) × (1 - e^(-k(ΔR-x₀))) / (1 + e^(-k(ΔR-x₀)))³

    This represents the rate of change of sensitivity.
    Zero at ΔR = x₀ (inflection point).
    """
    exp_term = np.exp(-k * (x - x0))
    numerator = -k**2 * exp_term * (1 - exp_term)
    denominator = (1 + exp_term) ** 3
    return numerator / denominator


def find_plateau_threshold(k, x0, epsilon=0.0001):
    """
    Find the plateau threshold ΔR* where dP/dΔR < ε.

    Beyond this threshold, further rating differences have
    negligible impact on win probability.

    Uses numerical search to find where derivative drops below epsilon.
    """
    print("\n" + "=" * 60)
    print("STEP 3: DERIVATIVE ANALYSIS")
    print("=" * 60)

    print(f"\nFirst Derivative: dP/dΔR = k × e^(-k(ΔR-x₀)) / (1 + e^(-k(ΔR-x₀)))²")
    print(f"Second Derivative: d²P/dΔR² = -k² × e^(-k(ΔR-x₀)) × (1-e^(-k(ΔR-x₀))) / (1+e^(-k(ΔR-x₀)))³")

    # Maximum derivative at x = x0
    max_deriv = dP_dDeltaR(x0, k, x0)
    print(f"\nMaximum sensitivity at ΔR = {x0:.1f}:")
    print(f"  dP/dΔR|_max = {max_deriv:.6f}")

    # Find plateau threshold
    print(f"\nFinding plateau threshold where dP/dΔR < {epsilon}...")

    delta_r_star = None
    for x in range(0, 1000, 1):
        if dP_dDeltaR(x + x0, k, x0) < epsilon:
            delta_r_star = x
            break

    if delta_r_star is None:
        delta_r_star = 500  # Default if not found

    print(f"\nPlateau Threshold: ΔR* = ±{delta_r_star}")
    print(f"  At ΔR = +{delta_r_star}: P_win = {logistic(delta_r_star + x0, k, x0):.4f}")
    print(f"  At ΔR = -{delta_r_star}: P_win = {logistic(-delta_r_star + x0, k, x0):.4f}")

    # Derivative values at key points
    print("\nDerivative values at key rating differences:")
    for dr in [0, 100, 200, 300, 400, 500]:
        dp = dP_dDeltaR(dr, k, x0)
        d2p = d2P_dDeltaR2(dr, k, x0)
        print(f"  ΔR = {dr:4d}: dP/dΔR = {dp:.6f}, d²P/dΔR² = {d2p:.8f}")

    return delta_r_star


# ============================================================================
# STEP 4: INTEGRATION ANALYSIS
# ============================================================================

def integration_analysis(k, x0):
    """
    Perform integration analysis.

    Computes: ∫P_win(ΔR) dΔR over the range [-400, 400]

    Average win probability: P̄ = (1/(b-a)) × ∫[a,b] P(ΔR) dΔR

    Uses scipy.integrate.quad for numerical integration.
    """
    print("\n" + "=" * 60)
    print("STEP 4: INTEGRATION ANALYSIS")
    print("=" * 60)

    a, b = -400, 400  # Integration bounds

    print(f"\nComputing: ∫[{a}, {b}] P_win(ΔR) dΔR")

    # Numerical integration
    integral, error = quad(lambda x: logistic(x, k, x0), a, b)

    # Average win probability
    avg_p_win = integral / (b - a)

    print(f"\nIntegration Results:")
    print(f"  ∫P dΔR = {integral:.4f}")
    print(f"  Integration error: ±{error:.2e}")
    print(f"  Average P_win over [{a}, {b}]: {avg_p_win:.4f}")

    # Segment breakdown
    print("\nSegment breakdown:")
    segments = [(-400, -200), (-200, 0), (0, 200), (200, 400)]
    segment_data = []

    for seg_a, seg_b in segments:
        seg_integral, _ = quad(lambda x: logistic(x, k, x0), seg_a, seg_b)
        seg_avg = seg_integral / (seg_b - seg_a)
        segment_data.append({
            'start': seg_a,
            'end': seg_b,
            'integral': seg_integral,
            'avg_p_win': seg_avg
        })
        print(f"  [{seg_a:4d}, {seg_b:4d}]: ∫P dΔR = {seg_integral:.4f}, avg P = {seg_avg:.4f}")

    return integral, avg_p_win, segment_data


# ============================================================================
# STEP 5: EXCEL OUTPUT
# ============================================================================

def create_excel_output(df, binned, k, x0, r_squared, delta_r_star, integral, avg_p_win, segment_data):
    """
    Create comprehensive Excel workbook with all analysis.

    Sheets:
    1. RawData - Sample of game data
    2. EmpiricalProbabilities - Binned data with P_win
    3. ModelParameters - Fitted parameters and statistics
    4. Predictions - Model predictions at various ΔR
    5. Integration - Integration analysis results
    6. Comparison - Empirical vs predicted values
    """
    print("\n" + "=" * 60)
    print("STEP 5: EXCEL OUTPUT")
    print("=" * 60)

    excel_path = os.path.join(RESULTS_DIR, "elo_analysis.xlsx")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1: Raw Data (sample)
        sample_df = df.sample(min(500, len(df)), random_state=42).copy()
        sample_df = sample_df[['white_elo', 'black_elo', 'delta_r', 'result', 'score']]
        sample_df.columns = ['White_Elo', 'Black_Elo', 'Delta_R', 'Result', 'Score']
        sample_df.to_excel(writer, sheet_name='1_RawData', index=False)
        print(f"  Sheet 1: RawData - {len(sample_df)} sample games")

        # Sheet 2: Empirical Probabilities
        emp_df = binned[['bin', 'p_win', 'n_games', 'se']].copy()
        emp_df.columns = ['Rating_Difference', 'Empirical_P_win', 'N_Games', 'Std_Error']
        emp_df.to_excel(writer, sheet_name='2_EmpiricalProbabilities', index=False)
        print(f"  Sheet 2: EmpiricalProbabilities - {len(emp_df)} bins")

        # Sheet 3: Model Parameters
        k_theoretical = np.log(10) / 400
        params_data = {
            'Parameter': ['k (slope)', 'x₀ (midpoint)', 'R²', 'ΔR* (plateau)',
                         'k_theoretical', 'k_ratio', 'N_games', 'N_bins'],
            'Value': [k, x0, r_squared, delta_r_star,
                     k_theoretical, k/k_theoretical, len(df), len(binned)],
            'Description': [
                'Slope of logistic curve',
                'Rating difference where P = 0.5',
                'Coefficient of determination',
                'Plateau threshold (where dP/dΔR < ε)',
                'Theoretical Elo k = ln(10)/400',
                'Fitted k / Theoretical k',
                'Total games analyzed',
                'Number of data bins'
            ]
        }
        pd.DataFrame(params_data).to_excel(writer, sheet_name='3_ModelParameters', index=False)
        print("  Sheet 3: ModelParameters")

        # Sheet 4: Predictions
        delta_r_values = list(range(-500, 501, 25))
        predictions = []
        for dr in delta_r_values:
            predictions.append({
                'Delta_R': dr,
                'P_win': logistic(dr, k, x0),
                'dP_dDeltaR': dP_dDeltaR(dr, k, x0),
                'd2P_dDeltaR2': d2P_dDeltaR2(dr, k, x0)
            })
        pd.DataFrame(predictions).to_excel(writer, sheet_name='4_Predictions', index=False)
        print(f"  Sheet 4: Predictions - {len(predictions)} points")

        # Sheet 5: Integration
        int_data = {
            'Segment': [f"[{s['start']}, {s['end']}]" for s in segment_data] + ['Total [-400, 400]'],
            'Integral': [s['integral'] for s in segment_data] + [integral],
            'Average_P_win': [s['avg_p_win'] for s in segment_data] + [avg_p_win]
        }
        pd.DataFrame(int_data).to_excel(writer, sheet_name='5_Integration', index=False)
        print("  Sheet 5: Integration")

        # Sheet 6: Comparison (Empirical vs Predicted)
        comparison = binned[['bin', 'p_win', 'n_games']].copy()
        comparison['predicted'] = logistic(comparison['bin'].values, k, x0)
        comparison['residual'] = comparison['p_win'] - comparison['predicted']
        comparison['residual_pct'] = 100 * comparison['residual'] / comparison['p_win']
        comparison.columns = ['Delta_R', 'Empirical_P', 'N_Games', 'Predicted_P', 'Residual', 'Residual_Pct']
        comparison.to_excel(writer, sheet_name='6_Comparison', index=False)
        print("  Sheet 6: Comparison")

    print(f"\nExcel workbook saved: {excel_path}")

    return excel_path


# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

def create_visualizations(binned, k, x0, r_squared, delta_r_star, integral, avg_p_win):
    """
    Create publication-quality visualizations.

    Figure 1: Logistic Fit - Empirical data with fitted curve
    Figure 2: Derivatives - First and second derivative plots
    Figure 3: Integration - Area under curve visualization
    """
    print("\n" + "=" * 60)
    print("STEP 6: VISUALIZATIONS")
    print("=" * 60)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14

    # -------------------------------------------------------------------------
    # Figure 1: Logistic Fit
    # -------------------------------------------------------------------------
    print("\nCreating Figure 1: Logistic Fit...")

    fig1, ax1 = plt.subplots(figsize=(10, 7))

    # Plot empirical data points (size proportional to sample size)
    sizes = binned['n_games'].values / binned['n_games'].max() * 200 + 20
    scatter = ax1.scatter(binned['bin'], binned['p_win'],
                         s=sizes, alpha=0.6, c='steelblue',
                         edgecolors='navy', linewidth=0.5,
                         label='Empirical Data')

    # Plot fitted logistic curve
    x_fit = np.linspace(-500, 500, 1000)
    y_fit = logistic(x_fit, k, x0)
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2.5, label=f'Logistic Fit (R² = {r_squared:.4f})')

    # Add theoretical Elo curve for comparison
    k_theo = np.log(10) / 400
    y_theo = logistic(x_fit, k_theo, 0)
    ax1.plot(x_fit, y_theo, 'g--', linewidth=1.5, alpha=0.7, label='Theoretical Elo')

    # Reference lines
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    # Labels and formatting
    ax1.set_xlabel('Rating Difference (ΔR = White Elo - Black Elo)')
    ax1.set_ylabel('Win Probability for White (P_win)')
    ax1.set_title('Logistic Model: Win Probability vs Rating Difference\nIB Mathematics HL Internal Assessment')
    ax1.set_xlim(-550, 550)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower right')

    # Add equation box
    eq_text = f'$P_{{win}}(\\Delta R) = \\frac{{1}}{{1 + e^{{-k(\\Delta R - x_0)}}}}$\n\n'
    eq_text += f'$k = {k:.6f}$\n$x_0 = {x0:.2f}$\n$R^2 = {r_squared:.4f}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, eq_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    fig1.savefig(os.path.join(VIZ_DIR, 'fig1_logistic_fit.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  Saved: fig1_logistic_fit.png")

    # -------------------------------------------------------------------------
    # Figure 2: Derivatives
    # -------------------------------------------------------------------------
    print("Creating Figure 2: Derivatives...")

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 9))

    x_deriv = np.linspace(-600, 600, 1000)

    # First derivative
    y_deriv1 = dP_dDeltaR(x_deriv, k, x0)
    ax2a.plot(x_deriv, y_deriv1, 'b-', linewidth=2, label='dP/dΔR')
    ax2a.fill_between(x_deriv, 0, y_deriv1, alpha=0.3)

    # Mark plateau threshold
    ax2a.axvline(x=delta_r_star, color='red', linestyle='--', linewidth=1.5,
                label=f'Plateau: ΔR* = ±{delta_r_star}')
    ax2a.axvline(x=-delta_r_star, color='red', linestyle='--', linewidth=1.5)
    ax2a.axhline(y=0.0001, color='orange', linestyle=':', alpha=0.7, label='ε = 0.0001')

    ax2a.set_xlabel('Rating Difference (ΔR)')
    ax2a.set_ylabel('dP/dΔR (Sensitivity)')
    ax2a.set_title('First Derivative: Sensitivity of Win Probability')
    ax2a.legend(loc='upper right')
    ax2a.set_xlim(-600, 600)

    # Second derivative
    y_deriv2 = d2P_dDeltaR2(x_deriv, k, x0)
    ax2b.plot(x_deriv, y_deriv2, 'purple', linewidth=2, label='d²P/dΔR²')
    ax2b.fill_between(x_deriv, 0, y_deriv2, where=(y_deriv2 > 0), alpha=0.3, color='green')
    ax2b.fill_between(x_deriv, 0, y_deriv2, where=(y_deriv2 < 0), alpha=0.3, color='red')

    # Mark inflection point
    ax2b.axvline(x=x0, color='orange', linestyle='--', linewidth=1.5,
                label=f'Inflection point: ΔR = {x0:.1f}')
    ax2b.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    ax2b.set_xlabel('Rating Difference (ΔR)')
    ax2b.set_ylabel('d²P/dΔR² (Curvature)')
    ax2b.set_title('Second Derivative: Rate of Change of Sensitivity')
    ax2b.legend(loc='upper right')
    ax2b.set_xlim(-600, 600)

    plt.tight_layout()
    fig2.savefig(os.path.join(VIZ_DIR, 'fig2_derivatives.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  Saved: fig2_derivatives.png")

    # -------------------------------------------------------------------------
    # Figure 3: Integration
    # -------------------------------------------------------------------------
    print("Creating Figure 3: Integration...")

    fig3, ax3 = plt.subplots(figsize=(10, 7))

    x_int = np.linspace(-400, 400, 1000)
    y_int = logistic(x_int, k, x0)

    # Plot the curve
    ax3.plot(x_int, y_int, 'b-', linewidth=2.5, label='P_win(ΔR)')

    # Fill the area under the curve
    ax3.fill_between(x_int, 0, y_int, alpha=0.3, color='steelblue',
                    label=f'∫P dΔR = {integral:.2f}')

    # Mark average
    ax3.axhline(y=avg_p_win, color='red', linestyle='--', linewidth=2,
               label=f'Average P_win = {avg_p_win:.4f}')

    # Segment markers
    for seg_x in [-200, 0, 200]:
        ax3.axvline(x=seg_x, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Rating Difference (ΔR)')
    ax3.set_ylabel('Win Probability (P_win)')
    ax3.set_title('Integration: Area Under Win Probability Curve\n∫[-400, 400] P_win(ΔR) dΔR')
    ax3.set_xlim(-450, 450)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc='lower right')

    # Add integral formula
    int_text = f'$\\int_{{-400}}^{{400}} P_{{win}}(\\Delta R) \\, d\\Delta R = {integral:.2f}$\n\n'
    int_text += f'$\\bar{{P}}_{{win}} = \\frac{{1}}{{800}} \\times {integral:.2f} = {avg_p_win:.4f}$'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax3.text(0.05, 0.25, int_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    fig3.savefig(os.path.join(VIZ_DIR, 'fig3_integration.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: fig3_integration.png")

    print(f"\nAll visualizations saved to: {VIZ_DIR}/")


# ============================================================================
# STEP 7: SUMMARY REPORT
# ============================================================================

def print_summary(df, k, x0, r_squared, delta_r_star, integral, avg_p_win):
    """
    Print comprehensive summary of all analysis results.
    """
    print("\n" + "=" * 60)
    print("STEP 7: SUMMARY REPORT")
    print("=" * 60)

    print("\n" + "─" * 60)
    print("IB MATHEMATICS HL INTERNAL ASSESSMENT")
    print("Elo Rating Analysis: Complete Results")
    print("─" * 60)

    print("\n▸ RESEARCH QUESTION:")
    print("  How does the Elo rating difference between two chess players")
    print("  affect the probability of winning, and at what rating difference")
    print("  does the advantage plateau?")

    print("\n▸ DATA:")
    print(f"  Games analyzed: {len(df)}")
    print(f"  Rating range: {df['white_elo'].min()}-{df['white_elo'].max()} Elo")
    print(f"  ΔR range: [{df['delta_r'].min()}, {df['delta_r'].max()}]")

    print("\n▸ LOGISTIC MODEL:")
    print(f"  P_win(ΔR) = 1 / (1 + e^(-{k:.6f}(ΔR - ({x0:.2f}))))")
    print(f"  ")
    print(f"  Parameters:")
    print(f"    k  = {k:.6f} (slope)")
    print(f"    x₀ = {x0:.2f} (midpoint)")
    print(f"    R² = {r_squared:.4f} (goodness of fit)")

    print("\n▸ KEY PREDICTIONS:")
    print("  ΔR        P_win      Interpretation")
    print("  " + "─" * 45)
    for dr in [-400, -200, -100, 0, 100, 200, 400]:
        p = logistic(dr, k, x0)
        if dr < 0:
            interp = "Black favored"
        elif dr > 0:
            interp = "White favored"
        else:
            interp = "Equal rating"
        print(f"  {dr:+4d}       {p:.4f}     {interp}")

    print("\n▸ DERIVATIVE ANALYSIS:")
    print(f"  Maximum sensitivity at ΔR = {x0:.1f}")
    print(f"  dP/dΔR|_max = {dP_dDeltaR(x0, k, x0):.6f}")
    print(f"  ")
    print(f"  Plateau threshold: ΔR* = ±{delta_r_star}")
    print(f"  Beyond this, further rating differences have")
    print(f"  negligible impact on win probability.")

    print("\n▸ INTEGRATION ANALYSIS:")
    print(f"  ∫[-400,400] P_win dΔR = {integral:.4f}")
    print(f"  Average P_win = {avg_p_win:.4f}")

    print("\n▸ ANSWER TO RESEARCH QUESTION:")
    print(f"  The win probability follows a logistic curve with")
    print(f"  R² = {r_squared:.4f}. The advantage plateaus at")
    print(f"  approximately ±{delta_r_star} rating points, where")
    print(f"  further differences no longer significantly impact")
    print(f"  the outcome.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    Runs all 7 steps of the analysis.
    """
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║  IB MATHEMATICS HL INTERNAL ASSESSMENT                   ║")
    print("║  Elo Rating Analysis                                     ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Step 0: Get data
    if not download_lichess_games():
        generate_theoretical_data()

    # Step 1: Data Preparation
    df = parse_pgn_file(PGN_FILE)
    df, binned = prepare_data(df)

    # Step 2: Mathematical Modeling
    k, x0, r_squared = fit_logistic_model(binned)

    # Step 3: Derivative Analysis
    delta_r_star = find_plateau_threshold(k, x0)

    # Step 4: Integration Analysis
    integral, avg_p_win, segment_data = integration_analysis(k, x0)

    # Step 5: Excel Output
    create_excel_output(df, binned, k, x0, r_squared, delta_r_star,
                       integral, avg_p_win, segment_data)

    # Step 6: Visualizations
    create_visualizations(binned, k, x0, r_squared, delta_r_star, integral, avg_p_win)

    # Step 7: Summary Report
    print_summary(df, k, x0, r_squared, delta_r_star, integral, avg_p_win)

    return {
        'k': k,
        'x0': x0,
        'r_squared': r_squared,
        'delta_r_star': delta_r_star,
        'integral': integral,
        'avg_p_win': avg_p_win,
        'n_games': len(df)
    }


if __name__ == "__main__":
    results = main()
