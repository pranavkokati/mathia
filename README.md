# IB Mathematics HL Internal Assessment: Elo Rating Analysis

## Research Question

> "How does the Elo rating difference between two chess players affect the probability of winning, and at what rating difference does the advantage plateau such that further differences no longer significantly impact outcomes?"

## Project Overview

This project performs a rigorous mathematical analysis of the Elo rating system using **real chess game data** from Lichess grandmaster games. The analysis follows the IB Mathematics HL IA framework with emphasis on:

- **Logistic Function Fitting** - Modeling win probability as a function of rating difference
- **Derivative Analysis** - Finding the plateau threshold where rating advantages diminish
- **Integration Analysis** - Computing average win probability across rating spectrum

## Mathematical Framework

### The Logistic Model

The probability of the higher-rated player (White) winning is modeled as:

$$P_{win}(\Delta R) = \frac{1}{1 + e^{-k(\Delta R - x_0)}}$$

Where:
- $\Delta R$ = Rating difference (White Elo - Black Elo)
- $k$ = Slope parameter (sensitivity)
- $x_0$ = Midpoint (rating difference where P = 0.5)

### Derivative Analysis

**First Derivative** (sensitivity of probability to rating difference):

$$\frac{dP}{d\Delta R} = \frac{k \cdot e^{-k(\Delta R - x_0)}}{(1 + e^{-k(\Delta R - x_0)})^2}$$

**Second Derivative** (rate of change of sensitivity):

$$\frac{d^2P}{d\Delta R^2} = \frac{-k^2 \cdot e^{-k(\Delta R - x_0)} \cdot (1 - e^{-k(\Delta R - x_0)})}{(1 + e^{-k(\Delta R - x_0)})^3}$$

### Plateau Threshold

The plateau threshold $\Delta R^*$ is defined as the point where:

$$\frac{dP}{d\Delta R} < \varepsilon$$

Beyond this threshold, further rating differences have negligible impact on win probability.

### Integration

The average win probability across a rating range is:

$$\overline{P_{win}} = \frac{1}{b-a} \int_a^b P_{win}(\Delta R) \, d\Delta R$$

## Results Summary

| Parameter | Value |
|-----------|-------|
| Games Analyzed | 1,600 |
| Slope (k) | 0.003326 |
| Midpoint (x₀) | -18.04 |
| R² (Goodness of Fit) | 0.8536 |
| Plateau Threshold (ΔR*) | ±448 |

### Key Predictions

| Rating Difference | Win Probability |
|-------------------|-----------------|
| ΔR = 0 (equal) | 0.5152 |
| ΔR = +100 | 0.5978 |
| ΔR = +200 | 0.6774 |
| ΔR = +400 | 0.8107 |

### Main Finding

**Beyond ±448 rating points, further rating differences have negligible additional impact on win probability.** This answers the research question about where the advantage "plateaus."

## Project Structure

```
mathia/
├── elo_analysis.py           # Main analysis script (all 7 steps)
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── data/
│   └── grandmaster_games.pgn # Real Lichess grandmaster game data
├── results/
│   └── elo_analysis.xlsx     # Excel workbook with all analysis
└── visualizations/
    ├── fig1_logistic_fit.png # Empirical data + logistic curve
    ├── fig2_derivatives.png  # First and second derivative plots
    └── fig3_integration.png  # Area under curve visualization
```

## Analysis Steps

The analysis follows 7 explicit steps:

### Step 1: Data Preparation
- Parse PGN file to extract White Elo, Black Elo, and Result
- Compute rating difference: ΔR = White Elo - Black Elo
- Bin ΔR in intervals of 50 points
- Calculate empirical probability: P_win = (Wins + 0.5 × Draws) / Total

### Step 2: Mathematical Modeling
- Fit logistic function using scipy.optimize.curve_fit
- Calculate R² goodness of fit
- Generate predictions at key rating differences

### Step 3: Derivative Analysis
- Compute first derivative (dP/dΔR) analytically
- Compute second derivative (d²P/dΔR²) analytically
- Find plateau threshold where dP/dΔR < ε

### Step 4: Integration Analysis
- Numerical integration using scipy.integrate.quad
- Calculate average win probability across rating spectrum
- Segment breakdown analysis

### Step 5: Excel Output
Creates `results/elo_analysis.xlsx` with 6 sheets:
1. **1_RawData** - Sample of 500 games
2. **2_EmpiricalProbabilities** - Binned P_win data
3. **3_ModelParameters** - k, x₀, R², ΔR*
4. **4_Predictions** - Model predictions with derivatives
5. **5_Integration** - Segment breakdown
6. **6_Comparison** - Empirical vs predicted values

### Step 6: Visualizations
- **Figure 1**: Scatter plot of empirical data with logistic fit overlay
- **Figure 2**: First and second derivative curves with plateau marked
- **Figure 3**: Area under curve (integration visualization)

### Step 7: Summary Report
- Final summary with all key findings printed to console

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mathia

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the complete analysis
python elo_analysis.py
```

The script will:
1. Parse the PGN data file
2. Perform all mathematical analysis
3. Generate Excel workbook with results
4. Create visualization figures
5. Print comprehensive summary to console

## Dependencies

- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scipy>=1.7.0` - Curve fitting and integration
- `matplotlib>=3.5.0` - Visualization
- `openpyxl` - Excel file writing

## Data Source

The analysis uses real chess game data from Lichess, containing 1,600 grandmaster-level games with:
- White and Black Elo ratings
- Game results (1-0, 0-1, 1/2-1/2)
- Rating differences ranging from -461 to +480

## Visualizations

### Figure 1: Logistic Fit
![Logistic Fit](visualizations/fig1_logistic_fit.png)

Scatter plot showing empirical win probabilities (point size proportional to sample size) with the fitted logistic curve overlay. R² = 0.8536 indicates strong model fit.

### Figure 2: Derivative Analysis
![Derivatives](visualizations/fig2_derivatives.png)

Top: First derivative showing sensitivity peaks at ΔR ≈ 0 and diminishes toward plateau regions.
Bottom: Second derivative showing inflection point and curvature changes.

### Figure 3: Integration
![Integration](visualizations/fig3_integration.png)

Area under the win probability curve from ΔR = -400 to +400, with average P_win marked.

## Mathematical Significance

This analysis demonstrates several IB Mathematics HL concepts:
- **Regression Analysis**: Fitting a nonlinear (logistic) model to data
- **Calculus**: Derivatives to analyze rate of change and find critical points
- **Integration**: Definite integrals to compute average values
- **Statistics**: R² for goodness of fit, empirical probability estimation

## Author

IB Mathematics HL Internal Assessment Project

## License

Educational use only.
