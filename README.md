# IB Mathematics HL Internal Assessment: Elo Rating Analysis

## The Mathematics of Chess: Analyzing How Elo Rating Differences Affect Win Probability

---

## Research Question

> **"How does the Elo rating difference between two chess players affect the probability of winning, and at what rating difference does the advantage plateau such that further differences no longer significantly impact outcomes?"**

---

## Project Overview

This project performs a rigorous mathematical analysis of the Elo rating system using **real chess game data** from Lichess grandmaster games. The analysis follows the IB Mathematics HL IA framework with emphasis on:

- **Logistic Function Fitting** - Modeling win probability as a function of rating difference
- **Derivative Analysis** - Finding the plateau threshold where rating advantages diminish
- **Integration Analysis** - Computing average win probability across rating spectrum

## Full IA Document

For the complete Internal Assessment write-up with all mathematical derivations, see:
- **[IB_MATH_IA.md](IB_MATH_IA.md)** - Complete ~4,000 word IA document

---

## Mathematical Framework

### The Logistic Model

The probability of White winning is modeled as:

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

---

## Results Summary

| Parameter | Value |
|-----------|-------|
| Games Analyzed | 127 |
| Slope (k) | 0.003734 |
| Midpoint (x₀) | -21.75 |
| R² (Goodness of Fit) | 0.6602 |
| Plateau Threshold (ΔR*) | ±955 |

### Fitted Model

$$P_{win}(\Delta R) = \frac{1}{1 + e^{-0.003734(\Delta R + 21.75)}}$$

### Key Predictions

| Rating Difference | Win Probability | Interpretation |
|-------------------|-----------------|----------------|
| ΔR = -400 | 0.196 | Black strongly favored |
| ΔR = -200 | 0.340 | Black favored |
| ΔR = -100 | 0.428 | Slight Black edge |
| ΔR = 0 | 0.520 | Even (slight White edge) |
| ΔR = +100 | 0.612 | White favored |
| ΔR = +200 | 0.696 | White strongly favored |
| ΔR = +400 | 0.829 | White dominant |

### Main Finding

**Beyond ±955 rating points, further rating differences have negligible additional impact on win probability.** At this threshold, P_win approaches 0.97 or 0.03, meaning the outcome is essentially predetermined.

---

## Project Structure

```
mathia/
├── elo_analysis.py           # Main analysis script (505 lines, 7 steps)
├── IB_MATH_IA.md             # Complete IA write-up (~4,000 words)
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── data/
│   └── grandmaster_games.pgn # Real Lichess grandmaster game data (127 games)
├── results/
│   └── elo_analysis.xlsx     # Excel workbook with all analysis (6 sheets)
└── visualizations/
    ├── fig1_logistic_fit.png # Empirical data + logistic curve
    ├── fig2_derivatives.png  # First and second derivative plots
    └── fig3_integration.png  # Area under curve visualization
```

---

## Analysis Steps

The analysis follows 7 explicit steps:

### Step 0: Data Acquisition
- Download real grandmaster games from Lichess API
- Players include DrNykterstein (Magnus Carlsen), GMWSO, nihalsarin, etc.

### Step 1: Data Preparation
- Parse PGN file to extract White Elo, Black Elo, and Result
- Compute rating difference: ΔR = White Elo - Black Elo
- Bin ΔR in intervals of 50 points
- Calculate empirical probability: P_win = (Wins + 0.5 × Draws) / Total

### Step 2: Mathematical Modeling
- Fit logistic function using scipy.optimize.curve_fit
- Calculate R² goodness of fit using weighted regression
- Compare to theoretical Elo formula (k = ln(10)/400)

### Step 3: Derivative Analysis
- Compute first derivative (dP/dΔR) analytically
- Compute second derivative (d²P/dΔR²) analytically
- Find plateau threshold where dP/dΔR < ε = 0.0001

### Step 4: Integration Analysis
- Numerical integration using scipy.integrate.quad
- Calculate average win probability: ∫P dΔR / (b-a)
- Segment breakdown analysis over [-400, 400]

### Step 5: Excel Output
Creates `results/elo_analysis.xlsx` with 6 sheets:
1. **1_RawData** - All 127 game records
2. **2_EmpiricalProbabilities** - Binned P_win data
3. **3_ModelParameters** - k, x₀, R², ΔR*, theoretical comparison
4. **4_Predictions** - Model predictions every 25 rating points with derivatives
5. **5_Integration** - Segment breakdown of integrals
6. **6_Comparison** - Empirical vs predicted values with residuals

### Step 6: Visualizations
- **Figure 1**: Scatter plot with logistic fit and theoretical Elo comparison
- **Figure 2**: First and second derivative curves with plateau marked
- **Figure 3**: Area under curve (shaded integration visualization)

### Step 7: Summary Report
- Comprehensive console output with all findings

---

## Installation

```bash
# Clone the repository
git clone https://github.com/pranavkokati/mathia.git
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
1. Download real games from Lichess (or use existing data)
2. Parse PGN data
3. Perform all mathematical analysis
4. Generate Excel workbook with results
5. Create 3 visualization figures
6. Print comprehensive summary to console

---

## Dependencies

```
numpy>=1.21.0      # Numerical computing
pandas>=1.3.0      # Data manipulation
scipy>=1.7.0       # Curve fitting and integration
matplotlib>=3.5.0  # Visualization
openpyxl           # Excel file writing
```

---

## Data Source

The analysis uses real chess game data from **Lichess.org**, the world's largest free chess platform:

- **Players**: DrNykterstein (Magnus Carlsen), GMWSO, nihalsarin, LyonBeast, etc.
- **Games**: 127 rated games
- **Rating range**: 1000 - 2525 Elo
- **ΔR range**: -889 to +490

---

## Visualizations

### Figure 1: Logistic Fit
![Logistic Fit](visualizations/fig1_logistic_fit.png)

Scatter plot showing empirical win probabilities (point size proportional to sample size) with:
- **Red line**: Fitted logistic curve (R² = 0.6602)
- **Green dashed**: Theoretical Elo curve for comparison

### Figure 2: Derivative Analysis
![Derivatives](visualizations/fig2_derivatives.png)

**Top panel**: First derivative dP/dΔR showing:
- Maximum sensitivity at ΔR ≈ 0
- Plateau threshold marked at ΔR* = ±955

**Bottom panel**: Second derivative d²P/dΔR² showing:
- Inflection point at ΔR = x₀
- Regions of positive/negative curvature

### Figure 3: Integration
![Integration](visualizations/fig3_integration.png)

Shaded area under the win probability curve from ΔR = -400 to +400:
- **Total integral**: ∫P dΔR = 413.77
- **Average P_win**: 0.5172

---

## Mathematical Significance

This analysis demonstrates several IB Mathematics HL concepts:

| Topic | Application |
|-------|-------------|
| **Functions** | Logistic function modeling |
| **Regression** | Nonlinear curve fitting with weighted least squares |
| **Differentiation** | First/second derivatives to find sensitivity and inflection |
| **Integration** | Definite integrals for average probability |
| **Statistics** | R² coefficient, empirical probability, standard error |

---

## Key Formulas Used

### Logistic Function
$$P(x) = \frac{1}{1 + e^{-k(x-x_0)}}$$

### First Derivative
$$\frac{dP}{dx} = \frac{k e^{-k(x-x_0)}}{(1+e^{-k(x-x_0)})^2}$$

### Second Derivative
$$\frac{d^2P}{dx^2} = \frac{-k^2 e^{-k(x-x_0)}(1-e^{-k(x-x_0)})}{(1+e^{-k(x-x_0)})^3}$$

### Theoretical Elo
$$k_{Elo} = \frac{\ln(10)}{400} \approx 0.00576$$

### Coefficient of Determination
$$R^2 = 1 - \frac{\sum w_i(y_i - \hat{y}_i)^2}{\sum w_i(y_i - \bar{y})^2}$$

---

## Answer to Research Question

### Q: How does the Elo rating difference affect win probability?

The relationship follows a **logistic (S-shaped) curve**. Near equal ratings, each 100-point rating advantage increases win probability by approximately 8-9 percentage points.

### Q: At what rating difference does the advantage plateau?

The advantage effectively plateaus at **±955 rating points**. Beyond this threshold:
- dP/dΔR < 0.0001 (less than 0.01% change per rating point)
- P_win approaches 0.97 (or 0.03 for the disadvantaged side)
- The outcome is essentially predetermined

---

## Author

IB Mathematics HL Internal Assessment Project

## License

Educational use only.

---

## References

1. Elo, A. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
2. Glickman, M. E. (1995). "A Comprehensive Guide to Chess Ratings." *American Chess Journal*.
3. Lichess.org Open Database. https://lichess.org/
