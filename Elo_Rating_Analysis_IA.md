# The Mathematics of Chess: How Rating Differences Determine Victory

---

*"Chess is the struggle against the error."*

~ Johannes Zukertort

---

## Introduction

When two chess players sit across from each other at a tournament, spectators and commentators often reference their Elo ratings to predict the likely outcome. The Elo rating system, a numerical representation of playing strength, has become the universal language of competitive chess. However, I have always wondered: at what point does a rating advantage become so overwhelming that the outcome is essentially predetermined? This question emerged from my own experience as a competitive chess player, where I frequently faced opponents of varying skill levels and noticed that while a 100-point advantage felt meaningful, a 400-point advantage seemed to guarantee victory.

This mathematical investigation will focus on modeling the relationship between Elo rating differences and win probability using real grandmaster game data. I chose this topic because of my personal interest in competitive chess and my curiosity about the mathematical foundations underlying the rating system that governs tournament play worldwide. Through this exploration, I aim to:

1. Fit a logistic regression model to empirical chess data to determine how win probability changes with rating difference
2. Apply differential calculus to find the "plateau threshold" — the rating difference beyond which further advantages have negligible impact
3. Use integral calculus to calculate average win probabilities across rating ranges
4. Compare my empirical findings with the theoretical Elo formula

This investigation requires understanding of the logistic function, derivatives, and definite integrals, concepts that extend beyond the standard IB Mathematics curriculum but are essential for modeling real-world probabilistic phenomena.

---

## Background Information

### The History of the Elo Rating System

The Elo rating system was developed by Arpad Elo, a Hungarian-American physics professor and chess master, in the 1960s.¹ Before Elo's system, chess ratings were largely arbitrary and inconsistent across different organizations. Elo applied statistical theory to create a self-correcting system where ratings adjust based on actual game outcomes relative to expected outcomes.

The system was officially adopted by the World Chess Federation (FIDE) in 1970 and has since become the standard for rating players in chess and many other competitive games, from Go to online video games.² The elegance of the Elo system lies in its use of the logistic function — the same S-shaped curve that appears in population growth models, neural network activation functions, and epidemiological studies.

### The Theoretical Foundation

According to Elo's original formulation, the expected score for Player A against Player B is given by:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

Where $R_A$ and $R_B$ are the ratings of Players A and B respectively.³ The constant 400 was chosen so that a 400-point rating difference corresponds to an expected score of approximately 0.91 (or about a 91% chance of winning for the higher-rated player, assuming draws are rare).

This formula can be rewritten in terms of the rating difference $\Delta R = R_A - R_B$:

$$E_A = \frac{1}{1 + 10^{-\Delta R/400}}$$

I was intrigued by whether real-world data from modern grandmaster games would conform to this theoretical model, or whether factors such as playing style, time pressure, and psychological elements might cause deviations.

---

## The Mathematical Framework

### The Logistic Function

The Elo formula is a specific case of the logistic function, which has the general form:

$$P(x) = \frac{1}{1 + e^{-k(x - x_0)}}$$

Where:
- $k$ is the slope parameter controlling the steepness of the curve
- $x_0$ is the midpoint where $P = 0.5$
- $e$ is Euler's number (≈ 2.71828)

The theoretical Elo system uses a base-10 exponential, which can be converted to the natural exponential form with:

$$k_{theoretical} = \frac{\ln(10)}{400} \approx 0.00576$$

This means the theoretical model predicts that each rating point changes the log-odds of winning by approximately 0.00576.

### Derivatives of the Logistic Function

To understand how sensitive win probability is to rating changes, I needed to derive the first derivative of the logistic function.

**First Derivative:**

Starting with $P = \frac{1}{1 + e^{-k(x-x_0)}}$, let $u = e^{-k(x-x_0)}$

Then $P = \frac{1}{1+u} = (1+u)^{-1}$

Using the chain rule:
$$\frac{dP}{dx} = -(1+u)^{-2} \cdot \frac{du}{dx}$$

Since $\frac{du}{dx} = -k \cdot e^{-k(x-x_0)} = -ku$:

$$\frac{dP}{dx} = -(1+u)^{-2} \cdot (-ku) = \frac{ku}{(1+u)^2}$$

Substituting back:

$$\boxed{\frac{dP}{d\Delta R} = \frac{k \cdot e^{-k(\Delta R - x_0)}}{(1 + e^{-k(\Delta R - x_0)})^2}}$$

This derivative represents the sensitivity of win probability to changes in rating difference. It reaches its maximum value at $\Delta R = x_0$, where the probability is changing most rapidly.

**Second Derivative:**

To find where the sensitivity itself is changing most rapidly (the inflection points), I derived the second derivative:

$$\frac{d^2P}{d\Delta R^2} = \frac{-k^2 \cdot e^{-k(\Delta R-x_0)} \cdot (1 - e^{-k(\Delta R-x_0)})}{(1 + e^{-k(\Delta R-x_0)})^3}$$

The second derivative equals zero at $\Delta R = x_0$, confirming this is the inflection point where the curve transitions from concave up to concave down.

---

## Data Collection and Methodology

### Data Source

To test the theoretical model against reality, I collected game data from Lichess.org, the world's largest free online chess platform with over 100 million registered users.⁴ I specifically targeted games from titled players, including:

- **DrNykterstein** (Magnus Carlsen's account)
- **GMWSO** (Wesley So)
- **nihalsarin** (Nihal Sarin)
- **LyonBeast** (Maxime Vachier-Lagrave)

Using the Lichess API, I downloaded rated classical and rapid games from these players.⁵

### Data Processing

For each game, I extracted three pieces of information:
- White player's Elo rating
- Black player's Elo rating
- Game result (1-0 for White win, 0-1 for Black win, 1/2-1/2 for draw)

I then calculated the rating difference:

$$\Delta R = \text{White Elo} - \text{Black Elo}$$

Results were converted to numerical scores for the White player:
- White win (1-0): Score = 1.0
- Draw (1/2-1/2): Score = 0.5
- Black win (0-1): Score = 0.0

### Binning the Data

To calculate empirical probabilities, I grouped games into bins of 50 rating points. For each bin, I calculated the empirical win probability:

$$P_{win} = \frac{\text{Number of White Wins} + 0.5 \times \text{Number of Draws}}{\text{Total Games in Bin}}$$

The 0.5 multiplier for draws reflects the fact that a draw is worth half a point in chess scoring.

### Sample Statistics

After processing, my dataset contained:

| Statistic | Value |
|-----------|-------|
| Total games analyzed | 127 |
| Rating range | 1000 - 2525 Elo |
| ΔR range | -889 to +490 |
| White wins | 64 (50.4%) |
| Draws | 6 (4.7%) |
| Black wins | 57 (44.9%) |

The mean rating difference was $\bar{\Delta R} = -13.8$, indicating a slight average rating advantage for Black in my sample.

---

## Mathematical Analysis

### Fitting the Logistic Model

I used nonlinear least squares regression to fit the logistic function to my empirical data. The model to be fitted was:

$$P_{win}(\Delta R) = \frac{1}{1 + e^{-k(\Delta R - x_0)}}$$

To account for varying sample sizes across bins (some bins had many more games than others), I applied weighted regression with weights proportional to $\sqrt{n}$, where $n$ is the number of games in each bin.

The optimization algorithm (scipy.optimize.curve_fit in Python) minimized the weighted sum of squared residuals:

$$\min_{k, x_0} \sum_i w_i \cdot (P_{empirical,i} - P_{model,i})^2$$

### Fitted Parameters

The regression yielded the following parameters:

| Parameter | Fitted Value | Standard Error |
|-----------|--------------|----------------|
| k (slope) | 0.003734 | ±0.001556 |
| x₀ (midpoint) | -21.75 | ±24.53 |

Therefore, my fitted model is:

$$\boxed{P_{win}(\Delta R) = \frac{1}{1 + e^{-0.003734(\Delta R + 21.75)}}}$$

### Goodness of Fit

To assess how well the model fits the data, I calculated the coefficient of determination (R²):

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum w_i(y_i - \hat{y}_i)^2}{\sum w_i(y_i - \bar{y})^2}$$

**Result: R² = 0.6602**

This indicates that approximately 66% of the variance in win probability is explained by the logistic model. While not perfect, this is reasonable given the inherent randomness in chess outcomes — even strong rating favorites occasionally lose.

### Comparison to Theoretical Elo

| Parameter | Theoretical | Fitted | Ratio |
|-----------|-------------|--------|-------|
| k | 0.00576 | 0.00373 | 0.649 |
| x₀ | 0 | -21.75 | — |

The fitted slope is approximately 65% of the theoretical value. This suggests that in practice, rating differences have somewhat less predictive power than theory suggests. Several factors could explain this:

1. **Rating inflation/deflation**: Online ratings may not perfectly reflect true playing strength
2. **Time pressure**: Rapid games introduce more randomness than classical games
3. **Psychological factors**: Players may perform differently against higher or lower-rated opponents

The non-zero midpoint ($x_0 = -21.75$) indicates a slight advantage for Black in my sample, which could be due to sampling bias toward games where strong players chose to play Black.

---

## Derivative Analysis: Finding the Plateau

### Calculating Sensitivity

Using my fitted parameters, I calculated the first derivative at various rating differences:

| ΔR | dP/dΔR | Interpretation |
|----|--------|----------------|
| 0 | 0.000932 | Near maximum sensitivity |
| ±100 | 0.000887 | High sensitivity |
| ±200 | 0.000790 | Moderate sensitivity |
| ±400 | 0.000531 | Reduced sensitivity |
| ±600 | 0.000308 | Low sensitivity |

At equal ratings ($\Delta R = 0$), each 1-point rating increase corresponds to approximately a 0.093% increase in win probability. This means a 100-point advantage translates to roughly a 9% increase in winning chances.

### Maximum Sensitivity

The derivative reaches its maximum at $\Delta R = x_0 = -21.75$:

$$\left.\frac{dP}{d\Delta R}\right|_{max} = \frac{k}{4} = \frac{0.003734}{4} = 0.000934$$

This maximum occurs because the logistic curve is steepest at its midpoint, where small changes in rating have the largest impact on probability.

### The Plateau Threshold

I defined the plateau threshold $\Delta R^*$ as the point where the derivative falls below a small value $\varepsilon$:

$$\frac{dP}{d\Delta R} < \varepsilon = 0.0001$$

Solving numerically, I found:

$$\boxed{\Delta R^* = \pm 955 \text{ rating points}}$$

At this threshold:
- $P_{win}(+955) = 0.9725$ (White has 97.25% chance of winning)
- $P_{win}(-955) = 0.0275$ (White has only 2.75% chance of winning)

**Interpretation:** Beyond approximately 955 rating points difference, further advantages have less than 0.01% effect per additional rating point. The outcome is essentially predetermined.

This answers my original research question: the advantage "plateaus" at roughly 950-1000 rating points, where the higher-rated player is virtually guaranteed to win.

---

## Integration Analysis

### Average Win Probability

To find the average win probability for White across a range of rating differences, I computed the definite integral:

$$\bar{P}_{win} = \frac{1}{b-a} \int_a^b P_{win}(\Delta R) \, d\Delta R$$

Using numerical integration (Simpson's rule) over the range $[-400, 400]$:

$$\int_{-400}^{400} P_{win}(\Delta R) \, d\Delta R = 413.77$$

Therefore:

$$\bar{P}_{win} = \frac{413.77}{800} = 0.5172$$

This average of 51.72% is slightly above 50%, reflecting the well-known first-move advantage in chess — White has a slight inherent edge due to moving first.⁶

### Segment Analysis

Breaking down the integral by region reveals how win probability distributes across different rating advantages:

| Segment | ∫P dΔR | Average P | Interpretation |
|---------|--------|-----------|----------------|
| [-400, -200] | 52.69 | 0.264 | Strong Black advantage |
| [-200, 0] | 85.66 | 0.428 | Slight Black advantage |
| [0, +200] | 122.11 | 0.611 | Slight White advantage |
| [+200, +400] | 153.32 | 0.767 | Strong White advantage |

This analysis shows the asymmetric nature of the logistic curve: the "area" contributed by White-favored positions exceeds that of Black-favored positions, even when the rating ranges are symmetric around zero.

---

## Model Predictions

Using my fitted model, I can predict win probabilities for any rating difference:

| Rating Difference | Win Probability | Practical Meaning |
|-------------------|-----------------|-------------------|
| ΔR = -400 | 19.6% | Black strongly favored |
| ΔR = -200 | 34.0% | Black favored |
| ΔR = -100 | 42.8% | Slight Black edge |
| ΔR = 0 | 52.0% | Approximately even |
| ΔR = +100 | 61.2% | White favored |
| ΔR = +200 | 69.6% | White strongly favored |
| ΔR = +400 | 82.9% | White dominant |

These predictions align well with practical chess experience. A player rated 200 points higher will win roughly 70% of games — enough to be considered a clear favorite, but not so dominant that upsets are rare.

---

## Evaluation and Limitations

### Strengths of the Investigation

1. **Real Data**: I used authentic grandmaster games rather than simulated data, ensuring my analysis reflects actual competitive play.

2. **Mathematical Rigor**: I applied proper curve fitting techniques with weighted regression to account for unequal sample sizes.

3. **Comprehensive Analysis**: The investigation combined descriptive statistics, nonlinear regression, differential calculus, and integral calculus.

4. **Practical Relevance**: The results have direct applications for chess players, tournament organizers, and rating system designers.

### Limitations

1. **Sample Size**: With only 127 games, statistical power is limited. A larger sample would provide more reliable estimates.

2. **Sampling Bias**: Games from specific elite players may not represent all skill levels. The relationship might differ for club-level players.

3. **Mixed Time Controls**: Combining classical and rapid games introduces variability, as time pressure affects outcomes differently.

4. **R² Value**: The 0.66 R² indicates substantial unexplained variance, suggesting factors beyond rating (form, preparation, psychology) significantly influence outcomes.

5. **Draw Treatment**: Counting draws as 0.5 wins simplifies the three-outcome nature of chess into a binary model.

### Sources of Error

| Error Source | Impact | Possible Mitigation |
|--------------|--------|---------------------|
| Random chess variance | High | Larger sample size |
| Rating accuracy | Medium | Use only classical games |
| Unequal bin sizes | Low | Weighted regression (applied) |
| Numerical precision | Negligible | Double-precision arithmetic |

---

## Conclusion

This investigation successfully applied logistic regression and calculus to analyze the relationship between chess Elo ratings and win probability. My key findings are:

1. **Win probability follows a logistic function** with fitted parameters $k = 0.003734$ and $x_0 = -21.75$, confirming the theoretical basis of the Elo system while revealing a slightly weaker relationship than theory predicts.

2. **The derivative analysis revealed** that sensitivity to rating differences is highest near equal ratings (where each point matters most) and diminishes toward the extremes (where outcomes become nearly certain).

3. **The plateau threshold is approximately ±955 rating points.** Beyond this difference, the higher-rated player has greater than 97% winning probability, and additional rating advantage has negligible practical impact.

4. **Integration showed** an average White win probability of 51.7% across the range [-400, 400], consistent with the known first-move advantage in chess.

5. **The model explains 66% of variance**, indicating that while Elo is meaningfully predictive, significant randomness remains in chess outcomes — a feature, not a bug, that keeps the game exciting.

This exploration satisfied my curiosity about the mathematical foundations of chess ratings. I was particularly pleased to discover that calculus provides tools to answer practical questions like "at what point is the outcome essentially decided?" The plateau threshold of ~955 points explains why watching a grandmaster play an amateur feels like a foregone conclusion, while games between similarly-rated players remain genuinely uncertain.

The investigation also reinforced my appreciation for how mathematics connects to competitive games. The same logistic function that models chess ratings appears in machine learning, epidemiology, and economics — illustrating how fundamental mathematical structures underlie diverse real-world phenomena.

If I were to extend this investigation, I would:
- Analyze 10,000+ games for stronger statistical conclusions
- Compare the relationship across different time controls (bullet, blitz, classical)
- Investigate whether the plateau threshold differs at various rating levels
- Apply machine learning techniques to incorporate additional predictive factors

The mathematics of Elo ratings elegantly captures the essence of competitive skill, and this investigation demonstrates how calculus provides deeper insights into probability models beyond simple curve fitting.

---

## Bibliography

¹ Elo, Arpad. *The Rating of Chessplayers, Past and Present*. New York: Arco Publishing, 1978. Print.

² Glickman, Mark E. "A Comprehensive Guide to Chess Ratings." *American Chess Journal* 3 (1995): 59-102. Print.

³ Elo, Arpad. "The Proposed USCF Rating System." *Chess Life* 22.8 (1967): 242-247. Print.

⁴ Lichess.org. "About Lichess." Web. 10 Feb. 2026. <https://lichess.org/about>.

⁵ Lichess.org. "Lichess API Documentation." Web. 10 Feb. 2026. <https://lichess.org/api>.

⁶ Regan, Kenneth W., and Guy McC. Haworth. "Intrinsic Chess Ratings." *Proceedings of the AAAI Conference on Artificial Intelligence*. 2011. Print.

⁷ Stewart, James. *Calculus: Early Transcendentals*. 8th ed. Boston: Cengage Learning, 2015. Print.

⁸ Weisstein, Eric W. "Logistic Equation." *MathWorld—A Wolfram Web Resource*. Web. 10 Feb. 2026. <https://mathworld.wolfram.com/LogisticEquation.html>.

---

**Word Count:** Approximately 3,200 words (excluding tables and equations)
