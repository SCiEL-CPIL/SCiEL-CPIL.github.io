"""
Strategy: Posterior Sampling / Thompson Sampling with Global Optimum Estimation.

Instead of using a heuristic like UCB (which adds a confidence bonus to the mean),
this strategy uses the probabilistic nature of the Gaussian Process directly.

The "Posterior Minimum" (or Expected Minimum) is the value the function is most
likely to take at its global minimum. For a GP, we can estimate this by sampling
from the posterior distribution at the candidate points and taking the minimum.
However, sampling 2000 points multiple times is expensive.

A more efficient and robust proxy for "Posterior Minimum" in the context of
minimization is to compute the **Lower Confidence Bound (LCB)** with a
**very aggressive exploration parameter** that is dynamically adjusted, OR
more specifically, use the **Expected Minimum** approximation.

Actually, the most distinct structural change from UCB is **Thompson Sampling**.
In TS, we sample a function $f \sim \mathcal{GP}$ and pick the minimum of that
sample.
Since we only have `mu` and `sigma` at 2000 points, we can simulate a single
draw from the joint Gaussian distribution at these points.
However, the covariance matrix of these 2000 points is not provided.
We only have the marginal `mu` and `sigma`.
Without the covariance between points, we cannot sample a consistent function
draw from the GP posterior directly.

BUT, we can use a different structural approach: **The "Optimistic" Lower Bound**.
Standard UCB uses `-mu + beta * sigma`. This is an upper bound on the potential
regret.
A different family is **Probability of Improvement (PoI)** or **Expected Improvement (EI)**.
But EI/UCB are "local" in the sense that they score individual points.

Let's look at the "Global Optimum Estimation" angle.
If we assume the points are independent (which they aren't, but let's see if a
heuristic based on independence works better), the probability that a point $x_i$
is the global minimum is related to $P(f(x_i) < f_{best})$.
Score = $P(f(x_i) < f_{best})$.
This is PoI.

However, the directive asks for a MATERIAL difference.
Let's try **Quantile-based Acquisition**.
Instead of linearly combining mean and std, we look at the **lower quantile** of the
posterior predictive distribution.
Score = $-\text{Quantile}_{0.1}(\mathcal{N}(\mu_i, \sigma_i^2))$.
This is equivalent to $-\mu_i + z_{0.1} \sigma_i$.
This is just LCB with a fixed $z$.

What is truly different?
**Hypervolume-based or Pareto-style?** No, single objective.

Let's reconsider the **Thompson Sampling** constraint.
If we cannot sample the joint GP, can we use a **Nearest Neighbor** heuristic?
No coordinates.

Let's look at the "Dynamic UCB kappa" finding: "Dynamic UCB kappa balances exploration
and exploitation better than fixed hybrid EI/LCB".
The current program uses adaptive beta.
What if we use a **non-linear** combination that is not linear in sigma?
For example, **Squared UCB**: $-\mu^2 + \beta \sigma^2$? No, units don't match well.

Let's try **Expected Regret** style but with a **Gaussian Kernel** on the score itself?
No.

Let's go back to **Thompson Sampling** but approximate it.
If we assume the points are conditionally independent given the GP hyperparameters (a common
simplification when only marginals are available), then a sample from the posterior
at point $i$ is $y_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$.
The "Thompson Sampling" choice would be the $x_i$ that minimizes $y_i$.
So, we generate a vector $Y$ of shape (2000,) where $Y_i = \mu_i + \sigma_i * Z_i$,
with $Z_i \sim \mathcal{N}(0,1)$.
Then we return $-Y$ as the scores?
Wait, if we return $-Y$, the evaluator picks the max score, which corresponds to the min $Y$.
This is exactly **Thompson Sampling**.

Why is this different from UCB?
UCB: Deterministic score $S_i = -\mu_i + \beta \sigma_i$. Always picks the same point if mu/sigma are fixed.
TS: Stochastic score $S_i = -(\mu_i + \sigma_i Z_i)$. The "best" point changes every call even if mu/sigma are identical, because $Z$ is random.
This stochasticity is a key structural feature of TS that UCB lacks. It naturally balances exploration and exploitation without a hand-tuned $\beta$.
Early on, sigma is large, so the random noise $Z \sigma$ dominates, leading to high exploration.
Later, sigma is small, so $\mu$ dominates, leading to exploitation.

This is a distinct optimization method (Posterior Sampling vs. Confidence Bound).

Implementation:
1. Generate standard normal random numbers $Z$ of shape (2000,).
2. Compute samples $Y = \mu + \sigma \odot Z$.
3. Return $-Y$ as scores.

We need to be careful with the random seed. The task says "For each seed 0, 1 and 2...".
The evaluator likely sets the global numpy random seed or we should manage it.
If we use `np.random.randn`, it consumes the global RNG state.
Since the evaluator runs 3 seeds, and each seed runs 30 steps, the RNG state will evolve.
This is standard for TS.

One potential issue: If `sigma` is very small, `Y` is just `mu`.
If `sigma` is large, `Y` is noisy.
This is exactly the desired behavior.

Let's verify the "higher meaning evaluate next" constraint.
We want to minimize $f$.
TS picks $x$ that minimizes the sample $y$.
So we want the acquisition function to be high for the $x$ that minimizes $y$.
Score $= -y$.
Max Score $\implies$ Min $y$.
Correct.

We must ensure finite scores.
$Y$ is finite if $\mu, \sigma, Z$ are finite.
$Z$ is always finite (practically).
$\mu, \sigma$ are sanitized.

Deadline guard:
TS is very fast (O(N)). No complex loops.

Let's implement this.
"""

import time
import numpy as np

_START_TIME = None
_HARD_LIMIT = 330.0
_MARGIN = 5.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling Acquisition Function.
    
    Strategy:
    Instead of deterministic UCB, we sample from the posterior predictive
    distribution at each candidate point. We pick the candidate that yields
    the lowest sampled value.
    
    Score = -Sampled_Value
    Sampled_Value_i = mu_i + sigma_i * Z_i, where Z_i ~ N(0, 1)
    
    This approach inherently balances exploration and exploitation:
    - When sigma is large (uncertain), the random term dominates, encouraging
      exploration of unknown regions.
    - When sigma is small (certain), the mean term dominates, encouraging
      exploitation of known good regions.
    
    This is structurally different from UCB because it is stochastic and
    does not rely on a fixed or tuned confidence parameter (beta).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard
    if _elapsed() > _HARD_LIMIT - _MARGIN:
        # Fallback to simple UCB if we are running out of time (though TS is fast)
        scores = -mu + sigma
        return np.where(np.isfinite(scores), scores, 0.0)

    # Generate standard normal samples
    # Using np.random.randn is standard. The evaluator's seed control will
    # ensure reproducibility across runs if it sets the global seed.
    # If the evaluator does NOT set the seed, the results will be non-deterministic
    # but valid. The task description implies a controlled environment ("For each seed...").
    # Assuming global numpy random state is managed by the harness.
    
    Z = np.random.randn(mu.shape[0])
    
    # Compute posterior samples
    # Y_i ~ N(mu_i, sigma_i^2)
    Y = mu + sigma * Z
    
    # Acquisition score: we want to pick the minimum Y.
    # The evaluator picks the maximum score.
    # So Score = -Y.
    scores = -Y
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
