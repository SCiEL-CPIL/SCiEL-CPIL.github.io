import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Knowledge Gradient (KG) approximation.
    
    Instead of Expected Improvement (EI), which is a myopic heuristic,
    we estimate the expected reduction in the best observed value (f_best)
    after one more evaluation. This accounts for the fact that the new
    observation will update the GP posterior, potentially leading to
    better future evaluations.
    
    For a single evaluation step, the KG is approximated by:
    KG(x) = E[ max(0, f_best - Y(x)) * w(x) ]
    where Y(x) ~ N(mu(x), sigma^2(x)) and w(x) is a weight that reflects
    the probability that this point will be the new best, adjusted for
    the fact that we are minimizing.
    
    A common practical approximation for KG in BO is to use a modified EI
    that incorporates the uncertainty in a way that favors points likely
    to significantly improve the lower bound. However, a more direct
    structural difference from EI is to use the "Expected Reduction in
    Variance" or a sample-based KG.
    
    Given the compute budget (330s) and the small dimensionality (1D),
    we can afford a lightweight Monte Carlo approximation of the KG.
    However, even simpler and often more effective for 1D is to use
    a "Greedy" KG approximation:
    
    KG(x) ≈ E[ min(f_best, Y(x)) - E[min(f_best, Y(x))] ] is not quite right.
    
    Let's use the standard definition:
    KG(x) = E[ V(X_{n+1}) - V(X_n) ]
    where V is the value of the optimal solution found so far.
    
    In practice, for a single step, KG is closely related to EI but with
    a different weighting. However, a truly different structural approach
    is to use **Probability of Improvement (PI)** combined with a
    **Thompson Sampling** style exploration, or better yet,
    **Upper Confidence Bound (UCB)** with an adaptive beta.
    
    Wait, the directive asks for a STRUCTURAL change. EI is the current one.
    Let's implement **Thompson Sampling** via the acquisition score.
    Thompson Sampling suggests sampling from the posterior and picking the
    argmin. But we must return a score for *each* candidate to let the
    evaluator pick the max.
    
    We can simulate the Thompson Sampling decision by estimating the
    probability that a candidate x is the minimizer of the posterior
    samples.
    
    Score(x) = P(x is the argmin of the posterior sample)
    
    This is a fundamentally different objective than EI. It directly
    optimizes the probability of selecting the true minimizer of the
    current belief, which naturally balances exploration (high variance
    regions have a non-zero probability of being the minimum) and
    exploitation.
    
    To compute P(x is argmin) efficiently:
    1. Draw S samples from the GP posterior at the candidate points.
       y_s ~ N(mu, sigma^2) for s=1..S
    2. For each sample s, find the index i_s that minimizes y_s.
    3. Score(x_i) = (1/S) * sum_s [ i_s == i ]
    
    This is O(S * N) where N=2000. With S=100, this is 200,000 operations,
    which is very fast.
    
    However, there is a subtlety: the candidates are fixed uniform points.
    The true minimizer might not be in the candidate set. But since we
    are constrained to choose from these 2000 points, this is the correct
    decision rule for the discrete choice.
    
    Let's implement this.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for sampling
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Number of Monte Carlo samples for Thompson Sampling
    # 100 samples is usually enough for a stable probability estimate in 1D
    n_samples = 100
    n_candidates = len(mu)
    
    # Initialize score array
    scores = np.zeros(n_candidates, dtype=np.float64)
    
    # Draw samples: shape (n_samples, n_candidates)
    # y = mu + sigma * z
    z = np.random.randn(n_samples, n_candidates)
    y = mu[None, :] + sigma_safe[None, :] * z
    
    # For each sample, find the index of the minimum value
    # argmin returns the index of the minimum in each row
    min_indices = np.argmin(y, axis=1)
    
    # Count how many times each candidate was the minimum
    # bincount is efficient for this
    counts = np.bincount(min_indices, minlength=n_candidates)
    
    # Normalize to get probabilities
    scores = counts / n_samples
    
    # The task requires "higher meaning evaluate next".
    # Thompson Sampling selects the point with the highest probability of being the minimizer.
    # So scores are already in the correct direction.
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
