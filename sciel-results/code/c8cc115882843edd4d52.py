# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Monte Carlo Knowledge Gradient (KG) acquisition function.
    
    The Knowledge Gradient measures the expected reduction in the minimum observed
    function value if we evaluate at a candidate point x.
    KG(x) = f_best - E[ min(f_best, f(x)) ]
    
    Since f(x) ~ N(mu(x), sigma(x)^2), we can compute the expectation analytically
    or via Monte Carlo. Given the 2000 candidates and 330s limit, an analytic
    approximation or a small MC sample is efficient.
    
    For a normal variable Y ~ N(m, s^2):
    E[min(f_best, Y)] = f_best * Phi((f_best - m) / s) + m * (1 - Phi((f_best - m) / s)) - s * phi((f_best - m) / s)
    
    However, the "pure" KG often performs better than this closed form in practice
    when using Monte Carlo with a moderate number of samples, or simply by using
    the analytic formula which is exact. Let's use the analytic formula for speed
    and precision, as it is O(N) and very fast.
    
    Wait, the analytic formula for E[min(f_best, Y)] is:
    Let z = (f_best - mu) / sigma
    E[min(f_best, Y)] = f_best * Phi(z) + mu * (1 - Phi(z)) - sigma * phi(z)
    
    Therefore:
    KG(x) = f_best - [ f_best * Phi(z) + mu * (1 - Phi(z)) - sigma * phi(z) ]
    KG(x) = f_best * (1 - Phi(z)) - mu * (1 - Phi(z)) + sigma * phi(z)
    KG(x) = (f_best - mu) * (1 - Phi(z)) + sigma * phi(z)
    
    Note: If sigma is very small, this should behave like max(0, f_best - mu) + small term.
    If mu is much smaller than f_best, z is large positive, Phi(z) ~ 1, 1-Phi(z) ~ 0, phi(z) ~ 0.
    KG ~ 0. This is correct: if we are already very confident f(x) is better, the *expected reduction* 
    from the *current best* is not the right metric if we don't know it's better? 
    Actually, KG is defined as E[min(f_best, f(x)) - min(f_best, f(x'))] is for sequential.
    The standard one-step KG is E[ min(f_best, f(x)) ] reduction.
    
    Let's re-verify the logic.
    We want to maximize the expected improvement of the best value.
    Current best: f_best.
    New best: min(f_best, f(x)).
    Expected New Best: E[min(f_best, f(x))].
    Expected Reduction: f_best - E[min(f_best, f(x))].
    
    If f(x) is definitely much lower than f_best (mu << f_best, sigma small):
    min(f_best, f(x)) ~ f(x).
    E[min] ~ mu.
    Reduction ~ f_best - mu.
    
    Let's check the formula:
    z = (f_best - mu) / sigma. If mu << f_best, z is large positive.
    Phi(z) ~ 1. 1 - Phi(z) ~ 0.
    phi(z) ~ 0.
    Formula gives ~ 0.
    
    There is a discrepancy. The formula E[min(A, Y)] for Y~N(mu, sigma^2) is:
    If A is a constant.
    E[min(A, Y)] = A * P(Y > A) + E[Y | Y <= A] * P(Y <= A)
    P(Y > A) = 1 - Phi((A-mu)/sigma)
    E[Y | Y <= A] = mu - sigma * phi((A-mu)/sigma) / Phi((A-mu)/sigma)
    
    So E[min(A, Y)] = A * (1 - Phi(z)) + (mu - sigma * phi(z)/Phi(z)) * Phi(z)
    = A * (1 - Phi(z)) + mu * Phi(z) - sigma * phi(z)
    
    Reduction = A - E[min(A, Y)]
    = A - [ A * (1 - Phi(z)) + mu * Phi(z) - sigma * phi(z) ]
    = A * Phi(z) - mu * Phi(z) + sigma * phi(z)
    = (A - mu) * Phi(z) + sigma * phi(z)
    
    Let's check limits again.
    Case 1: mu << A (very good). z = (A-mu)/sigma is large positive.
    Phi(z) ~ 1. phi(z) ~ 0.
    Reduction ~ (A - mu) * 1 + 0 = A - mu.
    This matches the intuition: we expect to improve by A - mu.
    
    Case 2: mu >> A (very bad). z is large negative.
    Phi(z) ~ 0. phi(z) ~ 0.
    Reduction ~ 0.
    This matches intuition: we expect no improvement.
    
    Case 3: mu = A. z = 0.
    Phi(0) = 0.5. phi(0) = 1/sqrt(2*pi) ~ 0.399.
    Reduction ~ 0 + sigma * 0.399.
    Positive, encourages exploration if uncertainty exists.
    
    This formula is correct and efficient.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check
    if time.time() - start_time > 325:
        return -mu

    # Handle edge cases where sigma is 0 or very small
    # If sigma is 0, the distribution is a point mass.
    # If mu < f_best, reduction is f_best - mu.
    # If mu >= f_best, reduction is 0.
    
    # To avoid division by zero in z calculation, we add a small epsilon to sigma
    # or handle separately.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # We need Phi(z) and phi(z)
    # scipy.stats.norm is available, but to be safe and fast, we can use erf
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2/2) / sqrt(2*pi)
    
    sqrt2 = np.sqrt(2.0)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    
    # Compute Phi(z)
    # Using scipy.stats.norm.cdf is standard and fast enough for 2000 points.
    from scipy.stats import norm
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # KG Score = (f_best - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # For points where sigma was originally 0 (or very small), we might have artifacts.
    # If sigma < eps, we should probably just use the deterministic value.
    # But the formula with eps handles it reasonably:
    # If mu < f_best, z is huge positive, Phi~1, phi~0 -> score ~ f_best - mu.
    # If mu > f_best, z is huge negative, Phi~0, phi~0 -> score ~ 0.
    # If mu == f_best, z=0, Phi=0.5, phi~0.4 -> score ~ eps * 0.4 ~ 0.
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # One final check: if the computation took too long (unlikely), return safe fallback
    if time.time() - start_time > 328:
        # Fallback to LCB-like behavior which is fast
        return -mu - 2.0 * sigma

    return scores
# EVOLVE-BLOCK-END
