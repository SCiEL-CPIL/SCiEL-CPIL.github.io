"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc, gammaln
from scipy.stats import norm
import time as _time

_DEADLINE_S = 280.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _student_t_ei(mu, sigma, f_best, nu=3.0):
    """
    Expected Improvement under a Student-t marginal for minimization.
    
    If y ~ t_nu(mu, sigma^2), then EI = E[max(f_best - y, 0)].
    
    For a t distribution with nu degrees of freedom, location mu, scale sigma:
    EI = (f_best - mu) * P(y < f_best) + E[y * 1(y < f_best)]
    
    We use a numerical quadrature-free approximation via the fact that
    for t_nu, the CDF and PDF can be computed, and we use the identity:
    EI = (f_best - mu) * F_t(z) + sigma * G_t(z)
    where z = (f_best - mu)/sigma, F_t is the t CDF, and G_t(z) = E[t * 1(t < z)] / 1
    ... this gets complex. Instead, we use a robust approximation:
    
    For small nu (heavy tails), EI is more sensitive to tail uncertainty.
    We approximate: EI_t ≈ EI_gaussian * (1 + 1/nu) * (1 + 0.5 * z^2 / nu)
    clipped to be non-negative.
    
    Actually, let's use a cleaner approach: compute EI via the t-distribution
    using the relationship that a t_nu variable can be written as N(0,1)/sqrt(Chi2/nu).
    But for vectorized 2000-point evaluation, let's use a practical approximation.
    
    Better: Use the fact that for t_nu, the CDF is available via scipy, and we can
    compute the truncated first moment numerically for the scalar case, but for 2000
    points we need speed. Let's use a Taylor expansion around the Gaussian EI.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    # Gaussian EI as baseline
    cdf = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf = norm.pdf(z)
    ei_gauss = d * cdf + sigma_safe * pdf
    
    # Student-t correction: for nu degrees of freedom, the t-distribution has
    # heavier tails. The EI under t is larger when z is in the tail region.
    # A practical approximation: multiply by a factor that accounts for kurtosis.
    # For t_nu, excess kurtosis = 6/(nu-4) for nu > 4.
    # We use a simple scaling: EI_t ≈ EI_gauss * (nu / (nu - 2)) for nu > 2
    # This is an upper bound adjustment.
    if nu > 2:
        t_factor = nu / (nu - 2.0)
    else:
        t_factor = 1.5  # cap for small nu
    
    # Also, when sigma is large relative to d (high uncertainty), the t-distribution
    # gives more weight to potential large improvements.
    # Add a small bonus proportional to sigma^2 / (1 + z^2) for heavy-tail effect.
    heavy_tail_bonus = 0.3 * sigma_safe / (1.0 + z * z)
    
    ei_t = ei_gauss * t_factor + heavy_tail_bonus
    ei_t = np.maximum(ei_t, 0.0)
    
    # Where sigma is very small, EI is essentially max(d, 0)
    ei_t = np.where(sigma < eps, np.maximum(d, 0.0), ei_t)
    
    return ei_t


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]

    # Deadline guard
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    # Compute Student-t robust EI
    # nu=3 gives moderate heavy tails; robust to early GP miscalibration
    ei = _student_t_ei(mu, sigma, f_best, nu=3.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Adaptive cost weighting:
    # The total budget is 30. We've already spent some cost.
    # cost is the cost of evaluating each candidate (in [1, 2]).
    # We don't know the exact remaining budget, but we can infer it from
    # the number of calls made (approximated by the magnitude of f_best
    # and the GP uncertainty).
    # 
    # Structural idea: The cost exponent should increase as we get closer
    # to budget exhaustion. Since we can't directly know remaining budget,
    # we use the following heuristic:
    # - Early in optimization: f_best is far from optimum, sigma is large.
    #   We can afford more expensive points. Use lower cost exponent.
    # - Late: f_best is close to optimum, sigma is small in explored regions.
    #   We should be very cost-efficient. Use higher cost exponent.
    #
    # Proxy for "how far along": the ratio of current best improvement to
    # total possible improvement. But we don't know the optimum.
    # Alternative: use the mean sigma as a proxy. High mean sigma = early.
    #
    # Let's use: alpha = 0.5 + 1.0 * (1 - mean_sigma / (mean_sigma + 0.1))
    # This gives alpha in [0.5, 1.5]. When sigma is large (early), alpha ~ 0.5.
    # When sigma is small (late), alpha ~ 1.5.
    
    mean_sigma = np.mean(sigma)
    # Normalize: when mean_sigma is large, we're early; when small, we're late.
    # Use a logistic-like transition centered at sigma ~ 0.1 (typical scale).
    sigma_norm = mean_sigma / (mean_sigma + 0.1)  # in (0, 1), 1 = large sigma
    # alpha: 0.5 when sigma_norm=1 (early), 1.5 when sigma_norm=0 (late)
    alpha_cost = 0.5 + 1.0 * (1.0 - sigma_norm)
    # Clamp to reasonable range
    alpha_cost = np.clip(alpha_cost, 0.4, 1.6)
    
    cost_safe = np.maximum(cost, 1.0)
    cost_factor = cost_safe ** alpha_cost
    
    ei_cost = ei / cost_factor
    
    # Additional exploration term: probability of improvement (PI) under t-distribution
    # PI = P(y < f_best) = F_t(z). For t, this is more spread out.
    # Use PI as a bounded exploration bonus.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    # Approximate t CDF using Gaussian CDF scaled (crude but fast)
    pi_approx = 0.5 * erfc(-z / np.sqrt(2.0))  # Gaussian CDF as proxy
    # Scale PI to be in [0, 1] and use as small bonus
    pi_bonus = 0.05 * pi_approx / cost_safe
    
    score = ei_cost + pi_bonus
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
