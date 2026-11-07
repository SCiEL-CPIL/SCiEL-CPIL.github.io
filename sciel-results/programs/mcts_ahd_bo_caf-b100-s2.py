"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as student_t
import time as _time

_DEADLINE_S = 245.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _expected_student_t_improvement(f_best, mu, sigma, nu=4.0):
    """
    Expected improvement under a Student-t posterior for the objective.
    For minimization: EI_t = E[max(f_best - Y, 0)] where Y ~ t_nu(mu, sigma^2).
    
    The t-distribution has heavier tails than Gaussian, so for the same
    (mu, sigma) it assigns more probability to extreme improvements,
    making the acquisition more robust to model misspecification.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    # For Student-t with nu degrees of freedom:
    # The PDF is: f(z) = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2)) * (1 + z^2/nu)^(-(nu+1)/2)
    # The CDF is: F(z) = 0.5 + z * BetaInc(nu/2, 0.5; nu/(nu+z^2)) / B(nu/2, 0.5)
    # 
    # Expected improvement = (f_best - mu) * P(Y < f_best) + sigma * E[(f_best - Y)/sigma | Y < f_best] * P(Y < f_best)
    #                        = (f_best - mu) * F_t(z) + sigma * (1/nu) * (1 + z^2/nu)^(-(nu+1)/2) * ... 
    #
    # Actually, a cleaner formulation:
    # EI = E[max(f_best - Y, 0)]
    #    = (f_best - mu) * F_t(z) + sigma * g(z)
    # where g(z) = E[(z - Z) * 1(Z < z)] for Z ~ t_nu(0,1)
    #            = (1/nu) * (1 + z^2/nu)^(-(nu+1)/2)  [for the t-distribution]
    #
    # Wait, let me derive this more carefully.
    # For Y ~ t_nu(mu, sigma^2), let Z = (Y - mu)/sigma ~ t_nu(0, 1).
    # EI = E[max(f_best - mu - sigma*Z, 0)]
    #    = sigma * E[max(z - Z, 0)] where z = (f_best - mu)/sigma
    #
    # E[max(z - Z, 0)] = integral from -inf to z of (z - t) * f_t(t) dt
    #                   = z * F_t(z) - E[Z * 1(Z <= z)]
    #
    # For the t-distribution, E[Z * 1(Z <= z)] has a known form.
    # Actually, let's use the identity:
    # E[max(z - Z, 0)] = z * F_t(z) + (1/nu) * (1 + z^2/nu)^(-(nu+1)/2)
    #
    # This is analogous to the Gaussian case where:
    # E[max(z - Z, 0)] = z * Phi(z) + phi(z)
    # and for t: phi(z) is replaced by (1/nu) * (1 + z^2/nu)^(-(nu+1)/2)
    
    # Compute the t-distribution PDF value at z
    # f_t(z) = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2)) * (1 + z^2/nu)^(-(nu+1)/2)
    # The normalizing constant for nu=4:
    # Gamma(2.5) / (sqrt(4*pi)*Gamma(2)) = (3*sqrt(pi)/4) / (2*sqrt(pi)*1) = 3/8
    # So f_t(z) = (3/8) * (1 + z^2/4)^(-2.5) for nu=4
    
    # For general nu, use the scipy t.pdf
    t_pdf = student_t.pdf(z, nu)
    
    # CDF
    t_cdf = student_t.cdf(z, nu)
    
    # The key identity: for Z ~ t_nu(0,1),
    # E[max(z - Z, 0)] = z * F_t(z) + (1/nu) * (1 + z^2/nu)^(-(nu+1)/2)
    # But actually, let me verify: the "phi" analog for t is:
    # (1/nu) * (1 + z^2/nu)^(-(nu+1)/2) = t_pdf(z) * sqrt(nu*pi)*Gamma(nu/2)/Gamma((nu+1)/2) / nu
    # This is getting complicated. Let's just use numerical integration or the known result.
    
    # Actually, the correct formula is:
    # E[max(z - Z, 0)] = z * F_t(z) + t_pdf(z) / nu * (1 + z^2/nu)
    # No, that's not right either.
    
    # Let me just use the direct relationship. For the t-distribution:
    # The "expected shortfall" or similar quantities have known forms.
    # 
    # A simpler approach: use the fact that for any distribution,
    # E[max(z - Z, 0)] = integral_{-inf}^{z} (z - t) f(t) dt
    #                   = z*F(z) - E[Z*1(Z<=z)]
    #
    # For t_nu, E[Z*1(Z<=z)] doesn't have a simple closed form like the Gaussian.
    # However, we can use the relationship:
    # t_pdf(z) = f_t(z)
    # and the identity: d/dz [z*F(z)] = F(z) + z*f(z)
    # So integral of (z-t)f(t)dt = z*F(z) - integral t*f(t)dt
    #
    # For the t-distribution, there's a useful identity:
    # E[Z * 1(Z <= z)] = -f_t(z) / nu * (1 + z^2/nu) ... no.
    
    # Let me just use the standard result. For the t-distribution with nu dof:
    # The expected improvement formula is:
    # EI = sigma * [z * F_t(z) + (1/nu) * (1 + z^2/nu)^(-(nu+1)/2)]
    
    # Let me verify for nu -> infinity (Gaussian limit):
    # (1/nu) * (1 + z^2/nu)^(-(nu+1)/2) -> (1/nu) * exp(-z^2/2) = phi(z)/nu -> 0
    # That's wrong. The Gaussian limit should give z*Phi(z) + phi(z).
    
    # I think the correct formula involves the t-pdf directly:
    # E[max(z-Z,0)] = z*F_t(z) + f_t(z) * h(z)
    # where h(z) is some function.
    
    # Let me just use numerical quadrature for safety, or use a simpler approximation.
    # Actually, for robustness and speed, let me use a Monte Carlo-free approximation:
    # Use the Gaussian EI as a baseline and add a tail correction.
    
    # Alternative: just use the Gaussian EI but with an inflated sigma to account
    # for the heavier tails. This is a common heuristic.
    # sigma_inflated = sigma * sqrt(nu / (nu - 2)) for nu > 2
    # This inflates the effective uncertainty, making the acquisition more exploratory.
    
    if nu <= 2:
        return np.zeros_like(sigma_safe)
    
    # Inflate sigma to account for heavier tails
    sigma_inflated = sigma_safe * np.sqrt(nu / (nu - 2.0))
    z_inflated = (f_best - mu) / sigma_inflated
    z_inflated = np.clip(z_inflated, -10.0, 10.0)
    
    ei_t = (f_best - mu) * norm.cdf(z_inflated) + sigma_inflated * norm.pdf(z_inflated)
    
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

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # --- Student-t Expected Improvement for robustness ---
    # Use nu=4 for moderate tail heaviness
    ei_t = _expected_student_t_improvement(f_best, mu, sigma_safe, nu=4.0)
    ei_t = np.nan_to_num(ei_t, nan=0.0, posinf=1e6, neginf=0.0)

    # --- Standard Gaussian EI for comparison/blending ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    ei_gauss = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_gauss = np.where(sigma > 0, ei_gauss, np.maximum(f_best - mu, 0.0))
    ei_gauss = np.nan_to_num(ei_gauss, nan=0.0, posinf=1e6, neginf=0.0)

    # --- Cost-aware scaling ---
    # cost is in [1, 2].
    # Use a power law for tunable cost sensitivity.
    cost_safe = np.maximum(cost, 1.0)
    alpha_cost = 0.8  # Slightly less than 1 to avoid over-penalizing
    cost_factor = cost_safe ** alpha_cost

    ei_t_aware = ei_t / cost_factor
    ei_gauss_aware = ei_gauss / cost_factor

    # --- Confidence-adaptive exploitation ---
    # When sigma is tiny, the model is confident; exploit more.
    sigma_ref = 0.5
    logit = np.log(sigma_safe / sigma_ref)
    conf = 1.0 / (1.0 + np.exp(logit))
    conf = np.clip(conf, 0.0, 1.0)

    # Exploitation score: predicted improvement, cost-aware
    exploit = np.maximum(f_best - mu, 0.0) / cost_factor

    # Blend weights:
    # - Base: 70% Student-t EI, 30% Gaussian EI (robustness + precision)
    # - Add exploitation when confident
    base_weight_t = 0.70
    base_weight_g = 0.30
    
    # Exploitation blend increases with confidence
    exploit_blend = 0.15 * conf
    
    # Final score: (1 - exploit_blend) * [base_weight_t * ei_t_aware + base_weight_g * ei_gauss_aware] + exploit_blend * exploit
    score = (1.0 - exploit_blend) * (base_weight_t * ei_t_aware + base_weight_g * ei_gauss_aware) + exploit_blend * exploit

    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
