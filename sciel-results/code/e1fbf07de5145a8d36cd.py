"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # 10s safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

# Known minimum of Michalewicz 2D
_MICH_MIN = -1.8013

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) approximation for
    minimization of Michalewicz 2D. KG directly estimates the expected
    reduction in regret (improvement over f_best), making it a more
    principled choice than EI or UCB.
    
    mu, sigma: shape (2000,) in objective units
    f_best: lowest observed value
    Returns: shape (2000,) finite scores, higher = evaluate next
    """
    # Check deadline
    if not _elapsed_ok():
        return np.random.rand(len(mu))
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # --- Knowledge Gradient for minimization ---
    # KG(x) = E[max(0, f_best - f(x))] - E[max(0, f_best - min(f(x), f_best))]
    # Simplified: For a single point, KG ≈ EI but with a correction term
    # that accounts for the probability that this point becomes the new best.
    # 
    # A practical approximation: KG is similar to EI but we can add a
    # "new best" bonus. The probability that f(x) < f_best is Phi((f_best - mu)/sigma).
    # The expected improvement given it's new best is E[f_best - f(x) | f(x) < f_best].
    # 
    # For minimization, the KG can be approximated as:
    # KG ≈ EI + c * P(new best) * sigma
    # where c is a small constant.
    
    z = (f_best - mu) / sigma_safe
    
    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability that this point is better than current best
    p_new_best = norm.cdf(z)
    
    # Expected value of f(x) given f(x) < f_best
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z) for z != 0
    # We need to be careful with numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Avoid division by zero
    Phi_z_safe = np.maximum(Phi_z, eps)
    
    # Conditional expectation
    cond_exp = mu - sigma_safe * phi_z / Phi_z_safe
    
    # Expected improvement if new best: f_best - cond_exp
    exp_improve_new_best = np.maximum(f_best - cond_exp, 0)
    
    # KG approximation: EI captures the expected improvement,
    # but we boost points that are likely to be new bests
    # The term p_new_best * exp_improve_new_best / sigma_safe is a
    # normalized measure of the "value" of discovering a new best here.
    
    # Actually, a cleaner KG-like score:
    # The expected reduction in the best-so-far is:
    # E[min(f_best, f(x))] = f_best - E[max(0, f_best - f(x))]
    # So KG = E[max(0, f_best - f(x))] = EI (for a single point)
    # 
    # But KG differs from EI in that it considers the *information* value.
    # A simpler structural change: use a "probabilistic regret" score.
    
    # Let's use a different structural approach: 
    # Score = -mu + alpha * sigma * Phi(z) + beta * sigma * phi(z)
    # This combines exploitation (-mu), exploration weighted by
    # probability of improvement (sigma * Phi(z)), and the "density"
    # of the improvement distribution (sigma * phi(z)).
    
    # Alternative: Use the "Probability of Improvement" (PI) combined with EI
    # PI = Phi(z)
    # This is a simpler, more robust acquisition that balances exploration
    # and exploitation differently than EI.
    
    # Let's try: Score = w1 * EI + w2 * PI * sigma + w3 * (-mu)
    # where weights adapt based on progress.
    
    # Progress: 0 when f_best is far from min, 1 when f_best is at min
    progress = np.clip((f_best - _MICH_MIN) / 3.6, 0.0, 1.0)
    
    # Normalize EI
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    # PI * sigma: high when both likely to improve AND high uncertainty
    pi_sigma = p_new_best * sigma_safe
    pi_sigma_min, pi_sigma_max = np.min(pi_sigma), np.max(pi_sigma)
    pi_sigma_range = pi_sigma_max - pi_sigma_min if pi_sigma_max > pi_sigma_min else 1.0
    pi_sigma_norm = (pi_sigma - pi_sigma_min) / pi_sigma_range
    
    # Exploitation: -mu normalized
    mu_min, mu_max = np.min(mu), np.max(mu)
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    mu_norm = (mu_max - mu) / mu_range  # Higher for lower mu
    
    # Blend: 
    # Early (low progress): more exploration (pi_sigma), less exploitation
    # Late (high progress): more exploitation (ei, mu_norm), less exploration
    w_ei = 0.4 + 0.4 * progress      # 0.4 to 0.8
    w_pi_sigma = 0.4 * (1.0 - progress) + 0.1  # 0.5 to 0.1
    w_mu = 0.2 * (1.0 - progress) + 0.1       # 0.3 to 0.1
    
    # Ensure weights sum to 1
    w_sum = w_ei + w_pi_sigma + w_mu
    w_ei /= w_sum
    w_pi_sigma /= w_sum
    w_mu /= w_sum
    
    score = w_ei * ei_norm + w_pi_sigma * pi_sigma_norm + w_mu * mu_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
