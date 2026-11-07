"""Cost-aware Knowledge Gradient acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time for the first call in a session
_kg_start_time = None
_KG_DEADLINE_S = 320.0  # Safety margin before 330s limit


def _kg_pair(mu1, s1, mu2, s2):
    """
    Compute E[max(X1, X2) - X1] where X1 ~ N(mu1, s1^2), X2 ~ N(mu2, s2^2).
    This is the expected improvement of the minimum after observing X2,
    given current best is X1.
    
    For minimization: KG(x) = E[min(f_best, f(x))] - min(f_best, f(x))
    But more precisely, if we have current best b, and evaluate x with
    f(x) ~ N(mu_x, sigma_x^2), the expected reduction in best is:
    E[min(b, f(x))] - min(b, E[f(x)])... no.
    
    Actually, KG for minimization:
    Current best observed value is b (deterministic).
    If we evaluate x, new best is min(b, f(x)).
    Expected new best = E[min(b, f(x))] where f(x) ~ N(mu, s^2).
    
    E[min(b, f(x))] = mu * Phi((b-mu)/s) + b * (1 - Phi((b-mu)/s)) - s * phi((b-mu)/s)
    
    Wait, let me derive:
    Let Y = f(x) ~ N(mu, s^2).
    E[min(b, Y)] = integral min(b, y) * (1/s) * phi((y-mu)/s) dy
    
    Split at y=b:
    = integral_{-inf}^{b} y * (1/s) * phi((y-mu)/s) dy + integral_{b}^{inf} b * (1/s) * phi((y-mu)/s) dy
    
    Let z = (y-mu)/s, so y = mu + s*z, dy = s*dz
    When y=b, z = (b-mu)/s = d
    
    First integral: integral_{-inf}^{d} (mu + s*z) * phi(z) dz
    = mu * Phi(d) + s * integral_{-inf}^{d} z*phi(z) dz
    = mu * Phi(d) + s * (-phi(d))  [since integral z*phi(z)dz = -phi(z)]
    = mu * Phi(d) - s * phi(d)
    
    Second integral: b * (1 - Phi(d))
    
    So E[min(b, Y)] = mu * Phi(d) - s * phi(d) + b * (1 - Phi(d))
    
    KG = b - E[min(b, Y)]  (expected reduction in best)
    = b - [mu * Phi(d) - s * phi(d) + b * (1 - Phi(d))]
    = b - mu*Phi(d) + s*phi(d) - b + b*Phi(d)
    = b*Phi(d) - mu*Phi(d) + s*phi(d)
    = (b - mu) * Phi(d) + s * phi(d)
    
    where d = (b - mu) / s
    """
    s = np.maximum(s1, 1e-9)
    d = (mu2 - mu1) / s  # Here mu1 is the candidate's mu, mu2 is b (f_best)
    # Wait, I need to be careful with notation.
    # In our context:
    # mu1 = mu (candidate GP mean)
    # s1 = sigma (candidate GP std)
    # mu2 = f_best (current best, deterministic)
    # s2 = 0 (f_best is deterministic)
    #
    # So X1 = f(x) ~ N(mu1, s1^2), X2 = f_best (constant)
    # E[min(X1, X2)] where X2 is constant = mu2
    # From above: E[min(b, Y)] = mu * Phi(d) - s * phi(d) + b * (1 - Phi(d))
    # where d = (b - mu) / s
    #
    # KG = b - E[min(b, Y)] = (b - mu) * Phi(d) + s * phi(d)
    
    d = (mu2 - mu1) / s
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    kg = (mu2 - mu1) * Phi_d + s * phi_d
    
    return np.maximum(kg, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient (KG) acquisition function.
    
    KG directly estimates the expected reduction in the best observed value
    after evaluating a candidate, which is the theoretically optimal criterion
    for sequential Bayesian optimization. We soften the cost penalty using
    cost^0.8 to allow slightly more expensive evaluations that offer higher
    expected improvement.
    """
    global _kg_start_time
    
    # Initialize deadline tracking
    if _kg_start_time is None:
        _kg_start_time = time.time()
    
    # Check deadline
    elapsed = time.time() - _kg_start_time
    if elapsed > _KG_DEADLINE_S:
        # Return uniform scores to avoid biasing, or zeros
        return np.zeros(len(mu))
    
    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-9)
    
    # Compute Knowledge Gradient: expected reduction in best value
    # mu: GP mean at candidate (shape: 2000,)
    # s: GP std at candidate (shape: 2000,)
    # f_best: scalar, current best observed value
    
    d = (f_best - mu) / s
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    kg = (f_best - mu) * Phi_d + s * phi_d
    
    # Ensure non-negative
    kg = np.maximum(kg, 0.0)
    
    # Cost awareness with softened penalty
    c = np.maximum(cost, 1e-9)
    cost_softened = np.power(c, 0.8)
    
    scores = kg / cost_softened
    
    # Safety: Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
