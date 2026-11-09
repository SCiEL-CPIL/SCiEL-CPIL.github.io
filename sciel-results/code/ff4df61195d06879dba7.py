# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_P(w, support):
    """Compute P[X1+X2+X3 < 2*X4] for given weights and support."""
    n = len(support)
    w = np.asarray(w)
    w = w / w.sum()
    
    # Precompute pairwise sums
    pair_sums = support[:, None] + support[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            mask = pair_sums < threshold
            Q_kl = np.sum(w[:, None] * w[None, :] * mask)
            P += w[k] * w[l] * Q_kl
    
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Uses a sparse support and optimizes the weights via SLSQP.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    # Design a sparse support - try various configurations
    # Start with a geometric progression
    support = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    n = len(support)
    
    def objective(w):
        return -_compute_P(w, support)
    
    # Initial guess: uniform weights
    w0 = np.ones(n) / n
    
    # Constraints: sum(w) = 1, w >= 0
    bounds = [(0, None)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-15}
    )
    
    w_opt = result.x
    w_opt = np.clip(w_opt, 0, None)
    w_opt = w_opt / w_opt.sum()
    
    return support, w_opt


# EVOLVE-BLOCK-END
