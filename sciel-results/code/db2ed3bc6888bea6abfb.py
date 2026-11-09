# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses a parametric ansatz P(x,y) ∝ exp(-α x² - β y² - γ xy) and optimizes
    (α, β, γ) via multi-start L-BFGS-B to maximize D/M where D = H_{-1} and M = max(H_0, H_1, H_2).

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    x_vals = np.arange(n)
    y_vals = np.arange(n)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    # P[y, x] so rows are y, columns are x
    X_grid = X  # shape (10, 10), X_grid[y, x] = x
    Y_grid = Y  # shape (10, 10), Y_grid[y, x] = y

    # Precompute projection indices for each slope
    slopes = [-1.0, 0.0, 1.0, 2.0]
    proj_indices = {}
    for slope in slopes:
        vals = X_grid + slope * Y_grid
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        proj_indices[slope] = inverse

    def compute_P(alpha, beta, gamma):
        """Compute normalized P given parameters."""
        logP = -alpha * X_grid**2 - beta * Y_grid**2 - gamma * X_grid * Y_grid
        # For numerical stability
        logP -= np.max(logP)
        P = np.exp(logP)
        s = np.sum(P)
        if s <= 0 or not np.isfinite(s):
            return None
        P /= s
        return P

    def entropy_of_projection(P, slope):
        """Compute Shannon entropy in bits of the projection x + slope*y."""
        inverse = proj_indices[slope]
        n_unique = len(np.unique(inverse))
        proj_probs = np.zeros(n_unique)
        np.add.at(proj_probs, inverse.ravel(), P.ravel())
        # Ignore zero probabilities
        nonzero = proj_probs > 1e-15
        if not np.any(nonzero):
            return 0.0
        p = proj_probs[nonzero]
        p = p / np.sum(p)  # renormalize
        return -np.sum(p * np.log2(p))

    def ratio_D_over_M(params):
        """Compute D/M where D = H_{-1} and M = max(H_0, H_1, H_2)."""
        alpha, beta, gamma = params
        P = compute_P(alpha, beta, gamma)
        if P is None:
            return 0.0
            
        D = entropy_of_projection(P, -1.0)
        H0 = entropy_of_projection(P, 0.0)
        H1 = entropy_of_projection(P, 1.0)
        H2 = entropy_of_projection(P, 2.0)
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return 0.0
            
        return D / M

    def objective(params):
        """Compute -D/M for minimization with penalty."""
        val = ratio_D_over_M(params)
        if val == 0.0:
            return 1e6  # Penalty for invalid
        return -val

    # Define bounds for the parameters
    # Widen bounds to allow for sharper distributions
    bounds = [(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)]

    # Multi-start optimization with L-BFGS-B
    rng = np.random.RandomState(42)
    best_score = -1e6
    best_params = None

    # Try several random starting points
    n_starts = 100
    for i in range(n_starts):
        # Random start within bounds
        x0 = np.array([
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1])
        ])
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
        
        score = ratio_D_over_M(result.x)
        if score > best_score:
            best_score = score
            best_params = result.x

    # Also try some structured starts
    structured_starts = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.5, -1.0],
        [2.0, 2.0, 0.0],
        [-2.0, -2.0, 0.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, -2.0],
        [3.0, 3.0, 1.0],
        [5.0, 5.0, -5.0],
        [-5.0, -5.0, 5.0],
        [10.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, -10.0],
    ]
    
    for x0 in structured_starts:
        x0 = np.array(x0, dtype=np.float64)
        # Clamp to bounds
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
        
        score = ratio_D_over_M(result.x)
        if score > best_score:
            best_score = score
            best_params = result.x

    # Refine the best found parameters with a second pass of optimization
    if best_params is not None:
        result = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16})
        score = ratio_D_over_M(result.x)
        if score > best_score:
            best_score = score
            best_params = result.x

    # Compute final P
    if best_params is None:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P = compute_P(*best_params)
        if best_P is None:
            best_P = np.ones((n, n), dtype=np.float64) / (n * n)
        else:
            # Clip and normalize
            best_P = np.clip(best_P, 0, None)
            s = np.sum(best_P)
            if s <= 1e-9:
                best_P = np.ones((n, n), dtype=np.float64) / (n * n)
            else:
                best_P /= s

    return best_P

# EVOLVE-BLOCK-END
