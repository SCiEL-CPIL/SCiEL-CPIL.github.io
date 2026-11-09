# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / total


def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    
    # D: H of x - y (slope -1)
    vals_d = X - Y
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = 0.0
    for c in counts_d:
        if c > 0:
            H_d -= c * np.log2(c)
    
    # M: max of H for slopes 0, 1, 2
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        counts = np.zeros(n + r * (n - 1) + 1)
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        for v in range(int(min_val), int(max_val) + 1):
            mask = (vals == v)
            counts[v - min_val] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _compute_ratio_neg(P):
    """Negative ratio for minimization."""
    return -_compute_ratio(P)


def _reconstruct_from_anti_diag(anti_diag_probs, coupling=0.5):
    """
    Reconstruct a 10x10 P from anti-diagonal (x-y) marginal probabilities.
    
    anti_diag_probs: array of 19 values, summing to 1, where index v corresponds to x-y = v-9
    coupling: parameter controlling how much weight goes to diagonal vs off-diagonal
    
    Strategy: For each anti-diagonal value d = x-y, distribute the probability mass
    across cells on that diagonal. Use a coupling parameter to control concentration.
    """
    n = 10
    P = np.zeros((n, n))
    
    for v in range(19):
        d = v - 9  # x - y = d
        prob = anti_diag_probs[v]
        if prob <= 0:
            continue
        
        # Find all cells where x - y = d
        cells = []
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                cells.append((y, x))
        
        if not cells:
            continue
        
        # Distribute prob across cells on this diagonal
        # Use a simple uniform distribution within each diagonal
        # This gives maximum entropy within each diagonal
        cell_prob = prob / len(cells)
        for (y, x) in cells:
            P[y, x] += cell_prob
    
    return _clip_and_normalize(P)


def _reconstruct_with_structure(anti_diag_probs, structure_type='diagonal'):
    """
    More sophisticated reconstruction that uses structure to maximize D while minimizing M.
    
    Key insight: To maximize D/M, we want:
    - D (entropy of x-y) to be high: spread mass across many x-y values
    - M (max of H_x, H_x+y, H_x+2y) to be low: concentrate mass in projections
    
    Strategy: Put mass on cells that have distinct x-y values but similar x, x+y, x+2y values.
    """
    n = 10
    P = np.zeros((n, n))
    
    if structure_type == 'diagonal':
        # Simple: uniform within each anti-diagonal
        return _reconstruct_from_anti_diag(anti_diag_probs)
    
    elif structure_type == 'corner':
        # Put mass on corners to minimize x, x+y, x+2y entropy
        # Corners: (0,0), (0,9), (9,0), (9,9)
        # x-y values: 0, -9, 9, 0
        # x values: 0, 0, 9, 9
        # x+y values: 0, 9, 9, 18
        # x+2y values: 0, 18, 9, 27
        # This gives D with 3 distinct values, M with 4 distinct values
        corner_cells = [(0,0), (0,9), (9,0), (9,9)]
        # Map anti-diag probs to corners
        # v=0: d=-9 -> (0,9)
        # v=9: d=0 -> (0,0) and (9,9)
        # v=18: d=9 -> (9,0)
        P[0, 0] = anti_diag_probs[9] * 0.5
        P[9, 9] = anti_diag_probs[9] * 0.5
        P[0, 9] = anti_diag_probs[0]
        P[9, 0] = anti_diag_probs[18]
        # Distribute remaining mass
        remaining = 1.0 - P.sum()
        if remaining > 1e-9:
            # Add to center for balance
            P[4, 4] += remaining * 0.5
            P[5, 5] += remaining * 0.5
        return _clip_and_normalize(P)
    
    elif structure_type == 'mixed':
        # Hybrid: use anti-diagonal structure but bias toward corners
        P = _reconstruct_from_anti_diag(anti_diag_probs)
        # Add corner bias
        corner_boost = 0.3
        P[0, 0] *= (1 + corner_boost)
        P[0, 9] *= (1 + corner_boost)
        P[9, 0] *= (1 + corner_boost)
        P[9, 9] *= (1 + corner_boost)
        return _clip_and_normalize(P)
    
    return _clip_and_normalize(P)


def _optimize_anti_diag(anti_diag_init, structure_type='diagonal', time_limit=30):
    """
    Optimize the 19 anti-diagonal probabilities to maximize D/M.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 0.0
        # Ensure params are valid probabilities
        params = np.abs(params)
        total = params.sum()
        if total < 1e-12:
            return 0.0
        params = params / total
        
        P = _reconstruct_with_structure(params, structure_type)
        return -_compute_ratio(P)
    
    # Initial point
    x0 = anti_diag_init.copy()
    x0 = np.abs(x0)
    x0 = x0 / x0.sum()
    
    # Constraints: sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    # Bounds: each prob in [0, 1]
    bounds = [(0.0, 1.0)] * 19
    
    # Try SLSQP
    try:
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        if result.success and result.fun < 0:
            optimal_params = np.abs(result.x)
            optimal_params = optimal_params / optimal_params.sum()
            return optimal_params
    except Exception:
        pass
    
    # Fallback: Nelder-Mead
    try:
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if result.fun < 0:
            optimal_params = np.abs(result.x)
            optimal_params = optimal_params / optimal_params.sum()
            return optimal_params
    except Exception:
        pass
    
    return x0


def _optimize_full_P(P_init, time_limit=20):
    """
    Optimize the full 10x10 P directly using SLSQP.
    """
    start_time = time.time()
    P_flat = P_init.flatten().copy()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 0.0
        P = params.reshape(10, 10)
        P = _clip_and_normalize(P)
        return -_compute_ratio(P)
    
    # Constraints: sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0.0, 1.0)] * 100
    
    try:
        result = minimize(
            objective, P_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-7}
        )
        if result.success and result.fun < 0:
            P_opt = np.clip(result.x.reshape(10, 10), 0, None)
            P_opt = P_opt / P_opt.sum()
            return P_opt
    except Exception:
        pass
    
    return P_init


def _random_anti_diag_init(method='uniform'):
    """Generate random initial anti-diagonal probabilities."""
    if method == 'uniform':
        return np.ones(19) / 19.0
    elif method == 'dirichlet':
        alpha = np.ones(19) * 2.0  # Slightly concentrated
        return np.random.dirichlet(alpha)
    elif method == 'corner_focused':
        # Emphasize corners (d=-9, 0, 9)
        probs = np.ones(19) * 0.1
        probs[0] = 1.0   # d=-9
        probs[9] = 1.0   # d=0
        probs[18] = 1.0  # d=9
        return probs / probs.sum()
    elif method == 'spread':
        # Emphasize spread across all diagonals
        probs = np.exp(-np.abs(np.arange(19) - 9) * 0.1)
        return probs / probs.sum()
    else:
        return np.random.dirichlet(np.ones(19))


def _generate_structured_P(method='anti_diag_uniform'):
    """Generate a structured initial P."""
    n = 10
    
    if method == 'uniform':
        return np.ones((n, n)) / (n * n)
    
    elif method == 'anti_diag_uniform':
        anti_diag = np.ones(19) / 19.0
        return _reconstruct_from_anti_diag(anti_diag)
    
    elif method == 'corner':
        P = np.zeros((n, n))
        P[0, 0] = 0.25
        P[0, 9] = 0.25
        P[9, 0] = 0.25
        P[9, 9] = 0.25
        return _clip_and_normalize(P)
    
    elif method == 'diagonal':
        P = np.zeros((n, n))
        for i in range(n):
            P[i, i] = 1.0 / n
        return _clip_and_normalize(P)
    
    elif method == 'random':
        P = np.random.dirichlet(np.ones(n * n))
        return P.reshape(n, n)
    
    else:
        return np.ones((n, n)) / (n * n)


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses a combination of anti-diagonal parameterization and direct optimization.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = 570.0  # seconds, with safety margin
    
    np.random.seed(42)
    
    # Track best
    best_P = np.ones((n, n)) / (n * n)
    best_ratio = _compute_ratio(best_P)
    
    # Phase 1: Optimize anti-diagonal parameterization
    structure_types = ['diagonal', 'corner', 'mixed']
    init_methods = ['uniform', 'dirichlet', 'corner_focused', 'spread']
    
    for struct in structure_types:
        if time.time() - start_time > deadline:
            break
        
        for init_method in init_methods:
            if time.time() - start_time > deadline:
                break
            
            anti_diag_init = _random_anti_diag_init(init_method)
            anti_diag_opt = _optimize_anti_diag(anti_diag_init, struct, time_limit=15)
            P_candidate = _reconstruct_with_structure(anti_diag_opt, struct)
            ratio = _compute_ratio(P_candidate)
            
            if ratio > best_ratio + 1e-9:
                best_P = P_candidate.copy()
                best_ratio = ratio
    
    # Phase 2: Direct optimization from best found so far
    if time.time() - start_time < deadline - 30:
        P_optimized = _optimize_full_P(best_P, time_limit=30)
        ratio = _compute_ratio(P_optimized)
        if ratio > best_ratio + 1e-9:
            best_P = P_optimized.copy()
            best_ratio = ratio
    
    # Phase 3: More direct optimization from different starts
    for init_method in ['uniform', 'corner', 'diagonal', 'random']:
        if time.time() - start_time > deadline - 20:
            break
        
        P_init = _generate_structured_P(init_method)
        P_optimized = _optimize_full_P(P_init, time_limit=20)
        ratio = _compute_ratio(P_optimized)
        
        if ratio > best_ratio + 1e-9:
            best_P = P_optimized.copy()
            best_ratio = ratio
    
    # Final validation
    best_P = _clip_and_normalize(best_P)
    return best_P


# EVOLVE-BLOCK-END
