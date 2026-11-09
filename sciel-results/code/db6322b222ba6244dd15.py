# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS  # range -9..9
_IDX_XM1Y = _XM1Y + 9 # 0..18

# Global state for annealing
_current_alpha = 1.0

def _entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 0.0]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())

def _compute_value(P):
    """Compute the ratio D/M for a given distribution P using hard max."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.bincount(_IDX_XM1Y, weights=p, minlength=19)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.bincount(_XS, weights=p, minlength=10)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.bincount(_XY, weights=p, minlength=19)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.bincount(_X2Y, weights=p, minlength=28)
    h2 = _entropy(h2_bins)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m

def _compute_value_smooth(P, alpha):
    """Compute the ratio D/M_smooth for a given distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.bincount(_IDX_XM1Y, weights=p, minlength=19)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.bincount(_XS, weights=p, minlength=10)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.bincount(_XY, weights=p, minlength=19)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.bincount(_X2Y, weights=p, minlength=28)
    h2 = _entropy(h2_bins)

    if alpha == np.inf:
        m = max(h0, h1, h2)
    else:
        # Soft max: (1/alpha) * log(sum(exp(alpha * Hi)))
        h_vals = np.array([h0, h1, h2])
        max_h = np.max(h_vals)
        # If max_h is 0, result is 0
        if max_h < 1e-15:
            m = 0.0
        else:
            # log-sum-exp trick
            exp_terms = np.exp(alpha * (h_vals - max_h))
            log_sum = np.log(np.sum(exp_terms))
            m = (max_h + log_sum / alpha)
            
    if m <= 1e-9:
        return 0.0
    return hd / m

def _build_from_theta(theta):
    """Convert log-probability parameters to a probability distribution."""
    theta_shifted = theta - np.max(theta)
    p = np.exp(theta_shifted)
    s = p.sum()
    if s <= 1e-30:
        return np.ones(_N * _N) / (_N * _N)
    p = p / s
    return p

def _objective_smooth(theta):
    """Negative of the value we want to maximize, using current alpha."""
    p = _build_from_theta(theta)
    P = p.reshape(_N, _N)
    val = _compute_value_smooth(P, _current_alpha)
    return -val

def _greedy_sparse_search(deadline, rng):
    """
    Greedy search for sparse support sets.
    Try to find good sparse distributions by iteratively adding cells.
    """
    best_P = None
    best_val = 0.0
    n_cells = _N * _N
    
    # Pre-compute cell indices
    all_cells = np.arange(n_cells)
    
    # Try different numbers of support points
    for n_support in [5, 8, 10, 12, 15, 19, 20]:
        if time.time() > deadline - 20:
            break
            
        # Multiple random restarts for greedy
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            # Start with a random single cell
            support = [rng.randint(n_cells)]
            
            # Greedily add cells
            for _ in range(n_support - 1):
                if time.time() > deadline - 10:
                    break
                    
                # Evaluate all candidate cells to add
                best_gain = -np.inf
                best_cell = None
                
                for cell in all_cells:
                    if cell in support:
                        continue
                    
                    # Compute value if we add this cell
                    new_support = support + [cell]
                    P_test = np.zeros(n_cells)
                    for c in new_support:
                        P_test[c] = 1.0
                    P_test = P_test / P_test.sum()
                    P_test_2d = P_test.reshape(_N, _N)
                    val = _compute_value(P_test_2d)
                    
                    if val > best_gain:
                        best_gain = val
                        best_cell = cell
                
                if best_cell is not None:
                    support.append(best_cell)
                
                # Early termination if no improvement possible
                if best_gain < best_val:
                    break
            
            # Evaluate final support
            if len(support) > 0:
                P_final = np.zeros(n_cells)
                for c in support:
                    P_final[c] = 1.0
                P_final = P_final / P_final.sum()
                P_final_2d = P_final.reshape(_N, _N)
                val = _compute_value(P_final_2d)
                
                if val > best_val:
                    best_val = val
                    best_P = P_final_2d.copy()
    
    return best_P, best_val

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Combines greedy sparse search with continuous optimization.
    """
    global _current_alpha
    deadline = time.time() + 550
    n_cells = _N * _N
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Greedy sparse search
    sparse_P, sparse_val = _greedy_sparse_search(deadline, rng)
    
    best_P = sparse_P if sparse_P is not None else np.ones((_N, _N), dtype=np.float64) / n_cells
    best_val = sparse_val if sparse_P is not None else _compute_value(best_P)
    
    # Phase 2: Continuous optimization from multiple starting points
    start_points = []
    
    # Start from uniform
    start_points.append(np.zeros(n_cells))
    
    # Start from sparse solution
    if sparse_P is not None:
        p_sparse = sparse_P.ravel()
        # Convert to log-probabilities
        p_safe = np.maximum(p_sparse, 1e-15)
        start_points.append(np.log(p_safe))
    
    # Random starts
    for _ in range(5):
        start_points.append(rng.randn(n_cells) * 0.5)
    
    # Annealing schedule
    alphas = [1.0, 5.0, 20.0, 100.0, 500.0]
    
    for alpha in alphas:
        if time.time() > deadline - 30:
            break
            
        _current_alpha = alpha
        
        for theta0 in start_points:
            if time.time() > deadline - 10:
                break
                
            result = minimize(
                _objective_smooth, theta0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            p_opt = _build_from_theta(result.x)
            P_opt = p_opt.reshape(_N, _N)
            
            # Evaluate with HARD max to see true score
            val_opt = _compute_value(P_opt)
            
            if val_opt > best_val:
                best_val = val_opt
                best_P = P_opt.copy()
    
    # Final polish with very high alpha
    if time.time() < deadline - 10:
        _current_alpha = 1000.0
        # Convert best_P to log-probabilities for final polish
        p_safe = np.maximum(best_P.ravel(), 1e-15)
        theta0 = np.log(p_safe)
        
        result = minimize(
            _objective_smooth, theta0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
