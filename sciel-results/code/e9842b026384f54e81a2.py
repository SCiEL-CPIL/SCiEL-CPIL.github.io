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

def _fast_greedy_sparse(deadline, rng):
    """
    Fast greedy search for sparse support sets.
    Uses random sampling instead of exhaustive search.
    """
    best_P = None
    best_val = 0.0
    n_cells = _N * _N
    all_cells = np.arange(n_cells)
    
    # Try different numbers of support points with fewer restarts
    for n_support in [8, 12, 15, 19]:
        if time.time() > deadline - 30:
            break
            
        for restart in range(2):
            if time.time() > deadline - 15:
                break
                
            # Start with a random single cell
            support = [rng.randint(n_cells)]
            
            # Greedily add cells using random candidate sampling
            for step in range(n_support - 1):
                if time.time() > deadline - 15:
                    break
                    
                best_gain = -np.inf
                best_cell = None
                
                # Sample 20 random candidates instead of all 100
                candidates = rng.choice(all_cells, size=min(20, n_cells - len(support)), replace=False)
                
                for cell in candidates:
                    if cell in support:
                        continue
                    
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

def _coordinate_polish(P, deadline, max_iters=200):
    """
    Coordinate descent polish on the hard objective.
    Try moving probability mass between pairs of cells.
    """
    n_cells = _N * _N
    p = P.ravel().copy()
    best_val = _compute_value(P)
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        improved = False
        
        # Try moving mass from each cell to each other cell
        for i in range(n_cells):
            if time.time() > deadline - 5:
                break
            if p[i] < 1e-15:
                continue
                
            for j in range(n_cells):
                if i == j:
                    continue
                if p[j] < 1e-15 and rng_random_check(it, i, j) > 0.9:
                    continue
                    
                # Try moving a small fraction of mass from i to j
                for frac in [0.1, 0.3, 0.5]:
                    p_new = p.copy()
                    p_new[i] = p[i] * (1 - frac)
                    p_new[j] = p[j] + p[i] * frac
                    total = p_new.sum()
                    if total <= 1e-30:
                        continue
                    p_new = p_new / total
                    P_new = p_new.reshape(_N, _N)
                    val_new = _compute_value(P_new)
                    
                    if val_new > best_val + 1e-12:
                        p = p_new.copy()
                        best_val = val_new
                        improved = True
                        break
        
        if not improved:
            break
    
    P_out = p.reshape(_N, _N)
    return P_out, best_val

def rng_random_check(it, i, j):
    """Deterministic pseudo-random for skipping some pairs."""
    return ((it * 7919 + i * 104729 + j * 1299709) % 10000) / 10000.0

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Combines fast greedy sparse search with continuous optimization and coordinate polish.
    """
    global _current_alpha
    deadline = time.time() + 550
    n_cells = _N * _N
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Fast greedy sparse search (limited time)
    greedy_deadline = time.time() + 30
    sparse_P, sparse_val = _fast_greedy_sparse(greedy_deadline, rng)
    
    best_P = sparse_P if sparse_P is not None else np.ones((_N, _N), dtype=np.float64) / n_cells
    best_val = sparse_val if sparse_P is not None else _compute_value(best_P)
    
    # Phase 2: Continuous optimization from multiple starting points
    start_points = []
    
    # Start from uniform
    start_points.append(np.zeros(n_cells))
    
    # Start from sparse solution
    if sparse_P is not None:
        p_sparse = sparse_P.ravel()
        p_safe = np.maximum(p_sparse, 1e-15)
        start_points.append(np.log(p_safe))
    
    # More diverse random starts
    for _ in range(8):
        start_points.append(rng.randn(n_cells) * 1.0)
    
    # Also try structured starts: diagonal, anti-diagonal, corners
    diag = np.zeros(n_cells)
    for i in range(_N):
        diag[i * _N + i] = 1.0
    start_points.append(np.log(np.maximum(diag, 1e-15)))
    
    anti_diag = np.zeros(n_cells)
    for i in range(_N):
        anti_diag[i * _N + (_N - 1 - i)] = 1.0
    start_points.append(np.log(np.maximum(anti_diag, 1e-15)))
    
    # Annealing schedule - fewer, more aggressive steps
    alphas = [5.0, 20.0, 100.0, 500.0]
    
    for alpha in alphas:
        if time.time() > deadline - 40:
            break
            
        _current_alpha = alpha
        
        for theta0 in start_points:
            if time.time() > deadline - 20:
                break
                
            result = minimize(
                _objective_smooth, theta0, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            p_opt = _build_from_theta(result.x)
            P_opt = p_opt.reshape(_N, _N)
            
            # Evaluate with HARD max to see true score
            val_opt = _compute_value(P_opt)
            
            if val_opt > best_val:
                best_val = val_opt
                best_P = P_opt.copy()
    
    # Phase 3: Final polish with very high alpha
    if time.time() < deadline - 30:
        _current_alpha = 1000.0
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
    
    # Phase 4: Coordinate descent polish on hard objective
    if time.time() < deadline - 10:
        best_P, best_val = _coordinate_polish(best_P, deadline)
    
    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
