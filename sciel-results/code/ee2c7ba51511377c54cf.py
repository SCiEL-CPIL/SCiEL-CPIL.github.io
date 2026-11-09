# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import random

_DEADLINE = 600.0  # safety margin before 630s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_reached():
    return _elapsed() > _DEADLINE


def _shannon_bits(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    nz = probs > 0.0
    if not np.any(nz):
        return 0.0
    p = probs[nz]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def _projection_probs(P, r):
    """
    P has shape (n, n) with P[y, x].
    Returns the pooled distribution of value = x + r*y.
    """
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[i,j]=x, Y[i,j]=y
    P_t = P.T  # P_t[i, j] = P[j, i]
    vals = X + r * Y
    order = np.argsort(vals.ravel())
    vals_flat = vals.ravel()[order]
    p_flat = P_t.ravel()[order]
    uniq, idx = np.unique(vals_flat, return_index=True)
    sums = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(sums, np.searchsorted(uniq, vals_flat), p_flat)
    total = sums.sum()
    if total <= 0:
        return np.zeros(len(uniq))
    return sums / total


def _objective_from_P(P):
    """
    Return D / M where D = H_{-1}, M = max(H_0, H_1, H_2).
    Returns -inf if invalid (M <= 1e-9).
    """
    P = np.asarray(P, dtype=np.float64)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return -np.inf
    P = P / s
    Hm1 = _shannon_bits(_projection_probs(P, -1.0))
    H0 = _shannon_bits(_projection_probs(P, 0.0))
    H1 = _shannon_bits(_projection_probs(P, 1.0))
    H2 = _shannon_bits(_projection_probs(P, 2.0))
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return -np.inf
    return Hm1 / M


def _parametric_P(alpha, beta, gamma, n=10):
    """
    P[y, x] ∝ exp(-alpha*x^2 - beta*y^2 - gamma*x*y)
    x, y in 0..n-1
    """
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[i,j]=x, Y[i,j]=y
    logw = -alpha * X**2 - beta * Y**2 - gamma * X * Y
    logw = logw - logw.max()
    w = np.exp(logw)
    w_t = w.T
    s = w_t.sum()
    if s <= 0:
        return np.ones((n, n)) / (n * n)
    return w_t / s


def _neg_obj_params(params):
    """Negative of the objective for the parametric model."""
    alpha, beta, gamma = params
    P = _parametric_P(alpha, beta, gamma)
    val = _objective_from_P(P)
    if not np.isfinite(val):
        return 1e9
    return -val


def _refine_with_bfgs(x0):
    """Local refinement using L-BFGS-B on the parametric model."""
    try:
        res = minimize(
            _neg_obj_params,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 4000},
        )
        if res.success or res.fun < 1e9:
            return res.x
    except Exception:
        pass
    return x0


def _greedy_set_cover(n=10, max_iter=5000):
    """
    Discrete greedy algorithm to select cells and assign weights.
    
    Strategy:
    1. Start with empty selection
    2. At each step, add the cell that maximizes marginal gain in objective
    3. Use random restarts with different initial seeds
    
    The key insight: we want x-y to be uniform (high H_{-1}) while keeping
    x, x+y, x+2y distributions concentrated (low H_0, H_1, H_2).
    """
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    
    # Precompute projection values for each cell
    val_m1 = X - Y  # x - y
    val_0 = X       # x
    val_1 = X + Y   # x + y
    val_2 = X + 2*Y # x + 2*y
    
    best_P = None
    best_val = -np.inf
    
    for restart in range(50):
        if _deadline_reached():
            break
        
        # Random initial selection of k cells
        k = random.randint(15, 35)
        indices = random.sample(range(n*n), k)
        selected = set(indices)
        
        # Build initial P with equal weights on selected cells
        P = np.zeros((n, n), dtype=np.float64)
        for idx in selected:
            y, x = divmod(idx, n)
            P[y, x] = 1.0
        P /= P.sum()
        
        current_val = _objective_from_P(P)
        
        # Greedy improvement: try adding/removing/swapping cells
        for it in range(max_iter):
            if _deadline_reached():
                break
            
            improved = False
            
            # Try adding a random unselected cell
            unselected = [i for i in range(n*n) if i not in selected]
            if unselected:
                idx_add = random.choice(unselected)
                y_add, x_add = divmod(idx_add, n)
                
                P_new = P.copy()
                P_new[y_add, x_add] = 1.0
                P_new /= P_new.sum()
                val_new = _objective_from_P(P_new)
                
                if val_new > current_val + 1e-10:
                    selected.add(idx_add)
                    P = P_new
                    current_val = val_new
                    improved = True
            
            # Try removing a random selected cell (if more than 5)
            if len(selected) > 5:
                idx_rem = random.choice(list(selected))
                y_rem, x_rem = divmod(idx_rem, n)
                
                P_new = P.copy()
                P_new[y_rem, x_rem] = 0.0
                if P_new.sum() > 0:
                    P_new /= P_new.sum()
                    val_new = _objective_from_P(P_new)
                    
                    if val_new > current_val + 1e-10:
                        selected.discard(idx_rem)
                        P = P_new
                        current_val = val_new
                        improved = True
            
            # Try swapping: remove one, add another
            if len(selected) >= 5 and unselected:
                idx_rem = random.choice(list(selected))
                idx_add = random.choice(unselected)
                y_rem, x_rem = divmod(idx_rem, n)
                y_add, x_add = divmod(idx_add, n)
                
                P_new = P.copy()
                P_new[y_rem, x_rem] = 0.0
                P_new[y_add, x_add] = 1.0
                if P_new.sum() > 0:
                    P_new /= P_new.sum()
                    val_new = _objective_from_P(P_new)
                    
                    if val_new > current_val + 1e-10:
                        selected.discard(idx_rem)
                        selected.add(idx_add)
                        P = P_new
                        current_val = val_new
                        improved = True
            
            # Weight adjustment: try adjusting weights of selected cells
            if len(selected) >= 3:
                idx_adj = random.choice(list(selected))
                y_adj, x_adj = divmod(idx_adj, n)
                
                # Try increasing or decreasing this cell's weight
                for factor in [0.5, 0.7, 1.3, 1.5, 2.0]:
                    P_new = P.copy()
                    P_new[y_adj, x_adj] *= factor
                    if P_new.sum() > 0:
                        P_new /= P_new.sum()
                        val_new = _objective_from_P(P_new)
                        
                        if val_new > current_val + 1e-10:
                            P = P_new
                            current_val = val_new
                            improved = True
                            break
            
            if not improved:
                break
        
        if current_val > best_val:
            best_val = current_val
            best_P = P.copy()
    
    return best_P, best_val


def _structured_ansatz_search(n=10):
    """
    Try various structured ansatze that exploit the problem structure.
    """
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    
    best_P = None
    best_val = -np.inf
    
    # Ansatz 1: Weight by (x-y) to flatten x-y distribution
    for c in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        if _deadline_reached():
            break
        w = 1.0 + c * (X - Y)**2
        P = w.T / w.T.sum()
        val = _objective_from_P(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()
    
    # Ansatz 2: Weight by |x-y|
    for c in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        if _deadline_reached():
            break
        w = 1.0 + c * np.abs(X - Y)
        P = w.T / w.T.sum()
        val = _objective_from_P(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()
    
    # Ansatz 3: Diagonal bands - weight cells where x-y is in certain ranges
    for band_width in [1, 2, 3, 4, 5]:
        for center in range(-band_width, band_width + 1):
            if _deadline_reached():
                break
            diff = X - Y
            mask = np.abs(diff - center) <= band_width
            P = mask.T.astype(float)
            if P.sum() > 0:
                P /= P.sum()
                val = _objective_from_P(P)
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best_P = P.copy()
    
    # Ansatz 4: Anti-diagonal weighting (favor cells where x+y is concentrated)
    for c in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        if _deadline_reached():
            break
        # Penalize cells where x+y is far from center
        center = (n-1)
        penalty = c * (X + Y - center)**2
        w = np.exp(-penalty)
        P = w.T / w.T.sum()
        val = _objective_from_P(P)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = P.copy()
    
    # Ansatz 5: Combine x-y flattening with x+y concentration
    for c1 in [0.1, 0.3, 0.5, 1.0]:
        for c2 in [0.1, 0.3, 0.5, 1.0]:
            if _deadline_reached():
                break
            w = (1.0 + c1 * (X - Y)**2) * np.exp(-c2 * (X + Y - (n-1))**2 / 10.0)
            P = w.T / w.T.sum()
            val = _objective_from_P(P)
            if np.isfinite(val) and val > best_val:
                best_val = val
                best_P = P.copy()
    
    return best_P, best_val


def _direct_polish(P_start, n=10):
    """
    Local optimization polish on the 100-dim P vector.
    """
    x0 = P_start.ravel().copy()
    
    def _direct_obj(x):
        P = np.clip(x, 0.0, None).reshape(n, n)
        s = P.sum()
        if s <= 1e-9:
            return 1e9
        P = P / s
        val = _objective_from_P(P)
        if not np.isfinite(val):
            return 1e9
        return -val
    
    best_P = P_start.copy()
    best_val = _objective_from_P(best_P)
    
    # Try Nelder-Mead
    try:
        res = minimize(
            _direct_obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10},
        )
        if res.fun < 1e9:
            P = np.clip(res.x, 0.0, None).reshape(n, n)
            s = P.sum()
            if s > 1e-9:
                P = P / s
                val = _objective_from_P(P)
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best_P = P.copy()
    except Exception:
        pass
    
    # Try Powell
    try:
        res = minimize(
            _direct_obj,
            x0,
            method='Powell',
            options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-8},
        )
        if res.fun < 1e9:
            P = np.clip(res.x, 0.0, None).reshape(n, n)
            s = P.sum()
            if s > 1e-9:
                P = P / s
                val = _objective_from_P(P)
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best_P = P.copy()
    except Exception:
        pass
    
    return best_P, best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    n = 10
    
    # Baseline: uniform
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_val = _objective_from_P(best_P)
    
    # Phase 1: Structured ansatz search (fast, deterministic)
    P_ansatz, val_ansatz = _structured_ansatz_search(n)
    if P_ansatz is not None and val_ansatz > best_val:
        best_P = P_ansatz.copy()
        best_val = val_ansatz
    
    if _deadline_reached():
        P = np.clip(best_P, 0.0, None)
        s = P.sum()
        if s > 1e-9:
            P = P / s
        return P
    
    # Phase 2: Greedy set-cover with random restarts (discrete combinatorial search)
    P_greedy, val_greedy = _greedy_set_cover(n, max_iter=3000)
    if P_greedy is not None and val_greedy > best_val:
        best_P = P_greedy.copy()
        best_val = val_greedy
    
    if _deadline_reached():
        P = np.clip(best_P, 0.0, None)
        s = P.sum()
        if s > 1e-9:
            P = P / s
        return P
    
    # Phase 3: Direct optimization polish from best P found
    P_polished, val_polished = _direct_polish(best_P, n)
    if val_polished > best_val:
        best_P = P_polished.copy()
        best_val = val_polished
    
    # Ensure validity
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s
    
    return P


# EVOLVE-BLOCK-END
