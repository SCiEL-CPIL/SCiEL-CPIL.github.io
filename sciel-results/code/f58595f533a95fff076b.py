# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

_START_TIME = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill

def _shannon_entropy(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _compute_D_M(P):
    """Compute D = H(x-y) and M = max(H(x), H(x+y), H(x+2y))."""
    n = 10
    P = np.asarray(P, dtype=np.float64)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s < 1e-15:
        return 0.0, 0.0
    P = P / s
    
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    # P[y, x] so we need to be careful with indexing
    # P is indexed P[y, x], so P[i,j] corresponds to y=i, x=j
    
    # Distribution of x - y: values range from -9 to 9 (19 values)
    diff = X - Y  # shape (n, n), diff[i,j] = x_j - y_i
    diff_vals = np.arange(-9, 10)  # -9 to 9
    p_diff = np.zeros(19)
    for k, d in enumerate(diff_vals):
        mask = (diff == d)
        p_diff[k] = P[mask].sum()
    D = _shannon_entropy(p_diff)
    
    # Distribution of x: values 0..9
    p_x = P.sum(axis=0)  # sum over y for each x
    H0 = _shannon_entropy(p_x)
    
    # Distribution of x + y: values 0..18 (19 values)
    s_vals = X + Y
    s_range = np.arange(0, 19)
    p_s = np.zeros(19)
    for k, sv in enumerate(s_range):
        mask = (s_vals == sv)
        p_s[k] = P[mask].sum()
    H1 = _shannon_entropy(p_s)
    
    # Distribution of x + 2y: values 0..27 (28 values)
    t_vals = X + 2 * Y
    t_range = np.arange(0, 28)
    p_t = np.zeros(28)
    for k, tv in enumerate(t_range):
        mask = (t_vals == tv)
        p_t[k] = P[mask].sum()
    H2 = _shannon_entropy(p_t)
    
    M = max(H0, H1, H2)
    return D, M

def _ratio_from_P(P):
    D, M = _compute_D_M(P)
    if M < 1e-9:
        return 0.0
    return D / M

def _objective_smooth(theta, soft_max_beta=50.0):
    """Differentiable objective using soft-max for M."""
    n = 10
    P = np.exp(theta)
    P = P / P.sum()
    P = np.clip(P, 0.0, None)
    
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # D = H(x-y)
    diff = X - Y
    diff_vals = np.arange(-9, 10)
    p_diff = np.zeros(19)
    for k, d in enumerate(diff_vals):
        p_diff[k] = P[diff == d].sum()
    p_diff = np.clip(p_diff, 1e-15, None)
    p_diff = p_diff / p_diff.sum()
    D = -np.sum(p_diff * np.log2(p_diff))
    
    # H0 = H(x)
    p_x = P.sum(axis=0)
    p_x = np.clip(p_x, 1e-15, None)
    p_x = p_x / p_x.sum()
    H0 = -np.sum(p_x * np.log2(p_x))
    
    # H1 = H(x+y)
    s_vals = X + Y
    s_range = np.arange(0, 19)
    p_s = np.zeros(19)
    for k, sv in enumerate(s_range):
        p_s[k] = P[s_vals == sv].sum()
    p_s = np.clip(p_s, 1e-15, None)
    p_s = p_s / p_s.sum()
    H1 = -np.sum(p_s * np.log2(p_s))
    
    # H2 = H(x+2y)
    t_vals = X + 2 * Y
    t_range = np.arange(0, 28)
    p_t = np.zeros(28)
    for k, tv in enumerate(t_range):
        p_t[k] = P[t_vals == tv].sum()
    p_t = np.clip(p_t, 1e-15, None)
    p_t = p_t / p_t.sum()
    H2 = -np.sum(p_t * np.log2(p_t))
    
    # Soft-max for M
    Hs = np.array([H0, H1, H2])
    beta = soft_max_beta
    log_M = np.log(np.sum(np.exp(beta * Hs))) / beta
    
    # Maximize D - log_M (equivalent to maximizing D/M in a smooth way)
    # Actually maximize D * exp(-log_M) = D / M
    val = D * np.exp(-log_M)
    return -val  # minimize

def _build_P_from_theta(theta):
    n = 10
    P = np.exp(theta)
    P = P / P.sum()
    return P

def _gradient_ascent(P_init, max_iter=2000, lr=0.05, tol=1e-8):
    """Simple projected gradient ascent on the simplex."""
    n = 10
    P = P_init.copy()
    P = np.clip(P, 1e-12, None)
    P = P / P.sum()
    
    best_P = P.copy()
    best_val = _ratio_from_P(P)
    
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    diff = X - Y
    diff_vals = np.arange(-9, 10)
    s_vals = X + Y
    s_range = np.arange(0, 19)
    t_vals = X + 2 * Y
    t_range = np.arange(0, 28)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        P = np.clip(P, 1e-12, None)
        P = P / P.sum()
        
        # Compute entropies
        p_diff = np.zeros(19)
        for k, d in enumerate(diff_vals):
            p_diff[k] = P[diff == d].sum()
        p_diff = np.clip(p_diff, 1e-15, None)
        p_diff = p_diff / p_diff.sum()
        D = -np.sum(p_diff * np.log2(p_diff))
        
        p_x = P.sum(axis=0)
        p_x = np.clip(p_x, 1e-15, None)
        p_x = p_x / p_x.sum()
        H0 = -np.sum(p_x * np.log2(p_x))
        
        p_s = np.zeros(19)
        for k, sv in enumerate(s_range):
            p_s[k] = P[s_vals == sv].sum()
        p_s = np.clip(p_s, 1e-15, None)
        p_s = p_s / p_s.sum()
        H1 = -np.sum(p_s * np.log2(p_s))
        
        p_t = np.zeros(28)
        for k, tv in enumerate(t_range):
            p_t[k] = P[t_vals == tv].sum()
        p_t = np.clip(p_t, 1e-15, None)
        p_t = p_t / p_t.sum()
        H2 = -np.sum(p_t * np.log2(p_t))
        
        M = max(H0, H1, H2)
        if M < 1e-9:
            break
        val = D / M
        
        if val > best_val:
            best_val = val
            best_P = P.copy()
        
        # Numerical gradient
        grad = np.zeros_like(P)
        eps = 1e-6
        for i in range(n):
            for j in range(n):
                P_plus = P.copy()
                P_plus[i, j] += eps
                P_plus = np.clip(P_plus, 0, None)
                s = P_plus.sum()
                if s < 1e-15:
                    continue
                P_plus /= s
                val_plus = _ratio_from_P(P_plus)
                grad[i, j] = (val_plus - val) / eps
                if time.time() > _DEADLINE - 5.0:
                    return best_P, best_val
        
        # Gradient ascent step
        P_new = P + lr * grad
        P_new = np.clip(P_new, 1e-12, None)
        P_new = P_new / P_new.sum()
        
        if np.max(np.abs(P_new - P)) < tol:
            break
        P = P_new
    
    return best_P, best_val

def _lbfgs_optimize(P_init, max_iter=500):
    """Use L-BFGS-B on log-softmax parameterization."""
    n = 10
    theta = np.log(P_init + 1e-15)
    
    def obj(theta):
        if time.time() > _DEADLINE - 5.0:
            return 0.0
        return _objective_smooth(theta)
    
    try:
        res = opt.minimize(obj, theta, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        P = _build_P_from_theta(res.x)
        P = np.clip(P, 0, None)
        P = P / P.sum()
        return P
    except Exception:
        return P_init

def _random_start(n=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    P = np.random.dirichlet(np.ones(n*n))
    return P.reshape(n, n)

def _diag_start(n=10, offset=0, width=2):
    """Start with mass concentrated along diagonals x-y = offset +/- width."""
    P = np.zeros((n, n))
    for dy in range(-width, width+1):
        for x in range(n):
            y = x - (offset + dy)
            if 0 <= y < n:
                P[y, x] = 1.0
    P = P / P.sum()
    return P

def _row_col_start(n=10):
    """Start with mass on a cross pattern."""
    P = np.zeros((n, n))
    P[0, :] = 1.0
    P[:, 0] = 1.0
    P[0, 0] = 1.0
    P = P / P.sum()
    return P

def _block_start(n=10, block_size=3):
    """Start with mass in a block."""
    P = np.zeros((n, n))
    P[:block_size, :block_size] = 1.0
    P = P / P.sum()
    return P

def _strip_start(n=10, axis='x', pos=3, width=2):
    """Start with mass in a horizontal or vertical strip."""
    P = np.zeros((n, n))
    if axis == 'x':
        P[:, pos-width:pos+width+1] = 1.0
    else:
        P[pos-width:pos+width+1, :] = 1.0
    P = P / P.sum()
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    best_P = np.ones((n, n)) / (n * n)
    best_val = _ratio_from_P(best_P)
    
    # Generate diverse initializations
    starts = []
    starts.append(best_P.copy())  # uniform
    
    # Diagonal starts
    for offset in range(-4, 5):
        for width in [1, 2, 3]:
            starts.append(_diag_start(n, offset, width))
    
    # Random starts
    for seed in [42, 123, 456, 789, 1024, 2048, 3333, 4444]:
        starts.append(_random_start(n, seed))
    
    # Structural starts
    starts.append(_row_col_start(n))
    starts.append(_block_start(n, 2))
    starts.append(_block_start(n, 3))
    starts.append(_block_start(n, 4))
    for pos in [2, 4, 6, 8]:
        starts.append(_strip_start(n, 'x', pos, 1))
        starts.append(_strip_start(n, 'y', pos, 1))
        starts.append(_strip_start(n, 'x', pos, 2))
        starts.append(_strip_start(n, 'y', pos, 2))
    
    # Remove duplicates approximately
    unique_starts = []
    for s in starts:
        is_dup = False
        for u in unique_starts:
            if np.max(np.abs(s - u)) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
    
    # Phase 1: Quick evaluation of all starts
    for P0 in unique_starts:
        if time.time() > _DEADLINE - 60.0:
            break
        val = _ratio_from_P(P0)
        if val > best_val:
            best_val = val
            best_P = P0.copy()
    
    # Phase 2: L-BFGS optimization from top starts
    # Sort starts by initial value
    scored_starts = []
    for P0 in unique_starts:
        val = _ratio_from_P(P0)
        scored_starts.append((val, P0))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Take top 15 starts for L-BFGS
    top_starts = [s[1] for s in scored_starts[:15]]
    
    for P0 in top_starts:
        if time.time() > _DEADLINE - 30.0:
            break
        P_opt = _lbfgs_optimize(P0, max_iter=300)
        val = _ratio_from_P(P_opt)
        if val > best_val:
            best_val = val
            best_P = P_opt.copy()
    
    # Phase 3: Gradient ascent polish on best
    if time.time() < _DEADLINE - 20.0:
        P_grad, val_grad = _gradient_ascent(best_P, max_iter=1000, lr=0.03)
        if val_grad > best_val:
            best_val = val_grad
            best_P = P_grad.copy()
    
    # Phase 4: Final L-BFGS polish with tighter settings
    if time.time() < _DEADLINE - 10.0:
        P_polish = _lbfgs_optimize(best_P, max_iter=500)
        val_polish = _ratio_from_P(P_polish)
        if val_polish > best_val:
            best_val = val_polish
            best_P = P_polish.copy()
    
    # Ensure validity
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s < 1e-15:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / s
    
    return best_P

# EVOLVE-BLOCK-END
