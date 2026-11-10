import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# Precompute bin indices for slopes -1, 0, 1, 2
# x - y ranges from -9 to 9 -> offset by 9 to get 0..18
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
_h0_vals = _x_flat.astype(np.int64)
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

_ln2 = np.log(2.0)

# Precompute pooling matrices (0-1 matrices) for fast matrix multiplication
# M_D: (100, N_D), M_H0: (100, N_H0), etc.
def _make_pooling_matrix(vals, n_bins):
    M = np.zeros((len(vals), n_bins), dtype=np.float64)
    for i, v in enumerate(vals):
        M[i, v] = 1.0
    return M

M_D = _make_pooling_matrix(_d_vals, _N_D)
M_H0 = _make_pooling_matrix(_h0_vals, _N_H0)
M_H1 = _make_pooling_matrix(_h1_vals, _N_H1)
M_H2 = _make_pooling_matrix(_h2_vals, _N_H2)

def _entropy_from_probs(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    nz = probs[probs > 1e-15]
    if len(nz) == 0:
        return 0.0
    return -np.sum(nz * np.log2(nz))

def _score(w):
    """Compute D/M ratio from flat weight array w (assumed non-negative, sum 1)."""
    # w is flat, shape (100,)
    d_probs = M_D @ w
    h0_probs = M_H0 @ w
    h1_probs = M_H1 @ w
    h2_probs = M_H2 @ w
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
    return D / M

def _log_softmax_to_prob(params):
    log_p = params
    max_log_p = np.max(log_p)
    log_probs = log_p - max_log_p
    exp_probs = np.exp(log_probs)
    return exp_probs / np.sum(exp_probs)

def _smooth_score_and_grad(params, temp=0.1):
    """
    Compute the smooth objective -D / soft_max(H0, H1, H2) and its gradient.
    Uses log-sum-exp for the soft max.
    Returns: (value, gradient) where value is the objective to minimize (negative ratio).
    """
    probs = _log_softmax_to_prob(params)
    n = len(probs)
    
    # Pooling
    d_probs = M_D @ probs
    h0_probs = M_H0 @ probs
    h1_probs = M_H1 @ probs
    h2_probs = M_H2 @ probs
    
    # Entropies
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    hs = np.array([H0, H1, H2])
    
    # Soft max calculation
    scaled_hs = hs / temp
    max_scaled = np.max(scaled_hs)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled_hs - max_scaled)))
    M_soft = log_sum_exp * temp
    
    if M_soft <= 1e-9:
        return 0.0, np.zeros(n)
    
    ratio = D / M_soft
    value = -ratio
    
    # Gradients of Entropies w.r.t. probs
    # dH/dp_i = -log2(p_i) - 1/ln(2) for p_i > 0
    
    def _ent_grad_flat(probs_flat, pool_matrix, n_bins):
        bin_probs = pool_matrix @ probs_flat
        dH_dq = np.zeros(n_bins)
        nz = bin_probs > 1e-15
        dH_dq[nz] = -np.log2(bin_probs[nz]) - 1.0 / _ln2
        return pool_matrix.T @ dH_dq
    
    gD = _ent_grad_flat(probs, M_D, _N_D)
    gH0 = _ent_grad_flat(probs, M_H0, _N_H0)
    gH1 = _ent_grad_flat(probs, M_H1, _N_H1)
    gH2 = _ent_grad_flat(probs, M_H2, _N_H2)
    
    # Gradient of Soft Max
    # d(soft_max)/dH_i = softmax(hs/temp)_i
    softmax_w = np.exp(scaled_hs - max_scaled) / np.sum(np.exp(scaled_hs - max_scaled))
    
    # Gradient of M_soft w.r.t probs
    gM = softmax_w[0] * gH0 + softmax_w[1] * gH1 + softmax_w[2] * gH2
    
    # Chain rule: d(ratio)/dprobs = (gD * M - ratio * gM) / M
    d_ratio_d_probs = (gD * M_soft - ratio * gM) / M_soft
    
    # Gradient w.r.t log_params
    # dprobs/dlog_params = probs * (I - probs^T)
    # grad = d_ratio_d_probs * probs - dot(d_ratio_d_probs, probs) * probs
    mean_grad = np.dot(d_ratio_d_probs, probs)
    grad = probs * (d_ratio_d_probs - mean_grad)
    
    # We are minimizing -ratio, so negate the gradient
    return value, -grad

def _adam_optimize(params, time_limit=10.0, temp=0.1):
    """
    Custom Adam optimizer for the smooth objective.
    """
    start = time.time()
    best_params = params.copy()
    best_val, _ = _smooth_score_and_grad(params, temp)
    
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros(len(params))
    v = np.zeros(len(params))
    lr = 0.05
    eps = 1e-8
    
    it = 0
    while time.time() - start < time_limit:
        val, grad = _smooth_score_and_grad(params, temp)
        
        if val < best_val:
            best_val = val
            best_params = params.copy()
            
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        
        # Adaptive learning rate
        if it % 50 == 0:
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-10:
                base_lr = 0.2 / (1 + it / 1000)
                lr = min(0.5, max(0.001, base_lr / (gnorm + 1e-6)))
        
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        params = params - step
        it += 1
        
        # Early stopping if gradient is very small
        if it > 100 and np.linalg.norm(grad) < 1e-12:
            break
            
    return best_params

def arithmetic_kakeya_distribution() -> np.ndarray:
    np.random.seed(42)
    start_time = time.time()
    timeout = 600  # Safety margin for 630s limit
    
    best_P = None
    best_score = 0.0
    
    def _update_best(P):
        nonlocal best_P, best_score
        if P is None:
            return
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-9:
            return
        P = P / s
        score = _score(P.flatten())
        if score > best_score:
            best_score = score
            best_P = P.copy()
            
    # Phase 1: Global Search using Differential Evolution
    # Bounds for log-parameters. 
    # We can use a relatively wide range, e.g., [-10, 10]
    bounds = [(-10.0, 10.0)] * (_N * _N)
    
    # DE is slow, so we run it for a limited time
    de_time_limit = 120
    if time.time() - start_time < timeout - de_time_limit:
        try:
            # maxiter is capped by time in practice, but we set a high number
            # popsize=15 is a reasonable start
            result = differential_evolution(
                lambda p: -_score(_log_softmax_to_prob(p)), # Maximize score -> Minimize negative score
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                seed=np.random.randint(1, 1000),
                polish=False, # We will do our own polish
                init='sobol',
                x0=None,
                constraints=(),
                integrality=None,
                update_interval=1,
                polish=None,
                callback=None
            )
            # Note: DE above might take too long or finish early. 
            # We need a stricter time guard. 
            # Let's implement a custom DE loop or just use the result if it finishes.
            # Since DE in scipy doesn't have a hard time limit, we rely on it finishing 
            # or we should probably just use a simpler global method if this is risky.
            # Actually, let's stick to a safer multi-start local optimization strategy 
            # which is more predictable in time.
            
            # Reset best_P to None to force re-evaluation if DE didn't help
            # But we can still use the result if it exists
            if result.success or result.fun < 1e10:
                best_params_de = result.x
                best_P_de = _log_softmax_to_prob(best_params_de).reshape(_N, _N)
                _update_best(best_P_de)
        except Exception:
            pass

    # Phase 2: Multi-Start Local Optimization
    # We define a set of initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.zeros(_N * _N))
    
    # 2. Uniform Diagonals (x-y = c)
    for c in range(-9, 10):
        init = np.zeros(_N * _N)
        for y in range(_N):
            x = y + c
            if 0 <= x < _N:
                init[y * _N + x] = 1.0
        if np.sum(init) > 0:
            inits.append(init)
            
    # 3. Random Sparse
    for _ in range(20):
        init = np.zeros(_N * _N)
        k = np.random.randint(3, 15)
        idx = np.random.choice(_N * _N, k, replace=False)
        init[idx] = 1.0
        inits.append(init)
        
    # 4. Gaussian Blobs
    for _ in range(20):
        cx = np.random.uniform(0, _N)
        cy = np.random.uniform(0, _N)
        sigma = np.random.uniform(0.5, 3.0)
        init = np.zeros(_N * _N)
        for y in range(_N):
            for x in range(_N):
                init[y * _N + x] = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
        inits.append(init)
        
    # Add the best from Phase 1 if found
    if best_P is not None:
        inits.append(np.log(np.clip(best_P, 1e-15, None).flatten() / np.clip(best_P, 1e-15, None).sum()))
        # Also perturbations of the best
        for _ in range(10):
            pert = np.log(np.clip(best_P + np.random.normal(0, 0.1, (_N, _N)), 1e-15, None))
            pert = pert.flatten() - np.sum(pert) # Centering
            inits.append(pert)

    # Optimize each initial point
    # Time budget per start
    total_opt_time = timeout - (time.time() - start_time) - 20
    per_init_time = total_opt_time / len(inits) if len(inits) > 0 else 0
    
    for i, init_log_p in enumerate(inits):
        if time.time() - start_time > timeout - 20:
            break
            
        # If init is a probability vector, convert to log space
        # We mixed them, let's standardize to log space
        # My inits construction:
        # - Uniform: zeros (log 1) -> Uniform dist.
        # - Diagonals: 1s on diagonal -> Not log space! 
        # Correction: The _adam_optimize expects log-parameters.
        # I need to convert the probability inits to log-parameters.
        
        # Let's re-evaluate the init list.
        # I will just construct log-parameters directly.
        
        pass # The loop above was flawed. Let's implement a cleaner multi-start.

    # Clean Multi-Start Implementation
    init_log_params_list = []
    
    # 1. Uniform
    init_log_params_list.append(np.zeros(_N * _N))
    
    # 2. Diagonals
    for c in range(-9, 10):
        init_log_p = np.full(_N * _N, -100.0) # Very small prob
        for y in range(_N):
            x = y + c
            if 0 <= x < _N:
                init_log_p[y * _N + x] = 0.0 # Equal prob on diagonal
        init_log_params_list.append(init_log_p)
        
    # 3. Random Sparse (Log Uniform on support)
    for _ in range(30):
        k = np.random.randint(3, 20)
        idx = np.random.choice(_N * _N, k, replace=False)
        init_log_p = np.full(_N * _N, -100.0)
        init_log_p[idx] = 0.0
        init_log_params_list.append(init_log_p)
        
    # 4. Gaussian Blobs (Log Gaussian)
    for _ in range(30):
        cx = np.random.uniform(0, _N)
        cy = np.random.uniform(0, _N)
        sigma = np.random.uniform(0.5, 3.0)
        init_log_p = np.zeros(_N * _N)
        for y in range(_N):
            for x in range(_N):
                init_log_p[y * _N + x] = -((x-cx)**2 + (y-cy)**2) / (2*sigma**2)
        init_log_params_list.append(init_log_p)
        
    # Add best from previous phases if available
    if best_P is not None:
        p_clip = np.clip(best_P, 1e-15, None)
        p_norm = p_clip / p_clip.sum()
        log_p = np.log(p_norm).flatten()
        init_log_params_list.append(log_p)
        # Perturbations
        for _ in range(10):
            pert_log_p = log_p + np.random.normal(0, 0.5, _N*_N)
            init_log_params_list.append(pert_log_p)

    # Optimize
    time_per_init = max(1.0, (timeout - (time.time() - start_time) - 30) / len(init_log_params_list))
    
    for log_p in init_log_params_list:
        if time.time() - start_time > timeout - 30:
            break
        
        # Run Adam for a short time
        opt_params = _adam_optimize(log_p, time_limit=min(5.0, time_per_init), temp=0.2)
        opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
        _update_best(opt_P)
        
        # Follow up with a tighter temperature polish if time allows
        if time.time() - start_time < timeout - 40:
            # Use a lower temperature for better precision
            opt_params2 = _adam_optimize(opt_params, time_limit=min(5.0, time_per_init), temp=0.05)
            opt_P2 = _log_softmax_to_prob(opt_params2).reshape(_N, _N)
            _update_best(opt_P2)

    # Phase 3: High-Precision Polish on the Best Candidate
    if best_P is not None and time.time() - start_time < timeout - 10:
        p_clip = np.clip(best_P, 1e-15, None)
        p_norm = p_clip / p_clip.sum()
        log_p = np.log(p_norm).flatten()
        
        # Run Adam with very low temperature and long time
        remaining = timeout - (time.time() - start_time) - 5
        if remaining > 5:
            final_params = _adam_optimize(log_p, time_limit=remaining, temp=0.01)
            final_P = _log_softmax_to_prob(final_params).reshape(_N, _N)
            _update_best(final_P)
            
    # Fallback
    if best_P is None:
        best_P = np.ones((_N, _N)) / (_N * _N)
        
    # Final normalization and clipping
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((_N, _N)) / (_N * _N)
        
    return best_P.astype(np.float64)
