"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _compute_ratio(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _ratio_and_grad(params, support):
    """Compute ratio and gradient w.r.t. log-amplitudes on support."""
    k = len(support)
    amps = np.exp(params)
    h = np.zeros(N)
    h[support] = amps
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, np.zeros(k)
    h_norm = h / l1
    l2_sq = np.sum(h_norm * h_norm)
    conv = np.correlate(h_norm, h_norm, mode='full')
    sup_norm = np.max(conv)
    if sup_norm < 1e-15:
        return 0.0, np.zeros(k)
    ratio = l2_sq / sup_norm  # since l1=1 after normalization
    
    # Gradient computation
    # d/dparams_j [log(ratio)] where h[support_j] = exp(params_j)
    # Let a_j = exp(params_j), h = a on support, 0 elsewhere
    # After normalization: h_i = a_i / S where S = sum(a)
    # l2_sq = sum(h_i^2) = sum(a_i^2) / S^2
    # conv[k] = sum_i h_i * h_{i+k}
    # We need d(conv[k])/da_j and d(l2_sq)/da_j
    
    S = l1
    # dh_i/da_j = delta_{i,j} - h_i/S (for i in support)
    # For i in support: dh_i/da_j = (delta_{i,j} * S - a_i) / S^2 = (delta_{i,j} - h_i) / S
    
    # d(l2_sq)/da_j = 2 * sum_i h_i * dh_i/da_j = 2 * sum_i h_i * (delta_{i,j} - h_i)/S
    # = 2 * (h_j - sum_i h_i^2) / S = 2 * (h_j - l2_sq) / S
    
    # d(conv[k])/da_j = sum_i dh_i/da_j * h_{i+k} + sum_i h_i * dh_{i+k}/da_j
    # For the first sum: sum_i (delta_{i,j} - h_i)/S * h_{i+k} = (h_{j+k} - sum_i h_i * h_{i+k})/S
    # = (h_{j+k} - conv[k])/S
    # For the second sum: sum_i h_i * (delta_{i+k,j} - h_{i+k})/S = (h_{j-k} - conv[k])/S
    # So d(conv[k])/da_j = (h_{j+k} + h_{j-k} - 2*conv[k]) / S
    
    # Now ratio = l2_sq / sup_norm (with l1=1)
    # d(ratio)/da_j = (d(l2_sq)/da_j * sup_norm - l2_sq * d(sup_norm)/da_j) / sup_norm^2
    # d(sup_norm)/da_j = d(conv[k*])/da_j where k* is argmax
    # d(log(ratio))/da_j = d(l2_sq)/da_j / l2_sq - d(sup_norm)/da_j / sup_norm
    
    k_star = np.argmax(conv)
    
    # d(l2_sq)/da_j = 2*(h_j - l2_sq)/S for j in support
    dl2 = np.zeros(k)
    for j, idx in enumerate(support):
        dl2[j] = 2.0 * (h_norm[idx] - l2_sq) / S
    
    # d(conv[k_star])/da_j = (h_{j+k_star} + h_{j-k_star} - 2*conv[k_star])/S
    dconv = np.zeros(k)
    for j, idx in enumerate(support):
        # h_{idx + k_star} and h_{idx - k_star}
        ip = idx + k_star
        im = idx - k_star
        hp = h_norm[ip] if 0 <= ip < N else 0.0
        hm = h_norm[im] if 0 <= im < N else 0.0
        dconv[j] = (hp + hm - 2.0 * conv[k_star]) / S
    
    # d(log(ratio))/da_j = dl2[j]/l2_sq - dconv[j]/sup_norm
    dlogratio_da = dl2 / l2_sq - dconv / sup_norm
    
    # da_j/dparams_j = a_j = exp(params_j)
    grad = dlogratio_da * amps
    
    return ratio, grad

def _build_sidon_set(n, seed=0):
    """Build a Sidon set of size n from [0, N-1]."""
    rng = np.random.RandomState(seed)
    best_sidon = None
    best_size = 0
    
    for _ in range(20):
        sidon = []
        used_sums = set()
        for i in range(N):
            valid = True
            # Check if adding i creates duplicate sums
            new_sums = set()
            for a in sidon:
                new_sums.add(a + i)
            new_sums.add(2 * i)
            if new_sums & used_sums:
                valid = False
            if valid:
                sidon.append(i)
                used_sums |= new_sums
                if len(sidon) == n:
                    break
        
        if len(sidon) > best_size:
            best_size = len(sidon)
            best_sidon = sidon[:]
            if best_size == n:
                break
    
    if best_sidon is None or len(best_sidon) < n:
        # Fallback: use a simple construction
        # Quadratic residues mod a prime
        p = 53  # prime > N
        residues = sorted(set((i * i) % p for i in range(p) if (i * i) % p < N))
        if len(residues) >= n:
            return residues[:n]
        else:
            # Just use first n indices
            return list(range(n))
    
    return best_sidon[:n]

def construct():
    """Construct 50 nonnegative heights to maximize the ratio.
    
    Strategy: 
    1. Build Sidon sets of various sizes
    2. For each, optimize amplitudes using gradient-based method
    3. Return the best solution found
    """
    best_ratio = 0.0
    best_h = None
    
    # Try different Sidon set sizes
    for k in [8, 10, 12, 14, 16, 18, 20]:
        for seed in range(3):
            support = _build_sidon_set(k, seed=seed)
            if len(support) < k:
                continue
            
            # Initialize with equal amplitudes
            log_amp0 = np.zeros(k)
            
            # Try multiple initializations
            for init_trial in range(3):
                if init_trial == 0:
                    x0 = np.zeros(k)
                elif init_trial == 1:
                    rng = np.random.RandomState(42 + seed + init_trial)
                    x0 = rng.randn(k) * 0.5
                else:
                    rng = np.random.RandomState(100 + seed + init_trial)
                    x0 = rng.randn(k) * 1.0
                
                def obj(x):
                    ratio, _ = _ratio_and_grad(x, support)
                    return -np.log(max(ratio, 1e-15))
                
                def obj_grad(x):
                    ratio, grad = _ratio_and_grad(x, support)
                    if ratio <= 0:
                        return 0.0, np.zeros(k)
                    return -np.log(ratio), -grad
                
                try:
                    res = minimize(obj_grad, x0, method='L-BFGS-B',
                                  jac=True,
                                  options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
                    h = np.zeros(N)
                    h[support] = np.exp(res.x)
                    h = np.maximum(h, 0)
                    if np.sum(h) < 1e-15:
                        continue
                    ratio = _compute_ratio(h)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_h = h.copy()
                except Exception:
                    pass
    
    # Also try a denser approach: optimize on a larger support
    for k in [25, 30, 35]:
        for seed in range(2):
            # Use a pseudo-random support with some structure
            rng = np.random.RandomState(200 + seed)
            support = sorted(rng.choice(N, size=k, replace=False).tolist())
            
            log_amp0 = np.zeros(k)
            
            def obj(x):
                ratio, _ = _ratio_and_grad(x, support)
                return -np.log(max(ratio, 1e-15))
            
            def obj_grad(x):
                ratio, grad = _ratio_and_grad(x, support)
                if ratio <= 0:
                    return 0.0, np.zeros(k)
                return -np.log(ratio), -grad
            
            try:
                res = minimize(obj_grad, log_amp0, method='L-BFGS-B',
                              jac=True,
                              options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
                h = np.zeros(N)
                h[support] = np.exp(res.x)
                h = np.maximum(h, 0)
                if np.sum(h) < 1e-15:
                    continue
                ratio = _compute_ratio(h)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h.copy()
            except Exception:
                pass
    
    # Final refinement: if we have a good solution, try to refine with all 50 entries
    if best_h is not None and best_ratio > 0.5:
        # Keep only entries above a threshold to maintain sparsity
        threshold = np.max(best_h) * 0.01
        mask = best_h > threshold
        if np.sum(mask) >= 5:
            support = np.where(mask)[0]
            k = len(support)
            log_amp0 = np.log(best_h[support] + 1e-15)
            
            def obj(x):
                ratio, _ = _ratio_and_grad(x, support)
                return -np.log(max(ratio, 1e-15))
            
            def obj_grad(x):
                ratio, grad = _ratio_and_grad(x, support)
                if ratio <= 0:
                    return 0.0, np.zeros(k)
                return -np.log(ratio), -grad
            
            try:
                res = minimize(obj_grad, log_amp0, method='L-BFGS-B',
                              jac=True,
                              options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-14})
                h = np.zeros(N)
                h[support] = np.exp(res.x)
                h = np.maximum(h, 0)
                if np.sum(h) > 1e-15:
                    ratio = _compute_ratio(h)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_h = h.copy()
            except Exception:
                pass
    
    if best_h is None:
        return [1.0] * N
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
