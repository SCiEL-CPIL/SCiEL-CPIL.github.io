"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-30:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _soft_norm_inf(g, beta):
    """Compute soft max via log-sum-exp for numerical stability."""
    max_g = np.max(g)
    if max_g < 1e-30:
        return 1e-30
    shifted = beta * (g - max_g)
    log_sum_exp = max_g + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return log_sum_exp

def _smooth_objective(h, beta):
    """
    Smooth objective: minimize -l2 / (norm_1 * norm_inf_smooth)
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(g) / (m + 1)
    
    norm_inf_smooth = _soft_norm_inf(g, beta)
    
    if norm_1 * norm_inf_smooth < 1e-30:
        return 0.0
    
    val = l2 / (norm_1 * norm_inf_smooth)
    return -val

def _analytical_gradient(h, beta):
    """
    Analytical gradient of _smooth_objective with respect to h.
    Uses the chain rule through convolution and log-sum-exp.
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    
    # Gradient of l2 with respect to g
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    # d(l2)/d(y_i) for each y
    dy_l2 = np.zeros(m + 2)
    # l2 = (w/3) * sum_i (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # d/dy[i] of term i: (w/3)*(2*y[i] + y[i+1])
    # d/dy[i] of term i-1: (w/3)*(y[i-1] + 2*y[i])
    # So d/dy[i] = (w/3)*(2*y[i] + y[i+1] + y[i-1] + 2*y[i]) = (w/3)*(4*y[i] + y[i+1] + y[i-1])
    # But we need to be careful with boundaries
    for i in range(m + 2):
        coeff = 0.0
        # From term i (if i <= m): y[i]^2 + y[i]*y[i+1] contributes 2*y[i] + y[i+1]
        if i <= m:
            coeff += 2.0 * y[i] + y[i+1]
        # From term i-1 (if i-1 >= 0 and i-1 <= m): y[i-1]*y[i] + y[i]^2 contributes y[i-1] + 2*y[i]
        if i >= 1 and i - 1 <= m:
            coeff += y[i-1] + 2.0 * y[i]
        dy_l2[i] = (w / 3.0) * coeff
    
    # Gradient of norm_1 w.r.t. g: 1/(m+1) for each g_j
    # Gradient of norm_inf_smooth w.r.t. g: softmax weights
    max_g = np.max(g)
    if max_g < 1e-30:
        return np.zeros_like(h)
    shifted = beta * (g - max_g)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    softmax_weights = exp_shifted / sum_exp  # d(log-sum-exp)/d(g_j) = softmax
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf_smooth = max_g + (1.0/beta) * np.log(sum_exp)
    
    # f = l2 / (norm_1 * norm_inf_smooth)
    # df/dg_j = (dl2/dg_j * norm_1 * norm_inf_smooth - l2 * (d(norm_1)/dg_j * norm_inf_smooth + norm_1 * d(norm_inf_smooth)/dg_j)) / (norm_1 * norm_inf_smooth)^2
    # = (dl2/dg_j) / (norm_1 * norm_inf_smooth) - l2 * (d(norm_1)/dg_j * norm_inf_smooth + norm_1 * d(norm_inf_smooth)/dg_j) / (norm_1 * norm_inf_smooth)^2
    
    # dl2/dg_j: g_j appears in y[j+1]
    # So dl2/dg_j = dy_l2[j+1]
    dl2_dg = np.zeros(m)
    dl2_dg[:] = dy_l2[1:m+1]
    
    dnorm1_dg = np.ones(m) / (m + 1)
    dnorminf_dg = softmax_weights
    
    # f = l2 / (norm_1 * norm_inf_smooth)
    # df/dg_j = dl2_dg[j] / (norm_1 * norm_inf_smooth) - l2 * (dnorm1_dg[j] * norm_inf_smooth + norm_1 * dnorminf_dg[j]) / (norm_1 * norm_inf_smooth)^2
    
    l2_val = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    denom = norm_1 * norm_inf_smooth
    if denom < 1e-30:
        return np.zeros_like(h)
    
    df_dg = dl2_dg / denom - l2_val * (dnorm1_dg * norm_inf_smooth + norm_1 * dnorminf_dg) / (denom * denom)
    
    # Now, g_j = sum_{k=0}^{N-1} h[k] * h[j-k] (with appropriate bounds)
    # dg_j/dh_i = h[j-i] + h[i-j] (with bounds checking, and if j-i == i-j then 2*h[i])
    # Actually g = convolve(h, h), so g[j] = sum_{i: 0<=i<N, 0<=j-i<N} h[i] * h[j-i]
    # dg_j/dh_i = h[j-i] if 0 <= j-i < N, else 0
    # But also if we vary h_i, it appears in g[j] as h[i]*h[j-i] and h[j-i]*h[i]
    # So dg_j/dh_i = h[j-i] + h[i-j] if both in range, but since convolution is symmetric:
    # g[j] = sum_{k} h[k]*h[j-k], so dg[j]/dh[i] = h[j-i] if 0 <= j-i < N
    
    # More precisely: g[j] = sum_{k=max(0,j-N+1)}^{min(j,N-1)} h[k] * h[j-k]
    # dg[j]/dh[i] = h[j-i] if 0 <= j-i < N, else 0
    
    # So df/dh_i = sum_j df_dg[j] * dg[j]/dh[i] = sum_j df_dg[j] * h[j-i] (if 0<=j-i<N)
    # = sum_{j: 0<=j-i<N} df_dg[j] * h[j-i]
    # Let k = j-i, so j = k+i, k ranges from max(0,-i) to min(m-1, N-1-i)
    # = sum_{k: max(0,-i) <= k <= min(m-1, N-1-i)} df_dg[k+i] * h[k]
    
    # This is a cross-correlation-like operation
    # df_dh = np.zeros(N)
    # for i in range(N):
    #     k_min = max(0, -i)  # j = k+i >= 0 => k >= -i
    #     k_max = min(m-1, N-1-i)  # j = k+i <= m-1 => k <= m-1-i; k < N => k <= N-1
    #     # Also j = k+i < m, so k < m-i
    #     k_min = max(0, -i)
    #     k_max = min(N-1, m-1-i)
    #     if k_min <= k_max:
    #         df_dh[i] = np.sum(df_dg[k_min+i:k_max+i+1] * h[k_min:k_max+1])
    
    # Vectorized version using convolution/correlation
    # df_dh[i] = sum_k df_dg[k+i] * h[k] for valid k
    # This is like a cross-correlation of df_dg and h
    
    # Let's compute it efficiently
    # df_dh = np.zeros(N)
    # We can use the fact that this is a valid convolution
    
    # Actually, let me think again. g = np.convolve(h, h) has length 2*N-1 = m
    # g[j] = sum_{i=0}^{N-1} h[i] * h[j-i] where h[k]=0 for k<0 or k>=N
    # dg[j]/dh[i] = h[j-i] (with h out of range = 0)
    
    # So df/dh[i] = sum_{j=0}^{m-1} df_dg[j] * h[j-i]
    # where h[j-i] = 0 if j-i < 0 or j-i >= N
    
    # This is: for each i, sum over j where 0<=j-i<N, i.e., i <= j < i+N
    # = sum_{j=i}^{i+N-1} df_dg[j] * h[j-i] (with bounds on j: 0<=j<m)
    
    # Let me compute this directly but efficiently
    df_dh = np.zeros(N)
    for i in range(N):
        j_start = max(0, i)
        j_end = min(m, i + N)
        if j_start < j_end:
            # h[j-i] for j in [j_start, j_end)
            # j-i ranges from j_start-i to j_end-1-i
            # j_start-i >= 0 (since j_start >= i or j_start=0 and i could be > 0... wait)
            # If j_start = i, then j-i starts at 0
            # If j_start = 0 and i > 0, then j-i starts at -i... but we said j_start = max(0,i)
            # So j_start = i if i >= 0, which is always true
            # So j-i ranges from 0 to j_end-1-i
            h_indices = np.arange(j_start - i, j_end - i)
            df_dh[i] = np.sum(df_dg[j_start:j_end] * h[h_indices])
    
    return -df_dh  # negative because we minimize -f

def _generate_initializations():
    """Generate diverse initializations."""
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Spike in middle
    spike = np.zeros(N)
    spike[N//2] = 1.0
    inits.append(spike)
    
    # Two spikes
    for offset in [5, 10, 15]:
        s = np.zeros(N)
        s[N//2 - offset] = 1.0
        s[N//2 + offset] = 1.0
        inits.append(s)
    
    # Triangular
    tri = np.linspace(1, 0, N)
    tri = np.concatenate([tri, np.linspace(0, 1, N-1)])
    tri = tri[:N]
    inits.append(tri)
    
    # Cosine window
    x = np.linspace(0, np.pi, N)
    inits.append(np.cos(x)**2 + 0.1)
    
    # Sparse random
    rng = np.random.RandomState(42)
    for _ in range(5):
        s = np.zeros(N)
        indices = rng.choice(N, size=rng.randint(3, 10), replace=False)
        s[indices] = rng.rand(len(indices)) + 0.1
        inits.append(s)
    
    # Gaussian-like
    x = np.arange(N)
    inits.append(np.exp(-0.1 * (x - N//2)**2) + 0.05)
    
    return inits

def _coordinate_descent_polish(h, max_iters=100):
    """Coordinate descent polish for the exact objective."""
    h = np.maximum(h, 0.0).copy()
    best_val = _compute_value(h)
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        improved = False
        for i in range(N):
            # Try a few perturbations of h[i]
            base = h[i]
            best_local = base
            best_local_val = _compute_value(h)
            
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                h[i] = max(0.0, base + delta)
                val = _compute_value(h)
                if val > best_local_val:
                    best_local_val = val
                    best_local = h[i]
            
            h[i] = best_local
            if best_local != base:
                improved = True
        
        if not improved:
            break
    
    return h

def construct():
    global START_TIME, TIME_LIMIT
    
    # Generate diverse initializations
    inits = _generate_initializations()
    
    best_h = None
    best_val = -np.inf
    
    # Evaluate all initializations
    for h0 in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.maximum(h0, 0.0)
        val = _compute_value(h0)
        if val > best_val:
            best_val = val
            best_h = h0.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = _compute_value(best_h)
    
    # Multi-stage optimization with increasing beta
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        def obj(x):
            return _smooth_objective(x, beta)
        
        def grad(x):
            return _analytical_gradient(x, beta)
        
        try:
            result = minimize(
                obj,
                current_h,
                jac=grad,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            
            if result.success or result.fun < obj(current_h):
                candidate = np.maximum(result.x, 0.0)
                val = _compute_value(candidate)
                if val > best_val:
                    best_val = val
                    best_h = candidate.copy()
                current_h = candidate
        except Exception:
            pass
        
        # Check time
        if time.time() - START_TIME > TIME_LIMIT:
            break
    
    # Final polish with coordinate descent
    if time.time() - START_TIME < TIME_LIMIT - 10:
        polished = _coordinate_descent_polish(best_h, max_iters=50)
        val = _compute_value(polished)
        if val > best_val:
            best_val = val
            best_h = polished
    
    # Try a few random perturbations from best
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng = np.random.RandomState(42)
        for _ in range(30):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            perturbation = rng.randn(N) * 0.05
            candidate = np.maximum(best_h + perturbation, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)

# Ensure the function is defined at module level for the evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Value: {_compute_value(result)}")
