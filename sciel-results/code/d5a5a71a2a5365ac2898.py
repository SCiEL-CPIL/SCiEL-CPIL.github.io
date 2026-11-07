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
    Smooth objective: maximize l2 / (norm_1 * norm_inf_smooth)
    where norm_inf_smooth = (1/beta) * log(sum(exp(beta*g_i)))
    We minimize the negative of this.
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    norm_inf_smooth = _soft_norm_inf(g, beta)
    
    if norm_1 * norm_inf_smooth < 1e-30:
        return 0.0
    
    val = l2 / (norm_1 * norm_inf_smooth)
    return -val  # minimize negative

def _analytical_gradient(h, beta):
    """
    Compute analytical gradient of the smooth objective.
    Uses the chain rule and properties of convolution.
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    # Compute l2 and its gradient
    y0 = y[:m+1]
    y1 = y[1:m+2]
    
    # l2 = (w/3) * sum(y0^2 + y0*y1 + y1^2)
    # dl2/dy0 = (w/3) * 2*(y0 + y1/2) = (w/3) * (2*y0 + y1)
    # dl2/dy1 = (w/3) * 2*(y1 + y0/2) = (w/3) * (2*y1 + y0)
    
    dl2_dy0 = (w / 3.0) * (2.0 * y0 + y1)
    dl2_dy1 = (w / 3.0) * (2.0 * y1 + y0)
    
    # y[1:m+1] = g, so dy/dg = I (identity for g elements)
    # dl2/dg = [dl2_dy1[0], dl2_dy0[0] + dl2_dy1[1], dl2_dy0[1] + dl2_dy1[2], ..., dl2_dy0[m-1]]
    # More carefully: y[i] for i in 1..m corresponds to g[i-1]
    # y[0] and y[m+1] are 0
    
    dl2_dg = np.zeros(m)
    # Contribution from y0 (indices 0..m of y, i.e., g[-1] doesn't exist, g[0]..g[m-1])
    # y0[i] = y[i] for i in 0..m
    # y0[0] = y[0] = 0, y0[1] = y[1] = g[0], ..., y0[m] = y[m] = g[m-1]
    # y1[i] = y[i+1] for i in 0..m
    # y1[0] = y[1] = g[0], y1[1] = y[2] = g[1], ..., y1[m-1] = y[m] = g[m-1], y1[m] = y[m+1] = 0
    
    # dl2/dg[j] = dl2_dy0[j+1] + dl2_dy1[j] for j in 0..m-1
    # (since g[j] appears in y[j+1] as y0[j+1] and in y[j+1] as y1[j])
    
    for j in range(m):
        contrib = 0.0
        # g[j] appears in y[j+1]
        # In y0: y0[j+1] = y[j+1] if j+1 <= m, i.e., j <= m-1
        if j + 1 <= m:
            contrib += dl2_dy0[j + 1]
        # In y1: y1[j] = y[j+1] if j <= m, i.e., j <= m-1
        if j <= m:
            contrib += dl2_dy1[j]
        dl2_dg[j] = contrib
    
    # norm_1 = sum(g) / (m+1), so d(norm_1)/dg = ones/(m+1)
    dnorm1_dg = np.ones(m) / (m + 1)
    
    # norm_inf_smooth = soft max, gradient is softmax
    max_g = np.max(g)
    shifted = beta * (g - max_g)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    softmax = exp_shifted / sum_exp  # d(soft max)/dg
    
    # val = l2 / (norm_1 * norm_inf_smooth)
    # dval/dg = (dl2/dg * norm_1 * norm_inf - l2 * (dnorm1/dg * norm_inf + norm_1 * dnorm_inf/dg)) / (norm_1 * norm_inf)^2
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = _soft_norm_inf(g, beta)
    
    if norm_1 * norm_inf < 1e-30:
        return np.zeros_like(h)
    
    dval_dg = (dl2_dg * norm_1 * norm_inf - 
               _smooth_objective(h, beta) * (-1.0) * (dnorm1_dg * norm_inf + norm_1 * softmax)) / \
              (norm_1 * norm_inf) ** 2
    
    # dval/dh: g = convolve(h, h), so g[k] = sum_i h[i]*h[k-i]
    # dg[k]/dh[j] = h[k-j] + h[j-k] (for valid indices)
    # dval/dh[j] = sum_k dval_dg[k] * (h[k-j] + h[j-k])
    
    dval_dh = np.zeros(N)
    for j in range(N):
        for k in range(m):
            idx1 = k - j
            idx2 = j - k
            contrib = 0.0
            if 0 <= idx1 < N:
                contrib += dval_dg[k] * h[idx1]
            if 0 <= idx2 < N:
                contrib += dval_dg[k] * h[idx2]
            dval_dh[j] += contrib
    
    return -dval_dh  # We're minimizing -val

def _generate_initializations():
    """Generate diverse initializations for multi-start optimization."""
    rng = np.random.RandomState(42)
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Spike at center
    spike = np.zeros(N)
    spike[N//2] = 1.0
    inits.append(spike)
    
    # Two spikes
    spike2 = np.zeros(N)
    spike2[N//4] = 1.0
    spike2[3*N//4] = 1.0
    inits.append(spike2)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 * 2)
    inits.append(gauss / np.max(gauss))
    
    # Random
    inits.append(rng.rand(N))
    
    # Triangular
    tri = 1.0 - np.abs(x)
    inits.append(tri / np.max(tri))
    
    # Random with structure
    for _ in range(3):
        base = rng.rand(N)
        base = np.maximum(base, 0.1)  # Ensure some minimum height
        inits.append(base / np.max(base))
    
    return inits

def construct():
    global START_TIME, TIME_LIMIT
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_value(best_h)
    
    # Multi-stage optimization with increasing beta
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    
    # Generate diverse initializations
    inits = _generate_initializations()
    
    # For each initialization, run the multi-stage optimization
    for init_idx, init_h in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        current_h = init_h.copy()
        init_val = _compute_value(current_h)
        
        if init_val > best_val:
            best_val = init_val
            best_h = current_h.copy()
        
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
    
    # Final high-precision polish with very high beta
    if time.time() - START_TIME < TIME_LIMIT - 10:
        beta_final = 1000.0
        current_h = best_h.copy()
        
        def obj(x):
            return _smooth_objective(x, beta_final)
        
        def grad(x):
            return _analytical_gradient(x, beta_final)
        
        try:
            result = minimize(
                obj,
                current_h,
                jac=grad,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000}
            )
            
            candidate = np.maximum(result.x, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()
        except Exception:
            pass
    
    # Final polish: try a few random perturbations from best
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng = np.random.RandomState(123)
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
