import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _objective_complex(x):
    """
    Objective function for complex parameterization.
    x is a real vector of length 2*N representing Re(G) and Im(G).
    h = |G|^2.
    We minimize log(||h||_1) + log(||h*h||_inf) subject to ||h||_2 = 1.
    Actually, since it's scale invariant, we can just normalize h to L2=1 inside.
    Score = 1 / (||h||_1 * ||h*h||_inf) when ||h||_2=1.
    So we minimize log(||h||_1 * ||h*h||_inf).
    
    To handle the non-smooth max, we use a soft-max approximation for the 
    log-sum-exp of the convolution peaks.
    """
    G = x[:N] + 1j * x[N:]
    h = np.abs(G) ** 2
    
    # Normalize h to L2 = 1
    l2_norm = np.sqrt(np.sum(h ** 2))
    if l2_norm < 1e-15:
        return 1e10
    h = h / (l2_norm ** 2) # Wait, if h = |G|^2, then ||h||_2^2 = sum(h^2). 
    # Let's be precise.
    # Let h_raw = |G|^2.
    # We want to evaluate ratio for h_normalized = h_raw / ||h_raw||_2.
    # Ratio is scale invariant. So we can just use h_raw and compute the ratio directly?
    # No, the ratio is R = ||h||_2^2 / (||h||_1 ||h*h||_inf).
    # If we scale h by c, R is invariant.
    # So we can just compute R for h_raw.
    # R = sum(h^2) / (sum(h) * max(conv(h)))
    # Minimize -log(R) = log(sum(h)) + log(max(conv(h))) - log(sum(h^2))
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
        
    l2_sq = np.sum(h ** 2)
    if l2_sq < 1e-15:
        return 1e10
        
    conv = _autoconvolution(h)
    
    # Use soft max for the peak of the convolution to make it differentiable
    # The peak is likely at the center (lag 0) or near it for non-negative h.
    # However, for complex G, h is non-negative, so conv is non-negative.
    # The max is a non-smooth function.
    # Let's use a log-sum-exp with a temperature parameter.
    # But N=50, so we can also just find the argmax and use that?
    # L-BFGS-B with numerical gradients can handle kinks if we are careful, 
    # but analytic gradients are better.
    # Let's try a smooth approximation:
    # log_max(x) approx max(x) + log(sum(exp((x-max)/T))) * T
    # For the objective, we want to minimize log(l1) + log(max_conv).
    # So we need log(max_conv).
    
    # If we use the exact max, the gradient is subgradient.
    # Let's use a high-temperature soft max to ensure smoothness, then anneal?
    # Or just use the exact max and rely on the optimizer's robustness?
    # The inspiration 1 used _smooth_max. Let's implement a smooth max.
    
    # However, for the final score, we must use the exact max.
    # For the optimization, smooth max is better.
    
    T = 0.1 # Temperature for soft max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    
    # log-sum-exp trick for stability
    # log(max_conv) approx m + T * log(sum(exp((conv - m)/T)))
    # But we want the actual log of the value.
    # Actually, if we just use m, it's non-smooth.
    # Let's use the soft max value itself as the "max" in the objective?
    # Objective: log(l1) + log(soft_max_conv)
    
    exp_conv = np.exp((conv - m) / T)
    log_sum_exp = m + T * np.log(np.sum(exp_conv))
    
    return np.log(l1) + np.log(log_sum_exp) - np.log(l2_sq)

def _construct_starts_complex():
    """Generate starting points for the complex parameterization."""
    starts = []
    rng = np.random.RandomState(42)
    
    # 1. Real positive Gaussian (Im=0)
    x = np.arange(N)
    g_real = np.exp(-((x - N//2)**2) / (2 * (N/6)**2))
    starts.append(np.concatenate([g_real, np.zeros(N)]))
    
    # 2. Uniform real
    starts.append(np.concatenate([np.ones(N), np.zeros(N)]))
    
    # 3. Random complex
    for _ in range(10):
        re = rng.normal(0, 1, N)
        im = rng.normal(0, 1, N)
        starts.append(np.concatenate([re, im]))
        
    # 4. Sparse real
    for _ in range(5):
        re = np.zeros(N)
        idx = rng.choice(N, rng.randint(5, 15), replace=False)
        re[idx] = rng.uniform(0.5, 1.0, len(idx))
        starts.append(np.concatenate([re, np.zeros(N)]))
        
    # 5. Sinc-like real
    for a in [0.5, 1.0, 2.0]:
        re = np.abs(np.sinc(a * (x - N//2) / N))
        starts.append(np.concatenate([re, np.zeros(N)]))
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    best_x = None
    
    starts = _construct_starts_complex()
    rng = np.random.RandomState(123)
    rng.shuffle(starts)
    
    for s in starts:
        if time.time() > deadline - 15:
            break
            
        x0 = np.asarray(s, dtype=float)
        
        # Optimize using L-BFGS-B
        # We don't have bounds for complex parameters (they can be any real/imag)
        try:
            res = minimize(
                _objective_complex, 
                x0, 
                method='L-BFGS-B', 
                # bounds=[(None, None)] * (2*N), # Unbounded
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        x_opt = res.x
        G = x_opt[:N] + 1j * x_opt[N:]
        h_opt = np.abs(G) ** 2
        
        # Ensure non-negative (should be by construction, but just in case of numerical issues)
        h_opt = np.maximum(h_opt, 0)
        
        # Normalize for score calculation? No, score is scale invariant.
        # But let's check if it's zero
        if np.sum(h_opt) == 0:
            continue
            
        score = _score(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
            best_x = x_opt.copy()
            
        # Local refinement from this result
        if time.time() > deadline - 10:
            break
        try:
            # Increase maxiter for polish
            res2 = minimize(
                _objective_complex, 
                x_opt, 
                method='L-BFGS-B', 
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_opt2 = res2.x
            G2 = x_opt2[:N] + 1j * x_opt2[N:]
            h_opt2 = np.abs(G2) ** 2
            h_opt2 = np.maximum(h_opt2, 0)
            
            score2 = _score(h_opt2)
            if score2 > best_score:
                best_score = score2
                best_h = h_opt2.copy()
                best_x = x_opt2.copy()
        except Exception:
            pass
    
    # Final polish with best found
    if best_score > 0 and best_x is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _objective_complex, 
                best_x, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_final = np.abs(res.x[:N] + 1j * res.x[N:] ** 2) # Wait, res.x is real vector
            G_final = res.x[:N] + 1j * res.x[N:]
            h_final = np.abs(G_final) ** 2
            h_final = np.maximum(h_final, 0)
            
            score_final = _score(h_final)
            if score_final > best_score:
                best_h = h_final
        except Exception:
            pass
    
    # Ensure non-negative and valid
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
