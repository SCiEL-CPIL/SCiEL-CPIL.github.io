import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 110.0  # Safety margin before 120s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _smooth_max_grad(heights, M, alpha):
    """Compute smooth max of autoconvolution and its gradient w.r.t. heights.
    Returns (smooth_max_value, gradient_array, peak_idx)."""
    n = len(heights)
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    m = np.max(conv)
    if m < 0:
        m = 0.0
    
    # Numerically stable log-sum-exp
    shifted = alpha * (conv - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp <= 0:
        sum_exp = 1e-300
    
    smooth_max = m + np.log(sum_exp) / alpha
    weights = exp_shifted / sum_exp
    
    # Gradient of smooth_max w.r.t. f
    # d(smooth_max)/d(f_i) = sum_k weights[k] * d(conv_k)/d(f_i)
    # conv_k = sum_j f_j * f_{k-j} (with zero-padding)
    # d(conv_k)/d(f_i) = f_{k-i} + f_{i-k} (symmetric)
    # So d(smooth_max)/d(f_i) = sum_k weights[k] * (f_{k-i} + f_{i-k})
    #                         = 2 * sum_k weights[k] * f_{k-i}  (by symmetry, with proper indexing)
    
    # Efficient computation via FFT:
    # correlation[i] = sum_k weights[k] * f_{k-i}
    W = np.fft.rfft(weights)
    corr = np.fft.irfft(W * np.conj(F), n=M)
    grad_f = 2.0 * corr[:n]
    
    # Now convert to gradient w.r.t. the normalized score
    S = np.sum(heights)
    w_val = 0.5 / n
    if S <= 1e-10:
        return smooth_max, np.zeros(n), 0
    
    # score = smooth_max * n / (S^2 * w_val)
    # d(score)/d(h_i) = (n / (S^2 * w_val)) * grad_f[i] - smooth_max * (2n / (S^3 * w_val))
    obj_grad = grad_f * (n / (S * S * w_val)) - smooth_max * (2.0 * n / (S * S * S * w_val))
    
    peak_idx = np.argmax(conv)
    return smooth_max, obj_grad, peak_idx

def _simulated_annealing_optimize(start_time, deadline):
    """Multi-start simulated annealing with adaptive step size."""
    def time_left():
        return deadline - time.time()
    
    w = 0.5 / N
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    def compute_score_fast(heights):
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        max_conv = np.max(conv)
        total = np.sum(heights)
        if total <= 0:
            return np.inf
        return max_conv * N / (total * total * w)
    
    best_score = np.inf
    best_h = None
    
    rng = np.random.default_rng(42)
    
    # Multiple starts from different initial configurations
    initializations = [
        np.ones(N, dtype=np.float64),
        np.exp(-((np.linspace(0, 1, N) - 0.5) / 0.3) ** 2),
        np.concatenate([np.ones(N//3), np.zeros(N//3), np.ones(N - 2*(N//3))]),
        np.maximum(1.0 - 0.5 * (2 * np.linspace(0, 1, N) - 1) ** 2, 0.0),
        np.concatenate([np.full(N//4, 0.5), np.ones(N//2), np.full(N - 3*(N//4), 0.5)]),
    ]
    
    for start_idx, h_init in enumerate(initializations):
        if time_left() < 25:
            break
        
        h = np.maximum(h_init, 0.0)
        if np.sum(h) <= 0:
            continue
        
        current_score = compute_score_fast(h)
        current_h = h.copy()
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
        
        # Simulated annealing parameters
        T_start = 0.005
        T_end = 0.00005
        n_iterations = 3000
        
        T = T_start
        step_scale = 0.03
        
        for it in range(n_iterations):
            if time_left() < 12:
                break
            
            # Adaptive step size: larger when T is high
            noise_scale = step_scale * (T / T_start) ** 0.5
            noise_scale = max(noise_scale, 0.001)
            
            h_new = current_h + rng.normal(0, noise_scale, size=N)
            h_new = np.maximum(h_new, 0.0)
            
            new_score = compute_score_fast(h_new)
            
            if new_score < current_score:
                current_h = h_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
            else:
                delta = new_score - current_score
                if delta < 1e-12:
                    accept = True
                else:
                    accept_prob = np.exp(-delta / T)
                    accept = rng.random() < accept_prob
                
                if accept:
                    current_h = h_new
                    current_score = new_score
            
            T = max(T * 0.998, T_end)
    
    return best_score, best_h

def _gradient_optimize(h_init, start_time, deadline, max_iter=500):
    """Gradient-based optimization with smooth max surrogate."""
    def time_left():
        return deadline - time.time()
    
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    w = 0.5 / N
    h = np.maximum(h_init, 0.0).copy()
    
    best_score = compute_score(h)
    best_h = h.copy()
    
    alpha = 50.0
    lr = 0.005
    momentum = 0.9
    v = np.zeros(N)
    
    prev_grad = None
    prev_obj = None
    
    for it in range(max_iter):
        if time_left() < 10:
            break
        
        sm, obj_grad, peak_idx = _smooth_max_grad(h, M, alpha)
        
        S = np.sum(h)
        if S <= 1e-10:
            break
        
        # Nesterov-like momentum
        if prev_grad is not None:
            v = momentum * v + (1 - momentum) * obj_grad
        else:
            v = obj_grad.copy()
        
        h_new = h - lr * v
        h_new = np.maximum(h_new, 0.0)
        
        new_score = compute_score(h_new)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
        
        # Adaptive learning rate
        if prev_obj is not None:
            obj_change = new_score - prev_obj
            if obj_change > 0:
                lr = max(lr * 0.85, 0.0001)
            else:
                lr = min(lr * 1.05, 0.05)
        
        h = h_new
        prev_grad = obj_grad
        prev_obj = new_score
        
        # Increase alpha over time for sharper max approximation
        if it % 50 == 0 and it > 0:
            alpha = min(alpha * 1.3, 1000.0)
    
    return best_score, best_h

def _structured_candidates():
    """Generate a set of structured candidate shapes."""
    x = np.linspace(0, 1, N)
    candidates = []
    
    # Tapered edges (cosine taper)
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        h_taper = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            h_taper[:taper_len] = ramp
            h_taper[-taper_len:] = ramp[::-1]
        candidates.append(h_taper)
    
    # Gaussian bump
    for width in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        h_gauss = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(h_gauss)
    
    # Parabolic
    for a in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        h_par = 1.0 - a * (2 * x - 1) ** 2
        h_par = np.maximum(h_par, 0.0)
        if np.max(h_par) > 0:
            candidates.append(h_par)
    
    # Linear taper
    for slope in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        h_lin = 1.0 - slope * np.abs(2 * x - 1)
        h_lin = np.maximum(h_lin, 0.0)
        if np.max(h_lin) > 0:
            candidates.append(h_lin)
    
    # Plateau with smooth edges
    for plateau_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        h_plat = np.ones(N)
        edge_len = int((1 - plateau_frac) * N / 2)
        if edge_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
            h_plat[:edge_len] = ramp
            h_plat[-edge_len:] = ramp[::-1]
        candidates.append(h_plat)
    
    # Two-block structures
    for frac1 in [0.2, 0.3, 0.4]:
        for gap in [0.1, 0.2, 0.3]:
            h_two = np.zeros(N)
            block1 = int(frac1 * N)
            block2 = int(frac1 * N)
            gap_len = int(gap * N)
            h_two[:block1] = 1.0
            h_two[N - block2:] = 1.0
            # Smooth edges
            if block1 > 10:
                edge = int(0.1 * block1)
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge)))
                h_two[:edge] = ramp
                h_two[block1-edge:block1] = ramp[::-1]
                h_two[N-block2:N-block2+edge] = ramp
                h_two[N-block2+block1-edge:] = ramp[::-1]
            candidates.append(h_two)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - start_time
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Phase 1: Evaluate structured candidates
    if time_left() < DEADLINE - 10:
        for h_cand in _structured_candidates():
            if time_left() < DEADLINE - 15:
                break
            h_cand = np.maximum(h_cand, 0.0)
            if np.sum(h_cand) <= 0:
                continue
            score = compute_score(h_cand)
            if score < best_score:
                best_score = score
                best_h = h_cand.copy()
    
    # Phase 2: Simulated annealing from multiple starts
    if time_left() < DEADLINE - 20:
        s_sa, h_sa = _simulated_annealing_optimize(start_time, deadline)
        if h_sa is not None and s_sa < best_score:
            best_score = s_sa
            best_h = h_sa.copy()
    
    # Phase 3: Gradient-based optimization from best found so far
    if time_left() < DEADLINE - 15:
        s_grad, h_grad = _gradient_optimize(best_h, start_time, deadline, max_iter=400)
        if s_grad < best_score:
            best_score = s_grad
            best_h = h_grad.copy()
    
    # Phase 4: Refine with SA from the gradient result
    if time_left() < DEADLINE - 10:
        s_refine, h_refine = _simulated_annealing_optimize(start_time, deadline)
        # Also try SA starting from best_h
        if h_refine is not None and s_refine < best_score:
            best_score = s_refine
            best_h = h_refine.copy()
    
    # Phase 5: Final gradient polish
    if time_left() < DEADLINE - 5:
        s_final, h_final = _gradient_optimize(best_h, start_time, deadline, max_iter=200)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    # Final cleanup
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
