import time
import numpy as np

N = 600
DEADLINE = 135.0

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
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

def _smooth_max_and_weights(conv, alpha):
    """Compute smooth max and weights for log-sum-exp."""
    m = np.max(conv)
    if m < 0:
        m = 0.0
    shifted = alpha * (conv - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp < 1e-300:
        weights = np.zeros_like(conv)
        weights[np.argmax(conv)] = 1.0
        sm = m
    else:
        weights = exp_shifted / sum_exp
        sm = m + np.log(sum_exp) / alpha
    return sm, weights

def _grad_smooth_max(weights, heights, M):
    """Compute gradient of smooth max autoconvolution w.r.t. heights."""
    W = np.fft.rfft(weights)
    F = np.fft.rfft(heights)
    corr = np.fft.irfft(W * np.conj(F), n=M)
    return 2.0 * corr[:N]

def _score_and_grad(heights, alpha):
    """Compute score and gradient in log-domain (x = log(h))."""
    h = np.exp(heights)
    S = np.sum(h)
    w = 0.5 / N
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:N] = h
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    sm, weights = _smooth_max_and_weights(conv, alpha)
    grad_f = _grad_smooth_max(weights, h, M)
    
    # Score = sm * N / (S^2 * w)
    # d(score)/d(x_i) = d(score)/d(h_i) * h_i
    # d(sm)/d(h_i) = grad_f[i]
    # d(S)/d(h_i) = 1
    # d(score)/d(h_i) = grad_f[i] * N / (S^2 * w) - sm * 2*N / (S^3 * w)
    
    if S < 1e-300:
        return 1e10, np.zeros(N)
    
    dscore_dh = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
    dscore_dx = dscore_dh * h
    
    score = sm * N / (S * S * w)
    return score, dscore_dx

def _bfgs_log_domain(x_init, alpha_start, alpha_end, max_iter, deadline, start_time):
    """BFGS optimization in log-domain for h = exp(x)."""
    def time_left():
        return deadline - time.time()
    
    x = x_init.copy()
    n = len(x)
    
    # BFGS memory
    m_mem = min(20, n)
    S_list = []  # step vectors
    Y_list = []  # gradient difference vectors
    
    # Initialize inverse Hessian as identity
    H = np.eye(n)
    
    best_score = np.inf
    best_x = x.copy()
    
    # Initial gradient
    score, grad = _score_and_grad(x, alpha_start)
    if score < best_score:
        best_score = score
        best_x = x.copy()
    
    for it in range(max_iter):
        if time_left() < 5:
            break
        
        # Check for convergence
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        # Line search (backtracking)
        alpha_ls = 0.5
        c1 = 1e-4
        step_dir = -H @ grad
        
        # Ensure step direction is descent
        if np.dot(step_dir, grad) > 0:
            step_dir = -grad
        
        x_trial = x + alpha_ls * step_dir
        score_trial, grad_trial = _score_and_grad(x_trial, alpha_start)
        
        # Backtracking
        max_ls = 20
        for ls_iter in range(max_ls):
            if score_trial < score + c1 * alpha_ls * np.dot(grad, step_dir):
                break
            alpha_ls *= 0.5
            x_trial = x + alpha_ls * step_dir
            score_trial, grad_trial = _score_and_grad(x_trial, alpha_start)
            if alpha_ls < 1e-10:
                break
        
        if alpha_ls < 1e-10:
            # Try smaller step
            alpha_ls = 1e-8
            x_trial = x + alpha_ls * step_dir
            score_trial, grad_trial = _score_and_grad(x_trial, alpha_start)
        
        # BFGS update
        s = x_trial - x
        y = grad_trial - grad
        sy = np.dot(s, y)
        
        if sy > 1e-12:
            S_list.append(s)
            Y_list.append(y)
            if len(S_list) > m_mem:
                S_list.pop(0)
                Y_list.pop(0)
            
            # Update H (inverse Hessian approximation)
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        x = x_trial
        score = score_trial
        grad = grad_trial
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Gradually increase alpha for sharper max approximation
        progress = (it + 1) / max_iter
        alpha_current = alpha_start * (alpha_end / alpha_start) ** progress
        # Recompute gradient with new alpha
        score, grad = _score_and_grad(x, alpha_current)
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    return best_x, best_score

def _parametric_init(rng, n_params=8):
    """Generate diverse initial points in log-domain."""
    x = np.zeros(N)
    # Start with log of a reasonable height profile
    base = np.ones(N) * 0.5
    # Add random smooth perturbations
    for _ in range(rng.integers(3, 8)):
        k = rng.integers(1, N // 4)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.1, 0.5)
        x += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
    # Ensure log-domain values are reasonable
    x = np.clip(x, -10, 5)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform in log-domain: log(1) = 0
    x_init = np.zeros(N)
    best_x = x_init.copy()
    best_score = compute_score(np.exp(x_init))
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Try several initializations
    n_restarts = 15
    max_iter_per_restart = 200
    
    for restart in range(n_restarts):
        if time_left() < 30:
            break
        
        if restart == 0:
            x_r = np.zeros(N)
        else:
            x_r = _parametric_init(rng)
        
        # Use BFGS in log-domain
        alpha_start = 30.0
        alpha_end = 300.0
        x_opt, score_opt = _bfgs_log_domain(
            x_r, alpha_start, alpha_end, 
            max_iter_per_restart, deadline, start_time
        )
        
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt.copy()
    
    # Final polish with high alpha
    if time_left() > 20:
        alpha_polish = 500.0
        x_polish = best_x.copy()
        score_p, grad_p = _score_and_grad(x_polish, alpha_polish)
        
        # Simple gradient descent with small steps
        lr = 0.001
        for it in range(100):
            if time_left() < 10:
                break
            
            x_polish -= lr * grad_p
            score_p, grad_p = _score_and_grad(x_polish, alpha_polish)
            
            if score_p < best_score:
                best_score = score_p
                best_x = x_polish.copy()
            
            lr *= 0.98
    
    # Convert back from log-domain
    h_final = np.exp(best_x)
    h_final = np.maximum(h_final, 0.0)
    
    # Final verification
    final_score = compute_score(h_final)
    if final_score < best_score:
        best_score = final_score
    
    return list(h_final)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
