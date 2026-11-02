"""Direct gradient optimization on signal domain with sparse-recovery-inspired regularization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad(x, beta=200.0):
    """
    Objective and gradient for direct optimization on x.
    Uses soft-max surrogate for max|ac|.
    
    ac[k] = sum_j x[j] * x[k-j] (linear autoconv)
    d(ac[k])/dx[i] = x[k-i] + x[i-k] for valid indices (but for linear conv, it's just 2*x[k-i] when 0<=k-i<N)
    
    Actually for linear autoconvolution:
    ac[k] = sum_{j} x[j]*x[k-j] where 0<=j<N and 0<=k-j<N
    d(ac[k])/dx[i] = sum_j [delta(j=i)*x[k-j] + x[j]*delta(k-j=i)]
                   = x[k-i] (if 0<=k-i<N) + x[k-i] (if 0<=k-i<N)
                   = 2*x[k-i] if 0<=k-i<N, else 0
    
    But wait - this double counts. Let me reconsider.
    ac[k] = sum_{j=max(0,k-N+1)}^{min(k,N-1)} x[j]*x[k-j]
    
    d(ac[k])/dx[i] = x[k-i] (from j=i term, if valid) + x[k-i] (from k-j=i term, if valid)
    
    If i is in [max(0,k-N+1), min(k,N-1)], then both terms contribute x[k-i].
    So d(ac[k])/dx[i] = 2*x[k-i] for i in valid range, 0 otherwise.
    """
    s = np.sum(x)
    s_sq = s * s
    if abs(s_sq) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    
    if max_ac < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max weights
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_ac + (1.0 / beta) * np.log(sum_exp)
    
    val = soft_max / s_sq
    
    # Gradient: d(val)/dx[i] = (1/s_sq) * d(soft_max)/dx[i] - 2*soft_max/s^3 * s[i]
    # where s[i] = 1 for all i (since s = sum x)
    
    # d(soft_max)/dx[i] = sum_k weights[k] * sign(ac[k]) * d(ac[k])/dx[i]
    # d(ac[k])/dx[i] = 2*x[k-i] if 0<=k-i<N, else 0
    
    grad_soft = np.zeros(N)
    for k in range(len(ac)):
        if weights[k] < 1e-10:
            continue
        sign_ac = np.sign(ac[k])
        # For this k, grad_soft[i] += weights[k] * sign_ac * 2 * x[k-i] for valid i
        valid_i = np.arange(max(0, k - N + 1), min(k + 1, N))
        if len(valid_i) > 0:
            grad_soft[valid_i] += weights[k] * sign_ac * 2.0 * x[k - valid_i]
    
    # d(val)/dx[i] = grad_soft[i]/s_sq - 2*soft_max/(s_sq*s)
    grad = grad_soft / s_sq - 2.0 * soft_max / (s_sq * s)
    
    return val, grad

def _gradient_descent(x0, max_iters=500, lr_init=0.01, beta=200.0):
    """
    Gradient descent with adaptive learning rate.
    """
    best_x = x0.copy()
    best_val, _ = _objective_grad(best_x, beta)
    
    lr = lr_init
    patience = 0
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_grad(best_x, beta)
        
        if val < best_val:
            best_val = val
            best_x = best_x.copy()
            patience = 0
        else:
            patience += 1
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step
        trial = best_x - lr * grad / grad_norm
        
        # Check if improvement
        trial_val = _objective(trial)
        if trial_val < best_val:
            best_x = trial.copy()
            best_val = trial_val
            patience = 0
            # Slightly increase lr if improving
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
            if patience > 20:
                break
    
    return best_x, best_val

def _sparse_recovery_init(rng):
    """
    Generate initial signals inspired by sparse signal recovery.
    Start with sparse patterns and expand.
    """
    candidates = []
    
    # All ones baseline
    candidates.append(np.ones(N))
    
    # Sparse + dense mix
    for sparsity in [0.1, 0.2, 0.3, 0.5]:
        n_sparse = int(N * sparsity)
        idx = rng.choice(N, size=n_sparse, replace=False)
        x = np.zeros(N)
        x[idx] = 1.0
        # Add small dense component
        x += 0.1 * rng.randn(N)
        candidates.append(x)
    
    # Block patterns
    for block_size in [5, 10, 20, 50]:
        n_blocks = N // block_size
        for pattern in ['alternate', 'random', 'gradient']:
            x = np.zeros(N)
            if pattern == 'alternate':
                for b in range(n_blocks):
                    if b % 2 == 0:
                        x[b*block_size:(b+1)*block_size] = 1.0
                    else:
                        x[b*block_size:(b+1)*block_size] = -0.5
            elif pattern == 'random':
                for b in range(n_blocks):
                    x[b*block_size:(b+1)*block_size] = rng.choice([-1, 1])
            else:  # gradient
                for b in range(n_blocks):
                    x[b*block_size:(b+1)*block_size] = 1.0 - 2.0 * b / n_blocks
            candidates.append(x)
    
    # Frequency-based patterns
    for freq in [1, 2, 3, 5, 8, 13]:
        t = np.arange(N)
        candidates.append(np.cos(2 * np.pi * freq * t / N))
        candidates.append(np.sign(np.cos(2 * np.pi * freq * t / N)))
    
    # Random smooth signals
    for _ in range(10):
        # Generate in frequency domain with smooth spectrum
        freqs = np.arange(N // 2 + 1)
        mask = rng.random(N // 2 + 1) > 0.3  # Keep 70% of frequencies
        X = np.zeros(N // 2 + 1, dtype=complex)
        X[mask] = rng.randn(mask.sum()) + 1j * rng.randn(mask.sum())
        # Smooth by zeroing high frequencies
        X[N//4:] = 0
        x = np.fft.irfft(X, n=N)
        candidates.append(x)
    
    return candidates

def construct():
    rng = np.random.RandomState(42)
    
    # Generate diverse initial candidates
    candidates = _sparse_recovery_init(rng)
    
    # Evaluate all candidates
    best_x = None
    best_val = 1e10
    for x0 in candidates:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if best_x is None:
        return np.ones(N)
    
    if _check_deadline():
        return best_x
    
    # Run gradient descent from multiple starting points
    # Start from the best candidate and a few others
    start_points = [best_x]
    # Add top 3 other candidates
    sorted_candidates = sorted(candidates, key=lambda x: _objective(x))
    for x0 in sorted_candidates[1:4]:
        start_points.append(x0)
    
    for x0 in start_points:
        if _check_deadline():
            break
        
        # Try different beta values
        for beta in [100, 200, 500]:
            if _check_deadline():
                break
            
            x_opt, val_opt = _gradient_descent(x0, max_iters=200, lr_init=0.01, beta=beta)
            
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Final polish with finer gradient descent
    best_x, best_val = _gradient_descent(best_x, max_iters=300, lr_init=0.005, beta=500)
    
    if _check_deadline():
        return best_x
    
    # Coordinate descent refinement
    best_x = best_x.copy()
    for _ in range(20):
        if _check_deadline():
            break
        
        improved = False
        for j in range(0, N, 2):  # Step by 2 for speed
            if _check_deadline():
                return best_x
            
            for delta in [0.05, 0.02, 0.01]:
                for sign in [-1, 1]:
                    trial = best_x.copy()
                    trial[j] += sign * delta
                    val = _objective(trial)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best_x
# EVOLVE-BLOCK-END
