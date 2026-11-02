import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog

N = 600
DEADLINE = 140.0

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    if n == 0:
        return np.inf
    
    # Linear autoconvolution via FFT
    M = 1 << (2*n - 1).bit_length()
    h_padded = np.zeros(M)
    h_padded[:n] = heights
    H = np.fft.fft(h_padded)
    auto_conv = np.fft.ifft(H * H).real[:2*n-1]
    
    total = np.sum(heights)
    if total <= 0:
        return np.inf
    
    # Normalization: for uniform heights=1, auto_conv[0]=1, total=N
    # score = max(auto_conv) * n / total^2 gives 1.0 for uniform
    score = np.max(auto_conv) * n / (total * total)
    return score

def _autoconv(h):
    """Compute linear autoconvolution of h."""
    n = len(h)
    M = 1 << (2*n - 1).bit_length()
    h_padded = np.zeros(M)
    h_padded[:n] = h
    H = np.fft.fft(h_padded)
    return np.fft.ifft(H * H).real[:2*n-1]

def _minimax_pushdown(h, max_iter=2000, time_limit=60.0, tol=1e-10):
    """Minimax optimization: iteratively push down the largest autoconvolution coefficient.
    
    At each step, identify the argmax of autoconv, then perform a projected gradient step
    to reduce it while keeping h >= 0.
    """
    start_time = time.time()
    h = h.copy().astype(np.float64)
    n = len(h)
    current_score = compute_score(h)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        ac = _autoconv(h)
        total = np.sum(h)
        if total < 1e-15:
            break
        
        # Normalize score
        score = np.max(ac) * n / (total * total)
        
        # Find the dominant coefficient
        k_max = np.argmax(ac)
        
        # Gradient of ac[k_max] w.r.t. h[i]:
        # ac[k] = sum_j h[j] * h[k-j] (with bounds)
        # dac[k]/dh[i] = h[k-i] + h[i] if valid
        grad = np.zeros(n)
        for i in range(n):
            j = k_max - i
            if 0 <= j < n:
                grad[i] += h[j]
            if 0 <= i < n:
                grad[i] += h[i]
        
        # Normalize gradient by total^2 for the score
        grad_norm = grad * n / (total * total)
        
        # Projected gradient descent step
        # Try multiple step sizes
        best_step = 0.0
        best_new_score = current_score
        best_new_h = h.copy()
        
        for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            h_trial = h - lr * grad_norm
            h_trial = np.maximum(h_trial, 0.0)
            new_score = compute_score(h_trial)
            if new_score < best_new_score:
                best_new_score = new_score
                best_step = lr
                best_new_h = h_trial.copy()
        
        if best_step == 0.0:
            break
        
        h = best_new_h
        current_score = best_new_score
        
        if it > 0 and abs(score - current_score) < tol:
            break
    
    return h, current_score

def _symmetric_optimize(h, time_limit=30.0):
    """Optimize assuming h is symmetric: h[i] = h[N-1-i].
    
    This reduces the effective dimension to N/2 and exploits the natural symmetry
    of the autoconvolution problem.
    """
    start_time = time.time()
    n = len(h)
    half = n // 2
    
    # Extract symmetric part
    h_sym = np.zeros(n)
    h_sym[:half] = h[:half]
    h_sym[half:] = h[:half][::-1]
    
    current_score = compute_score(h_sym)
    
    # Optimize the first half, then mirror
    h_half = h[:half].copy()
    
    for it in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Reconstruct full symmetric h
        h_full = np.zeros(n)
        h_full[:half] = h_half
        h_full[half:] = h_half[::-1]
        
        ac = _autoconv(h_full)
        total = np.sum(h_full)
        if total < 1e-15:
            break
        
        k_max = np.argmax(ac)
        
        # Gradient w.r.t. h_full
        grad_full = np.zeros(n)
        for i in range(n):
            j = k_max - i
            if 0 <= j < n:
                grad_full[i] += h_full[j]
            grad_full[i] += h_full[i]
        
        # Map gradient back to h_half (sum gradients from symmetric pairs)
        grad_half = np.zeros(half)
        for i in range(half):
            grad_half[i] = grad_full[i]
            if i != n - 1 - i:
                grad_half[i] += grad_full[n - 1 - i]
        
        grad_half_norm = grad_half * n / (total * total)
        
        # Projected gradient step on h_half
        best_step = 0.0
        best_new_score = current_score
        best_new_half = h_half.copy()
        
        for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            h_half_trial = h_half - lr * grad_half_norm
            h_half_trial = np.maximum(h_half_trial, 0.0)
            
            h_full_trial = np.zeros(n)
            h_full_trial[:half] = h_half_trial
            h_full_trial[half:] = h_half_trial[::-1]
            
            new_score = compute_score(h_full_trial)
            if new_score < best_new_score:
                best_new_score = new_score
                best_step = lr
                best_new_half = h_half_trial.copy()
        
        if best_step == 0.0:
            break
        
        h_half = best_new_half
        current_score = best_new_score
    
    # Reconstruct
    h_result = np.zeros(n)
    h_result[:half] = h_half
    h_result[half:] = h_half[::-1]
    return h_result, current_score

def _lowfreq_optimize(h, time_limit=40.0):
    """Optimize in a low-frequency basis to reduce effective dimensionality.
    
    Project h onto the first K Fourier modes (cosine basis for real, symmetric signals),
    optimize in that reduced space, then project back.
    """
    start_time = time.time()
    n = len(h)
    
    # Use cosine basis (DCT-like) for real signals
    # h[i] = sum_k a_k * cos(pi * k * i / n)
    K = 30  # number of basis functions
    
    # Build basis matrix
    i = np.arange(n)
    basis = np.zeros((n, K))
    for k in range(K):
        basis[:, k] = np.cos(np.pi * k * i / n)
    
    # Project h onto basis
    # a = (B^T B)^{-1} B^T h
    BtB = basis.T @ basis
    BtB_inv = np.linalg.inv(BtB + 1e-10 * np.eye(K))
    a = BtB_inv @ (basis.T @ h)
    
    current_score = compute_score(h)
    
    for it in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Reconstruct h from coefficients
        h_full = basis @ a
        h_full = np.maximum(h_full, 0.0)
        
        ac = _autoconv(h_full)
        total = np.sum(h_full)
        if total < 1e-15:
            break
        
        k_max = np.argmax(ac)
        
        # Gradient w.r.t. h_full
        grad_full = np.zeros(n)
        for i in range(n):
            j = k_max - i
            if 0 <= j < n:
                grad_full[i] += h_full[j]
            grad_full[i] += h_full[i]
        
        # Map to coefficient space: grad_a = B^T grad_full
        grad_a = basis.T @ grad_full * n / (total * total)
        
        # Gradient step in coefficient space
        best_step = 0.0
        best_new_score = current_score
        best_new_a = a.copy()
        
        for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
            a_trial = a - lr * grad_a
            h_trial = basis @ a_trial
            h_trial = np.maximum(h_trial, 0.0)
            new_score = compute_score(h_trial)
            if new_score < best_new_score:
                best_new_score = new_score
                best_step = lr
                best_new_a = a_trial.copy()
        
        if best_step == 0.0:
            break
        
        a = best_new_a
        current_score = best_new_score
    
    h_result = basis @ a
    h_result = np.maximum(h_result, 0.0)
    return h_result, current_score

def _two_block_optimize():
    """Optimize a two-block step function: h = [a]*k + [b]*(n-k).
    Returns best (score, heights).
    """
    best_score = np.inf
    best_h = np.ones(N)
    
    # Try different split points and ratios
    for k in range(N//4, 3*N//4, 5):
        for ratio in np.linspace(0.05, 15.0, 80):
            h = np.zeros(N)
            h[:k] = ratio
            h[k:] = 1.0
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Also try the reverse
    for k in range(N//4, 3*N//4, 5):
        for ratio in np.linspace(0.05, 15.0, 80):
            h = np.zeros(N)
            h[:k] = 1.0
            h[k:] = ratio
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    return best_score, best_h

def _coordinate_descent(h, max_iter=500, time_limit=20.0):
    """Coordinate descent with geometric line search including zero."""
    start_time = time.time()
    h = h.copy()
    n = len(h)
    current_score = compute_score(h)
    
    def candidate_values(val):
        if val <= 1e-15:
            return [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
        else:
            return [0.0] + [val * r for r in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]]
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            old_val = h[i]
            best_val = old_val
            best_score = current_score
            
            for cand in candidate_values(old_val):
                h[i] = cand
                new_score = compute_score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_val = cand
            
            if best_val != old_val:
                h[i] = best_val
                current_score = best_score
                improved = True
        
        if not improved:
            break
    
    return h, current_score

def _zero_out_pass(h, threshold=0.001):
    """Remove small values to reduce noise and potentially improve score."""
    h = h.copy()
    n = len(h)
    current_score = compute_score(h)
    
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]:
        h_trial = h.copy()
        h_trial[h_trial < thresh] = 0.0
        
        if np.sum(h_trial) > 0:
            score_trial = compute_score(h_trial)
            if score_trial < current_score:
                h = h_trial
                current_score = score_trial
    
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start from uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    best_score = compute_score(h0)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Two-block optimization for a good starting point
    if time_left() > 110:
        two_block_score, two_block_h = _two_block_optimize()
        if two_block_score < best_score:
            best_score = two_block_score
            best_h = two_block_h.copy()
    
    # Phase 2: Minimax pushdown from best starting point
    if time_left() > 80:
        mm_h, mm_score = _minimax_pushdown(best_h, max_iter=3000, time_limit=30.0)
        if mm_score < best_score:
            best_score = mm_score
            best_h = mm_h.copy()
    
    # Phase 3: Symmetric optimization
    if time_left() > 60:
        sym_h, sym_score = _symmetric_optimize(best_h, time_limit=20.0)
        if sym_score < best_score:
            best_score = sym_score
            best_h = sym_h.copy()
    
    # Phase 4: Low-frequency optimization
    if time_left() > 45:
        lf_h, lf_score = _lowfreq_optimize(best_h, time_limit=15.0)
        if lf_score < best_score:
            best_score = lf_score
            best_h = lf_h.copy()
    
    # Phase 5: Another round of minimax pushdown
    if time_left() > 30:
        mm2_h, mm2_score = _minimax_pushdown(best_h, max_iter=2000, time_limit=15.0)
        if mm2_score < best_score:
            best_score = mm2_score
            best_h = mm2_h.copy()
    
    # Phase 6: Coordinate descent polish
    if time_left() > 20:
        cd_h, cd_score = _coordinate_descent(best_h, max_iter=300, time_limit=10.0)
        if cd_score < best_score:
            best_score = cd_score
            best_h = cd_h.copy()
    
    # Phase 7: Zero-out pass
    if time_left() > 12:
        zo_h, zo_score = _zero_out_pass(best_h)
        if zo_score < best_score:
            best_score = zo_score
            best_h = zo_h.copy()
    
    # Phase 8: Final minimax polish
    if time_left() > 8:
        mm3_h, mm3_score = _minimax_pushdown(best_h, max_iter=1000, time_limit=5.0)
        if mm3_score < best_score:
            best_score = mm3_score
            best_h = mm3_h.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
