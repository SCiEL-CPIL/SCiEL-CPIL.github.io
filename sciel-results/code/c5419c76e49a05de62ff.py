import numpy as np
import time

N = 400
DEADLINE = 140  # seconds, with 10s safety margin

def _autoconv(x):
    """Compute full autoconvolution (correlation) of x with itself."""
    n = len(x)
    # C[k] = sum_i x[i] * x[i+k] for k = 0..n-1
    # Use FFT for speed
    fft_len = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, fft_len)
    C = np.fft.irfft(X * np.conj(X), fft_len)[:n]
    return C

def _score(x):
    """Compute max |autoconv| / (sum(x))^2"""
    C = _autoconv(x)
    peak = np.max(np.abs(C))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    return peak / (s * s)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Phase 1: Saddle point / minimax on continuous relaxation
    np.random.seed(42)
    x = np.random.uniform(-1, 1, N)
    
    best_x = x.copy()
    best_score = _score(x)
    
    # Temperature for softmax
    tau = 1.0
    lr = 0.01
    n_iter_max = 500
    
    for it in range(n_iter_max):
        if time_left() < 20:
            break
        
        C = _autoconv(x)
        abs_C = np.abs(C)
        s = np.sum(x)
        
        if abs(s) < 1e-10:
            # Nudge away from zero sum
            x[0] += 0.1
            continue
        
        # Softmax distribution over lags
        # Scale to avoid overflow
        max_abs = np.max(abs_C)
        if max_abs < 1e-10:
            p = np.ones(N) / N
        else:
            # p[k] proportional to exp(|C[k]| / tau)
            exponents = abs_C / tau
            exponents -= np.max(exponents)
            p = np.exp(exponents)
            p /= np.sum(p)
        
        # Objective: sum_k p[k] * |C[k]| / s^2
        # Gradient with respect to x:
        # d/dx_j [sum_k p[k] |C[k]|] / s^2 - 2 * sum_k p[k] |C[k]| * s_j / s^3
        # 
        # C[k] = sum_i x[i] x[i+k]
        # dC[k]/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
        # 
        # This is getting complex. Let me use a simpler approach:
        # Just do gradient descent on the smooth objective.
        
        # Actually, let me use a simpler heuristic: 
        # Perturb x in the direction that reduces the weighted peak
        # 
        # Simpler: use the gradient of sum_k p[k] * C[k]^2 / s^2
        # This is smoother than |C[k]|
        
        C_sq = C * C
        obj = np.sum(p * C_sq) / (s * s)
        
        # Gradient of sum_k p[k] C[k]^2 / s^2
        # = (2 * sum_k p[k] C[k] * dC[k]/dx) / s^2 - 2 * (sum_k p[k] C[k]^2) / s^3 * ds/dx
        # ds/dx = 1 for all j
        
        # dC[k]/dx[j] = x[j+k] (if j+k<N) + x[j-k] (if j-k>=0)
        # sum_k p[k] C[k] dC[k]/dx[j] = sum_k p[k] C[k] (x[j+k] + x[j-k])
        # = sum_k p[k] C[k] x[j+k] + sum_k p[k] C[k] x[j-k]
        # = (p*C)[j] convolved with x... 
        # 
        # Let me just compute it directly for now (N=400 is small enough)
        
        # Actually, let's use a more efficient approach.
        # The gradient of C[k]^2 w.r.t x[j]:
        # d(C[k]^2)/dx[j] = 2*C[k] * (x[j+k] + x[j-k]) [with bounds]
        
        # Let me compute grad for the full objective
        grad = np.zeros(N)
        for k in range(N):
            if p[k] < 1e-10:
                continue
            # dC[k]/dx[j] for j in range: x[j+k] if j+k<N, x[j-k] if j-k>=0
            # C[k] = sum_{i=0}^{N-1-k} x[i] x[i+k]
            # dC[k]/dx[j] = x[j+k] (if j+k < N and j >= 0) + x[j-k] (if j-k >= 0 and j < N-k... wait)
            
            # Actually for C[k] = sum_i x[i]*x[i+k], the term x[j] appears when:
            # i = j: contributes x[j+k] (if j+k < N)
            # i+k = j: i = j-k: contributes x[j-k] (if j-k >= 0)
            # So dC[k]/dx[j] = (x[j+k] if j+k<N else 0) + (x[j-k] if j-k>=0 else 0)
            
            dk = np.zeros(N)
            if j := 0:  # can't use walrus like this in a loop easily
                pass
            
            # This is O(N^2) per iteration which is 160k ops, fine for 400
            # But let me vectorize
            pass
        
        # Let me just do a simpler perturbation-based approach
        # Use finite differences or just random perturbations guided by the loss
        
        # Simpler approach: just do random restarts with local search on ±1
        if it == 0:
            # Skip the complex gradient, go straight to local search
            break
        
        # Update temperature
        tau *= 0.99
    
    # Phase 2: Local search on ±1 sequences
    # Start from best continuous solution rounded to ±1
    x_rounded = np.sign(best_x)
    x_rounded[x_rounded == 0] = 1.0
    best_x1 = x_rounded.copy()
    best_score1 = _score(best_x1)
    
    # Local search: flip bits to reduce peak autoconvolution
    C = _autoconv(best_x1)
    peak_lag = np.argmax(np.abs(C))
    
    improved = True
    max_local_iters = 50000
    local_iter = 0
    
    while improved and local_iter < max_local_iters:
        if time_left() < 10:
            break
        improved = False
        local_iter += 1
        
        C = _autoconv(best_x1)
        abs_C = np.abs(C)
        peak = np.max(abs_C)
        s = np.sum(best_x1)
        current_score = peak / (s * s)
        
        # Try flipping each bit
        best_flip = -1
        best_new_score = current_score
        
        for j in range(N):
            # Flip bit j
            old_val = best_x1[j]
            best_x1[j] = -old_val
            
            C_new = _autoconv(best_x1)
            peak_new = np.max(np.abs(C_new))
            s_new = np.sum(best_x1)
            
            if abs(s_new) > 1e-10:
                score_new = peak_new / (s_new * s_new)
            else:
                score_new = np.inf
            
            best_x1[j] = old_val
            
            if score_new < best_new_score - 1e-10:
                best_new_score = score_new
                best_flip = j
        
        if best_flip >= 0:
            best_x1[best_flip] = -best_x1[best_flip]
            improved = True
            best_score1 = best_new_score
    
    # Also try random restarts with local search
    n_restarts = 20
    for r in range(n_restarts):
        if time_left() < 10:
            break
        
        x_rand = np.random.choice([-1.0, 1.0], N)
        # Bias toward more +1s to get larger |sum|
        # Actually, let's try different biases
        bias = np.random.uniform(0.1, 0.4)
        x_rand = np.where(np.random.random(N) < 0.5 + bias, 1.0, -1.0)
        
        # Local search from this start
        x_local = x_rand.copy()
        C = _autoconv(x_local)
        s = np.sum(x_local)
        if abs(s) < 1e-10:
            continue
        local_score = np.max(np.abs(C)) / (s * s)
        
        local_improved = True
        local_iters = 0
        while local_improved and local_iters < 5000:
            if time_left() < 10:
                break
            local_improved = False
            local_iters += 1
            
            C = _autoconv(x_local)
            abs_C = np.abs(C)
            peak = np.max(abs_C)
            s = np.sum(x_local)
            if abs(s) < 1e-10:
                break
            cur_score = peak / (s * s)
            
            # Try flipping each bit, pick best improvement
            best_j = -1
            best_s = cur_score
            
            for j in range(N):
                old = x_local[j]
                x_local[j] = -old
                C_t = _autoconv(x_local)
                peak_t = np.max(np.abs(C_t))
                s_t = np.sum(x_local)
                if abs(s_t) > 1e-10:
                    sc_t = peak_t / (s_t * s_t)
                else:
                    sc_t = np.inf
                x_local[j] = old
                
                if sc_t < best_s - 1e-10:
                    best_s = sc_t
                    best_j = j
            
            if best_j >= 0:
                x_local[best_j] = -x_local[best_j]
                local_improved = True
        
        final_score = _score(x_local)
        if final_score < best_score1:
            best_score1 = final_score
            best_x1 = x_local.copy()
    
    return best_x1
