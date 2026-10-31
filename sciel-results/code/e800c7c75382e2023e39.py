import numpy as np
import time

N = 600
DEADLINE = 115  # seconds

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def normalize(h):
        s = np.sum(h)
        if s == 0:
            return np.ones(N) / N
        return h / s
    
    def calc_autoconv(h):
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        return np.fft.ifft(H * H).real
    
    def smooth_max(c, beta=100.0):
        # Log-sum-exp approximation of max
        c_max = np.max(c)
        if c_max > 1e10:
            return c_max
        return c_max + np.log(np.sum(np.exp(beta * (c - c_max)))) / beta
    
    def calc_metric_smooth(h, beta=100.0):
        autoconv = calc_autoconv(h)
        return smooth_max(autoconv, beta)
    
    def calc_metric_exact(h):
        autoconv = calc_autoconv(h)
        return np.max(autoconv)
    
    # Baseline
    uniform_h = np.ones(N) / N
    best_h = uniform_h.copy()
    best_val = calc_metric_exact(uniform_h)
    
    # Multiple restarts
    np.random.seed(123)
    n_restarts = 15
    max_iter = 300
    
    for r in range(n_restarts):
        if time_left() < 15:
            break
        
        # Initialize
        if r == 0:
            h0 = uniform_h.copy()
        elif r == 1:
            i = np.arange(N)
            h0 = np.exp(-((i - N/2) ** 2) / (2 * (N/8) ** 2))
        elif r == 2:
            i = np.arange(N)
            h0 = 1 + 20 * (1 - np.sin(np.pi * i / N))
        elif r == 3:
            h0 = np.random.dirichlet(np.ones(N) * 0.5)
        elif r == 4:
            h0 = np.random.dirichlet(np.ones(N) * 0.1)
        else:
            # Mix
            alpha = np.random.random()
            h0 = alpha * uniform_h + (1 - alpha) * np.random.dirichlet(np.ones(N) * np.random.random() * 10 + 0.1)
        
        h0 = normalize(h0)
        h0_val = calc_metric_exact(h0)
        if h0_val < best_val:
            best_h = h0.copy()
            best_val = h0_val
        
        h = h0.copy()
        h_val = h0_val
        eta = 0.05
        beta = 50.0
        
        for k in range(max_iter):
            if time_left() < 5:
                break
            
            # Compute gradient of smooth max
            autoconv = calc_autoconv(h)
            c_max = np.max(autoconv)
            # Softmax weights
            weights = np.exp(beta * (autoconv - c_max))
            weights /= np.sum(weights)
            
            # Gradient of smooth_max w.r.t autoconv is weights
            # Gradient of autoconv w.r.t h
            # autoconv[k] = sum_i h[i] h[i+k]
            # d autoconv[k] / d h[j] = h[k-j] + h[j-k]
            # So gradient of smooth_max w.r.t h[j] = sum_k weights[k] * (h[k-j] + h[j-k])
            
            # This is a convolution of weights with h, plus h convolved with weights
            # Let's compute it efficiently
            # grad[j] = sum_k weights[k] * h[k-j] + sum_k weights[k] * h[j-k]
            # = (weights * h)[j] + (weights * h)[j] by symmetry? Not exactly.
            
            # Let's compute it directly using FFT or direct sum
            # Since N=600, direct sum is O(N^2) = 360k, which is fast.
            
            idx = np.arange(N)
            grad = np.zeros(N)
            
            # For each k, weights[k] * (h[k-j] + h[j-k]) for j in 0..N-1
            # This is equivalent to: for each k, add weights[k] * h shifted by k
            # and weights[k] * h shifted by -k
            
            # Use the fact that it's a convolution
            # Let's create a full weights vector of size 2N
            w_full = np.zeros(2 * N)
            w_full[:N] = weights  # But weights is size 2N? No, autoconv is size 2N
            # Wait, autoconv has size 2N (from FFT), but the linear conv is in 0..2N-2
            # weights is size 2N
            # We need to convolve weights with h, but h is size N
            
            # Simpler: direct computation
            # grad[j] = sum_{k=0}^{2N-1} weights[k] * (h[k-j] if 0<=k-j<N else 0 + h[j-k] if 0<=j-k<N else 0)
            
            # This is O(N * 2N) = O(N^2) = 720k for N=600, which is fine.
            
            for k in range(2 * N):
                if weights[k] < 1e-12:
                    continue
                # h[k-j] for j such that 0 <= k-j < N => k-N < j <= k
                j_start = max(0, k - N + 1)
                j_end = min(N, k + 1)
                if j_start < j_end:
                    j_indices = np.arange(j_start, j_end)
                    h_indices = k - j_indices
                    grad[j_indices] += weights[k] * h[h_indices]
                
                # h[j-k] for j such that 0 <= j-k < N => k <= j < k+N
                j_start2 = max(0, k)
                j_end2 = min(N, k + N)
                if j_start2 < j_end2:
                    j_indices2 = np.arange(j_start2, j_end2)
                    h_indices2 = j_indices2 - k
                    grad[j_indices2] += weights[k] * h[h_indices2]
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad = grad / grad_norm
            
            # Backtracking
            improved = False
            for _ in range(15):
                new_h = h - eta * grad
                new_h = np.maximum(new_h, 0)
                s = np.sum(new_h)
                if s > 0:
                    new_h = new_h / s
                else:
                    new_h = uniform_h.copy()
                
                new_val = calc_metric_smooth(new_h, beta)
                if new_val < h_val:
                    h = new_h
                    h_val = new_val
                    improved = True
                    if calc_metric_exact(h) < best_val:
                        best_h = h.copy()
                        best_val = calc_metric_exact(h)
                    break
                else:
                    eta *= 0.5
            
            if not improved:
                eta *= 0.5
                if eta < 1e-10:
                    break
            
            # Increase beta over time for sharper max
            if k % 50 == 0:
                beta *= 1.5
                if beta > 500:
                    beta = 500
        
        # Final exact evaluation
        final_val = calc_metric_exact(h)
        if final_val < best_val:
            best_h = h.copy()
            best_val = final_val
    
    return best_h.tolist()
