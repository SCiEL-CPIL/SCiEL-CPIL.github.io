import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    N = 100
    
    def compute_max_overlap(h):
        """Compute max overlap of h with shifted 1-h using FFT."""
        # overlap_k = sum_i h[i] * (1 - h[(i+k)%N])
        # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        # = N/2 - autocorr[k]
        # So max overlap = N/2 - min_k autocorr[k]
        # But we need to be careful: for the uniform case h=0.5, autocorr[k] = N*0.25 = 25 for all k
        # max overlap = 50 - 25 = 25
        
        # Use FFT for circular autocorrelation
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        # autocorr[k] = sum_i h[i] * h[(i+k)%N] (circular)
        min_autocorr = np.min(autocorr)
        return N / 2.0 - min_autocorr
    
    # Try binary sequence optimization
    # Represent as continuous variables, then round
    
    def objective(x):
        """x is in [0,1]^N, sum = N/2. We penalize deviation from binary and maximize max overlap."""
        h = x
        # Add penalty for non-binary values to encourage binary solutions
        # But first, let's just compute the max overlap
        return compute_max_overlap(h)
    
    def objective_grad(x):
        h = x
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        min_idx = np.argmin(autocorr)
        # Gradient of -autocorr[min_idx] w.r.t. h
        # autocorr[k] = sum_i h[i]*h[(i+k)%N]
        # d/dh[j] autocorr[k] = h[(j+k)%N] + h[(j-k)%N]
        k = min_idx
        grad = np.zeros(N)
        for j in range(N):
            grad[j] = h[(j + k) % N] + h[(j - k) % N]
        return -grad  # We want to minimize max overlap = N/2 - min_autocorr
    
    # Start with a random binary-like initialization
    np.random.seed(42)
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap(best_h)
    
    # Try several random binary initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Create random binary sequence with exactly N/2 ones
        ones_pos = np.random.choice(N, N // 2, replace=False)
        h = np.zeros(N)
        h[ones_pos] = 1.0
        
        current_score = compute_max_overlap(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Try local search: flip pairs of bits
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping a 1 to 0 and a 0 to 1
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            
            # Try a subset of flips for efficiency
            np.random.shuffle(ones)
            np.random.shuffle(zeros)
            
            tried = 0
            for i in range(min(200, len(ones))):
                for j in range(min(200, len(zeros))):
                    if time.time() > deadline:
                        break
                    a = ones[i]
                    b = zeros[j]
                    if a == b:
                        continue
                    
                    h[a] = 0
                    h[b] = 1
                    new_score = compute_max_overlap(h)
                    if new_score < current_score - 1e-10:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_h = h.copy()
                        # Update ones and zeros
                        ones = np.where(h == 1)[0]
                        zeros = np.where(h == 0)[0]
                        np.random.shuffle(ones)
                        np.random.shuffle(zeros)
                    else:
                        h[a] = 1
                        h[b] = 0
                    tried += 1
                if time.time() > deadline:
                    break
        
        if time.time() > deadline:
            break
    
    # Also try continuous optimization from best binary solution
    if time.time() < deadline - 10:
        h0 = best_h.copy()
        # Ensure sum is exactly N/2
        h0 = h0 / np.sum(h0) * (N / 2.0)
        
        def obj(x):
            return compute_max_overlap(x)
        
        res = minimize(obj, h0, method='L-BFGS-B', 
                       bounds=[(0, 1)] * N,
                       options={'maxiter': 50, 'maxfun': 1000})
        
        if time.time() < deadline:
            h_opt = res.x
            h_opt = np.clip(h_opt, 0, 1)
            # Normalize to sum = N/2
            h_opt = h_opt / np.sum(h_opt) * (N / 2.0)
            score_opt = compute_max_overlap(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
    
    # Final normalization to ensure sum is exactly N/2
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h / np.sum(best_h) * (N / 2.0)
    
    # Verify
    assert abs(np.sum(best_h) - N / 2.0) < 1e-6
    assert np.all(best_h >= 0) and np.all(best_h <= 1)
    
    return best_h
