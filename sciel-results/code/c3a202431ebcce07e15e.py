import time
import numpy as np

N = 600

def construct():
    """
    Returns a sequence of nonnegative step heights on equal intervals of [-1/4, 1/4].
    Uses spectral amplitude shaping with zero phase to minimize the normalized
    maximum of the autoconvolution.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with uniform heights
    h = np.ones(N, dtype=np.float64)
    
    # Compute the DFT
    H = np.fft.fft(h)
    
    # Fix phase to zero (real positive) and optimize magnitudes
    # For zero phase, h = IFFT(|H|) should be real and non-negative
    # We want to minimize the peak of the autoconvolution relative to its mean
    
    # The autoconvolution of h is IFFT(|FFT(h)|^2)
    # With zero phase: h = IFFT(|H|), so FFT(h) = |H| (real, non-negative)
    # Autoconvolution = IFFT(|H|^2)
    
    # Let's work with log-magnitudes for better conditioning
    log_mag = np.log(np.abs(H) + 1e-10)
    
    best_h = h.copy()
    best_score = float('inf')
    
    # Compute initial score
    def compute_score(h_arr):
        """Compute normalized max of autoconvolution"""
        H = np.fft.fft(h_arr)
        auto_conv = np.fft.ifft(np.abs(H)**2).real
        # Normalize by mean
        mean_val = np.mean(h_arr)
        if mean_val <= 0:
            return float('inf')
        # The normalized max is max(auto_conv) / (mean_val * N * interval_width)
        # But for our purposes, we can just look at max(auto_conv) / sum(h)^2 * N
        # Actually, let's use the standard normalization
        total = np.sum(h_arr)
        if total <= 0:
            return float('inf')
        return np.max(auto_conv) / (total * total / N)
    
    best_score = compute_score(h)
    
    # Gradient descent on log-magnitudes
    lr = 0.01
    step = 0
    
    while time.time() < deadline:
        # Update log-magnitudes
        mag = np.exp(log_mag)
        
        # Compute h from magnitudes with zero phase
        h_candidate = np.fft.ifft(mag).real
        
        # Ensure non-negativity
        h_candidate = np.maximum(h_candidate, 0)
        
        # Compute score
        score = compute_score(h_candidate)
        
        if score < best_score:
            best_score = score
            best_h = h_candidate.copy()
        
        # Compute gradient of score w.r.t. log_mag
        # This is complex, so let's use a simple finite difference approach
        # Or use the fact that we want to flatten the autoconvolution
        
        # Alternative: directly optimize h with constraints
        # Let's switch to optimizing h directly with projection
        
        # Compute autoconvolution
        H_cand = np.fft.fft(h_candidate)
        auto_conv = np.fft.ifft(np.abs(H_cand)**2).real
        
        # The peak is at some location; we want to reduce it
        # Gradient descent: h_new = h - lr * grad
        # For simplicity, use a momentum-based approach on h directly
        
        if step % 10 == 0:
            # Recompute from scratch to avoid drift
            H_current = np.fft.fft(h_candidate)
            phase = np.angle(H_current)
            mag_current = np.abs(H_current)
            
            # Perturb magnitudes to reduce peak
            # Find the peak of autoconvolution
            peak_idx = np.argmax(auto_conv)
            
            # Reduce magnitudes that contribute to the peak
            # This is heuristic; let's just do simple gradient descent on h
            
            # Project h to be non-negative
            h_candidate = np.maximum(h_candidate, 0)
            
            # Normalize to maintain some scale
            if np.sum(h_candidate) > 0:
                h_candidate = h_candidate * (N / np.sum(h_candidate))
        
        # Simple update: reduce peaks in h that correlate with autoconv peaks
        # Use a soft approach: h = h - lr * (auto_conv > threshold) * h
        threshold = np.mean(auto_conv) + 0.5 * np.std(auto_conv)
        mask = (auto_conv > threshold).astype(np.float64)
        
        h_candidate = h_candidate - lr * mask * h_candidate
        h_candidate = np.maximum(h_candidate, 0)
        
        # Normalize
        if np.sum(h_candidate) > 0:
            h_candidate = h_candidate * (N / np.sum(h_candidate))
        
        step += 1
        
        # Occasionally re-evaluate
        if step % 50 == 0:
            score = compute_score(h_candidate)
            if score < best_score:
                best_score = score
                best_h = h_candidate.copy()
    
    # Return the best result found
    return list(best_h)

if __name__ == "__main__":
    # Simple self-check
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Best score: {compute_score(np.array(res))}")
