import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_matrix(h):
    """Compute all cyclic cross-correlations sum_i h_i * (1 - h_{i+k}) for k=0..N-1"""
    # Use FFT for efficiency
    # C(k) = sum_i h_i * (1 - h_{(i+k) % N})
    # = sum_i h_i - sum_i h_i * h_{(i+k)%N}
    # = sum(h) - autocorr_h(k)
    # autocorr_h(k) = sum_i h_i * h_{(i+k)%N}
    # For cyclic autocorrelation, use FFT:
    # A(k) = ifft(fft(h) * conj(fft(h)))  -- but need to be careful with definition
    
    H = np.fft.fft(h)
    # Cyclic autocorrelation: A(k) = sum_i h_i * h_{(i+k) mod N}
    # This is ifft(|H|^2) but with appropriate indexing
    A = np.fft.ifft(H * np.conj(H)).real
    # A[k] should be sum_i h_i * h_{(i+k) mod N}
    # Actually, the standard DFT autocorrelation gives:
    # ifft(fft(x) * conj(fft(x)))[k] = sum_i x_i * x_{(i-k) mod N}
    # So A[k] = sum_i h_i * h_{(i-k) mod N}
    # We want sum_i h_i * h_{(i+k) mod N} = A[-k mod N] = A[N-k] for k>0
    # But since we take max over all k, it doesn't matter (it's symmetric in magnitude for real signals)
    
    C = np.sum(h) - A  # C[k] = sum_i h_i * (1 - h_{(i+k) mod N}) using the right indexing
    # Actually let me just compute directly for correctness
    C = np.zeros(N)
    for k in range(N):
        C[k] = np.sum(h * (1.0 - np.roll(h, -k)))
    return C

def _max_overlap(h):
    C = _overlap_matrix(h)
    return np.max(C)

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    
    # Try a simple gradient-based approach with projected constraints
    # Use smooth surrogate: soft-max via log-sum-exp
    
    def soft_max(x, tau=0.01):
        m = np.max(x)
        return m + tau * np.log(np.sum(np.exp((x - m) / tau)))
    
    def objective(h):
        C = np.zeros(N)
        for k in range(N):
            C[k] = np.sum(h * (1.0 - np.roll(h, -k)))
        return soft_max(C, tau=0.01)
    
    def grad_objective(h):
        # Gradient of soft-max of C with respect to h
        C = np.zeros(N)
        for k in range(N):
            C[k] = np.sum(h * (1.0 - np.roll(h, -k)))
        tau = 0.01
        m = np.max(C)
        exp_C = np.exp((C - m) / tau)
        weights = exp_C / np.sum(exp_C)
        
        # dC[k]/dh_j: C[k] = sum_i h_i - sum_i h_i * h_{(i+k)%N}
        # dC[k]/dh_j = 1 - sum over i where j=i or j=(i+k)%N of contributions
        # dC[k]/dh_j = 1 - [h_{(j-k)%N} + h_j] (from the product term)
        # Wait: C[k] = sum_i h_i*(1-h_{(i+k)%N})
        # dC[k]/dh_j = (1-h_{(j+k)%N}) + sum_i h_i * (-1) * [j == (i+k)%N]
        # = (1-h_{(j+k)%N}) - h_{(j-k)%N}
        
        grad = np.zeros(N)
        for k in range(N):
            w = weights[k]
            # dC[k]/dh_j = (1 - h_{(j+k)%N}) - h_{(j-k)%N}
            dhdk = (1.0 - np.roll(h, k)) - np.roll(h, -k)
            grad += w * dhdk
        return grad
    
    # Projected gradient descent
    h = h0.copy()
    lr = 0.01
    best_h = h.copy()
    best_score = _max_overlap(h)
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        g = grad_objective(h)
        h_new = h - lr * g
        
        # Project onto feasible set: sum = N/2, 0 <= h_i <= 1
        # Simple projection: adjust mean, then clip
        h_new = h_new - np.mean(h_new) + 0.5
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Re-project to satisfy sum constraint more carefully
        # Use iterative water-filling
        for _ in range(50):
            h_new = h_new - (np.sum(h_new) - N/2) / N
            h_new = np.clip(h_new, 0.0, 1.0)
        
        score = _max_overlap(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
        h = h_new
    
    return best_h
