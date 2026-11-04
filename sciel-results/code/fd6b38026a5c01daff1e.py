import time
import numpy as np
from scipy.optimize import minimize
import random

N = 600
DEADLINE = 140.0  # safety margin before 150s kill

def _raw_score(heights):
    """Compute the raw value to be minimized."""
    dx = 1.0 / (N * 2)
    s = np.sum(heights)
    if s <= 0:
        return float('inf')
    conv = np.convolve(heights, heights, mode='full') * dx
    return np.max(conv) / (s * dx)**2

def construct():
    start_time = time.time()
    
    # Uniform baseline
    uniform = np.ones(N)
    uniform_score = _raw_score(uniform)
    
    best_heights = uniform.copy()
    best_score = uniform_score
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    # Use a smooth surrogate: soft-max via log-sum-exp
    # Objective: soft_max(conv) / (sum(h)*dx)^2
    # With scale fixed (sum(h) = N), minimize soft_max(conv)
    
    dx = 1.0 / (N * 2)
    
    def soft_max(conv, tau=0.01):
        """Smooth approximation to max using log-sum-exp."""
        m = np.max(conv)
        return m + tau * np.log(np.sum(np.exp((conv - m) / tau)))
    
    def objective_smooth(h, tau=0.01):
        """Smooth objective: soft_max(conv) / (sum(h)*dx)^2"""
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s <= 0:
            return 1e10
        conv = np.convolve(h, h, mode='full') * dx
        sm = soft_max(conv, tau)
        return sm / (s * dx)**2
    
    def gradient_smooth(h, tau=0.01):
        """Analytical gradient of the smooth objective."""
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s <= 0:
            return np.zeros(N)
        
        conv = np.convolve(h, h, mode='full') * dx
        # soft_max gradient: softmax(conv/tau) / tau * tau = softmax(conv/tau)
        m = np.max(conv)
        exp_vals = np.exp((conv - m) / tau)
        softmax = exp_vals / np.sum(exp_vals)
        d_sm_d_conv = softmax  # d(soft_max)/d(conv_k)
        
        # conv_k = dx * sum_i h[i] * h[k-i]
        # d(conv_k)/d(h[j]) = dx * (h[k-j] + h[j-k+N-1]) for linear conv
        # Actually for np.convolve with mode='full', length is 2N-1
        # conv[k] = sum_{i} h[i] * h[k-i] where valid indices
        # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N else 0, plus h[j-(k-N+1)]... 
        # Let me be more careful.
        # np.convolve(a, b, 'full')[k] = sum_i a[i] * b[k-i]
        # For a=b=h, conv[k] = sum_i h[i]*h[k-i]
        # d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
        # But since it's the same array, these are symmetric.
        
        # Build gradient by accumulating contributions
        grad_conv = np.zeros(N)
        # For each k in conv, d(conv[k])/d(h[j]) involves h[k-j] and h[j-(k)]
        # This is complex. Let me use a different approach: 
        # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] where indices are valid
        # But k ranges from 0 to 2N-2, and j from 0 to N-1
        # k-j valid when 0 <= k-j < N, i.e., j <= k < j+N
        # j-k valid when 0 <= j-k < N, i.e., k <= j < k+N
        
        # Alternative: use the fact that conv is a quadratic form
        # conv[k] = sum_i h[i]*h[k-i]
        # This can be written as h^T A_k h where A_k is a permutation-like matrix
        # d(conv[k])/dh = 2 * A_k h (by symmetry)
        
        # Let me just compute the gradient numerically but efficiently using FFT
        # Actually, let me use a simpler approach: finite differences on a subset
        # or use the structure.
        
        # For efficiency, let me compute the gradient of soft_max(conv) w.r.t. h
        # d/dh_j [sum_k softmax_k * conv_k] = sum_k softmax_k * d(conv_k)/d(h_j)
        # d(conv_k)/d(h_j) = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
        # = h[k-j] + h[j-k] with proper bounds
        
        # Let me build this using convolution-like operations
        # d(conv)/d(h) for the full convolution:
        # The gradient of conv w.r.t. h is [convolve(h, ones)] in some sense
        # Actually: d(sum_k w_k * conv_k)/d(h_j) = sum_k w_k * (h[k-j] + h[j-k])
        # = sum_k w_k * h[k-j] + sum_k w_k * h[j-k]
        # First term: let m = k-j, sum_m w_{m+j} * h[m] = (w_shifted * h)[j]... 
        # This is a cross-correlation.
        
        # Let me just use numerical gradient with a small number of perturbations
        # for the outer optimization, and use the smooth objective.
        
        # Actually, for L-BFGS-B, let me provide an analytical gradient.
        # Let me compute it properly.
        
        # conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[k-i]
        # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N else 0, PLUS h[j-k] if 0<=j-k<N else 0
        # But wait, when i=j in the sum, we get h[j]*h[k-j], and differentiating gives h[k-j].
        # When i=k-j, we get h[k-j]*h[j], differentiating gives h[j].
        # So if j != k-j, both terms contribute: h[k-j] + h[j-k]... no.
        # Let me redo: conv[k] = sum_i h[i]*h[k-i]
        # d/dh[j] = sum_i [delta(i,j)*h[k-i] + h[i]*delta(k-i,j)]
        # = h[k-j] (from first term, i=j) + h[j] (from second term, k-i=j => i=k-j)
        # So d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] if 0<=k-j<N
        # Wait no: from second term, i=k-j, so h[i] = h[k-j], not h[j].
        # d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] when 0<=k-j<N
        # But this double counts when j = k-j, i.e., 2j=k.
        # Actually no, the two terms are distinct contributions from the two h's in the product.
        # When i=j: derivative of h[j]*h[k-j] w.r.t. h[j] is h[k-j]
        # When i=k-j: derivative of h[k-j]*h[j] w.r.t. h[j] is h[k-j]
        # These are the same term in the sum only if j=k-j, i.e., k=2j.
        # In that case, the term is h[j]^2, and its derivative is 2*h[j].
        # So: d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N, except when k=2j where it's 2*h[j] (same formula).
        # Wait, I think it's just: d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N, else 0.
        # Let me verify: if k-j is valid, both terms contribute h[k-j], giving 2*h[k-j].
        # If k-j is not valid, neither term contributes.
        # But what about when j-k is valid? That's the same condition: k-j valid <=> j-k valid (just negation).
        # Hmm, no. k-j valid means 0<=k-j<N. j-k valid means 0<=j-k<N. These are different unless k=j.
        # Let me re-examine. The sum is over i from 0 to N-1, and we need k-i to also be in [0,N-1].
        # So valid i: max(0, k-(N-1)) <= i <= min(N-1, k).
        # d(conv[k])/d(h[j]) = [j in valid range] * h[k-j] + [k-j in valid range] * h[j]
        # First term: j valid means max(0,k-N+1) <= j <= min(N-1,k), and h[k-j]
        # Second term: k-j valid means max(0,k-N+1) <= k-j <= min(N-1,k), i.e., k-min(N-1,k) <= j <= k-max(0,k-N+1)
        # This is getting complicated. Let me just use a numerical approach for the gradient
        # but make it efficient by using the structure.
        
        # For now, let me use a simple numerical gradient with central differences
        # on a subset of coordinates, or just use the smooth objective with L-BFGS-B
        # and let it compute the gradient numerically (but that's slow for N=600).
        
        # Better approach: reduce dimensionality using Fourier modes.
        return None  # We'll use a different approach
    
    # Strategy: Use a low-dimensional parameterization (Fourier modes + bias)
    # h[i] = max(0, a0 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N))
    
    K = 10  # number of Fourier modes
    
    def fourier_to_heights(params):
        """Convert Fourier parameters to nonnegative heights."""
        a0 = params[0]
        heights = np.full(N, a0)
        for k in range(1, K+1):
            a_k = params[1 + 2*(k-1)]
            b_k = params[2 + 2*(k-1)]
            angles = 2 * np.pi * k * np.arange(N) / N
            heights += a_k * np.cos(angles) + b_k * np.sin(angles)
        return np.maximum(heights, 0)
    
    def objective_fourier(params, tau=0.01):
        h = fourier_to_heights(params)
        s = np.sum(h)
        if s <= 0:
            return 1e10
        conv = np.convolve(h, h, mode='full') * dx
        sm = soft_max(conv, tau)
        return sm / (s * dx)**2
    
    # Optimize in Fourier parameter space
    random.seed(42)
    np.random.seed(42)
    
    num_starts = 0
    max_starts = 30
    
    while check_time() and num_starts < max_starts:
        num_starts += 1
        
        if num_starts == 1:
            # Start near uniform: a0=1, rest zero
            x0 = np.zeros(1 + 2*K)
            x0[0] = 1.0
        elif num_starts % 4 == 0:
            # Small random perturbation around uniform
            x0 = np.zeros(1 + 2*K)
            x0[0] = 1.0
            x0[1:] = 0.05 * np.random.randn(2*K)
        elif num_starts % 4 == 1:
            # Moderate random
            x0 = np.zeros(1 + 2*K)
            x0[0] = 0.5 + 0.5 * np.random.rand()
            x0[1:] = 0.3 * np.random.randn(2*K)
        else:
            # Larger random
            x0 = np.zeros(1 + 2*K)
            x0[0] = np.random.uniform(0.2, 2.0)
            x0[1:] = 0.5 * np.random.randn(2*K)
        
        try:
            result = minimize(objective_fourier, x0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if check_time():
                h_opt = fourier_to_heights(result.x)
                s_opt = _raw_score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_heights = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: coordinate descent on the actual heights
    if check_time():
        h = best_heights.copy()
        s = np.sum(h)
        if s > 0:
            h = h / s * N  # normalize to sum = N
            
            # Coordinate descent with smooth objective
            tau = 0.01
            improved = True
            while improved and check_time():
                improved = False
                for i in range(0, N, 3):  # step by 3 for speed
                    if not check_time():
                        break
                    best_val = objective_smooth(h, tau)
                    best_h = h[i]
                    
                    # Try a few perturbations
                    for delta in [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]:
                        h[i] = max(0, best_h + delta)
                        val = objective_smooth(h, tau)
                        if val < best_val:
                            best_val = val
                            best_h = h[i]
                    
                    h[i] = best_h
                    if best_val < objective_smooth(h, tau) - 1e-12:
                        pass
                    # Check if we improved
                    if best_val < _raw_score(h) * 0.999:
                        improved = True
            
            s_final = _raw_score(h)
            if s_final < best_score:
                best_score = s_final
                best_heights = h.copy()
    
    # Final check
    final_score = _raw_score(best_heights)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_heights)
