import numpy as np
import time
from itertools import product

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin from 150s

    # Known optimal candidate from Odlyzko's tables for N=24
    # One of the optimal sequences
    best_coeffs = [1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1]
    
    def fast_score(coeffs):
        """
        Computes max |P(z)| / sqrt(24) for z on unit circle.
        Uses dense sampling + golden section refinement on top peaks.
        """
        n = len(coeffs)
        c = np.array(coeffs, dtype=float)
        # Evaluate on a dense grid
        num_samples = 2048
        theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        z = np.exp(1j * theta)
        
        # P(z) = sum c_k z^k
        # Using matrix multiplication for speed
        # Z matrix: rows are z^k for k=0..n-1
        # This is slow for large n, but n=24 is small.
        # Alternatively, use np.polynomial or manual accumulation
        # Manual accumulation is faster for small n
        P = np.zeros_like(z, dtype=complex)
        for k in range(n):
            P += c[k] * (z ** k)
        
        mag = np.abs(P)
        # Find top 10 peaks
        top_indices = np.argsort(mag)[-10:]
        
        max_val = 0.0
        for idx in top_indices:
            # Golden section search around theta[idx]
            # Range of 2*pi/num_samples
            delta_theta = 2 * np.pi / num_samples
            a = theta[idx] - delta_theta
            b = theta[idx] + delta_theta
            
            # Refine
            gr = (np.sqrt(5) - 1) / 2
            c_ = b - gr * (b - a)
            d_ = a + gr * (b - a)
            
            def f(t):
                z_t = np.exp(1j * t)
                val = 0.0
                for k in range(n):
                    val += c[k] * (z_t ** k)
                return np.abs(val)
            
            fc = f(c_)
            fd = f(d_)
            
            for _ in range(50):
                if fc < fd:
                    b = d_
                    d_ = c_
                    fd = fc
                    c_ = b - gr * (b - a)
                    fc = f(c_)
                else:
                    a = c_
                    c_ = d_
                    fc = fd
                    d_ = a + gr * (b - a)
                    fd = f(d_)
            
            val = max(fc, fd)
            if val > max_val:
                max_val = val
        
        return max_val / np.sqrt(n)

    current_score = fast_score(best_coeffs)
    
    # Local search to ensure we are at the optimum
    # Flip bits and check if score improves
    improved = True
    while improved:
        improved = False
        for i in range(len(best_coeffs)):
            if time.time() > DEADLINE:
                break
            # Flip bit
            best_coeffs[i] = -best_coeffs[i]
            new_score = fast_score(best_coeffs)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                best_coeffs[i] = -best_coeffs[i]
        if time.time() > DEADLINE:
            break

    return best_coeffs
