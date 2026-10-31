"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s limit
    
    N = 512  # Power of 2 for FFT efficiency
    
    def autoconv_fft(f):
        """Compute linear autoconvolution using FFT."""
        size = 1
        while size < 2 * N - 1:
            size *= 2
        F = np.fft.rfft(f, n=size)
        C = np.fft.irfft(F * F, n=size)
        return C[:2 * N - 1]
    
    def score(f):
        S = np.sum(f)
        if abs(S) < 1e-12:
            return 1e10
        C = autoconv_fft(f)
        return np.max(np.abs(C)) / (S * S)
    
    # Strategy: Use a parametric continuous family that approximates the 
    # Euler-Lagrange optimal shape, then refine discretely.
    # The optimal shape for minimizing peak autoconvolution ratio has 
    # properties similar to a raised cosine or Hann window, but with 
    # specific curvature to minimize sidelobes.
    
    # Parametric family: f(x) = A * (1 + alpha * cos(2*pi*x/N)) ** beta
    # where alpha and beta are optimized to minimize the score.
    
    n = np.arange(N)
    best_score = 1e10
    best_f = np.ones(N)
    
    # Try different parametric shapes
    # Shape 1: Raised cosine with varying depth
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for beta in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            if time.time() > DEADLINE:
                break
            # Continuous shape: (1 + alpha*cos(2*pi*n/N))**beta
            shape = (1.0 + alpha * np.cos(2.0 * np.pi * n / N)) ** beta
            # Normalize to have unit sum
            S = np.sum(shape)
            if abs(S) < 1e-12:
                continue
            f = shape / S
            s = score(f)
            if s < best_score:
                best_score = s
                best_f = f.copy()
    
    # Shape 2: Gaussian-like windows
    for sigma in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if time.time() > DEADLINE:
            break
        center = N / 2.0
        shape = np.exp(-0.5 * ((n - center) / (sigma * N / 4.0)) ** 2)
        S = np.sum(shape)
        if abs(S) < 1e-12:
            continue
        f = shape / S
        s = score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Shape 3: Sinc-like (prolate spheroidal approximation)
    for M in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > DEADLINE:
            break
        # Approximate prolate spheroidal wave function
        shape = np.sinc(n / M)
        S = np.sum(shape)
        if abs(S) < 1e-12:
            continue
        f = shape / S
        s = score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Shape 4: Quadratic phase (Gauss sum related)
    for M in [4, 8, 16, 32, 64, 128, 256]:
        if time.time() > DEADLINE:
            break
        phase = np.pi * (n / M) ** 2
        shape = np.abs(np.cos(phase))  # Use magnitude to keep positive
        S = np.sum(shape)
        if abs(S) < 1e-12:
            continue
        f = shape / S
        s = score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Now discretize to signed values and refine
    # First, try to convert to {-1, 1} while preserving the shape
    f_disc = np.sign(best_f)
    if np.sum(f_disc) == 0:
        f_disc = np.ones(N)
    
    s_disc = score(f_disc)
    if s_disc < best_score:
        best_score = s_disc
        best_f = f_disc.copy()
    
    # Local search on the discrete sequence
    f = best_f.copy()
    S = np.sum(f)
    C = autoconv_fft(f)
    cur_score = np.max(np.abs(C)) / (S * S) if abs(S) > 1e-12 else 1e10
    
    no_improve = 0
    max_iter = 5000
    iter_count = 0
    
    while iter_count < max_iter:
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0
        best_idx = -1
        best_new_score = cur_score
        
        # Sample a subset of indices for efficiency
        if N <= 1024:
            indices = np.arange(N)
        else:
            indices = np.random.choice(N, size=min(N, 2048), replace=False)
        
        for i in indices:
            if time.time() > DEADLINE:
                break
            
            v = f[i]
            new_v = -v
            delta = new_v - v
            
            # Update convolution incrementally
            C_new = C.copy()
            
            # C[k] = sum_j f[j] * f[j+k]
            # When f[i] changes by delta, C[k] changes by:
            # delta * f[i+k] for k >= 0
            # delta * f[i-k] for k < 0 (but C is symmetric, so handle carefully)
            
            # For linear autoconvolution C[k] = sum_j f[j] * f[j+k]
            # C_new[k] = C[k] + delta * (f[i+k] + f[i-k]) for valid indices
            # But this is approximate; let's do it properly
            
            # Actually, C[k] = sum_j f[j] * f[j+k]
            # If f[i] changes by delta, then:
            # C[k] gets delta * f[i+k] (from j=i term)
            # and delta * f[i-k] (from j+k=i, i.e., j=i-k term)
            
            for k in range(2 * N - 1):
                j1 = i + k
                j2 = i - k
                contrib = 0
                if 0 <= j1 < N:
                    contrib += f[j1]
                if 0 <= j2 < N:
                    contrib += f[j2]
                C_new[k] += delta * contrib
            
            # Special case: when k=0, the term f[i]*f[i] changes
            # C[0] = sum f[j]^2, so C[0] changes by delta * (f[i] + new_v) = delta * (2*f[i] + delta)
            # But our incremental update already handles this if we're careful
            # Let's just recompute C[0]
            C_new[0] = np.sum(f * f) + delta * (f[i] + new_v)
            
            new_S = S + delta
            if abs(new_S) < 1e-12:
                continue
            
            new_max = np.max(np.abs(C_new))
            new_score = new_max / (new_S * new_S)
            
            if new_score < best_new_score - 1e-12:
                best_new_score = new_score
                best_improvement = cur_score - new_score
                best_idx = i
        
        if best_idx >= 0:
            v = f[best_idx]
            f[best_idx] = -v
            S += -2.0 * v
            C = autoconv_fft(f)  # Recompute for accuracy
            cur_score = np.max(np.abs(C)) / (S * S)
            no_improve = 0
            iter_count += 1
        else:
            no_improve += 1
            if no_improve > 50:
                break
            # Try a random perturbation to escape local minimum
            if np.random.rand() < 0.3:
                idx = np.random.randint(N)
                f[idx] = -f[idx]
                S = np.sum(f)
                C = autoconv_fft(f)
                cur_score = np.max(np.abs(C)) / (S * S) if abs(S) > 1e-12 else 1e10
                iter_count += 1
    
    # Final check
    final_score = score(f)
    if final_score < best_score:
        best_score = final_score
        best_f = f.copy()
    
    # Return the sequence
    return best_f
# EVOLVE-BLOCK-END
