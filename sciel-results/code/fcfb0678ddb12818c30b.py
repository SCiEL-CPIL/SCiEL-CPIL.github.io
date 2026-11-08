import time
import numpy as np
from scipy import signal

N = 50

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf)"""
    h = np.array(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Autoconvolution
    conv = signal.convolve(h, h, mode='full')
    conv_inf = np.max(np.abs(conv))
    if conv_inf == 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _ratio_normalized(h):
    """Compute ratio with L1-normalized h (shape-based, scale-invariant)."""
    h = np.array(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    conv = signal.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(np.abs(conv))
    if conv_inf == 0:
        return 0.0
    return l2_sq / conv_inf

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    best_h = [1.0] * N  # Seed as fallback
    best_ratio = _compute_ratio(best_h)
    
    # Try various structured configurations
    candidates = []
    
    # 1. Uniform
    candidates.append([1.0] * N)
    
    # 2. Single spike
    for pos in [0, N//2, N-1]:
        h = [0.0] * N
        h[pos] = 1.0
        candidates.append(h)
    
    # 3. Two spikes at various separations
    for sep in range(1, N):
        h = [0.0] * N
        h[0] = 1.0
        h[sep] = 1.0
        candidates.append(h)
    
    # 4. Block of consecutive ones
    for block_len in range(1, N+1):
        h = [0.0] * N
        h[:block_len] = [1.0] * block_len
        candidates.append(h)
        # Also centered
        if block_len < N:
            h2 = [0.0] * N
            start = (N - block_len) // 2
            h2[start:start+block_len] = [1.0] * block_len
            candidates.append(h2)
    
    # 5. Alternating patterns
    for period in range(2, 11):
        h = [0.0] * N
        for i in range(N):
            if i % period == 0:
                h[i] = 1.0
        candidates.append(h)
    
    # 6. Gaussian-like
    center = N // 2
    for sigma in [1, 2, 3, 5, 7, 10, 15, 20]:
        h = [0.0] * N
        for i in range(N):
            h[i] = np.exp(-0.5 * ((i - center) / sigma)**2)
        candidates.append(h)
    
    # 7. Cosine window
    for freq in [1, 2, 3, 4, 5, 7]:
        h = [0.0] * N
        for i in range(N):
            val = np.cos(2 * np.pi * freq * i / N)
            h[i] = max(0.0, val)
        candidates.append(h)
    
    # 8. Two-point clusters with varying ratios (key finding: 2:1 ratio helps)
    for sep in range(1, N):
        for ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            h = [0.0] * N
            h[0] = 1.0
            h[sep] = ratio
            candidates.append(h)
    
    # 9. Three-point clusters
    for sep1 in [1, 2, 3, 5, 10]:
        for sep2 in [sep1+1, sep1+2, sep1+5, sep1+10]:
            if sep2 < N:
                for r1 in [0.5, 1.0, 2.0]:
                    for r2 in [0.5, 1.0, 2.0]:
                        h = [0.0] * N
                        h[0] = 1.0
                        h[sep1] = r1
                        h[sep2] = r2
                        candidates.append(h)
    
    # 10. Random configurations
    np.random.seed(42)
    num_random = 500
    for r in range(num_random):
        if time.time() > deadline - 40.0:
            break
        # Random sparse with 1-5 nonzero
        num_nonzero = np.random.randint(1, 6)
        positions = np.random.choice(N, size=num_nonzero, replace=False)
        h = [0.0] * N
        for pos in positions:
            h[pos] = np.random.uniform(0.1, 3.0)
        candidates.append(h)
        
        # Random dense
        h = list(np.random.uniform(0.0, 1.0, N))
        candidates.append(h)
    
    # Evaluate all candidates using raw ratio
    for h in candidates:
        if time.time() > deadline - 10.0:
            break
        ratio = _compute_ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h
    
    # Local optimization: perturb best solution
    if time.time() < deadline - 5.0:
        best_arr = np.array(best_h, dtype=np.float64)
        np.random.seed(123)
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            # Random perturbation
            perturbed = best_arr.copy()
            idx = np.random.randint(N)
            perturbed[idx] = max(0.0, perturbed[idx] + np.random.normal(0, 0.5))
            # Occasionally set to 0 or increase sparsity
            if np.random.random() < 0.3:
                perturbed[np.random.randint(N)] = 0.0
            
            ratio = _compute_ratio(perturbed)
            if ratio > best_ratio:
                best_ratio = ratio
                best_arr = perturbed
    
    return list(best_arr)
