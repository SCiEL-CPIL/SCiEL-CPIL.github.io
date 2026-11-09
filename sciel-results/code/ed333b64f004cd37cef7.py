import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """Compute the normalized maximum of autoconvolution."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Structural change: try multiple structural families and pick the best.
    # Family 1: Single spike + uniform background (current approach)
    # Family 2: k spikes + uniform background, optimize k and H
    # Family 3: Two-level distribution with optimized ratio, try different k
    # Family 4: Smoothed spike using softplus-like distribution
    
    best_h = None
    best_score = float('inf')
    
    # Family 1: Single spike
    M = 1.0 / np.sqrt(N)
    h1 = np.full(N, (1.0 - M) / (N - 1))
    h1[0] = M
    s1 = compute_score(h1)
    if s1 < best_score:
        best_score = s1
        best_h = h1.copy()
    
    # Family 2: k spikes, optimize H for each k
    for k in range(2, min(20, N // 10)):
        if time.time() > deadline:
            break
        # For k spikes of height H, N-k valleys of height L
        # sum = kH + (N-k)L = 1 => L = (1-kH)/(N-k)
        # Need L >= 0 => H <= 1/k
        # Need L <= H => (1-kH)/(N-k) <= H => 1-kH <= H(N-k) => 1 <= HN => H >= 1/N
        # Score = (kH^2 + (N-k)L^2) / H = kH + (N-k)L^2/H
        # Substitute L:
        # f(H) = kH + (N-k) * ((1-kH)/(N-k))^2 / H = kH + (1-kH)^2 / ((N-k)H)
        # Optimize H in [1/N, 1/k]
        
        # Analytical derivative:
        # f'(H) = k + [2(1-kH)(-k)H - (1-kH)^2] / ((N-k)H^2)
        # = k - (1-kH)[2kH + 1-kH] / ((N-k)H^2)
        # = k - (1-kH)(1+kH) / ((N-k)H^2)
        # = k - (1-k^2H^2) / ((N-k)H^2)
        # Set to 0: k(N-k)H^2 = 1 - k^2H^2
        # k(N-k)H^2 + k^2H^2 = 1
        # kNH^2 = 1
        # H = 1/sqrt(kN)
        
        H_opt = 1.0 / np.sqrt(k * N)
        
        # Check bounds
        if H_opt < 1.0/N:
            H_opt = 1.0/N
        if H_opt > 1.0/k:
            H_opt = 1.0/k
            
        L_opt = (1.0 - k * H_opt) / (N - k)
        
        if L_opt < 0:
            continue
            
        h2 = np.full(N, L_opt)
        h2[:k] = H_opt
        
        s2 = compute_score(h2)
        if s2 < best_score:
            best_score = s2
            best_h = h2.copy()
    
    # Family 3: Try a smooth transition - use a Gaussian-like bump
    # h[i] = A * exp(-((i - c)^2) / (2 * sigma^2)) + B
    # Normalize so sum = 1
    # Optimize c (center), sigma (width), and the ratio A/B
    
    for sigma_frac in [0.1, 0.2, 0.3, 0.5]:
        if time.time() > deadline:
            break
        sigma = sigma_frac * N
        i = np.arange(N)
        c = N // 2
        gauss = np.exp(-((i - c) ** 2) / (2 * sigma ** 2))
        
        # Try different amplitudes
        for amp in [0.5, 1.0, 2.0, 5.0, 10.0]:
            h3 = amp * gauss + 1.0
            h3 = h3 / np.sum(h3)  # Normalize to sum=1
            
            s3 = compute_score(h3)
            if s3 < best_score:
                best_score = s3
                best_h = h3.copy()
    
    if best_h is None:
        # Fallback to Family 1
        best_h = h1
    
    return list(best_h)
