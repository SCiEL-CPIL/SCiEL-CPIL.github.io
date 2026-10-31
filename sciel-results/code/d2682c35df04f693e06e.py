import numpy as np
import time
from itertools import product

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def shape_score(h):
    """Scale-invariant: minimize ||h*h||_inf * ||h||_1 / ||h||_2^2"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 0.0
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l1 * conv_peak / l2_sq

def make_comb(period, offset, width=1):
    """Create a periodic comb pattern."""
    h = np.zeros(N)
    for i in range(N):
        if (i - offset) % period < width:
            h[i] = 1.0
    if np.sum(h) > 0:
        h /= np.sum(h)
    return h

def make_gaussian(center, sigma):
    """Create a Gaussian-like profile."""
    x = np.arange(N)
    h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    h += 1e-15
    h /= np.sum(h)
    return h

def make_block(start, length):
    """Create a block of consecutive ones."""
    h = np.zeros(N)
    end = min(start + length, N)
    if end > start:
        h[start:end] = 1.0
    if np.sum(h) > 0:
        h /= np.sum(h)
    return h

def make_mixture(h1, h2, alpha):
    """Mix two distributions."""
    h = alpha * h1 + (1 - alpha) * h2
    if np.sum(h) > 0:
        h /= np.sum(h)
    return h

def make_two_peaks(center1, sigma1, center2, sigma2):
    """Two Gaussian peaks."""
    x = np.arange(N)
    h = np.exp(-0.5 * ((x - center1) / sigma1) ** 2) + np.exp(-0.5 * ((x - center2) / sigma2) ** 2)
    h += 1e-15
    h /= np.sum(h)
    return h

def make_alternating_weighted(period, weight_on):
    """Alternating pattern with weighted on/off."""
    h = np.zeros(N)
    for i in range(N):
        if i % period == 0:
            h[i] = weight_on
        else:
            h[i] = 1.0 - weight_on
    h = np.maximum(h, 0)
    if np.sum(h) > 0:
        h /= np.sum(h)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N) / N
    best_shape = shape_score(best_h)
    
    def try_candidate(h):
        nonlocal best_h, best_shape
        if time.time() > deadline:
            return
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-15:
            return
        h = h / s
        sc = shape_score(h)
        if sc < best_shape:
            best_shape = sc
            best_h = h.copy()
    
    # Phase 1: Exhaustive structured family search
    
    # 1a. Periodic combs: period, offset, width
    for period in range(2, 16):
        if time.time() > deadline:
            break
        for offset in range(period):
            for width in range(1, min(period, 4)):
                h = make_comb(period, offset, width)
                try_candidate(h)
    
    # 1b. Gaussian profiles
    for center in range(0, N, 1):
        if time.time() > deadline:
            break
        for sigma in np.logspace(-1, 2, 20):
            h = make_gaussian(center, sigma)
            try_candidate(h)
    
    # 1c. Block structures
    for start in range(0, N, 1):
        if time.time() > deadline:
            break
        for length in range(1, N - start + 1, 1):
            h = make_block(start, length)
            try_candidate(h)
    
    # 1d. Two peaks
    for c1 in range(0, N, 2):
        if time.time() > deadline:
            break
        for c2 in range(c1 + 2, N, 2):
            for s1 in [1.0, 2.0, 3.0, 5.0, 8.0]:
                for s2 in [1.0, 2.0, 3.0, 5.0, 8.0]:
                    h = make_two_peaks(c1, s1, c2, s2)
                    try_candidate(h)
    
    # 1e. Alternating weighted patterns
    for period in range(2, 12):
        if time.time() > deadline:
            break
        for w in np.linspace(0.0, 1.0, 11):
            h = make_alternating_weighted(period, w)
            try_candidate(h)
    
    # 1f. Mixtures of comb and Gaussian
    for period in [3, 4, 5, 6, 7, 8, 10, 12]:
        if time.time() > deadline:
            break
        for offset in range(period):
            comb = make_comb(period, offset, 1)
            for center in [N//4, N//2, 3*N//4]:
                for sigma in [2.0, 4.0, 6.0, 10.0]:
                    gauss = make_gaussian(center, sigma)
                    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                        h = make_mixture(comb, gauss, alpha)
                        try_candidate(h)
    
    # 1g. Mixtures of two combs
    for p1 in [3, 4, 5, 6, 7, 8, 10]:
        if time.time() > deadline:
            break
        for p2 in [p1 + 1, p1 + 2, p1 + 3]:
            if p2 > 15:
                continue
            for o1 in range(p1):
                for o2 in range(p2):
                    c1 = make_comb(p1, o1, 1)
                    c2 = make_comb(p2, o2, 1)
                    for alpha in [0.2, 0.4, 0.5, 0.6, 0.8]:
                        h = make_mixture(c1, c2, alpha)
                        try_candidate(h)
    
    # Phase 2: Local refinement around best structured candidate
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)
        
        # Gradient-free local search with decreasing step sizes
        step_sizes = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
        
        for psize in step_sizes:
            if time.time() > deadline:
                break
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Try all single-index adjustments
                for i in range(N):
                    if time.time() > deadline:
                        break
                    for j in range(N):
                        if i == j:
                            continue
                        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                            delta = h[i] * psize * frac
                            if delta < 1e-12 or h[i] - delta < -1e-12:
                                continue
                            h_new = h.copy()
                            h_new[i] = max(0, h_new[i] - delta)
                            h_new[j] += delta
                            sc = shape_score(h_new)
                            if sc < current_score:
                                h = h_new
                                current_score = sc
                                improved = True
                                if current_score < best_shape:
                                    best_shape = current_score
                                    best_h = h.copy()
                                break
    
    # Phase 3: Final polish with very small steps
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)
        
        for _ in range(500):
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            delta = h[i] * 0.005 * np.random.uniform(0.1, 1.0)
            if delta < 1e-12 or h[i] - delta < -1e-12:
                continue
            
            h_new = h.copy()
            h_new[i] = max(0, h_new[i] - delta)
            h_new[j] += delta
            
            sc = shape_score(h_new)
            if sc < current_score:
                h = h_new
                current_score = sc
                if current_score < best_shape:
                    best_shape = current_score
                    best_h = h.copy()
    
    # Return the normalized (L1=1) best shape
    # The ratio is scale-invariant, so L1=1 is fine
    final_h = best_h.copy()
    final_h = np.maximum(final_h, 0)
    s = np.sum(final_h)
    if s > 0:
        final_h /= s
    
    return final_h.tolist()
