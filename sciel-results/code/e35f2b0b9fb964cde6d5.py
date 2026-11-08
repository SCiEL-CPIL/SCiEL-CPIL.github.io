import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for a given height array."""
    h = np.array(h, dtype=np.float64)
    if np.sum(h) == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    linf = np.max(conv)
    if l1 * linf == 0:
        return 0.0
    return l2_sq / (l1 * linf)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = [1.0] * N
    best_score = compute_ratio(best_h)
    
    # Strategy: Try various sparse patterns
    # Key insight: concentrate mass in few points
    
    # Try single spike
    h_single = [0.0] * N
    h_single[0] = 1.0
    s = compute_ratio(h_single)
    if s > best_score:
        best_score = s
        best_h = h_single[:]
    
    # Try 2-point clusters with various ratios and positions
    for pos in range(N - 1):
        for ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 0.25, 0.75, 4.0]:
            h = [0.0] * N
            h[pos] = 1.0
            h[pos + 1] = ratio
            s = compute_ratio(h)
            if s > best_score:
                best_score = s
                best_h = h[:]
            if time.time() > deadline:
                return best_h
    
    # Try 3-point clusters
    for pos in range(N - 2):
        for r1 in [0.5, 1.0, 2.0]:
            for r2 in [0.5, 1.0, 2.0]:
                h = [0.0] * N
                h[pos] = 1.0
                h[pos + 1] = r1
                h[pos + 2] = r2
                s = compute_ratio(h)
                if s > best_score:
                    best_score = s
                    best_h = h[:]
                if time.time() > deadline:
                    return best_h
    
    # Try 4-point clusters
    for pos in range(N - 3):
        for r1 in [0.5, 1.0, 2.0]:
            for r2 in [0.5, 1.0, 2.0]:
                for r3 in [0.5, 1.0, 2.0]:
                    h = [0.0] * N
                    h[pos] = 1.0
                    h[pos + 1] = r1
                    h[pos + 2] = r2
                    h[pos + 3] = r3
                    s = compute_ratio(h)
                    if s > best_score:
                        best_score = s
                        best_h = h[:]
                    if time.time() > deadline:
                        return best_h
    
    # Try symmetric patterns centered
    center = N // 2
    for width in [1, 2, 3, 4, 5]:
        for peak_ratio in [1.0, 2.0, 3.0, 5.0]:
            h = [0.0] * N
            for i in range(width):
                offset = i - width // 2
                pos = center + offset
                if 0 <= pos < N:
                    # Triangular or peak profile
                    if width == 1:
                        h[pos] = peak_ratio
                    else:
                        # Peak at center, decay outward
                        dist = abs(offset)
                        h[pos] = peak_ratio * (1.0 - dist / (width / 2.0 + 0.001))
                        if h[pos] < 0:
                            h[pos] = 0.0
            s = compute_ratio(h)
            if s > best_score:
                best_score = s
                best_h = h[:]
            if time.time() > deadline:
                return best_h
    
    return best_h
