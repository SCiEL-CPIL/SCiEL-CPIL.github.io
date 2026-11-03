import numpy as np

N = 50

def calc_score(h):
    h = np.array(h, dtype=float)
    s = np.sum(h)
    if s == 0:
        return 0.0
    h = h / s
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv == 0:
        return 0.0
    return l2_sq / max_conv

def construct():
    best_h = np.ones(N)
    best_score = calc_score(best_h)
    
    # Search over power law exponents
    for p in np.linspace(0.5, 3.0, 200):
        h = np.arange(1, N+1)**p
        s = calc_score(h)
        if s > best_score:
            best_score = s
            best_h = h
    
    # Search over "ramp with flat top"
    # h = i for i<k, k for i>=k
    for k in range(1, N+1):
        h = np.zeros(N)
        for i in range(N):
            h[i] = min(i+1, k)
        s = calc_score(h)
        if s > best_score:
            best_score = s
            best_h = h
            
    return best_h.tolist()
