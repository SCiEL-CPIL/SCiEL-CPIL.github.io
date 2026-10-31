import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    # Existing searches...
    # 2-point
    for pos_pair in [(0, 1), (N//2 - 1, N//2), (0, N//2)]:
        for a in np.linspace(0.01, 0.99, 100):
            h = np.zeros(N)
            h[pos_pair[0]] = a
            h[pos_pair[1]] = 1.0 - a
            sc = compute_ratio(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()
            if time.time() > deadline - 10.0:
                return list(best_h)

    # 3-point symmetric
    center = N // 2
    for a in np.linspace(0.01, 0.49, 100):
        b = 1.0 - 2.0 * a
        if b < 0: continue
        h = np.zeros(N)
        h[center-1] = a
        h[center] = b
        h[center+1] = a
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
            if time.time() > deadline - 10.0:
                return list(best_h)

    # New: Random sparse search
    # Try 4 to 8 non-zero elements
    np.random.seed(42)
    for _ in range(5000):
        if time.time() > deadline - 10.0:
            break
        k = np.random.randint(4, 9)
        indices = np.random.choice(N, k, replace=False)
        weights = np.random.dirichlet(np.ones(k))
        h = np.zeros(N)
        h[indices] = weights
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    return list(best_h)
