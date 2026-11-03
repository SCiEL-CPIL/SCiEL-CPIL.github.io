import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 145.0

    best_h = np.zeros(N)
    best_score = 0.0

    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        if np.all(h == 0):
            return
        sc = _score_raw(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    def neg_obj(h):
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-15:
            return 0.0
        return -(l2_sq / (l1 * mc))

    bounds = [(0.0, None)] * N

    def polish(h_init, maxiter=200):
        h_init = np.asarray(h_init, dtype=np.float64)
        try:
            res = minimize(neg_obj, h_init, method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-10})
            if res.success:
                h_cand = np.clip(res.x, 0, None)
                try_candidate(h_cand)
            else:
                try_candidate(h_init)
        except Exception:
            try_candidate(h_init)

    rng = np.random.RandomState(42)

    # Sidon set initialization
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    sidon = [0]
    for i in range(1, N):
        if is_sidon(sidon + [i]):
            sidon.append(i)
    
    h_sidon = np.zeros(N)
    h_sidon[sidon] = 1.0
    try_candidate(h_sidon)
    polish(h_sidon, maxiter=300)

    # Random sparse initializations
    for K in range(3, 18):
        if time.time() > deadline - 10:
            break
        for trial in range(8):
            if time.time() > deadline - 5:
                break
            support = rng.choice(N, size=K, replace=False)
            h_init = np.zeros(N)
            h_init[support] = 1.0
            try_candidate(h_init)
            polish(h_init, maxiter=150)

    # Polish the best found so far
    if time.time() < deadline - 5:
        polish(best_h, maxiter=500)

    return list(best_h)
