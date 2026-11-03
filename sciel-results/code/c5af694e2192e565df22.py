import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return DEADLINE - 5.0 - _elapsed()

def _get_fft_size(n):
    # next power of 2 >= n
    m = 1
    while m < n:
        m <<= 1
    return m

def _autoconv(h):
    # linear autoconvolution via FFT
    n = len(h)
    M = _get_fft_size(2 * n - 1)
    H = np.fft.rfft(h, M)
    ac = np.fft.irfft(H * H, M)
    return ac[:2 * n - 1]

def _score(h):
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    ac = _autoconv(h)
    return float(np.max(np.abs(ac)) / (S * S))

def _soft_max_abs(ac, beta=50.0):
    # log-sum-exp of |ac|^2 scaled, gives smooth max
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-12:
        return 0.0
    # smooth max of |a_k|
    s = np.log(np.sum(np.exp(beta * (a - m)))) / beta + m
    return float(s)

def _objective_smooth(x, beta=50.0):
    # x is continuous vector; we project to signs via tanh at end for score
    # but for smoothness use soft sign: h = tanh(x)
    h = np.tanh(x)
    S = np.sum(h)
    if abs(S) < 1e-8:
        return 1e6 + beta
    ac = _autoconv(h)
    smax = _soft_max_abs(ac, beta)
    return smax / (S * S)

def _signs_from_x(x):
    return np.sign(x)

def _local_search(h, max_time=20.0):
    # sign-flip local search
    h = h.copy()
    best = _score(h)
    t_start = time.time()
    improved = True
    while improved and (time.time() - t_start) < max_time and _remaining() > 10.0:
        improved = False
        # random order
        order = np.random.permutation(N)
        for i in order:
            if (time.time() - t_start) > max_time or _remaining() < 10.0:
                break
            old = h[i]
            h[i] = -old
            s = _score(h)
            if s < best - 1e-12:
                best = s
                improved = True
            else:
                h[i] = old
    return h, best

def _simulated_annealing(h, max_time=30.0):
    h = h.copy()
    best_h = h.copy()
    best = _score(h)
    t_start = time.time()
    T = 0.01
    T_min = 1e-5
    cooling = 0.995
    while (time.time() - t_start) < max_time and _remaining() > 10.0 and T > T_min:
        i = np.random.randint(N)
        old = h[i]
        h[i] = -old
        s = _score(h)
        delta = s - best
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-12)):
            if s < best:
                best = s
                best_h = h.copy()
        else:
            h[i] = old
        T *= cooling
    return best_h, best

def construct():
    np.random.seed(42)
    best_h = None
    best_score = np.inf

    # Initial candidates
    candidates = []

    # 1. Random signs
    candidates.append(np.random.choice([-1, 1], N))

    # 2. Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))

    # 3. Block signs (quarter waves)
    candidates.append(np.sign(np.sin(2 * np.pi * np.arange(N) / N)))
    candidates.append(np.sign(np.sin(2 * np.pi * 2 * np.arange(N) / N)))
    candidates.append(np.sign(np.cos(2 * np.pi * np.arange(N) / N)))

    # 4. Zadoff-Chu-like (m-sequence approximation)
    m = np.random.choice([-1, 1], N)
    candidates.append(m)

    # Evaluate initial
    for c in candidates:
        c = c.astype(float)
        s = _score(c)
        if s < best_score:
            best_score = s
            best_h = c.copy()

    # Continuous optimization from best
    x0 = np.arctanh(np.clip(best_h, -0.99, 0.99))
    
    if _remaining() > 30.0:
        # L-BFGS-B on smooth surrogate
        try:
            res = minimize(_objective_smooth, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxfun': 5000, 'disp': False})
            if res.success or res.fun < best_score:
                x_opt = res.x
                h_opt = np.sign(x_opt)
                s = _score(h_opt)
                if s < best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Local search polish
    if _remaining() > 15.0:
        h_ls, s_ls = _local_search(best_h, max_time=min(20.0, _remaining() - 10.0))
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls

    # Simulated annealing
    if _remaining() > 20.0:
        h_sa, s_sa = _simulated_annealing(best_h, max_time=min(30.0, _remaining() - 10.0))
        if s_sa < best_score:
            best_score = s_sa
            best_h = h_sa

    # Final local search
    if _remaining() > 10.0:
        h_ls2, s_ls2 = _local_search(best_h, max_time=min(10.0, _remaining() - 5.0))
        if s_ls2 < best_score:
            best_score = s_ls2
            best_h = h_ls2

    # Ensure output is clean signs
    result = np.sign(best_h)
    # If all zeros (shouldn't happen), fallback
    if np.all(result == 0):
        result = np.ones(N)
    
    return result

if __name__ == "__main__":
    # Quick self-test
    h = construct()
    print("Score:", _score(h))
    print("Time:", _elapsed())
