import numpy as np
import time

N = 400
FTLEN = 2 * N  # padding length

def autocorr_max_offdiag(f):
    # f: array of length N with values ±1
    F = np.fft.fft(f, n=FTLEN)
    ac = np.fft.ifft(F * np.conj(F)).real[:N]
    # ac[0] = sum f^2 = N, ac[k] for k>=1
    return np.max(np.abs(ac[1:]))

def thue_morse(n):
    # Thue-Morse sequence mapped to ±1
    # t_i = parity of number of 1s in binary i
    tm = np.array([bin(i).count('1') % 2 for i in range(n)])
    return 1 - 2 * tm  # 0 -> 1, 1 -> -1

def score(f):
    return autocorr_max_offdiag(f)

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill (and 150s hard)
    
    # Initial candidates
    candidates = []
    candidates.append(thue_morse(N))
    rng = np.random.default_rng(0)
    for _ in range(8):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    candidates.append(np.ones(N))
    
    best = None
    best_val = np.inf
    for c in candidates:
        v = score(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    
    # Local search + annealing
    f = best.copy()
    cur_val = best_val
    T = 0.5  # initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    while time.time() < deadline:
        # One pass of local search: try each flip
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            f[i] = -f[i]
            new_val = score(f)
            if new_val <= cur_val:
                cur_val = new_val
                improved = True
                if new_val < best_val:
                    best_val = new_val
                    best = f.copy()
            else:
                # Metropolis accept with probability
                if T > T_min:
                    delta = new_val - cur_val
                    if np.random.random() < np.exp(-delta / T):
                        cur_val = new_val
                    else:
                        f[i] = -f[i]
        T *= cooling
    
    return best
