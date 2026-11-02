import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
MARGIN = 10.0
DEADLINE = TIME_LIMIT - MARGIN

def _autoconv_full(x):
    size = 1
    while size < 2 * N:
        size <<= 1
    f = np.zeros(size, dtype=np.complex128)
    f[:N] = x
    F = np.fft.fft(f)
    ac = np.fft.ifft(F * np.conj(F)).real[:2 * N - 1]
    return ac

def _c3_bound(x):
    ac = _autoconv_full(x)
    return float(np.max(np.abs(ac[1:])))

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    def now():
        return time.time()

    def remaining():
        return DEADLINE - (now() - t0)

    x = rng.choice([-1.0, 1.0], size=N)
    ac = _autoconv_full(x)
    cur_val = _c3_bound(x)
    best_x = x.copy()
    best_val = cur_val
    
    k = np.arange(2*N-1)
    
    def try_flip(i, T):
        nonlocal x, ac, cur_val, best_x, best_val
        
        xi = x[i]
        
        S = np.zeros(2*N-1)
        
        # x[i+k] contribution
        # k such that 0 <= i+k < N
        # -i <= k < N-i
        start1 = max(0, -i)
        end1 = min(2*N-2, N-1-i)
        if start1 <= end1:
            S[start1:end1+1] += x[i+start1 : i+end1+1]
            
        # x[i-k] contribution
        # k such that 0 <= i-k < N
        # i-N < k <= i
        start2 = max(0, i-N+1)
        end2 = min(2*N-2, i)
        if start2 <= end2:
            S[start2:end2+1] += x[i-start2 : i-end2-1 : -1]
            
        # Update ac
        # ac_new = ac - 2*xi*S
        # We modify ac in place
        ac -= 2.0 * xi * S
        
        new_val = float(np.max(np.abs(ac[1:])))
        
        if new_val < cur_val or (T > 0 and rng.random() < np.exp(-(new_val - cur_val) / T)):
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
            x[i] = -xi # Apply flip to x
            return True
        else:
            # Revert ac
            ac += 2.0 * xi * S
            # x[i] was not changed
            return False

    # Simulated Annealing
    T_start = 0.5
    T_end = 1e-4
    n_steps = 800000
    
    for i_step in range(n_steps):
        if remaining() < 20.0:
            break
        T = T_start * (T_end / T_start) ** (i_step / n_steps)
        idx = int(rng.integers(0, N))
        try_flip(idx, T)
        
    # Greedy Polish
    improved = True
    while improved and remaining() > 5.0:
        improved = False
        for i in range(N):
            if remaining() < 5.0:
                break
            if try_flip(i, 0.0):
                improved = True

    return best_x
