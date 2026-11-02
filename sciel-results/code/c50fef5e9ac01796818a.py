"""
Signed step functions minimizing max |autoconv| / (sum)^2.
Structural approach: low-autocorrelation sequence initialization + simulated annealing
with O(N) incremental autoconvolution updates.
"""

import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _autoconv_direct(x):
    """Compute autoconvolution directly: C[m] = sum_i x[i]*x[i+m] for m in 0..N-1, and C[N-1-m] = C[m] by symmetry for full 2N-1."""
    L = 2 * N - 1
    C = np.zeros(L)
    for m in range(N):
        C[m] = np.dot(x[:N-m], x[m:])
        C[L-1-m] = C[m]
    return C

def _incremental_flip_update(C, x, pos):
    """
    When we flip x[pos] (from v to -v), update autoconvolution C.
    C[m] = sum_i x[i] * x[i+m]
    Flipping x[pos] affects C[m] for all m where pos is in [0, N-1] and pos+m is in [0, N-1], 
    OR where pos-m is in [0, N-1] (i.e., m = pos - i for some i, meaning x[i]*x[pos] term).
    
    More carefully: C[m] contains term x[i]*x[i+m]. 
    x[pos] appears when i = pos (then x[i+m] = x[pos+m]) or when i+m = pos (then x[i] = x[pos-m]).
    So C[m] changes by:
      - If pos+m < N: term x[pos]*x[pos+m] was in C[m], now becomes -x[pos]*x[pos+m], delta = -2*x[pos]*x[pos+m]
      - If pos-m >= 0: term x[pos-m]*x[pos] was in C[m], now becomes -x[pos-m]*x[pos], delta = -2*x[pos]*x[pos-m]
    
    Note: when m=0, both conditions apply (i=pos and i+m=pos), so delta = -4*x[pos]^2 = -4 (since x[pos]^2=1)
    But wait, for m=0: C[0] = sum x[i]^2 = N always. Flipping doesn't change C[0].
    Let me recheck: C[0] = sum_i x[i]*x[i] = N. Yes, invariant.
    
    For m > 0:
      - pos+m < N contributes -2*x[pos]*x[pos+m] to C[m]
      - pos-m >= 0 contributes -2*x[pos]*x[pos-m] to C[m]
    
    Also need to update C[L-1-m] = C[m] by symmetry.
    """
    v = x[pos]
    L = len(C)
    for m in range(1, N):
        delta = 0.0
        if pos + m < N:
            delta -= 2.0 * v * x[pos + m]
        if pos - m >= 0:
            delta -= 2.0 * v * x[pos - m]
        if delta != 0.0:
            C[m] += delta
            C[L - 1 - m] += delta
    return C

def _max_abs_ac(C):
    return np.max(np.abs(C))

def _construct_msequence(N):
    """Generate an m-sequence (maximum length sequence) of length close to N, pad/truncate to N."""
    # m-sequence of length 2^m - 1
    # Find m such that 2^m - 1 >= N
    m = 1
    while (2**m - 1) < N:
        m += 1
    mlen = 2**m - 1
    
    # Generate m-sequence using a primitive polynomial
    # For small m, use known primitive polynomials
    # m=9: x^9 + x^4 + 1
    # m=10: x^10 + x^7 + 1
    # m=11: x^11 + x^9 + 1
    # m=12: x^12 + x^6 + x^4 + x + 1
    # m=13: x^13 + x^4 + x^3 + x^2 + 1
    # m=14: x^14 + x^10 + x^6 + x^3 + 1
    # m=15: x^15 + x + 1
    
    # Use a simple LFSR
    # Primitive polynomial for m=9: 0x11 (x^9 + x^4 + 1)
    # Let's use m=9 (length 511) and truncate to N=400
    
    m_val = 9
    # Primitive polynomial: x^9 + x^4 + 1 -> bits at positions 9 and 4
    poly = (1 << m_val) | (1 << 4) | 1  # 0b100010001
    
    state = 1
    seq = []
    for _ in range(2**m_val - 1):
        seq.append(1 if (state & 1) else -1)
        # LFSR step
        if state & 1:
            state = ((state >> 1) ^ poly) & ((1 << m_val) - 1)
        else:
            state = state >> 1
    
    # Truncate or pad to N
    if len(seq) >= N:
        x = np.array(seq[:N], dtype=np.float64)
    else:
        # Pad with alternating signs
        pad_len = N - len(seq)
        pad = np.array([1 if i % 2 == 0 else -1 for i in range(pad_len)], dtype=np.float64)
        x = np.concatenate([np.array(seq, dtype=np.float64), pad])
    
    return x

def _construct_legendre(N):
    """Legendre sequence: x[n] = 1 if n is quadratic residue mod p, else -1."""
    # Find prime p >= N
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    p = N
    while not is_prime(p):
        p += 1
    
    x = np.ones(N, dtype=np.float64)
    for n in range(1, N):
        # Legendre symbol: n^((p-1)/2) mod p
        ls = pow(n, (p - 1) // 2, p)
        if ls == p - 1:  # -1 mod p
            x[n] = -1.0
    return x

def _simulated_annealing(x, max_iter=500000, seed=42):
    """
    Simulated annealing with O(N) incremental updates.
    """
    rng = np.random.default_rng(seed)
    
    C = _autoconv_direct(x)
    current_val = _max_abs_ac(C) / (np.sum(x) ** 2)
    best_x = x.copy()
    best_val = current_val
    
    T_start = 1.0
    T_end = 0.01
    T = T_start
    
    for it in range(max_iter):
        if _check_deadline():
            break
        
        # Cooling
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Choose position to flip
        pos = rng.integers(0, N)
        
        # Compute delta in objective
        old_sum = np.sum(x)
        new_sum = old_sum - 2 * x[pos]
        
        if abs(new_sum) < 1e-10:
            # Skip flips that would make sum zero
            continue
        
        old_peak = _max_abs_ac(C)
        old_obj = old_peak / (old_sum ** 2)
        
        # Update C
        C = _incremental_flip_update(C, x, pos)
        x[pos] = -x[pos]
        
        new_peak = _max_abs_ac(C)
        new_obj = new_peak / (new_sum ** 2)
        
        delta = new_obj - old_obj
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept
            if new_obj < best_val:
                best_val = new_obj
                best_x = x.copy()
        else:
            # Reject: revert
            x[pos] = -x[pos]
            C = _incremental_flip_update(C, x, pos)
    
    return best_x, best_val

def construct():
    # Multiple initializations
    candidates = []
    
    # 1. All ones
    x1 = np.ones(N)
    candidates.append(x1)
    
    # 2. M-sequence based
    x2 = _construct_msequence(N)
    candidates.append(x2)
    
    # 3. Legendre sequence
    x3 = _construct_legendre(N)
    candidates.append(x3)
    
    # 4. Random balanced sequence
    rng = np.random.default_rng(123)
    x4 = rng.choice([-1.0, 1.0], size=N)
    candidates.append(x4)
    
    # Evaluate all
    best_x = None
    best_val = 1e10
    for x in candidates:
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    if _check_deadline():
        return best_x
    
    # Simulated annealing from best candidate
    x_sa, val_sa = _simulated_annealing(best_x, max_iter=200000, seed=42)
    if val_sa < best_val:
        best_val = val_sa
        best_x = x_sa
    
    if _check_deadline():
        return best_x
    
    # Simulated annealing from m-sequence
    x_sa2, val_sa2 = _simulated_annealing(x2, max_iter=200000, seed=123)
    if val_sa2 < best_val:
        best_val = val_sa2
        best_x = x_sa2
    
    if _check_deadline():
        return best_x
    
    # Simulated annealing from Legendre
    x_sa3, val_sa3 = _simulated_annealing(x3, max_iter=200000, seed=456)
    if val_sa3 < best_val:
        best_val = val_sa3
        best_x = x_sa3
    
    if _check_deadline():
        return best_x
    
    # Final SA from best
    x_final, val_final = _simulated_annealing(best_x, max_iter=300000, seed=789)
    if val_final < best_val:
        best_x = x_final
    
    return best_x
# EVOLVE-BLOCK-END
