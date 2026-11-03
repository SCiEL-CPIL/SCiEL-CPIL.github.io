import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence as +/- 1."""
    tm = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        tm[i] = tm[i >> 1] ^ (i & 1)
    # Convert 0/1 to +1/-1
    return np.where(tm == 0, 1.0, -1.0)

def _legendre_sequence(n):
    """Generate Legendre symbol sequence as +/- 1."""
    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        for i in range(3, int(k**0.5) + 1, 2):
            if k % i == 0:
                return False
        return True
    
    def legendre(a, p):
        if p == 2:
            return 1.0
        if a % p == 0:
            return 0.0
        if pow(a, (p - 1) // 2, p) == p - 1:
            return -1.0
        return 1.0
    
    # Use primes to generate a sequence
    s = np.ones(n)
    idx = 0
    p = 2
    while idx < n:
        if is_prime(p):
            for k in range(1, p):
                if idx < n:
                    val = legendre(k, p)
                    s[idx] = val if val != 0 else 1.0
                    idx += 1
        p += 1
    return s

def _sine_init(freq, phase):
    """Sine wave initialization."""
    x = np.arange(N)
    return np.sign(np.sin(2 * np.pi * freq * x / N + phase))

def _analytic_update(s, i):
    """
    Compute the change in max_abs_acf if we flip s[i].
    Uses the fact that flipping s[i] changes A[k] by -2*s[i]*(s[i+k] + s[i-k] adjustments).
    Actually, A[k] = sum_j s[j]*s[j+k]. Flipping s[i] changes:
    - For A[k]: terms involving s[i]: s[i]*s[i+k] and s[i-k]*s[i]
    So delta_A[k] = -2*s[i]*(s[i+k] + s[i-k]) for valid indices.
    We want to find if this reduces the max abs.
    """
    s_i = s[i]
    # The autocorrelation A[k] = sum_{j=0}^{N-1-k} s[j] * s[j+k]
    # Flipping s[i] changes A[k] by:
    #   -2 * s_i * s[i+k] (if i+k < N)
    #   -2 * s_i * s[i-k] (if i-k >= 0)
    # We need to check if the new max is lower.
    # For efficiency, we can compute the new A values that change.
    
    # This is O(N) per flip, which is too slow for many iterations.
    # Let's use a different approach: compute the full A once, then for each flip,
    # compute the delta and check if it helps.
    pass

def _greedy_polish(s, deadline, max_passes=200):
    """
    Greedy local search: try flipping each bit, keep if it improves.
    Use analytic delta computation for speed.
    """
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    
    local_start = time.time()
    improved = True
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Flip s[i]
            s[i] *= -1.0
            
            # Recompute A efficiently: 
            # A[k] changes by -2*s[i]*(s[i+k] + s[i-k])
            # But we need to be careful with boundaries.
            # Let's just recompute for simplicity but use a faster method.
            # Actually, let's compute the delta analytically.
            
            s_i = s[i]  # new value after flip
            old_s_i = -s_i
            
            # Compute delta A
            delta_A = np.zeros(N)
            # For each k, delta_A[k] = -2 * old_s_i * (s[i+k] if i+k<N else 0 + s[i-k] if i-k>=0 else 0)
            # Wait, the change is: new_A[k] - old_A[k] = -2 * old_s_i * (contribution from s[i])
            # contribution from s[i] to A[k] is: s[i]*s[i+k] (if i+k < N) + s[i-k]*s[i] (if i-k >= 0)
            # So delta = -2 * old_s_i * (s[i+k] if i+k<N else 0 + s[i-k] if i-k>=0 else 0)
            
            # Vectorized:
            # For k where i+k < N: k in [0, N-1-i]
            # For k where i-k >= 0: k in [i, N-1]
            
            # Let's compute it properly
            # delta_A[k] = -2 * old_s_i * (s[i+k] if i+k < N else 0 + s[i-k] if i-k >= 0 else 0)
            
            # This is O(N) per bit, so O(N^2) per pass. With N=400, that's 160k ops per pass.
            # 200 passes = 32M ops. Should be fast enough.
            
            # Actually, let's just recompute A from scratch since N is small.
            A_new = _calc_acf(s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val - 1e-12:
                current_val = val_new
                A = A_new
                improved = True
            else:
                # Revert
                s[i] *= -1.0
    
    return s, current_val

def _fast_sa(s, deadline, t_start=0.5, t_end=0.001, cooling=0.995):
    """
    Fast simulated annealing with single-bit flips.
    """
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    best_val = current_val
    best_s = s.copy()
    
    local_start = time.time()
    t = t_start
    
    while time.time() < deadline and t > t_end:
        # Random single-bit flip
        i = np.random.randint(0, N)
        s[i] *= -1.0
        
        A_new = _calc_acf(s)
        val_new = _max_abs_acf_from_array(A_new)
        
        delta = val_new - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            current_val = val_new
            A = A_new
            if val_new < best_val:
                best_val = val_new
                best_s = s.copy()
        else:
            s[i] *= -1.0
        
        t *= cooling
    
    return best_s, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try structured initializations
    inits = []
    
    # Thue-Morse
    inits.append(_thue_morse_sequence(N))
    
    # Legendre
    inits.append(_legendre_sequence(N))
    
    # Sine waves with various frequencies
    for freq in [1, 2, 3, 5, 7, 13, 21, 33, 53]:
        for phase in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
            inits.append(_sine_init(freq, phase))
    
    # Random
    for _ in range(10):
        inits.append(rng.choice([-1.0, 1.0], size=N))
    
    for s_init in inits:
        if time.time() > deadline - 30.0:
            break
        
        s = s_init.copy()
        s, current_val = _greedy_polish(s, deadline - 20.0, max_passes=50)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 10.0:
        s_anneal, val_anneal = _fast_sa(best_seq.copy(), deadline - 5.0)
        if val_anneal < best_val:
            best_val = val_anneal
            best_seq = s_anneal.copy()
    
    # Phase 3: More greedy passes from best
    if time.time() < deadline - 2.0:
        s_final, val_final = _greedy_polish(best_seq.copy(), deadline - 1.0, max_passes=100)
        if val_final < best_val:
            best_val = val_final
            best_seq = s_final.copy()
    
    return best_seq
