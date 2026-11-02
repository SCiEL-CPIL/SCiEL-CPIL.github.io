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

def _soft_max_acf_from_array(A, lam=10.0):
    # Soft max of squared ACF to smooth the objective
    # We want to minimize this
    # Using log-sum-exp for stability
    x = A[1:]**2
    m = np.max(x)
    if m == 0: return 0.0
    # log(sum(exp(x - m))) + m
    # We can just return m + log(sum(exp(x-m)))
    # But for local search, we just need to compare values.
    # A simpler smooth proxy: sum(x) is not good (sum is constant for +/-1? No, sum of squares is constant N-1).
    # So we need a function that penalizes peaks.
    # sum(exp(lam * x)) is good.
    return np.log(np.sum(np.exp(lam * x)))

def _flip_delta(s, A, i):
    """
    Compute the change in ACF if we flip s[i] -> -s[i].
    Returns delta array where A_new = A + delta.
    """
    delta = np.zeros(N)
    old_si = s[i]
    factor = -2.0 * old_si
    
    if i < N:
        delta[:N-i] += factor * s[i:N]
    
    if i >= 0:
        delta[:i+1] += factor * s[i::-1]
    
    return delta

def _halton_init(base, N):
    """Generate a sign sequence based on Halton sequence in given base."""
    s = np.zeros(N)
    for n in range(N):
        # Calculate van der Corput sequence value for n in given base
        f = 0.0
        p = 1.0 / base
        num = n
        while num > 0:
            f += p * (num % base)
            num //= base
            p /= base
        # Map to sign
        if f < 0.5:
            s[n] = 1.0
        else:
            s[n] = -1.0
    return s

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Use multiple restarts with different initializations
    while time.time() < deadline:
        # Try different initializations
        init_type = np.random.choice(4)
        if init_type == 0:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        elif init_type == 1:
            # Sine-based initialization
            t = np.linspace(0, 4*np.pi, N, endpoint=False)
            s = np.sign(np.sin(t))
            s[s == 0] = 1
        elif init_type == 2:
            # Random with some structure
            s = np.random.randint(0, 2, size=N) * 2 - 1
            # Apply a few random flips to break symmetry
            for _ in range(N // 10):
                idx = np.random.randint(N)
                s[idx] = -s[idx]
        else:
            # Halton-based initialization (Base 3 or 5)
            base = np.random.choice([3, 5, 7])
            s = _halton_init(base, N)
        
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search with efficient delta updates
        improved = True
        passes = 0
        max_passes = 200
        patience = 0
        max_patience = 5000
        
        while improved and passes < max_passes and patience < max_patience and time.time() < deadline:
            improved = False
            passes += 1
            patience = 0
            
            # Random order to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Compute delta
                delta = _flip_delta(s, A, i)
                
                # New value would be max(|A[k] + delta[k]|) for k=1..N-1
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    # Accept the flip
                    s[i] = -s[i]
                    A = A_new
                    current_val = val_new
                    improved = True
                    patience = 0
                else:
                    patience += 1
        
        # Simulated annealing polish
        if time.time() < deadline:
            T = 0.1
            T_min = 0.001
            cooling = 0.995
            sa_start = time.time()
            # Increase SA time to 10s to allow more exploration
            sa_deadline = min(deadline, sa_start + 10.0)
            
            while T > T_min and time.time() < sa_deadline:
                i = np.random.randint(N)
                delta = _flip_delta(s, A, i)
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val or np.random.random() < np.exp(-(val_new - current_val) / T):
                    s[i] = -s[i]
                    A = A_new
                    current_val = val_new
                
                T *= cooling
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()

    return best_seq
