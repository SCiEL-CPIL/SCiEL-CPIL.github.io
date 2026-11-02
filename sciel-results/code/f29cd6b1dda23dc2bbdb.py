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

def _flip_delta(s, A, i):
    """
    Compute the change in ACF if we flip s[i] -> -s[i].
    Returns delta array where A_new = A + delta.
    """
    delta = np.zeros(N)
    old_si = s[i]
    factor = -2.0 * old_si
    
    # For lag k, the term involves s[i+k] (if i+k < N) and s[i-k] (if i-k >= 0)
    # A[k] = sum_j s[j] * s[j+k]
    # When we flip s[i], the terms that change are those where j=i or j+k=i
    # i.e., j=i (contributes s[i]*s[i+k]) and j=i-k (contributes s[i-k]*s[i])
    # So delta[k] = factor * (s[i+k] + s[i-k]) for valid indices
    
    # Part 1: j = i, so k = j' - i where j' = i + k, valid when i+k < N
    # k ranges from 0 to N-1-i
    # delta[k] += factor * s[i+k]
    if i < N:
        delta[:N-i] += factor * s[i:N]
    
    # Part 2: j+k = i, so j = i-k, valid when i-k >= 0
    # k ranges from 0 to i
    # delta[k] += factor * s[i-k]
    if i >= 0:
        delta[:i+1] += factor * s[i::-1]
    
    return delta

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Use multiple restarts with different initializations
    while time.time() < deadline:
        # Try different initializations
        init_type = np.random.choice(3)
        if init_type == 0:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        elif init_type == 1:
            # Sine-based initialization
            t = np.linspace(0, 4*np.pi, N, endpoint=False)
            s = np.sign(np.sin(t))
            s[s == 0] = 1
        else:
            # Random with some structure
            s = np.random.randint(0, 2, size=N) * 2 - 1
            # Apply a few random flips to break symmetry
            for _ in range(N // 10):
                idx = np.random.randint(N)
                s[idx] = -s[idx]
        
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
            sa_deadline = min(deadline, sa_start + 5.0)
            
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
