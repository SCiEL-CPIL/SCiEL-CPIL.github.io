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

def _flip_delta(s, i):
    """
    Compute the change in ACF if we flip s[i] -> -s[i].
    Returns delta array where A_new = A + delta.
    """
    delta = np.zeros(N)
    old_si = s[i]
    factor = -2.0 * old_si
    
    # For lags k=0..N-1-i: A[k] changes by factor * s[i+k]
    if i < N:
        delta[:N-i] += factor * s[i:N]
    
    # For lags k=1..i: A[k] also changes by factor * s[i-k] (from the conjugate part)
    if i > 0:
        delta[1:i+1] += factor * s[i-1::-1]
    
    return delta

def _block_flip_delta(s, A, start, length):
    """
    Compute the change in ACF if we flip a block of signs from start to start+length-1.
    """
    delta = np.zeros(N)
    block = s[start:start+length]
    factor = -2.0
    
    # For each position j in the block, its flip contributes to delta
    for j in range(length):
        i = start + j
        old_si = s[i]
        f = -2.0 * old_si
        
        if i < N:
            delta[:N-i] += f * s[i:N]
        
        if i > 0:
            delta[1:i+1] += f * s[i-1::-1]
    
    return delta

def _halton_init(base, N):
    """Generate a sign sequence based on Halton sequence in given base."""
    s = np.zeros(N)
    for n in range(N):
        f = 0.0
        p = 1.0 / base
        num = n
        while num > 0:
            f += p * (num % base)
            num //= base
            p /= base
        if f < 0.5:
            s[n] = 1.0
        else:
            s[n] = -1.0
    return s

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
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
            for _ in range(N // 10):
                idx = np.random.randint(N)
                s[idx] = -s[idx]
        else:
            # Halton-based initialization
            base = np.random.choice([3, 5, 7])
            s = _halton_init(base, N)
        
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search with efficient delta updates
        improved = True
        passes = 0
        max_passes = 100
        patience = 0
        max_patience = 3000
        
        while improved and passes < max_passes and patience < max_patience and time.time() < deadline:
            improved = False
            passes += 1
            patience = 0
            
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                delta = _flip_delta(s, i)
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    s[i] = -s[i]
                    A = A_new
                    current_val = val_new
                    improved = True
                    patience = 0
                else:
                    patience += 1
        
        # Block flip moves to escape local minima
        if time.time() < deadline:
            block_deadline = min(deadline, time.time() + 5.0)
            T = 0.05
            T_min = 0.001
            cooling = 0.999
            
            while T > T_min and time.time() < block_deadline:
                # Random block flip
                length = np.random.choice([2, 3, 4, 5, 8, 10, 15, 20])
                if length >= N:
                    continue
                start = np.random.randint(0, N - length + 1)
                
                # Compute delta for block flip
                delta = np.zeros(N)
                for j in range(length):
                    i = start + j
                    f = -2.0 * s[i]
                    if i < N:
                        delta[:N-i] += f * s[i:N]
                    if i > 0:
                        delta[1:i+1] += f * s[i-1::-1]
                
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val or (T > T_min and np.random.random() < np.exp(-(val_new - current_val) / T)):
                    s[start:start+length] = -s[start:start+length]
                    A = A_new
                    current_val = val_new
                
                T *= cooling
        
        # Final single-flip polish
        if time.time() < deadline:
            polish_deadline = min(deadline, time.time() + 3.0)
            for _ in range(50):
                if time.time() > polish_deadline:
                    break
                improved = False
                indices = np.random.permutation(N)
                for i in indices:
                    if time.time() > polish_deadline:
                        break
                    delta = _flip_delta(s, i)
                    A_new = A + delta
                    val_new = _max_abs_acf_from_array(A_new)
                    if val_new < current_val:
                        s[i] = -s[i]
                        A = A_new
                        current_val = val_new
                        improved = True
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()

    return best_seq
