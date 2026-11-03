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

def _max_abs_acf(A):
    return float(np.max(np.abs(A[1:])))

def _soft_max_acf(A, lam=5.0):
    """
    Smooth surrogate: log-sum-exp of squared ACF values.
    This is differentiable and helps gradient-free search escape local minima.
    """
    x = A[1:] ** 2
    m = np.max(x)
    if m == 0:
        return 0.0
    return float(m + np.log(np.sum(np.exp(lam * (x - m)))))

def _flip_delta(s, i):
    """
    Compute the change in ACF if we flip s[i] -> -s[i].
    A_new[k] = A[k] + delta[k]
    """
    delta = np.zeros(N)
    old_si = s[i]
    factor = -2.0 * old_si

    # A[k] = sum_j s[j]*s[j+k] for k>=0
    # Flipping s[i] affects terms where j=i (s[i]*s[i+k]) or j+k=i (s[j]*s[i])
    # For k from 0 to N-1-i: term s[i]*s[i+k] changes
    # For k from 0 to i: term s[i-k]*s[i] changes (when j = i-k, j+k = i)
    
    if i < N:
        delta[:N-i] += factor * s[i:N]
    
    if i >= 0:
        delta[:i+1] += factor * s[i::-1]
    
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
    deadline = 130.0  # Safety margin before 150s kill

    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Soft-max lambda: start moderate, can increase for sharper focus
    lam_soft = 5.0
    
    # Use multiple restarts with different initializations
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Try different initializations
        init_type = np.random.choice(5)
        if init_type == 0:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        elif init_type == 1:
            # Sine-based initialization with random frequency
            freq = np.random.uniform(0.5, 5.0)
            t = np.linspace(0, freq * 2 * np.pi, N, endpoint=False)
            s = np.sign(np.sin(t))
            s[s == 0] = 1.0
        elif init_type == 2:
            # Random with some structure
            s = np.random.randint(0, 2, size=N) * 2 - 1
            # Apply a few random flips to break symmetry
            for _ in range(N // 10):
                idx = np.random.randint(N)
                s[idx] = -s[idx]
        elif init_type == 3:
            # Halton-based initialization
            base = np.random.choice([3, 5, 7, 11])
            s = _halton_init(base, N)
        else:
            # Thue-Morse-like sequence
            s = np.ones(N)
            for n in range(1, N):
                # Count trailing zeros in binary representation of n
                m = n
                tz = 0
                while m % 2 == 0:
                    tz += 1
                    m //= 2
                s[n] = 1.0 if tz % 2 == 0 else -1.0
        
        A = _calc_acf(s)
        current_val = _max_abs_acf(A)
        current_soft = _soft_max_acf(A, lam_soft)
        
        # Phase 1: Local search using soft-max surrogate (smoother landscape)
        improved = True
        passes = 0
        max_passes = 300
        patience = 0
        max_patience = 8000
        lam_current = lam_soft
        
        while improved and passes < max_passes and patience < max_patience and time.time() < deadline:
            improved = False
            passes += 1
            patience = 0
            
            # Random order to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                delta = _flip_delta(s, i)
                A_new = A + delta
                soft_new = _soft_max_acf(A_new, lam_current)
                
                if soft_new < current_soft:
                    # Accept the flip
                    s[i] = -s[i]
                    A = A_new
                    current_soft = soft_new
                    current_val = _max_abs_acf(A)
                    improved = True
                    patience = 0
                else:
                    patience += 1
            
            # Gradually increase lambda to sharpen the objective
            if passes % 10 == 0:
                lam_current = min(lam_current * 1.2, 50.0)
        
        # Phase 2: Greedy polish with hard max (exact local optimum)
        if time.time() < deadline:
            improved = True
            hard_passes = 0
            max_hard_passes = 100
            while improved and hard_passes < max_hard_passes and time.time() < deadline:
                improved = False
                hard_passes += 1
                indices = np.random.permutation(N)
                for i in indices:
                    if time.time() > deadline:
                        break
                    delta = _flip_delta(s, i)
                    A_new = A + delta
                    val_new = _max_abs_acf(A_new)
                    if val_new < current_val:
                        s[i] = -s[i]
                        A = A_new
                        current_val = val_new
                        improved = True
        
        # Phase 3: Simulated annealing polish with hard max
        if time.time() < deadline:
            T = 0.05
            T_min = 0.0005
            cooling = 0.99
            sa_deadline = min(deadline, time.time() + 8.0)
            
            while T > T_min and time.time() < sa_deadline:
                i = np.random.randint(N)
                delta = _flip_delta(s, i)
                A_new = A + delta
                val_new = _max_abs_acf(A_new)
                
                if val_new < current_val or np.random.random() < np.exp(-(val_new - current_val) / T):
                    s[i] = -s[i]
                    A = A_new
                    current_val = val_new
                
                T *= cooling
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
        
        # If we've found a good value, use less time per restart
        if best_val < 4.0 and restart_count >= 10:
            break

    return best_seq
