import numpy as np
import time

N = 400

def _compute_autocorr(f):
    """
    Compute linear autocorrelation R[k] for k=0..N-1.
    R[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    """
    n = len(f)
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.rfft(f, size)
    R = np.fft.irfft(F * np.conj(F), size)
    return R[:n]

def _score(f):
    """
    Upper bound on C3: max absolute autocorrelation at non-zero lags.
    """
    R = _compute_autocorr(f)
    if len(R) <= 1:
        return 0.0
    return float(np.max(np.abs(R[1:])))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to +/-1."""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count set bits
        count = 0
        x = i
        while x:
            count += x & 1
            x >>= 1
        seq[i] = 1.0 if count % 2 == 0 else -1.0
    return seq

def _legendre_sequence(n):
    """
    Generate a Legendre symbol sequence of length n.
    """
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    p = n
    while not is_prime(p):
        p += 1
        
    seq = np.zeros(n, dtype=np.float64)
    for i in range(1, n + 1):
        val = pow(i, (p - 1) // 2, p)
        if val == 1:
            seq[i - 1] = 1.0
        elif val == p - 1:
            seq[i - 1] = -1.0
        else:
            seq[i - 1] = 1.0
    return seq

def _fast_local_search(f, time_limit=5.0):
    """
    Fast local search using incremental autocorrelation updates.
    """
    f = f.copy()
    R = _compute_autocorr(f)
    current_score = float(np.max(np.abs(R[1:])))
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Try random order to avoid bias
        order = np.random.permutation(N)
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            old_val = f[i]
            # Calculate delta R for flipping i
            # R[k] = sum f[j] f[j+k]
            # Flipping f[i] affects terms where i is involved.
            # Term 1: j=i, k such that i+k < N. f[i]f[i+k] -> -f[i]f[i+k]. Change: -2 f[i] f[i+k]
            # Term 2: j+k=i, k such that i-k >= 0. f[i-k]f[i] -> -f[i-k]f[i]. Change: -2 f[i] f[i-k]
            
            # We need to update R[k] for all k.
            # Note: R is defined for k=0..N-1.
            # R[k] = sum_{j=0}^{N-1-k} f[j]f[j+k]
            
            # Let's compute the change vector dR
            dR = np.zeros(N)
            
            # Contributions from j=i (f[i] is first factor)
            # k ranges from 0 to N-1-i
            if i < N:
                # f[i] * f[i+k] for k=0..N-1-i
                # Change is -2 * f[i] * f[i+k]
                # Indices in R: k
                # Values: f[i+k]
                end_k = N - i
                if end_k > 0:
                    dR[:end_k] -= 2.0 * old_val * f[i:i+end_k]
            
            # Contributions from j+k=i (f[i] is second factor)
            # k ranges from 0 to i
            # f[i-k] * f[i] for k=0..i
            # Change is -2 * f[i] * f[i-k]
            # Indices in R: k
            # Values: f[i-k]
            if i >= 0:
                # k=0..i
                # f[i-k] corresponds to f[i], f[i-1], ..., f[0]
                # So dR[0] gets -2 f[i] f[i]
                # dR[1] gets -2 f[i] f[i-1]
                # ...
                # dR[i] gets -2 f[i] f[0]
                dR[:i+1] -= 2.0 * old_val * f[i::-1]
            
            # Update R
            new_R = R + dR
            new_score = float(np.max(np.abs(new_R[1:])))
            
            if new_score < current_score - 1e-12:
                f[i] = -old_val
                R = new_R
                current_score = new_score
                improved = True
            # else: keep old f, R unchanged
            
    return f, current_score

def _simulated_annealing_fast(f, time_limit=10.0):
    """
    Simulated annealing with fast autocorrelation updates.
    """
    f = f.copy()
    R = _compute_autocorr(f)
    current_score = float(np.max(np.abs(R[1:])))
    best_f = f.copy()
    best_score = current_score
    start_time = time.time()
    
    T = 2.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time < time_limit):
        i = np.random.randint(N)
        old_val = f[i]
        
        # Calculate dR
        dR = np.zeros(N)
        end_k = N - i
        if end_k > 0:
            dR[:end_k] -= 2.0 * old_val * f[i:i+end_k]
        if i >= 0:
            dR[:i+1] -= 2.0 * old_val * f[i::-1]
            
        new_R = R + dR
        new_score = float(np.max(np.abs(new_R[1:])))
        
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            f[i] = -old_val
            R = new_R
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
                
        T *= cooling
        
    return best_f, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    """
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    best_f = None
    best_score = float('inf')
    
    # Initializations
    initializations = []
    initializations.append(_thue_morse(N))
    initializations.append(_legendre_sequence(N))
    initializations.append(np.random.choice([-1, 1], size=N).astype(np.float64))
    initializations.append(np.random.choice([-1, 1], size=N).astype(np.float64))
    
    # Add a periodic-ish initialization
    period = 16
    base = np.random.choice([-1, 1], size=period).astype(np.float64)
    initializations.append(np.tile(base, N // period + 1)[:N])
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
            
        # Phase 1: Local Search
        f_opt, score = _fast_local_search(init, time_limit=5.0)
        
        # Phase 2: Simulated Annealing
        if time.time() < deadline - 5:
            f_opt, score = _simulated_annealing_fast(f_opt, time_limit=10.0)
            
        # Phase 3: Final Local Search Polish
        if time.time() < deadline - 2:
            f_opt, score = _fast_local_search(f_opt, time_limit=2.0)
            
        if score < best_score:
            best_score = score
            best_f = f_opt.copy()
            
    # If we still have time, do more random restarts
    while time.time() < deadline - 15:
        init = np.random.choice([-1, 1], size=N).astype(np.float64)
        f_opt, score = _fast_local_search(init, time_limit=3.0)
        if time.time() < deadline - 5:
            f_opt, score = _simulated_annealing_fast(f_opt, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_f = f_opt.copy()
            
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
