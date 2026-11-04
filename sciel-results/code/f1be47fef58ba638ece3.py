"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(f):
    """
    Compute linear autocorrelation R[k] for k=0..N-1.
    R[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    Using FFT for speed.
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
    # t(i) = parity of popcount(i)
    # We can generate this iteratively or using bit operations.
    # For n=400, a simple loop is fine.
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
    We find a prime p >= n.
    f[i] = (i/p) for i=1..n.
    Legendre symbol (a/p) is 1 if a is QR mod p, -1 if non-QR, 0 if a%p==0.
    Since i < p for i in 1..n (if p>n), 0 never occurs.
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
        # Legendre symbol (i/p)
        # Euler's criterion: i^((p-1)/2) mod p
        # Result is 1 or p-1 (which is -1 mod p)
        val = pow(i, (p - 1) // 2, p)
        if val == 1:
            seq[i - 1] = 1.0
        elif val == p - 1:
            seq[i - 1] = -1.0
        else:
            seq[i - 1] = 1.0 # Should not happen for i < p
            
    return seq

def _refine_greedy(f, time_limit=10.0):
    """
    Fast greedy local search.
    We maintain the autocorrelation vector R.
    Flipping f[i] changes R by:
    delta_R[k] = -2 * f[i] * (f[i-k] + f[i+k]) for valid indices.
    Note: R is symmetric in the sense of the sum, but R[k] here is one-sided.
    Actually, R[k] = sum_{j=0}^{N-1-k} f[j]f[j+k].
    If we flip f[i], the terms involving f[i] are:
    1. j=i, term k: f[i]f[i+k] -> -f[i]f[i+k] (change -2 f[i] f[i+k])
    2. j+k=i, term k: f[i-k]f[i] -> -f[i-k]f[i] (change -2 f[i] f[i-k])
    So for a fixed lag k, the change is -2 f[i] (f[i+k] + f[i-k]) where terms are 0 if out of bounds.
    
    We will iterate passes. In each pass, we try to find a flip that reduces the peak.
    To be efficient, we can just do a full scan of i=0..N-1, and for each i,
    check if flipping it reduces the current peak.
    Computing the new peak exactly for each i is O(N) if we update R and recompute max.
    Total O(N^2) per pass. With N=400, N^2 = 160k.
    If we do 1000 passes, that's 160M ops. Might be tight in Python.
    
    Optimization:
    Instead of recomputing max(R) from scratch, we can track the top few peaks.
    Or, we can just do a few passes and rely on the structure.
    
    Let's try a simpler approach first:
    Repeat:
      improved = False
      for i in range(N):
        old_f = f[i]
        f[i] = -f[i]
        R = _compute_autocorr(f)
        new_score = np.max(np.abs(R[1:]))
        if new_score < current_score:
            current_score = new_score
            improved = True
        else:
            f[i] = old_f
      if not improved: break
    This is O(N^2 log N) per pass due to FFT. 400^2 * log(800) ~ 160k * 10 = 1.6M ops per pass.
    100 passes = 160M ops. This should be fast enough in NumPy/Python if we are careful.
    Actually, _compute_autocorr does an FFT of size 800.
    Let's implement this and time it.
    """
    current_f = f.copy()
    current_score = _score(current_f)
    start_time = time.time()
    
    # We will do full passes until no improvement or time limit.
    # To avoid getting stuck in 2-cycles, we can use a random order or just accept the first improvement.
    # Let's try random order to escape local minima slightly.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Random permutation of indices to try
        order = np.random.permutation(N)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            old_val = current_f[i]
            current_f[i] = -old_val
            
            new_score = _score(current_f)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current_f[i] = old_val
                
        if not improved:
            break
            
    return current_f, current_score

def _simulated_annealing(f, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    current_f = f.copy()
    current_score = _score(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start_time < time_limit:
        i = np.random.randint(N)
        trial_f = current_f.copy()
        trial_f[i] = -trial_f[i]
        
        new_score = _score(trial_f)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_f = trial_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = trial_f.copy()
        
        T *= cooling
        
    return best_f, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Initialize with Thue-Morse and Legendre sequences.
    2. Use fast greedy local search to refine.
    3. Use simulated annealing to escape local minima.
    4. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_f = None
    best_score = float('inf')
    
    # Phase 1: Thue-Morse initialization
    h_tm = _thue_morse(N)
    h_opt, score = _refine_greedy(h_tm, time_limit=10.0)
    if time.time() < deadline - 10:
        h_opt, score = _simulated_annealing(h_opt, time_limit=10.0)
    
    if score < best_score:
        best_score = score
        best_f = h_opt.copy()
        
    # Phase 2: Legendre initialization
    h_leg = _legendre_sequence(N)
    h_opt, score = _refine_greedy(h_leg, time_limit=10.0)
    if time.time() < deadline - 10:
        h_opt, score = _simulated_annealing(h_opt, time_limit=10.0)
    
    if score < best_score:
        best_score = score
        best_f = h_opt.copy()
        
    # Phase 3: Random initializations
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
        h_opt, score = _refine_greedy(h_rand, time_limit=5.0)
        if time.time() < deadline - 5:
            h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_f = h_opt.copy()
            
    # Final polish if time permits
    if best_f is not None and time.time() < deadline - 5:
        best_f, best_score = _refine_greedy(best_f, time_limit=10.0)
        if time.time() < deadline - 5:
            best_f, best_score = _simulated_annealing(best_f, time_limit=10.0)
            
    # Fallback
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
# EVOLVE-BLOCK-END
