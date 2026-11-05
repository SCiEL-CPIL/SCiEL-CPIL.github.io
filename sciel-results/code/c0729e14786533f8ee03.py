"""Rewritten program: minimize autoconvolution upper bound via structured search."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(h):
    """
    Upper bound on C3: max absolute autocorrelation at non-zero lags.
    We use circular autocorrelation via FFT for efficiency.
    """
    n = len(h)
    if n <= 1:
        return 0.0
    # Next power of 2 for FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # For circular autocorrelation of length n, lags 1..n-1
    # acf[k] for k in 1..n-1
    return float(np.max(np.abs(acf[1:n])))

def _thue_morse(n):
    """Thue-Morse sequence mapped to +/-1."""
    t = np.zeros(n)
    for i in range(n):
        # Parity of number of 1s in binary representation
        x = i
        parity = 0
        while x:
            parity ^= x & 1
            x >>= 1
        t[i] = 1.0 if parity == 0 else -1.0
    return t

def _zadoff_chu(n, root=1):
    """Zadoff-Chu sequence (complex, take real part mapped to sign)."""
    # For real-valued step functions, we use a different approach
    # Instead, generate a chirp-like sequence
    k = np.arange(n)
    phase = 2 * np.pi * root * k * (k + 1) / (2 * n)
    return np.sign(np.cos(phase))

def _local_search(h, time_limit=5.0, max_passes=50):
    """
    Local search: try flipping individual bits to reduce score.
    Use random ordering to avoid getting stuck.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    for _ in range(max_passes):
        if time.time() - start > time_limit:
            break
        improved = False
        # Random permutation of indices
        order = np.random.permutation(n)
        for i in order:
            if time.time() - start > time_limit:
                break
            trial = best_h.copy()
            trial[i] = -trial[i]  # Flip sign
            s = _score(trial)
            if s < best_score - 1e-10:
                best_h = trial
                best_score = s
                improved = True
                break
        if not improved:
            # Try swapping pairs
            improved2 = False
            for _ in range(100):
                if time.time() - start > time_limit:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                trial = best_h.copy()
                trial[i], trial[j] = trial[j], trial[i]
                s = _score(trial)
                if s < best_score - 1e-10:
                    best_h = trial
                    best_score = s
                    improved2 = True
                    break
            if not improved2:
                break
    
    return best_h, best_score

def _simulated_annealing(h, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = 10.0
    cooling = 0.99
    
    while T > 0.1 and time.time() - start < time_limit:
        i = np.random.randint(n)
        trial = current_h.copy()
        trial[i] = -trial[i]
        
        new_score = _score(trial)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling
    
    return best_h, best_score

def _logistic_chaos_sequence(n, r, x0=0.5):
    """
    Generate a sequence from the logistic map in chaotic regime.
    x_{k+1} = r * x_k * (1 - x_k)
    Map to +/-1 based on sign of (x_k - 0.5)
    """
    x = x0
    seq = np.zeros(n)
    for i in range(n):
        x = r * x * (1 - x)
        seq[i] = 1.0 if x > 0.5 else -1.0
    return seq

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Try multiple initializations: Thue-Morse, Zadoff-Chu, logistic chaos, random.
    2. Refine with local search and simulated annealing.
    3. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Thue-Morse initialization
    h_tm = _thue_morse(N)
    h_opt, score = _local_search(h_tm, time_limit=8.0)
    if time.time() < deadline - 10:
        h_opt, score = _simulated_annealing(h_opt, time_limit=10.0)
    if score < best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Phase 2: Zadoff-Chu / chirp initialization
    if time.time() < deadline - 20:
        h_zc = _zadoff_chu(N, root=1)
        h_opt, score = _local_search(h_zc, time_limit=5.0)
        if time.time() < deadline - 10:
            h_opt, score = _simulated_annealing(h_opt, time_limit=8.0)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Phase 3: Logistic chaos with different r values
    if time.time() < deadline - 30:
        for r in [3.9, 3.99, 3.999]:
            if time.time() > deadline - 15:
                break
            h_lc = _logistic_chaos_sequence(N, r, x0=0.5)
            h_opt, score = _local_search(h_lc, time_limit=5.0)
            if time.time() < deadline - 10:
                h_opt, score = _simulated_annealing(h_opt, time_limit=8.0)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Phase 4: Random initializations
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        h_rand = np.random.choice([-1, 1], size=N)
        h_opt, score = _local_search(h_rand, time_limit=5.0)
        if time.time() < deadline - 10:
            h_opt, score = _simulated_annealing(h_opt, time_limit=8.0)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Phase 5: Final polish on best solution
    if best_h is not None and time.time() < deadline - 5:
        best_h, best_score = _local_search(best_h, time_limit=10.0)
        if time.time() < deadline - 5:
            best_h, best_score = _simulated_annealing(best_h, time_limit=10.0)
    
    # Fallback
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
