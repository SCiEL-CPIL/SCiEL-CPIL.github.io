import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _logsumexp_abs(x, beta=100.0):
    """
    Smooth approximation to max(|x|) using log-sum-exp.
    As beta -> infinity, this approaches max(|x|).
    """
    ax = np.abs(x)
    m = np.max(ax)
    if m < 1e-12:
        return 0.0
    shifted = ax - m
    # Use log-sum-exp for stability
    log_sum = m + np.log(np.sum(np.exp(beta * shifted)) / len(x)) / beta
    return log_sum

def _smooth_score(f, beta=100.0):
    """
    Smooth surrogate for the score using log-sum-exp approximation.
    """
    S = np.sum(f)
    if abs(S) < 1e-6:
        return 1e10
    C = _autoconv_fft(f)
    # Use smooth max of |C|
    smooth_max = _logsumexp_abs(C, beta)
    return smooth_max / (S * S)

def _smooth_objective(x, beta=100.0):
    """
    Objective for continuous optimization.
    x is the continuous variable; we project to sign(x) for evaluation.
    """
    f = np.sign(x)
    # Avoid zero signs
    f[f == 0] = 1.0
    return _smooth_score(f, beta)

def _local_search_binary_fast(f, max_iter=1000, time_budget=5.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_flip_idx = -1
        best_new_C = None
        best_new_S = S
        
        for i in range(N):
            v = f[i]
            # Flip f[i]
            delta = -2.0 * v
            
            C_new = C.copy()
            
            k_start = i
            k_end = i + N
            
            if k_start < len(C_new):
                end_idx = min(k_end, len(C_new))
                start_idx = k_start
                f_segment = f[:end_idx - start_idx]
                C_new[start_idx:end_idx] += 2.0 * delta * f_segment
            
            if 2 * i < len(C_new):
                C_new[2 * i] += delta * delta
                
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
                best_new_C = C_new
                best_new_S = new_S
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S = best_new_S
            C = best_new_C
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                break
                
    return f, score

def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p)

def _generate_legendre_sequence(p):
    """
    Generate a sequence based on Legendre symbols modulo p.
    """
    seq = np.zeros(N)
    for i in range(N):
        val = _legendre_symbol(i, p)
        if val == 0:
            seq[i] = 1.0
        elif val == 1:
            seq[i] = 1.0
        else:  # -1
            seq[i] = -1.0
    return seq

def _quadratic_phase_seq(N, k):
    """
    Generate a sequence based on quadratic phase e^(i*pi*k*i^2/N).
    Map to signs.
    """
    i = np.arange(N)
    phase = np.pi * k * i * i / N
    return np.sign(np.cos(phase)).astype(np.float64)

def _barker_sequence(n):
    """
    Generate a Barker-like sequence of length n.
    Barker sequences are binary sequences with minimal peak sidelobe levels.
    Known Barker lengths: 2, 3, 4, 5, 7, 11, 13.
    For longer sequences, we use a recursive construction or map to nearest.
    """
    # Known Barker sequences
    barker_dict = {
        2: [1, 1],
        3: [1, 1, -1],
        4: [1, 1, -1, 1],
        5: [1, 1, -1, -1, 1],
        7: [1, 1, 1, -1, 1, -1, 1],
        11: [1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1],
        13: [1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1]
    }
    
    if n in barker_dict:
        return np.array(barker_dict[n], dtype=np.float64)
    
    # For lengths not in Barker dict, use a pseudo-Barker construction
    # Use Legendre symbol for prime lengths
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(np.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True
    
    # Find nearest prime >= n
    p = n
    while not is_prime(p):
        p += 1
    
    # Generate Legendre sequence of length p, take first n
    seq = np.zeros(n)
    for i in range(n):
        val = _legendre_symbol(i, p)
        if val == 0:
            seq[i] = 1.0
        elif val == 1:
            seq[i] = 1.0
        else:
            seq[i] = -1.0
    return seq

def _golay_complementary_pair(n):
    """
    Generate a Golay complementary pair of length n.
    Golay sequences have the property that the sum of their autoconvolutions
    is a delta function (except at zero lag).
    We use one of the sequences as our candidate.
    """
    # Known Golay pairs for small lengths
    golay_dict = {
        2: [[1, 1], [1, -1]],
        4: [[1, 1, 1, -1], [1, 1, -1, -1]],
        8: [[1, 1, 1, -1, 1, 1, -1, 1], [1, 1, 1, -1, 1, -1, 1, 1]],
        16: [[1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1],
              [1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1]]
    }
    
    if n in golay_dict:
        return np.array(golay_dict[n][0], dtype=np.float64)
    
    # For longer lengths, use recursive construction or fall back to Barker-like
    # Use a recursive Golay construction
    if n == 1:
        return np.array([1.0])
    
    # Recursive: if (a,b) is a Golay pair of length n/2, then
    # (a, b) and (a, -b) are Golay pairs of length n
    half_n = n // 2
    if half_n * 2 != n:
        # Not a power of 2, use Barker-like
        return _barker_sequence(n)
    
    a_half, b_half = _golay_complementary_pair(half_n)
    a = np.concatenate([a_half, b_half])
    return a

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # 1. Structured Initializations
    candidates = []
    
    # a) All ones
    candidates.append(np.ones(N))
    
    # b) Legendre sequences for primes near N
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = []
    for p in range(max(3, N - 20), N + 20):
        if is_prime(p):
            primes.append(p)
            
    for p in primes:
        candidates.append(_generate_legendre_sequence(p))
        
    # c) Quadratic phase sequences
    for k in [1, 2, 3, 5, 7, 11]:
        candidates.append(_quadratic_phase_seq(N, k))
        
    # d) Random binary sequences
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # e) Alternating patterns
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    candidates.append(np.array([1.0 if i % 3 < 2 else -1.0 for i in range(N)]))
    
    # f) Barker-like sequences (key addition)
    candidates.append(_barker_sequence(N))
    
    # g) Golay complementary sequence (key addition)
    candidates.append(_golay_complementary_pair(N))
    
    # h) Concatenated Barker sequences
    # Split N into chunks and use Barker patterns
    chunk_sizes = [7, 11, 13, 7, 11, 13, 7, 11, 13, 7, 11, 13, 7, 11, 13, 7, 11, 13, 7, 11]
    # 19 chunks of varying sizes, total should be ~400
    # Let's just use a single Barker-like for now
    
    # 2. Local Search from each candidate
    for f0 in candidates:
        if time.time() > DEADLINE:
            break
            
        f_bin = np.sign(f0)
        if np.sum(f_bin) == 0:
            f_bin = np.ones(N)
            
        f_local, score_local = _local_search_binary_fast(f_bin, max_iter=2000, time_budget=4.0)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()

    # 3. Continuous optimization using smooth surrogate
    time_elapsed = time.time() - start_time
    time_remaining = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
    
    if time_remaining > 20:
        # Try continuous optimization from best solution
        x0 = best_f.copy()
        
        # Use L-BFGS-B with smooth objective
        def objective(x):
            if time.time() > DEADLINE:
                return best_score
            return _smooth_objective(x, beta=100.0)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 500}
            )
            
            if time.time() < DEADLINE:
                f_cont = np.sign(result.x)
                f_cont[f_cont == 0] = 1.0
                score_cont = _score(f_cont)
                
                if score_cont < best_score:
                    best_score = score_cont
                    best_f = f_cont.copy()
                    
                    # Polish with local search
                    if time.time() < DEADLINE:
                        f_polished, score_polished = _local_search_binary_fast(
                            f_cont, max_iter=5000, time_budget=10.0
                        )
                        if score_polished < best_score:
                            best_score = score_polished
                            best_f = f_polished.copy()
        except Exception:
            pass

    # 4. Final random restarts with local search
    time_elapsed = time.time() - start_time
    time_remaining = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
    
    if time_remaining > 10:
        num_extra = int(time_remaining / 5)
        for _ in range(num_extra):
            if time.time() > DEADLINE:
                break
            f_rand = rng.choice([-1.0, 1.0], size=N)
            if abs(np.sum(f_rand)) < N * 0.2:
                continue
            f_local, score_local = _local_search_binary_fast(f_rand, max_iter=2000, time_budget=5.0)
            if score_local < best_score:
                best_score = score_local
                best_f = f_local.copy()

    return best_f
