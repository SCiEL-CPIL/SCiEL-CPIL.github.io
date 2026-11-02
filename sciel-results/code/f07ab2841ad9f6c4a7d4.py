import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    """Compute the score for a given set U."""
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    
    # Use FFT for sumset and diffset sizes
    vec = np.zeros(M + 1, dtype=np.float64)
    vec[U] = 1.0
    
    # Sumset U+U
    f = np.fft.rfft(vec, n=2 * (M + 1))
    s = np.fft.irfft(f * f, n=2 * (M + 1))
    sumset = int(np.sum(s >= 0.5))
    
    # Diffset U-U
    g = np.fft.irfft(f * np.conj(f), n=2 * (M + 1))
    diffset = int(np.sum(g >= 0.5))
    
    ratio = diffset / sumset if sumset > 0 else 0.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _primes_up_to(n):
    """Generate primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _gap_multiplication_construct(gaps):
    """Construct U from cumulative sum of gaps."""
    U = [0]
    for g in gaps:
        U.append(U[-1] + g)
    return U

def _greedy_prime_gaps(num_gaps, max_prime):
    """
    Greedy selection of prime gaps to maximize the difference/sum ratio.
    Each gap is a product of distinct primes.
    """
    primes = _primes_up_to(max_prime)
    if not primes:
        return [1] * num_gaps
    
    # Start with smallest primes as gaps
    gaps = [primes[0]]
    used_primes = {primes[0]}
    
    for _ in range(num_gaps - 1):
        best_gap = None
        best_score = -1
        best_prime = None
        
        # Try adding each unused prime as a new gap
        for p in primes:
            if p in used_primes:
                continue
            trial_gaps = gaps + [p]
            U = _gap_multiplication_construct(trial_gaps)
            if max(U) > 5000:  # Keep max bounded for FFT speed
                break
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_gap = p
                best_prime = p
        
        if best_prime is not None:
            gaps.append(best_prime)
            used_primes.add(best_prime)
        else:
            # Try products of two primes
            for i, p1 in enumerate(primes):
                for p2 in primes[i+1:]:
                    if p1 in used_primes or p2 in used_primes:
                        continue
                    prod = p1 * p2
                    trial_gaps = gaps + [prod]
                    U = _gap_multiplication_construct(trial_gaps)
                    if max(U) > 5000:
                        break
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_gap = prod
                        best_prime = prod
                    if s > best_score:
                        break
                if best_prime is not None:
                    break
        
        if best_prime is None:
            # Just use next prime
            for p in primes:
                if p not in used_primes:
                    gaps.append(p)
                    used_primes.add(p)
                    break
            else:
                gaps.append(gaps[-1])
    
    return gaps

def construct():
    best_U = [0]
    best_score = 0.0
    
    # Strategy 1: Gap-multiplication with prime gaps
    for num_gaps in range(3, 15):
        if _elapsed() > 60:
            break
        gaps = _greedy_prime_gaps(num_gaps, 50)
        U = _gap_multiplication_construct(gaps)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Strategy 2: Golomb ruler based sets
    # Known good Golomb rulers
    golomb_rulers = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 28],
        [0, 1, 4, 10, 12, 17, 22, 28, 34],
        [0, 1, 4, 9, 11, 17, 22],
        [0, 1, 4, 9, 11, 17, 22, 28],
    ]
    
    for ruler in golomb_rulers:
        for scale in range(1, 10):
            U = [x * scale for x in ruler]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Strategy 3: Random sparse sets with local search
    rng = np.random.RandomState(42)
    for max_val in [20, 50, 100, 200, 400]:
        if _elapsed() > 80:
            break
        for size in range(3, min(12, max_val // 5 + 1)):
            for _ in range(20):
                candidates = rng.choice(range(1, max_val + 1), size=size, replace=False)
                U = sorted([0] + list(candidates))
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U
    
    # Local search improvement
    U = list(best_U)
    cur_score = best_score
    improved = True
    max_iters = 0
    while improved and _elapsed() < 115 and max_iters < 200:
        improved = False
        max_iters += 1
        max_val = max(U) if U else 1
        
        # Try adding elements
        for _ in range(30):
            if _elapsed() > 118:
                break
            c = rng.randint(1, max_val * 2 + 1)
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
        
        # Try removing elements
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > 118:
                    break
                trial = U[:idx] + U[idx+1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
        
        # Try swapping elements
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > 118:
                    break
                for _ in range(10):
                    c = rng.randint(1, max_val * 2 + 1)
                    if c not in U:
                        trial = U[:idx] + [c] + U[idx+1:]
                        trial = sorted(trial)
                        if len(set(trial)) == len(trial):
                            s = _score_set(trial)
                            if s > cur_score + 1e-12:
                                cur_score = s
                                U = trial
                                best_U = U
                                best_score = cur_score
                                improved = True
                                break
    
    return best_U
