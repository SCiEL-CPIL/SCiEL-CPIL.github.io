"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """Compute max absolute value of aperiodic autocorrelation (excluding lag 0)."""
    n = len(f)
    # Use next fast FFT size for efficiency
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.fft(f, n=size)
    # Aperiodic autocorrelation via circular convolution
    # R[k] = sum_{i=0}^{n-1-k} f[i] f[i+k]
    # This is equivalent to the first n points of the circular autocorrelation
    # of f padded with zeros to size 2n.
    conv = np.fft.ifft(F * F).real[:n]
    # conv[0] is the self-convolution at lag 0 (sum of squares = n)
    # We want max |conv[k]| for k=1..n-1
    if n > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0


def _legendre_sequence(n):
    """Generate a Legendre sequence of length n."""
    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        for i in range(3, int(np.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    # Find a prime p such that p >= n and p == 3 mod 4
    # Legendre symbol is defined for primes p.
    # We need p such that p-1 >= n.
    p = n + 1
    while True:
        if is_prime(p) and p % 4 == 3:
            break
        p += 1
    
    # Legendre symbol (a/p)
    # For a=1..p-1, (a/p) is 1 if a is a quadratic residue mod p, -1 otherwise.
    # We only need the first n values.
    # Note: The standard Legendre sequence is defined on the field GF(p).
    # The sequence is x_i = (i/p) for i=1..p-1.
    # We take the first n terms.
    
    seq = np.zeros(n)
    for i in range(1, n + 1):
        # Compute Legendre symbol (i/p)
        # Using Euler's criterion: (i/p) = i^((p-1)/2) mod p
        # If result is 1, it's a residue (1), if p-1, it's non-residue (-1)
        res = pow(i, (p - 1) // 2, p)
        if res == 1:
            seq[i - 1] = 1.0
        else:
            seq[i - 1] = -1.0
            
    return seq


def _m_sequence(n):
    """Generate an m-sequence (maximum length LFSR sequence) of length n."""
    # For N=400, we can't have an exact m-sequence length (which is 2^k-1).
    # We can generate a long m-sequence and truncate.
    # A 9-bit LFSR gives length 511.
    # Primitive polynomial for degree 9: x^9 + x^4 + 1 (octal 035) or similar.
    # Let's use a known good primitive polynomial.
    # For simplicity, we'll use a 9-bit LFSR with polynomial 0x35 (x^9 + x^4 + 1).
    
    lfsr = 1
    poly = 0x35  # x^9 + x^4 + 1
    seq = []
    for _ in range(n):
        # Output bit
        seq.append(1.0 if (lfsr & 1) else -1.0)
        # Shift
        lfsr >>= 1
        # Feedback
        if lfsr & 0x100:  # If bit 8 (9th bit) is set
            lfsr ^= poly
    
    return np.array(seq[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    # Initialize with multiple candidate starting points
    candidates = []
    
    # 1. Random initialization
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. Legendre sequence
    candidates.append(_legendre_sequence(N))
    
    # 3. M-sequence
    candidates.append(_m_sequence(N))
    
    # 4. Randomized Legendre (flip some bits)
    leg = _legendre_sequence(N)
    mask = rng.rand(N) < 0.1
    leg_rand = leg.copy()
    leg_rand[mask] *= -1.0
    candidates.append(leg_rand)
    
    # Evaluate all candidates
    best_f = None
    best_score = float('inf')
    for c in candidates:
        score = _autoconv_max_abs(c)
        if score < best_score:
            best_score = score
            best_f = c.copy()
    
    current_f = best_f.copy()
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    # Precompute FFT size for speed
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    def _fast_autoconv_max_abs(f):
        F = np.fft.fft(f, n=fft_size)
        conv = np.fft.ifft(F * F).real[:N]
        return np.max(np.abs(conv[1:]))
    
    # Update best score with faster function
    current_score = _fast_autoconv_max_abs(current_f)
    best_score = current_score
    
    while time.time() < deadline - 10.0:
        # Choose move: single flip or block flip
        if rng.rand() < 0.5:
            # Single flip
            i = rng.randint(0, N)
            new_f = current_f.copy()
            new_f[i] *= -1.0
        else:
            # Block flip
            block_len = rng.randint(1, 10)
            start = rng.randint(0, N - block_len + 1)
            new_f = current_f.copy()
            new_f[start:start + block_len] *= -1.0
            
        new_score = _fast_autoconv_max_abs(new_f)
        
        delta = new_score - current_score
        if delta < 0 or (T > T_min and rng.rand() < np.exp(-delta / T)):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = new_f.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final greedy local search with single flips
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 1.0:
                break
            old_val = best_f[i]
            best_f[i] = -old_val
            score_new = _fast_autoconv_max_abs(best_f)
            if score_new < best_score:
                best_score = score_new
                improved = True
            else:
                best_f[i] = old_val
    
    return best_f
# EVOLVE-BLOCK-END
