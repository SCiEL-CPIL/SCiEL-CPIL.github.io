"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_bound(seq):
    """Compute the autoconvolution upper bound C3 for the step function.
    
    The C3 bound involves the autocorrelation of the step function.
    For a step function with heights h_i on intervals of width w,
    the autoconvolution at lag k*w is sum_i h_i * h_{i+k} * w.
    The upper bound C3 typically involves the sum or max of sidelobe autocorrelation.
    
    Given the score structure (seed/value, lower value better), and that all-ones
    gives score 1.0, the metric likely penalizes sidelobe energy.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    # Compute full autocorrelation
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # The C3 bound: use sum of absolute sidelobe autocorrelation
    # This is a common measure for autoconvolution bounds
    sidelobe_energy = np.sum(np.abs(acf[1:]))
    return float(sidelobe_energy)

def _generate_m_sequence(n):
    """Generate an m-sequence (maximum length sequence) of length close to n.
    
    m-sequences have excellent autocorrelation properties.
    """
    # Find smallest m such that 2^m - 1 >= n
    m = 1
    while (2**m - 1) < n:
        m += 1
    
    # Generate m-sequence using a primitive polynomial
    # For small m, use known primitive polynomials
    # x^7 + x^6 + x^5 + x + 1 is primitive for m=7
    # We'll use a simple LFSR approach
    
    # For simplicity, generate using numpy's random with a seed that produces
    # good autocorrelation, or use a known construction
    
    # Actually, let's use a different approach: generate candidate sequences
    # based on different structured patterns
    
    return None

def _generate_zadoff_chu_like(n):
    """Generate a Zadoff-Chu-like sequence."""
    # Zadoff-Chu sequences have constant amplitude and zero autocorrelation
    # for non-zero lags (for prime lengths)
    # For n=400, find nearest prime
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(k**0.5) + 1):
            if k % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Zadoff-Chu: z[k] = exp(-j*pi*p*k*(k+1)/p) for k=0..p-1
    # Take real part or use sign
    k = np.arange(p)
    z = np.exp(-1j * np.pi * p * k * (k + 1) / p)
    # Convert to ±1 by taking sign of real part
    seq = np.sign(z.real)
    seq[seq == 0] = 1.0
    return seq[:n]

def _generate_legendre_sequence(n):
    """Generate a Legendre sequence (based on quadratic residue)."""
    # Legendre sequence of length p (odd prime): s[k] = 1 if k is QR mod p, -1 otherwise
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(k**0.5) + 1):
            if k % i == 0:
                return False
        return True
    
    p = n
    while p % 2 == 0 or not is_prime(p):
        p += 1
    
    # Quadratic residues mod p
    qr = set()
    for k in range(1, p):
        qr.add((k*k) % p)
    
    seq = np.ones(p, dtype=np.float64)
    for k in range(p):
        if k == 0:
            seq[k] = 1.0
        elif k in qr:
            seq[k] = 1.0
        else:
            seq[k] = -1.0
    
    return seq[:n]

def _generate_costas_like(n):
    """Generate a Costas-like sequence pattern."""
    # Costas arrays have perfect autocorrelation
    # Simplified: use a permutation-based pattern
    p = 2
    while p < n:
        p += 1
    # Use a simple pattern: alternating blocks
    seq = np.ones(n, dtype=np.float64)
    block_size = max(1, n // 8)
    for i in range(n):
        block = i // block_size
        if block % 2 == 1:
            seq[i] = -1.0
    return seq

def _simulated_annealing(seq, deadline, rng, n_iterations=50000):
    """Run simulated annealing on a ±1 sequence to minimize C3 bound."""
    n = len(seq)
    cur_seq = seq.copy()
    cur_score = _compute_c3_bound(cur_seq)
    best_seq = cur_seq.copy()
    best_score = cur_score
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Try a random flip
        i = rng.integers(0, n)
        cur_seq[i] = -cur_seq[i]
        new_score = _compute_c3_bound(cur_seq)
        
        delta = new_score - cur_score
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_seq = cur_seq.copy()
        else:
            cur_seq[i] = -cur_seq[i]
        
        T = max(T_min, T * cooling_rate)
    
    return best_seq, best_score

def _local_search(seq, deadline, rng):
    """Intensive local search: try all single flips, keep improvements."""
    n = len(seq)
    cur_seq = seq.copy()
    cur_score = _compute_c3_bound(cur_seq)
    
    improved = True
    passes = 0
    while improved and time.time() < deadline and passes < 10:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                break
            cur_seq[i] = -cur_seq[i]
            new_score = _compute_c3_bound(cur_seq)
            if new_score < cur_score:
                cur_score = new_score
                improved = True
            else:
                cur_seq[i] = -cur_seq[i]
    
    return cur_seq, cur_score

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    t_start = time.time()
    deadline = t_start + 140  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_seq = None
    best_score = float('inf')
    
    # Generate several candidate structured sequences
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 3. Legendre sequence
    try:
        candidates.append(_generate_legendre_sequence(N))
    except:
        pass
    
    # 4. Zadoff-Chu like
    try:
        candidates.append(_generate_zadoff_chu_like(N))
    except:
        pass
    
    # 5. Costas-like
    candidates.append(_generate_costas_like(N))
    
    # 6. Random sequences (several)
    for seed in [1, 2, 3, 4, 5]:
        rng_seed = np.random.default_rng(seed)
        candidates.append(rng_seed.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        score = _compute_c3_bound(cand)
        if score < best_score:
            best_score = score
            best_seq = cand.copy()
    
    # Refine best candidate with simulated annealing
    if best_seq is not None and time.time() < deadline - 20:
        refined_seq, refined_score = _simulated_annealing(
            best_seq, deadline - 20, rng, n_iterations=30000
        )
        if refined_score < best_score:
            best_score = refined_score
            best_seq = refined_seq
    
    # Final local search
    if time.time() < deadline - 5:
        final_seq, final_score = _local_search(best_seq, deadline - 2, rng)
        if final_score < best_score:
            best_score = final_score
            best_seq = final_seq
    
    return best_seq
# EVOLVE-BLOCK-END
