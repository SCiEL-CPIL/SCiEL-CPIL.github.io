import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) using Euler's criterion."""
    if n % p == 0:
        return 0
    return (n ** ((p - 1) // 2)) % p
    # Returns 1 if n is quadratic residue mod p, -1 otherwise (after mapping)

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

def _local_search_binary_fast(f, max_iter=2000, time_budget=10.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    Includes multi-flip moves to escape local minima.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    rng = np.random.default_rng(hash(f.tobytes()) % (2**32))
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_flip_idx = -1
        best_new_C = None
        best_new_S = S
        
        # Single flip moves
        for i in range(N):
            v = f[i]
            delta = -2.0 * v
            
            C_new = C.copy()
            
            k_start = i
            k_end = i + N
            end_idx = min(k_end, len(C_new))
            start_idx = k_start
            
            if start_idx < len(C_new):
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
        
        # Multi-flip moves (try pairs of flips)
        if it % 5 == 0 and no_improve_count > 3:
            # Try random pairs of flips
            for _ in range(20):
                i1 = rng.integers(0, N)
                i2 = rng.integers(0, N)
                if i1 == i2:
                    continue
                    
                C_new = C.copy()
                new_S = S
                
                for idx in [i1, i2]:
                    v = f[idx]
                    delta = -2.0 * v
                    k_start = idx
                    k_end = idx + N
                    end_idx = min(k_end, len(C_new))
                    start_idx = k_start
                    
                    if start_idx < len(C_new):
                        f_segment = f[:end_idx - start_idx]
                        C_new[start_idx:end_idx] += 2.0 * delta * f_segment
                        
                    if 2 * idx < len(C_new):
                        C_new[2 * idx] += delta * delta
                    
                    new_S -= 2.0 * v
                
                if abs(new_S) < 1e-6:
                    continue
                    
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_flip_idx = i1  # We'll handle multi-flip separately
                    best_new_C = C_new
                    best_new_S = new_S
                    # Mark as multi-flip
                    best_flip_idx = -1  # Use -1 to indicate multi-flip
                    # Store both indices
                    best_flip_idx2 = i2
                    best_flip_idx = i1
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S = best_new_S
            C = best_new_C
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 15:
                break
                
    return f, score

def _quadratic_phase_seq(M, shift=0):
    """
    Generate a signed sequence based on quadratic phase: sign(exp(i*pi*n^2/M)).
    This is related to Gauss sums and has good autocorrelation properties.
    """
    n = np.arange(N) + shift
    phase = np.pi * n * n / M
    return np.sign(np.cos(phase)).astype(np.float64)

def _linear_phase_seq(M, shift=0):
    """
    Generate a signed sequence based on linear phase: sign(exp(i*pi*n/M)).
    """
    n = np.arange(N) + shift
    phase = np.pi * n / M
    return np.sign(np.cos(phase)).astype(np.float64)

def _walsh_hadamard_seq(k):
    """
    Generate a Walsh-Hadamard sequence of order k.
    These have perfect autocorrelation (only peak at zero).
    """
    # Walsh-Hadamard matrix rows have good autocorrelation
    # For length 2^m, use the m-th row
    m = int(np.log2(k)) if k > 0 else 0
    if m < 1:
        return np.ones(N)
    
    # Generate the k-th row of the Walsh-Hadamard matrix
    # Using the recursive definition
    def _wh_row(row_idx, length):
        if length == 1:
            return np.array([1.0])
        half = length // 2
        left = _wh_row(row_idx % half, half)
        right = _wh_row(row_idx // 2, half) * (-1.0 if (row_idx % half) >= half else 1.0)
        # Actually, let's use the standard Walsh matrix construction
        # W(0) = [1], W(n+1) = [[W(n), W(n)], [W(n), -W(n)]]
        # Row k of W(m) for length 2^m
        # This is equivalent to: sign of product of (-1)^(bit_i * bit_j)
        pass
    
    # Simpler: use the definition directly
    # f(n) = (-1)^(popcount(n & k))
    # This is a Walsh function
    n = np.arange(N)
    # Compute popcount of (n & k)
    def _popcount(arr):
        arr = arr.astype(np.int32)
        count = np.zeros_like(arr, dtype=np.int32)
        while np.any(arr > 0):
            count += arr & 1
            arr >>= 1
        return count
    
    pc = _popcount(n & k)
    return np.where(pc % 2 == 0, 1.0, -1.0)

def _generate_legendre_sequence(p):
    """
    Generate a signed sequence based on Legendre symbol modulo prime p.
    f(n) = (n/p) for n=0..N-1, where (n/p) is the Legendre symbol.
    These have provably low off-peak autocorrelation (Weil bound).
    """
    seq = np.zeros(N)
    for i in range(N):
        n = i % p
        if n == 0:
            seq[i] = 0.0  # Or map to 1 or -1; we'll handle in scoring
        else:
            leg = (n ** ((p - 1) // 2)) % p
            if leg == 1:
                seq[i] = 1.0
            else:  # leg == p-1 which is -1 mod p
                seq[i] = -1.0
    # Map zeros to 1 to ensure binary {-1, 1}
    seq[seq == 0] = 1.0
    return seq

def _m_sequence_seq(p, c):
    """
    Generate a maximal-length sequence (m-sequence) from a linear feedback shift register.
    These have excellent autocorrelation properties (three-valued autocorrelation).
    """
    # For N=400, we need a suitable LFSR
    # Use a primitive polynomial of degree m where 2^m - 1 >= N
    # m=9 gives 511, which is close to 400
    # Primitive polynomial x^9 + x^4 + 1 (in GF(2))
    # This is one of the known primitive polynomials
    
    # Let's use a simpler approach: generate from a known m-sequence
    # and truncate to N
    
    # For degree 9, primitive polynomial x^9 + x^4 + 1
    # Tap positions: 9 and 4 (0-indexed from LSB)
    # LFSR: new_bit = bit[9-1] XOR bit[4-1] = bit[8] XOR bit[3]
    
    m = 9
    # Initial state: non-zero
    state = [1] * m  # all 1s
    
    seq = []
    for _ in range(2**m - 1):
        seq.append(1.0 if state[-1] == 1 else -1.0)
        # Compute new bit
        new_bit = state[m-1] ^ state[m-1-4]  # bit[8] XOR bit[3] for m=9
        # Shift right
        state = [new_bit] + state[:-1]
    
    # Truncate to N
    if len(seq) >= N:
        return np.array(seq[:N])
    else:
        # Pad with repeats
        repeats = int(np.ceil(N / len(seq)))
        return np.array(seq * repeats)[:N]

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # 1. Generate number-theoretic constructions
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N))
    
    # Quadratic phase sequences with various M
    for M in [2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400]:
        for shift in [0, 1, 2]:
            candidates.append(_quadratic_phase_seq(M, shift))
    
    # Linear phase sequences
    for M in [2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400]:
        candidates.append(_linear_phase_seq(M, 0))
    
    # Walsh-Hadamard sequences
    for k in range(1, 32):
        candidates.append(_walsh_hadamard_seq(k))
    
    # M-sequences
    candidates.append(_m_sequence_seq(9, 0))
    
    # Legendre symbol sequences for various primes
    primes = [401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    for p in primes:
        if p >= N:
            candidates.append(_generate_legendre_sequence(p))
    
    # 2. Evaluate all candidates and keep the best few
    scored_candidates = []
    for c in candidates:
        if time.time() > DEADLINE:
            break
        s = np.sum(c)
        if abs(s) < N * 0.1:
            # Flip to increase sum magnitude
            if s < 0:
                c = -c
            s = np.sum(c)
            if abs(s) < N * 0.1:
                # Add bias
                c = np.sign(c + 0.5)
                s = np.sum(c)
                if abs(s) < N * 0.1:
                    continue
        score = _score(c)
        scored_candidates.append((score, c))
    
    # Sort by score and keep top 20
    scored_candidates.sort(key=lambda x: x[0])
    top_candidates = [c for _, c in scored_candidates[:20]]
    
    # 3. Local search from each top candidate
    for c in top_candidates:
        if time.time() > DEADLINE:
            break
        
        time_remaining = DEADLINE - time.time()
        budget = min(15.0, max(2.0, time_remaining / 3.0))
        
        f_local, score_local = _local_search_binary_fast(c, max_iter=3000, time_budget=budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # 4. Additional random restarts with local search
    remaining_time = DEADLINE - time.time()
    if remaining_time > 20:
        n_restarts = min(10, int(remaining_time / 15))
        for _ in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            f_rand = rng.choice([-1.0, 1.0], size=N)
            S_rand = np.sum(f_rand)
            if abs(S_rand) < N * 0.2:
                continue
                
            time_remaining = DEADLINE - time.time()
            budget = min(10.0, max(2.0, time_remaining / 2.0))
            
            f_rand_local, score_rand_local = _local_search_binary_fast(f_rand, max_iter=1500, time_budget=budget)
            
            if score_rand_local < best_score:
                best_score = score_rand_local
                best_f = f_rand_local.copy()
    
    # 5. Final polish pass with tighter tolerance
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        # Try more iterations with tighter convergence
        f_polish, score_polish = _local_search_binary_fast(best_f, max_iter=5000, time_budget=min(10.0, remaining_time - 2.0))
        if score_polish < best_score:
            best_score = score_polish
            best_f = f_polish.copy()
    
    return best_f
