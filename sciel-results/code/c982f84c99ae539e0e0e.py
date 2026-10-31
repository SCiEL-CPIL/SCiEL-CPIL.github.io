"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p) for prime p."""
    if a % p == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    if r == 1:
        return 1
    elif r == p - 1:
        return -1
    else:
        return 0

def _is_prime(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    """Find the next prime >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 == 1 else n + 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate

def _legendre_sequence(p, length):
    """Generate a Legendre sequence of given length using prime p."""
    seq = np.zeros(length)
    for i in range(length):
        a = (i + 1) % p
        if a == 0:
            seq[i] = 1.0
        else:
            seq[i] = float(_legendre_symbol(a, p))
    return seq

def _local_search_binary_fast(f_init, time_budget=10.0, deadline=None):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    Supports single and double flips for better escape from local minima.
    """
    f = np.sign(f_init).astype(np.float64)
    f[f == 0] = 1.0
    
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 8
    
    while time.time() - start_time < time_budget:
        if deadline is not None and time.time() > deadline:
            break
            
        best_improvement = 0
        best_move = None  # (indices, new_C, new_S)
        
        # Try single flips
        for i in range(N):
            v = f[i]
            C_new = C.copy()
            k_start = i
            k_end = i + N
            
            if k_start < len(C_new):
                end_idx = min(k_end, len(C_new))
                f_segment = f[:end_idx - k_start]
                C_new[k_start:end_idx] -= 4.0 * v * f_segment
            if 2 * i < len(C_new):
                C_new[2 * i] += 4.0
                
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_move = ([i], C_new, new_S)
        
        # If no single flip helps, try double flips (sampled for speed)
        if best_move is None and no_improve_count >= 3:
            # Sample pairs to try
            n_pairs = 50
            for _ in range(n_pairs):
                i1 = np.random.randint(0, N)
                i2 = np.random.randint(0, N)
                if i1 == i2:
                    continue
                
                v1 = f[i1]
                v2 = f[i2]
                
                C_new = C.copy()
                
                # Apply flip at i1
                k_start1 = i1
                k_end1 = i1 + N
                if k_start1 < len(C_new):
                    end_idx1 = min(k_end1, len(C_new))
                    C_new[k_start1:end_idx1] -= 4.0 * v1 * f[:end_idx1 - k_start1]
                if 2 * i1 < len(C_new):
                    C_new[2 * i1] += 4.0
                    
                # Apply flip at i2
                k_start2 = i2
                k_end2 = i2 + N
                if k_start2 < len(C_new):
                    end_idx2 = min(k_end2, len(C_new))
                    C_new[k_start2:end_idx2] -= 4.0 * v2 * f[:end_idx2 - k_start2]
                if 2 * i2 < len(C_new):
                    C_new[2 * i2] += 4.0
                    
                new_S = S - 2.0 * v1 - 2.0 * v2
                if abs(new_S) < 1e-6:
                    continue
                
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_move = ([i1, i2], C_new, new_S)
        
        if best_move is not None:
            indices, C_new, new_S = best_move
            for idx in indices:
                f[idx] = -f[idx]
            S = new_S
            C = C_new
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    return f, score

def _spectral_init(freq_scale=1.0, n_harmonics=10, random_seed=None):
    """
    Generate a sequence based on a windowed sinc-like spectral construction.
    This approach aims to minimize sidelobes in the power spectral density,
    which corresponds to minimizing the peak autoconvolution.
    
    We construct h(x) such that its Fourier transform has a smooth, low-sidelobe profile.
    Specifically, we use a sum of cosines with carefully chosen frequencies and phases
    to create a sequence with low autoconvolution peaks.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    x = np.arange(N)
    h = np.zeros(N)
    
    # Use a weighted sum of cosines to create a smooth signal
    # The weights decay to suppress high-frequency components
    for k in range(1, n_harmonics + 1):
        freq = k * freq_scale
        # Random phase to break symmetry
        phase = np.random.uniform(0, 2 * np.pi)
        # Weight decays with frequency
        weight = 1.0 / (k ** 0.5)
        h += weight * np.cos(2 * np.pi * freq * x / N + phase)
    
    # Normalize and binarize
    h = np.sign(h)
    h[h == 0] = 1.0
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Initialize best with the known baseline: all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Find primes near N for Legendre sequences
    primes_near_N = []
    p = _next_prime(N - 50)
    while p < N + 50 and len(primes_near_N) < 10:
        primes_near_N.append(p)
        p = _next_prime(p + 1)
    
    np.random.seed(42)
    
    n_restarts = 300  # Try many, break on deadline
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a diverse initialization
        init_type = r % 9
        if init_type == 0:
            # Legendre sequence using prime near N
            p = primes_near_N[r % len(primes_near_N)]
            h = _legendre_sequence(p, N)
        elif init_type == 1:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif init_type == 2:
            # Low frequency cosine
            k = (r // 9) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif init_type == 3:
            # Chirp-like
            x = np.arange(N)
            freq = 1.0 + (r // 9) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2*N)))
            h[h == 0] = 1.0
        elif init_type == 4:
            # Block structure
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        elif init_type == 5:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif init_type == 6:
            # Legendre with random shift
            p = primes_near_N[r % len(primes_near_N)]
            full_seq = _legendre_sequence(p, 2*N)
            shift = np.random.randint(0, N)
            h = full_seq[shift:shift+N]
        elif init_type == 7:
            # Spectral initialization: windowed sinc-like
            freq_scale = 0.5 + (r // 9) * 0.1
            n_harm = 5 + (r // 9) % 10
            h = _spectral_init(freq_scale, n_harm, random_seed=r)
        else:
            # Spectral initialization with different parameters
            freq_scale = 1.0 + (r // 9) * 0.05
            n_harm = 3 + (r // 9) % 8
            h = _spectral_init(freq_scale, n_harm, random_seed=r + 1000)
        
        # Ensure the sum is not zero
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization with faster direct convolution
        h_opt, val_opt = _local_search_binary_fast(h, time_budget=5.0, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END
