"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import comb

def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p."""
    if n % p == 0:
        return 0
    return pow(n, (p - 1) // 2, p) if pow(n, (p - 1) // 2, p) == 1 else -1

def _legendre_sequence(p):
    """Generate Legendre symbol sequence of length p (indices 1..p-1, plus 0)."""
    seq = np.zeros(p, dtype=np.int8)
    for i in range(1, p):
        seq[i] = _legendre_symbol(i, p)
    seq[0] = 1  # Convention
    return seq

def _compute_score(h):
    """Compute the ratio: max |autoconv| / (sum h)^2."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    # Compute autoconvolution
    # A(k) = sum_i h[i] * h[i+k] for k = 0..N-1
    # Use FFT for efficiency
    n = len(h)
    fft_size = 2 * n
    H = np.fft.rfft(h.astype(np.float64), fft_size)
    autoconv = np.fft.irfft(H * np.conj(H), fft_size)[:n]
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find a prime near N for Legendre sequence
    # Legendre sequences require prime length
    # Find nearest prime <= N or use N if N is prime
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Try primes near N
    best_h = None
    best_score = float('inf')
    
    # First, try the seed (all ones)
    h_seed = np.ones(N, dtype=np.int8)
    best_h = h_seed.copy()
    best_score = _compute_score(best_h)
    
    # Try Legendre sequences of prime length near N
    for p in range(N, max(10, N - 20), -1):
        if is_prime(p):
            leg = _legendre_sequence(p)
            # Pad or truncate to length N
            if p == N:
                h = leg.copy()
            elif p < N:
                h = np.zeros(N, dtype=np.int8)
                h[:p] = leg
                h[p:] = 1  # Pad with 1s
            else:
                h = leg[:N]
            score = _compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
            break  # Use the first (largest) prime <= N
    
    # Local search: try flipping bits to improve score
    h = best_h.astype(np.int8).copy()
    current_score = best_score
    
    # Precompute autoconv efficiently
    def _autoconv_max_abs(h):
        n = len(h)
        fft_size = 2 * n
        H = np.fft.rfft(h.astype(np.float64), fft_size)
        autoconv = np.fft.irfft(H * np.conj(H), fft_size)[:n]
        return np.max(np.abs(autoconv))
    
    def _score_from_parts(h):
        s = np.sum(h)
        if s == 0:
            return float('inf')
        max_abs = _autoconv_max_abs(h)
        return max_abs / (s * s)
    
    # Local search with bit flips
    n = len(h)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try random bit flips
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            idx = np.random.randint(0, n)
            old_val = h[idx]
            h[idx] = -old_val
            
            new_score = _score_from_parts(h)
            if new_score < current_score:
                current_score = new_score
                best_h = h.copy()
                best_score = current_score
                improved = True
                no_improve = 0
            else:
                h[idx] = old_val
                no_improve += 1
        
        if not improved:
            no_improve += 100
    
    return best_h
# EVOLVE-BLOCK-END
