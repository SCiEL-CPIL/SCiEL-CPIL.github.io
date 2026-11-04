"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(h):
    """Compute the max absolute value of the autoconvolution (excluding DC)."""
    n = len(h)
    # Use FFT for autoconvolution
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is the DC component (sum of squares = n for ±1 values)
    # We want max |acf[k]| for k=1..n-1
    if n > 1:
        return np.max(np.abs(acf[1:n]))
    return 0.0


def _legendre_sequence(n):
    """Generate a Legendre sequence of length n."""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = i + 1
        if x == 1:
            seq[i] = 1.0
        else:
            # Legendre symbol: 1 if x is quadratic residue mod p, -1 if non-residue
            # For composite x, use Jacobi symbol
            a = x
            m = 2 * n - 1  # Use a prime-like modulus for the construction
            # Actually, let's use the standard Legendre sequence approach
            # For position i (0-indexed), use i+1 and check quadratic residuosity
            # Simple approach: use the Legendre symbol (x/p) where p is prime
            # But we need a fixed prime. Let's use a different approach.
            # 
            # Standard Legendre sequence of length p-1 for prime p:
            # s[i] = (i+1 | p) for i=0..p-2
            # We need p-1 >= n, so find smallest prime p >= n+1
            pass
    # Simpler: generate a prime-based Legendre sequence
    # Find smallest prime >= n+1
    def is_prime(num):
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(num)) + 1, 2):
            if num % i == 0:
                return False
        return True
    
    p = n + 1
    while not is_prime(p):
        p += 1
    
    # Generate Legendre sequence of length p-1, take first n
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = i + 1
        # Legendre symbol (x/p)
        # Use Euler's criterion: (x/p) = x^((p-1)/2) mod p
        # If result is 1, it's a quadratic residue (+1)
        # If result is p-1, it's a non-residue (-1)
        # If x % p == 0, it's 0 (but x < p so this won't happen)
        val = pow(x, (p - 1) // 2, p)
        if val == 1:
            seq[i] = 1.0
        elif val == p - 1:
            seq[i] = -1.0
        else:
            seq[i] = 1.0  # Should not happen for x not divisible by p
    return seq


def _simulated_annealing(h, max_time=100.0):
    """Improve h using simulated annealing on sign flips."""
    n = len(h)
    h = h.copy()
    best_h = h.copy()
    best_score = _autoconvolution_max_abs(h)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    T = 1.0  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    current_score = best_score
    while T > T_min:
        if time.time() > deadline - 5.0:
            break
        
        # Try a batch of flips
        batch_size = max(1, int(n * 0.01))
        for _ in range(batch_size):
            if time.time() > deadline - 2.0:
                break
            
            i = rng.integers(0, n)
            h[i] *= -1
            
            new_score = _autoconvolution_max_abs(h)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[i] *= -1  # Revert
        
        T *= cooling_rate
    
    return best_h, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Legendre sequence
    h = _legendre_sequence(N)
    current_score = _autoconvolution_max_abs(h)
    
    # Run simulated annealing
    h, current_score = _simulated_annealing(h, max_time=130.0)
    
    # If we have time, try a few more restarts with random initialization
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            h_rand = rng.choice([-1.0, 1.0], size=N)
            h_rand, score_rand = _simulated_annealing(h_rand, max_time=20.0)
            if score_rand < current_score:
                current_score = score_rand
                h = h_rand
    
    return h
# EVOLVE-BLOCK-END
