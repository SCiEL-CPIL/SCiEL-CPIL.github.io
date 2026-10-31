import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 10.0

# Precompute DFT basis for Z_3^n
_dft_cache = {}
_n_current = None
_priority_cache = {}
_time_exceeded = False

def _time_check():
    global _time_exceeded
    if _time_exceeded:
        return True
    if (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN):
        _time_exceeded = True
        return True
    return False

def _get_dft_basis(n):
    """Precompute the DFT basis for Z_3^n."""
    global _dft_cache, _n_current
    if _n_current == n and n in _dft_cache:
        return _dft_cache[n]
    
    _n_current = n
    _priority_cache.clear()
    
    # For Z_3^n, the DFT character is chi_k(v) = omega^{k·v} where omega = exp(2*pi*i/3)
    # We'll work with real-valued approximations using cos and sin components.
    # The magnitude squared of the Fourier coefficient at frequency k for a single vector v
    # is |chi_k(v)|^2 = 1 for all k, but we care about the *phase* and how vectors combine.
    
    # Alternative approach: Use the Walsh-Hadamard-like transform for Z_3.
    # For each vector v, compute its "spectral signature" as the vector of
    # chi_k(v) for a subset of low-frequency k's.
    
    # To keep this tractable, we'll use a random projection of the full DFT.
    # Generate random frequency vectors k_j in Z_3^n, and compute the projection
    # of v onto these frequencies.
    
    rng = np.random.RandomState(42 + n)
    num_freqs = min(3 * n, 50)  # Number of random frequency projections
    freqs = rng.randint(0, 3, size=(num_freqs, n)).astype(np.float64)
    
    # Precompute omega = exp(2*pi*i/3)
    omega = np.exp(2j * np.pi / 3)
    
    # For each frequency k, the character is omega^{k·v mod 3}
    # We'll compute the real and imaginary parts of the projected Fourier coefficient.
    
    _dft_cache[n] = (freqs, omega)
    return _dft_cache[n]

def _compute_spectral_score(vector, n):
    """Compute a spectral-based priority score for a vector."""
    freqs, omega = _get_dft_basis(n)
    
    v = np.array(vector, dtype=np.float64)
    
    # Compute dot products k·v mod 3 for each frequency k
    # Since values are in {0,1,2}, we can compute directly
    dots = np.mod(freqs @ v, 3)
    
    # Compute omega^{dots} for each frequency
    # omega = exp(2*pi*i/3), so omega^0 = 1, omega^1 = -0.5 + i*sqrt(3)/2, omega^2 = -0.5 - i*sqrt(3)/2
    # We can compute this efficiently using lookup
    real_parts = np.ones(len(dots))
    imag_parts = np.zeros(len(dots))
    
    # omega^0 = 1
    # omega^1 = cos(2*pi/3) + i*sin(2*pi/3) = -0.5 + i*0.866...
    # omega^2 = cos(4*pi/3) + i*sin(4*pi/3) = -0.5 - i*0.866...
    mask1 = (dots == 1)
    mask2 = (dots == 2)
    
    real_parts[mask1] = -0.5
    real_parts[mask2] = -0.5
    imag_parts[mask1] = math.sqrt(3) / 2
    imag_parts[mask2] = -math.sqrt(3) / 2
    
    # The "spectral magnitude" is the norm of the projected Fourier coefficient
    # We want to favor vectors that have balanced spectral contributions
    # A vector with many 0s has a simpler spectrum; a vector with mixed values has a richer spectrum.
    
    # Compute the spectral norm
    spectral_norm = np.sqrt(np.sum(real_parts ** 2 + imag_parts ** 2))
    
    # Also compute the "spectral entropy" - how evenly spread the energy is
    energies = real_parts ** 2 + imag_parts ** 2
    total_energy = np.sum(energies)
    if total_energy > 0:
        probs = energies / total_energy
        # Shannon entropy of the energy distribution
        nonzero = probs > 0
        entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
        max_entropy = math.log(len(energies))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = 0
    
    # Score: favor vectors with high spectral norm (rich structure) and high entropy (balanced)
    # This encourages selecting vectors that are "spectrally diverse"
    score = spectral_norm * 100.0 + normalized_entropy * 50.0
    
    return score

def priority(vector, n):
    global _priority_cache, _time_exceeded
    
    key = (tuple(vector), n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    if _time_check():
        # Fallback: simple deterministic score
        result = float(sum(vector))
        _priority_cache[key] = result
        return result
    
    # Compute spectral score
    score = _compute_spectral_score(vector, n)
    
    # Add a small hash-based tiebreaker for determinism
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    hash_score = h / (10**9 + 7) * 0.01
    
    result = float(score + hash_score)
    
    _priority_cache[key] = result
    return result
