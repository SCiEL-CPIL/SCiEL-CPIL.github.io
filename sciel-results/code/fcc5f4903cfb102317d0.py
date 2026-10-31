"""Program for the cap set task using multi-frequency additive character theory prioritization."""
import numpy as np
from functools import lru_cache

# Precompute multiple fixed pseudo-random frequency vectors in F_3^n
# We use a higher number of frequencies to create a more complex priority landscape

_NUM_FREQS = 5

def _get_frequency_vectors(n):
    """Generate a fixed set of pseudo-random frequency vectors in F_3^n."""
    rng = np.random.RandomState(seed=n * 7919 + 13)
    # Generate _NUM_FREQS vectors of length n with values in {0, 1, 2}
    freqs = []
    for _ in range(_NUM_FREQS):
        freq = tuple(rng.randint(0, 3, size=n).tolist())
        freqs.append(freq)
    return tuple(freqs)

# Cache frequency vectors
_freq_cache = {}

def _get_freqs(n):
    if n not in _freq_cache:
        _freq_cache[n] = _get_frequency_vectors(n)
    return _freq_cache[n]

def priority(vector, n):
    """
    Priority based on the magnitude of the sum of additive characters
    with respect to multiple fixed pseudo-random frequency vectors.
    
    We compute the complex character sum for each frequency vector and
    sum them up. The magnitude of this total sum indicates how "aligned"
    the vector is with the random frequencies. Vectors with a smaller
    magnitude are more "random" and thus preferred, as they are less
    likely to form structured patterns (like 3-APs) that are detected
    by these frequencies.
    """
    freqs = _get_freqs(n)
    
    # omega = exp(2*pi*i/3)
    # We can represent the character values as complex numbers or just use
    # the real and imaginary parts.
    # omega^0 = 1 + 0j
    # omega^1 = -0.5 + sqrt(3)/2 * j
    # omega^2 = -0.5 - sqrt(3)/2 * j
    
    sqrt3_over_2 = np.sqrt(3) / 2
    
    sum_real = 0.0
    sum_imag = 0.0
    
    for freq in freqs:
        # Compute weighted sum mod 3
        weighted_sum = 0
        for i in range(n):
            weighted_sum += vector[i] * freq[i]
        weighted_sum %= 3
        
        if weighted_sum == 0:
            sum_real += 1.0
            sum_imag += 0.0
        elif weighted_sum == 1:
            sum_real += -0.5
            sum_imag += sqrt3_over_2
        else: # weighted_sum == 2
            sum_real += -0.5
            sum_imag += -sqrt3_over_2
            
    # Magnitude of the sum
    magnitude = np.sqrt(sum_real * sum_real + sum_imag * sum_imag)
    
    # We want to prioritize vectors with SMALL magnitude (more random)
    # So we invert it.
    # To keep the scale reasonable, we can map it to a range.
    # Max magnitude is _NUM_FREQS.
    # Let's return -magnitude + _NUM_FREQS so that higher is better.
    
    return _NUM_FREQS - magnitude
