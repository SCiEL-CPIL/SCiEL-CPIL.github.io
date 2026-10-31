import zlib
import math

def priority(vector, n):
    # Calculate L2 distance to the center (1, 1, ..., 1)
    dist = 0
    # Count symbol frequencies for spectral analysis
    counts = [0, 0, 0]
    for v in vector:
        diff = v - 1
        dist += diff * diff
        counts[v] += 1
    
    # Spectral penalty: measure deviation from uniform distribution
    # Uniform would be n/3 for each symbol
    # Higher variance in counts = more structured = more likely to form APs
    mean_count = n / 3.0
    variance = 0.0
    for c in counts:
        variance += (c - mean_count) ** 2
    variance /= n  # Normalize
    
    # Calculate a pseudo-random tie-breaker
    b = bytes(vector)
    h = zlib.crc32(b)
    rand_part = (h % 100000) / 100000.0
    
    # Priority: Higher is better.
    # We want to pick low distance and low variance vectors first.
    # Combine: -dist dominates, then -variance, then random tie-break
    # Use soft combination to allow some trade-off
    priority_val = -float(dist) - 0.3 * variance + rand_part * 0.1
    
    return priority_val
