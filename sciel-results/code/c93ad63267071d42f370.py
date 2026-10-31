"""Cap set construction via greedy selection with structured priorities."""
import time
import hashlib
from itertools import product

# EVOLVE-BLOCK-START

def _hash_priority(vector, n):
    """Deterministic pseudo-random priority via MD5 hash."""
    # Pack vector bytes: each coordinate as one byte (0, 1, or 2)
    key = bytes(vector)
    digest = hashlib.md5(key).digest()
    # Take first 8 bytes as a 64-bit integer
    val = int.from_bytes(digest[:8], 'big')
    return val / (1 << 64)

def _structured_priority(vector, n):
    """Structured priority that balances Hamming weights and coordinates."""
    # Prefer vectors with moderate weight (not all 0s or all 2s)
    weight = sum(1 for v in vector if v != 0)
    # Score: higher for weight near n/2, plus coordinate diversity
    center = abs(weight - n / 2.0)
    # Coordinate sum mod 3 gives additional variation
    coord_sum = sum(vector) % 3
    # Combine: minimize distance to center, break ties with hash
    base = 1.0 / (1.0 + center)
    h = _hash_priority(vector, n)
    return base * 0.7 + h * 0.3

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Higher priority = selected earlier.
    Uses a combination of structured heuristics and pseudo-random hashing.
    """
    # Time guard: if we've been computing too long, return a simple hash
    # (this shouldn't normally trigger for a single call)
    return _structured_priority(vector, n)

# EVOLVE-BLOCK-END
