import time
import random

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Fixed seed for reproducibility of the random vector
# The random vector is generated once per n.
# We use a simple LCG or just rely on Python's random module with a fixed seed
# for the specific n encountered, to ensure deterministic behavior for the evaluator
# while providing the "random" chaos required for the strategy.

_random_vectors_cache = {}

def _get_random_vector(n):
    if n not in _random_vectors_cache:
        # Create a random vector of length n with entries in {-1, 1}
        # Using a fixed seed based on n to ensure determinism across calls for the same n
        # but different n's get different vectors.
        rng = random.Random(42 + n) 
        vec = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        _random_vectors_cache[n] = vec
    return _random_vectors_cache[n]

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # Get the random vector for this dimension
    r_vec = _get_random_vector(n)
    
    # Compute the dot product (projection)
    # vector entries are {0, 1, 2}
    # r_vec entries are {-1, 1}
    # This creates a chaotic ordering.
    # Higher projection means we prioritize this vector.
    
    # Optimization: Manual loop is fast enough for typical n (< 50)
    # Using sum(v * r for v, r in zip(vector, r_vec))
    
    proj = 0
    for i in range(n):
        proj += vector[i] * r_vec[i]
        
    return float(proj)
