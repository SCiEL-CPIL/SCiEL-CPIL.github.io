def priority(vector, n):
    # Fixed pseudo-random weights for each coordinate position
    # Use a deterministic sequence based on primes
    h = 0
    weights = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    for i in range(min(n, len(weights))):
        h += weights[i] * vector[i]
    # For n > len(weights), continue with more primes
    if n > len(weights):
        # Use a simple recurrence for additional weights
        w = 131
        for i in range(len(weights), n):
            h += w * vector[i]
            w = (w * 2 + 1) % 999983
    return -h
