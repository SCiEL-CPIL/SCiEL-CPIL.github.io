def priority(vector, n):
    # Deterministic pseudo-random priority
    # Using a simple LCG-like step on the base-3 value
    val = 0
    for i in range(n):
        val = val * 3 + vector[i]
    # Knuth's multiplicative hash
    return (val * 2654435761) % 1000000007
