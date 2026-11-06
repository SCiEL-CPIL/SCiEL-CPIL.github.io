def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    # Primary: balance of symbol counts
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    # Secondary: penalize vectors with many repeated adjacent values
    # (heuristic: such vectors may be more likely to form APs with others)
    repeat_penalty = 0.0
    for i in range(n - 1):
        if vector[i] == vector[i + 1]:
            repeat_penalty += 1.0
    
    # Tertiary: decorrelated hash for tie-breaking
    h = 0
    for i, v in enumerate(vector):
        h = (h * 131 + v * 7 + i * 3) & 0x7FFFFFFF
    
    # Combine: balance dominates, then repeat penalty, then hash
    eps1 = 0.01  # repeat penalty weight (small relative to dev scale)
    eps2 = 1e-9  # hash weight (tiny)
    return -dev - eps1 * repeat_penalty + eps2 * (h / 0x7FFFFFFF)
