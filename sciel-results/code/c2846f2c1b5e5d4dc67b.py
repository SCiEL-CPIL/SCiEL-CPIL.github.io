def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    # Use a softer, more discriminative balance metric
    # Penalize deviations with a smooth nonlinearity
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    # Add a secondary term that penalizes extreme imbalances more
    max_dev = max(abs(c0 - target), abs(c1 - target), abs(c2 - target))
    dev_combined = dev + 0.5 * max_dev**2
    
    s = sum(vector) % 3
    
    # Count transitions: changes in value as we scan the vector
    # Vectors with fewer transitions may be more "structured" and cap-set-friendly
    transitions = 0
    for i in range(1, n):
        if vector[i] != vector[i-1]:
            transitions += 1
    # Normalize transitions to [0, 1]
    trans_norm = transitions / max(n - 1, 1)
    # Penalize high transition count (prefer smoother vectors)
    trans_penalty = -0.1 * trans_norm
    
    # Use a fast deterministic tie-breaker based on vector pattern
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) & 0x7FFFFFFF
    
    h_norm = h / 0x7FFFFFFF
    
    sum_bonus = {1: 2.0, 2: 1.0, 0: 0.0}[s]
    
    # Combined priority: balance is primary, sum and structure are secondary
    return -dev_combined + sum_bonus * 0.01 + trans_penalty + 1e-9 * h_norm
