<<<<<<< SEARCH
def _generate_value_candidates(n, max_val=10000):
    """Generate a smaller, more focused set of high-quality candidates."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Smooth numbers (highly composite)
    smooth_14 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    if len(smooth_14) == n:
        candidates.append(smooth_14)
        
    # 3. Another smooth set
    smooth_14b = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    if len(smooth_14b) == n:
        candidates.append(smooth_14b)
        
    # 4. Powers of 2
    candidates.append([2**i for i in range(n)])
    
    # 5. Random small numbers
    random.seed(42)
    for _ in range(15):
        vals = sorted(random.sample(range(1, 50), n))
        if len(set(vals)) == n:
            candidates.append(vals)
            
    # 6. Geometric progression
    base = 1.5
    vals = [max(1, int(base**i)) for i in range(n)]
    seen = set()
    unique_vals = []
    for v in vals:
        while v in seen:
            v += 1
        seen.add(v)
        unique_vals.append(v)
    if len(unique_vals) == n and all(v <= max_val for v in unique_vals):
        candidates.append(sorted(unique_vals))

    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates
=======
def _generate_value_candidates(n, max_val=10000):
    """Generate a smaller, more focused set of high-quality candidates."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Smooth numbers (highly composite)
    smooth_14 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    if len(smooth_14) == n:
        candidates.append(smooth_14)
        
    # 3. Another smooth set
    smooth_14b = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    if len(smooth_14b) == n:
        candidates.append(smooth_14b)
        
    # 4. Powers of 2
    candidates.append([2**i for i in range(n)])
    
    # 5. Random small numbers with more variety
    random.seed(42)
    for _ in range(20):
        # Mix of small and medium values
        vals = sorted(random.sample(range(1, 80), n))
        if len(set(vals)) == n:
            candidates.append(vals)
    
    # 6. Geometric progression
    base = 1.5
    vals = [max(1, int(base**i)) for i in range(n)]
    seen = set()
    unique_vals = []
    for v in vals:
        while v in seen:
            v += 1
        seen.add(v)
        unique_vals.append(v)
    if len(unique_vals) == n and all(v <= max_val for v in unique_vals):
        candidates.append(sorted(unique_vals))
    
    # 7. Values with many common sum/product patterns
    # Use numbers that share factors to reduce distinct products
    factor_based = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
    if len(factor_based) == n:
        candidates.append(factor_based)
    
    # 8. Near-smooth with slight variations
    near_smooth = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    if len(near_smooth) == n:
        candidates.append(near_smooth)

    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates
>>>>>>> REPLACE

