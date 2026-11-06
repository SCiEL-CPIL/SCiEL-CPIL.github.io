<<<<<<< SEARCH
def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers (powers of 2, 3, 5)
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Geometric Progressions
    for ratio in [1.2, 1.3, 1.5, 1.8, 2.0]:
        for start in [1, 2, 3]:
            A = []
            current = float(start)
            for _ in range(20):
                val = int(round(current))
                if val in A:
                    val += 1
                    while val in A:
                        val += 1
                if val > 10000:
                    break
                A.append(val)
                current *= ratio
            if len(A) == 20 and _is_valid(A):
                candidates.append(A)
                
    # 3. Arithmetic Progressions
    for start in [1, 2, 5]:
        for step in [1, 2, 3]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
                
    # 4. Random
    candidates.append(random.sample(range(1, 10001), 20))
    candidates.append(random.sample(range(1, 1001), 20))
    
    return candidates
=======
def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers (powers of 2, 3, 5)
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Rational geometric progressions (structured, deterministic)
    try:
        geo_cands = _rational_geometric_candidates()
        candidates.extend(geo_cands)
    except:
        pass
    
    # 3. Arithmetic Progressions
    for start in [1, 2, 5]:
        for step in [1, 2, 3]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
                
    # 4. Random
    candidates.append(random.sample(range(1, 10001), 20))
    candidates.append(random.sample(range(1, 1001), 20))
    
    return candidates
>>>>>>> REPLACE

