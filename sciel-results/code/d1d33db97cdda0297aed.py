def construct():
    deadline = time.time() + 135.0
    best = None
    
    # Prioritize n=46 and n=45
    targets = [46, 45, 44, 43, 42, 41, 40, 39]
    
    for n in targets:
        min_degree_target = n - 39
        if min_degree_target < 1:
            min_degree_target = 1
            
        # Estimate time remaining
        time_remaining = deadline - time.time()
        if time_remaining <= 0:
            break
            
        # Allocate a portion of the remaining time to this n
        # Give more time to larger n
        # Simple heuristic: 1 second per attempt, try as many as fit in allocated slice
        # Slice: 50% of remaining for first, 30% for second, etc?
        # Or just fixed attempts?
        # Let's use a time slice.
        slice_time = time_remaining * 0.5 if n == 46 else time_remaining * 0.3 if n == 45 else time_remaining * 0.1
        
        slice_deadline = time.time() + slice_time
        
        seed_offset = n * 100000
        found = False
        
        while time.time() < slice_deadline:
            s = seed_offset
            seed_offset += 1
            rng = random.Random(s)
            res = _try_build(n, rng, min_degree_target)
            if res is not None:
                deg = [0] * n
                for e in res['edges']:
                    deg[e[0]] += 1
                    deg[e[1]] += 1
                md = min(deg)
                
                if best is None or n > best[0] or (n == best[0] and md > best[1]):
                    best = (n, md, res)
                found = True
                break
        
        if found:
            break
            
    if best is not None:
        return best[2]
        
    return {'n': 39, 'edges': []}
