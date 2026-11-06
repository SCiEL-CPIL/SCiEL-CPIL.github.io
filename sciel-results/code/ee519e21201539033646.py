def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    # 1. Generate standard candidates
    candidates = _construct_candidates()
    
    # 2. Generate deterministic GP candidates
    # Allocate 10s for deterministic search
    det_time = 10.0
    det_candidates = _deterministic_search(det_time)
    
    # Combine and deduplicate
    all_candidates = candidates + det_candidates
    
    # Filter valid
    valid_candidates = []
    for A in all_candidates:
        if _is_valid(A):
            valid_candidates.append(A)
            
    # Evaluate initial objectives
    scored_candidates = []
    for A in valid_candidates:
        try:
            obj = _objective_fast(A)
            scored_candidates.append((obj, A))
        except:
            pass
    
    # Sort by objective
    scored_candidates.sort(key=lambda x: x[0])
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # Process top candidates with local search
    # We have time_budget left. 
    # Let's try top 5 candidates.
    max_attempts = 5
    
    for i, (obj, A) in enumerate(scored_candidates[:max_attempts]):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
            
        if obj < best_obj:
            best_A = list(A)
            best_obj = obj
            
        # Allocate time for local search proportional to remaining time
        # But ensure we don't spend all time on one.
        # Simple: use half remaining time for first, less for subsequent?
        # Or just fixed 10s per attempt?
        # Let's use 15s per attempt if time allows.
        ls_time = min(15.0, remaining - 1.0)
        
        if ls_time > 1.0:
            refined_A, refined_obj = _local_search(A, ls_time, rng)
            if refined_obj < best_obj:
                best_obj = refined_obj
                best_A = refined_A
                
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement with remaining time
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2.0:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
