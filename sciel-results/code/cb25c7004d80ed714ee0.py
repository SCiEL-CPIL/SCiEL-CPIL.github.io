def dispatch(ready_operations, time):
    if not ready_operations:
        return 0 # or handle empty
    
    best_idx = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Get min duration among machine options
        min_dur = min(op["machine_options"].values())
        
        # Heuristic: Shortest Processing Time (SPT)
        # Ties broken by Most Work Remaining First (MWRF) to keep baseline behavior for ties?
        # Or maybe break by least remaining ops?
        
        # Let's try SPT primarily.
        score = min_dur
        
        # To make it "dynamic" and potentially capture slack-like behavior:
        # If we assume that shorter tasks have more slack, we prefer them.
        
        if score < best_score:
            best_score = score
            best_idx = i
        elif score == best_score:
            # Tie-breaking: prefer the one with more remaining work? (Baseline)
            # Or less remaining work?
            # Let's stick to baseline tie-breaking: max remaining_work
            if op["remaining_work"] > ready_operations[best_idx]["remaining_work"]:
                best_idx = i
                
    return int(best_idx)
