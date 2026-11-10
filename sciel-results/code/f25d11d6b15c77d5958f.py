    def compute_max_discrepancy_fast(s):
        n_local = len(s)
        max_abs = 0
        for step in range(1, n_local + 1):
            # For this step, indices with same i % step form an AP
            # Actually, AP with step 'step' starting at 'start' has indices start, start+step, ...
            # These are exactly indices congruent to start-1 mod step (0-indexed)
            # For each residue r in 0..step-1, the sequence s[r], s[r+step], s[r+2*step], ...
            # is an AP starting at r+1
            for r in range(step):
                running = 0
                idx = r
                while idx < n_local:
                    running += s[idx]
                    if abs(running) > max_abs:
                        max_abs = abs(running)
                    idx += step
        return max_abs
    
    current_max = compute_max_discrepancy_fast(signs)
    
    # Step 3: Local search to improve
    # Try flipping individual signs and see if it reduces the max discrepancy
    # We need an efficient way to evaluate the effect of a flip
    
    # For local search, we'll use a simpler approach: 
    # For each position, try flipping it and compute the new max discrepancy.
    # But computing max discrepancy from scratch is O(n^2 log n) or similar.
    # For n=256, let's estimate: sum over step of (n/step) * step = n * n = n^2 = 65536 operations per evaluation.
    # With 256 positions and maybe 100 iterations, that's 256 * 100 * 65536 ≈ 1.6e9, too slow.
    
    # Better: incremental update. When we flip sign at position i, only APs containing i are affected.
    # But tracking all affected APs is complex. Let's try a different approach.
    
    # Alternative: use simulated annealing or hill climbing with a smarter cost function.
    # Let's just do a few rounds of hill climbing with the fast computation, limiting iterations.
    
    rng = random.Random(42)
    
    best_signs = signs[:]
    best_max = current_max
    
    # Try flipping each sign and keep if it improves
    improved = True
    iteration = 0
    max_iterations = 200  # Limit to ensure we finish in time
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n):
            # Try flipping sign at position i
            old_val = best_signs[i]
            best_signs[i] = -old_val
            new_max = compute_max_discrepancy_fast(best_signs)
            if new_max < best_max:
                best_max = new_max
                improved = True
            else:
                best_signs[i] = old_val
    
    # Try some random restarts with local search
    for restart in range(5):
        # Start from a perturbed version of the best solution
        trial = best_signs[:]
        # Flip a few random signs
        for _ in range(10):
            idx = rng.randint(0, n - 1)
            trial[idx] = -trial[idx]
        
        trial_max = compute_max_discrepancy_fast(trial)
        
        # Local search on trial
        t_improved = True
        t_iteration = 0
        while t_improved and t_iteration < 50:
            t_improved = False
            t_iteration += 1
            for i in range(n):
                old_val = trial[i]
                trial[i] = -old_val
                new_max = compute_max_discrepancy_fast(trial)
                if new_max < trial_max:
                    trial_max = new_max
                    t_improved = True
                else:
                    trial[i] = old_val
        
        if trial_max < best_max:
            best_max = trial_max
            best_signs = trial[:]
    
    return best_signs
