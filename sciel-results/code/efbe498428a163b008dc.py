<<<<<<< SEARCH
    # --- Search Strategy ---
    # Start with Min Slack
    best_scores = base_priority.copy()
    best_ms = simulate_sgs(best_scores)
    
    # Hill Climbing / Random Search
    # We will perturb the scores directly.
    # Since scores are relative, we can add noise or swap values.
    
    current_scores = best_scores.copy()
    current_ms = best_ms
    
    # Temperature for Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    max_iterations = 5000
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Generate neighbor
        # Option 1: Swap two random activities' scores
        # Option 2: Add noise to one activity's score
        # Option 3: Linear combination of base features with random weights
        
        # Let's try swapping scores of two activities that are "close" in priority or random.
        if random.random() < 0.5:
            # Swap
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], current_scores[i1]
        else:
            # Perturb one
            i1 = random.randint(0, n-1)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.1)
            # Ensure scores remain in a reasonable range? 
            # Actually, only relative order matters.
        
        # Evaluate neighbor
        neighbor_ms = simulate_sgs(neighbor_scores)
        
        if neighbor_ms is float('inf'):
            continue
            
        # Accept if better, or with probability exp(-Delta/T)
        delta = neighbor_ms - current_ms
        if delta < 0:
            # Better
            current_scores = neighbor_scores
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_scores = current_scores.copy()
        else:
            # Worse, accept with prob
            if random.random() < np.exp(-delta / T):
                current_scores = neighbor_scores
                current_ms = neighbor_ms
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_scores
=======
    # --- Search Strategy ---
    # Start with combined base priority
    best_scores = base_priority.copy()
    best_ms = simulate_sgs(best_scores)
    
    # Hill Climbing with Simulated Annealing
    current_scores = best_scores.copy()
    current_ms = best_ms
    
    # Temperature for Simulated Annealing
    T = 1.0
    T_min = 0.005
    cooling = 0.995
    
    # Use more iterations since simulator is faster
    max_iterations = 20000
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Generate neighbor: mix of swaps and perturbations
        r = random.random()
        if r < 0.4:
            # Swap two random activities' scores
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], current_scores[i1]
        elif r < 0.7:
            # Perturb one activity
            i1 = random.randint(0, n-1)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.15)
        else:
            # Perturb two activities
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.15)
            neighbor_scores[i2] += random.gauss(0, 0.15)
        
        # Evaluate neighbor
        neighbor_ms = simulate_sgs(neighbor_scores)
        
        if neighbor_ms is float('inf') or np.isinf(neighbor_ms):
            continue
            
        # Accept if better, or with probability exp(-Delta/T)
        delta = neighbor_ms - current_ms
        if delta < 0:
            # Better
            current_scores = neighbor_scores
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_scores = current_scores.copy()
        else:
            # Worse, accept with probability
            if T > T_min:
                if random.random() < np.exp(-delta / T):
                    current_scores = neighbor_scores
                    current_ms = neighbor_ms
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_scores
>>>>>>> REPLACE
