import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit
    
    # Ensure numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues, though task specifies (5, 100)
    if prizes.shape != weights.shape:
        # Fallback: try to broadcast or just return simple heuristic
        prize_per_item = prizes.mean(axis=0)
        return np.where(prize_per_item > 0, prize_per_item, 1e-9)
        
    num_constraints, num_items = weights.shape
    capacity = 25.0
    
    # Generate a diverse pool of heuristic candidates
    # Each candidate is a function of prizes and weights
    # We want to explore different trade-offs between prize and weight
    # and different ways of aggregating multi-constraint weights
    
    candidates = []
    
    # Basic aggregates
    prize_mean = prizes.mean(axis=0)
    prize_max = prizes.max(axis=0)
    prize_min = prizes.min(axis=0)
    
    weight_mean = weights.mean(axis=0)
    weight_max = weights.max(axis=0)
    weight_min = weights.min(axis=0)
    weight_sum = weights.sum(axis=0)
    
    # Normalize weights to [0,1] roughly for stability, or just use raw
    # Since weights are uniform [0,1], raw is fine.
    
    # Candidate 1: Current baseline (mean prize / (0.7 max + 0.3 mean weight))
    denom1 = 0.7 * weight_max + 0.3 * weight_mean + 1e-6
    candidates.append(prize_mean / denom1)
    
    # Candidate 2: Mean prize / mean weight
    candidates.append(prize_mean / (weight_mean + 1e-6))
    
    # Candidate 3: Max prize / max weight
    candidates.append(prize_max / (weight_max + 1e-6))
    
    # Candidate 4: Mean prize / sum weight
    candidates.append(prize_mean / (weight_sum + 1e-6))
    
    # Candidate 5: (Mean prize)^2 / mean weight
    candidates.append((prize_mean ** 2) / (weight_mean + 1e-6))
    
    # Candidate 6: Mean prize / (weight_mean + weight_max)/2
    candidates.append(prize_mean / ((weight_mean + weight_max)/2 + 1e-6))
    
    # Candidate 7: Geometric mean prize / arithmetic mean weight
    # Geometric mean of prizes across constraints
    # Avoid log of 0
    prize_safe = np.clip(prizes, 1e-9, None)
    geo_prize = np.exp(np.mean(np.log(prize_safe), axis=0))
    candidates.append(geo_prize / (weight_mean + 1e-6))
    
    # Candidate 8: Mean prize / (weight_max ** 0.5)
    candidates.append(prize_mean / (np.sqrt(weight_max) + 1e-6))
    
    # Candidate 9: Mean prize / (weight_min + 1e-6) - risky if min is 0, but uniform [0,1] so ok
    candidates.append(prize_mean / (weight_min + 1e-6))
    
    # Candidate 10: (Mean prize) / (weight_mean) * (1 / weight_max)
    candidates.append((prize_mean / (weight_mean + 1e-6)) / (weight_max + 1e-6))
    
    # Candidate 11: Mean prize / (weight_mean * weight_max)
    candidates.append(prize_mean / (weight_mean * weight_max + 1e-6))
    
    # Candidate 12: Sum prize / Sum weight
    candidates.append(prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-6))
    
    # Candidate 13: Mean prize / (weight_mean ** 1.5)
    candidates.append(prize_mean / (weight_mean ** 1.5 + 1e-6))
    
    # Candidate 14: Mean prize / (weight_max ** 0.75)
    candidates.append(prize_mean / (weight_max ** 0.75 + 1e-6))
    
    # Candidate 15: Max prize / (weight_sum)
    candidates.append(prize_max / (weights.sum(axis=0) + 1e-6))
    
    # Ensure all candidates are positive
    for i in range(len(candidates)):
        candidates[i] = np.where(candidates[i] > 0, candidates[i], 1e-9)
    
    # Fast Vectorized ACO Simulation for each candidate
    # We simulate a simplified ACO to estimate the quality of the heuristic.
    # Parameters: 5 ants, 10 iterations.
    # Pheromone update: decay 0.9, add best_prize/10.
    # Selection: prob proportional to pheromone * heuristic.
    
    num_ants = 5
    num_iters = 10
    
    # Pre-compute random numbers for reproducibility and speed within this call?
    # The task says "seeded per instance", but we don't have the seed.
    # We should use a fixed seed for our internal simulation to be deterministic,
    # or just use np.random. Since we are evaluating candidates, consistent randomness
    # across candidates is key for fair comparison.
    rng = np.random.RandomState(42)
    
    best_candidate_score = -1
    best_candidate_heuristic = candidates[0]
    
    for h in candidates:
        # Run simulation
        pheromone = np.ones(num_items)
        best_iter_prize = 0
        
        for _ in range(num_iters):
            # Each ant
            best_ant_prize_this_iter = 0
            for _ in range(num_ants):
                # Ant builds a solution
                remaining_cap = np.full(num_constraints, capacity)
                total_prize = 0.0
                selected = np.zeros(num_items, dtype=bool)
                
                # Greedy-like selection with randomness based on pheromone*heuristic
                # We pick items one by one.
                # To speed up, we can't easily vectorize the "fit" check per step in a loop
                # without complex masking, but 5 ants * 10 iters * 100 items is small.
                # However, pure Python loop for selection is slow.
                # Let's try a vectorized greedy step:
                # At each step, we have a set of available items.
                # We calculate probability for each available item.
                # We pick one.
                # We update remaining capacity.
                
                # Actually, a simpler approximation for "ACO quality" might be:
                # Simulate the greedy selection driven by the heuristic * pheromone.
                # Since pheromone starts at 1 and decays, and heuristic is fixed,
                # the first iteration is basically greedy by heuristic.
                # Subsequent iterations are influenced by previous bests.
                
                # Let's implement a fast greedy approximation for the simulation:
                # Instead of full stochastic ACO, we can simulate the "greedy" path
                # that the ACO would likely converge to, or just run the ACO steps
                # with vectorized operations where possible.
                
                # Given the small size (100 items), a Python loop for selection is acceptable
                # if we do it for 50 candidates * 50 steps = 2500 steps.
                # 2500 steps * 100 items check = 250,000 operations. This is fast in Python if vectorized checks are used.
                
                available = np.ones(num_items, dtype=bool)
                
                while True:
                    # Find items that fit
                    # weights[i, j] <= remaining_cap[j] for all j
                    # This is a matrix comparison
                    fits = np.all(weights <= remaining_cap[:, np.newaxis], axis=0)
                    fits = fits & available
                    
                    if not np.any(fits):
                        break
                    
                    # Calculate attractiveness: pheromone * heuristic
                    att = pheromone * h
                    att[~fits] = 0
                    
                    # Normalize to probabilities
                    total_att = np.sum(att)
                    if total_att == 0:
                        break
                    probs = att / total_att
                    
                    # Select item
                    item_idx = rng.choice(num_items, p=probs)
                    
                    # Update state
                    total_prize += prizes[:, item_idx].sum() # Wait, prize is per constraint?
                    # Task: "prizes and weights, of shape (5, 100)".
                    # "highest total prize found by any ant".
                    # Usually in knapsack, prize is a single value per item.
                    # Here prizes is (5, 100). Does "total prize" mean sum of all prizes?
                    # "mean over instances of the highest total prize".
                    # If prizes are per constraint, maybe the "prize" of an item is the sum?
                    # Or is it a multi-objective?
                    # "Item scores for ant colony optimization of the multiple-constraint knapsack"
                    # "highest total prize found by any ant"
                    # Let's assume total prize is the sum of all prize components for the selected items.
                    # prizes[:, item_idx] is a vector of 5 prizes. Sum them.
                    
                    # Actually, looking at the current program:
                    # prize_per_item = prizes.mean(axis=0)
                    # It uses mean.
                    # But the score is "highest total prize".
                    # If the evaluator sums the prizes, then sum is correct.
                    # If the evaluator uses mean, then mean is correct.
                    # "prizes ... uniform in [0,1]".
                    # Let's assume the "prize" of an item is the sum of its components, or maybe just one component?
                    # Given "multiple-constraint knapsack", usually there is ONE value and MULTIPLE weights.
                    # Here we have 5 prizes and 5 weights.
                    # This looks like a multi-objective or just a dense representation.
                    # "highest total prize" suggests a scalar.
                    # Let's use sum of prizes for the simulation's reward calculation, as it's a common "total value" metric.
                    # If the evaluator uses mean, sum is proportional, so ranking might be similar.
                    
                    total_prize += np.sum(prizes[:, item_idx])
                    
                    # Update capacity
                    remaining_cap -= weights[:, item_idx]
                    available[item_idx] = False
                    # Optional: remove items that no longer fit due to capacity reduction
                    # This is handled by the `fits` check in the next iteration.
                    
                if total_prize > best_ant_prize_this_iter:
                    best_ant_prize_this_iter = total_prize
            
            # Update pheromone
            pheromone *= 0.9
            pheromone += best_ant_prize_this_iter / 10.0
            
            # Early exit if time is running out
            if time.time() > deadline:
                break
        
        # Calculate average best prize over iterations for this candidate?
        # Or just the final best?
        # The task score is "mean over instances of the highest total prize found by any ant" in the REAL run.
        # Our simulation is a proxy. We want to pick the heuristic that leads to high prize.
        # Let's track the max prize seen in the simulation.
        
        # I didn't track max prize across all ants/iters in the loop above, I only updated pheromone.
        # Let's re-structure to track the best prize found in the simulation for this candidate.
        
    # The loop above was a bit messy. Let's refactor for clarity and correctness of the metric.
    
    best_candidate_score = -1
    best_candidate_heuristic = candidates[0]
    
    for h in candidates:
        pheromone = np.ones(num_items)
        sim_best_prize = 0.0
        
        for _ in range(num_iters):
            iter_best_prize = 0.0
            for _ in range(num_ants):
                remaining_cap = np.full(num_constraints, capacity)
                total_prize = 0.0
                available = np.ones(num_items, dtype=bool)
                
                while True:
                    fits = np.all(weights <= remaining_cap[:, np.newaxis], axis=0)
                    fits = fits & available
                    
                    if not np.any(fits):
                        break
                    
                    att = pheromone * h
                    att[~fits] = 0
                    total_att = np.sum(att)
                    if total_att == 0:
                        break
                    probs = att / total_att
                    
                    item_idx = rng.choice(num_items, p=probs)
                    
                    total_prize += np.sum(prizes[:, item_idx])
                    remaining_cap -= weights[:, item_idx]
                    available[item_idx] = False
                    
                if total_prize > iter_best_prize:
                    iter_best_prize = total_prize
                
            sim_best_prize = max(sim_best_prize, iter_best_prize)
            
            pheromone *= 0.9
            pheromone += iter_best_prize / 10.0
            
            if time.time() > deadline:
                break
        
        if sim_best_prize > best_candidate_score:
            best_candidate_score = sim_best_prize
            best_candidate_heuristic = h
        
        if time.time() > deadline:
            break
    
    # Return the best heuristic found
    return np.where(best_candidate_heuristic > 0, best_candidate_heuristic, 1e-9)
