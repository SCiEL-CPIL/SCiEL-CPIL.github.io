"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Empirical Utility Estimation via Stochastic Greedy Simulation.
    1. The ACO heuristic is computed once and static. A static rank (like prize/weight)
       doesn't account for the complex interactions of 5 constraints.
    2. We run a fast, lightweight simulation of the knapsack packing process (LPT-style
       greedy) multiple times with random tie-breaking.
    3. For each item, we track:
       - How often it gets packed (selection frequency).
       - The average prize it contributes when packed.
    4. The heuristic is the expected contribution: freq * avg_prize.
       This directly estimates the marginal value of an item in a realistic packing scenario,
       accounting for constraint conflicts (e.g., an item might have high prize/weight but
       rarely fits because it conflicts with other high-value items in multiple constraints).
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Time budget guard
    start_time = time.time()
    time_limit = 30.0  # 30 seconds for heuristic computation is plenty (total limit 300s)
    
    # We will run multiple simulations.
    # Each simulation is O(N^2) worst case if we do naive checks, but N=100 is small.
    # We can vectorize the "can fit" check.
    
    # Pre-compute a base attractiveness for initial sorting/tie-breaking to speed up simulation?
    # Or just use random order?
    # Using a random order is unbiased but might be slow to converge.
    # Using a "prize/weight" order as a base is faster to find good solutions,
    # but we want to estimate the *actual* utility in the *final* mix.
    # Let's do a mix: sort by prize/weight with some noise, or just random.
    # Random is safer for "empirical" distribution.
    
    num_sims = 0
    total_prize_contrib = np.zeros(n_items)
    selection_count = np.zeros(n_items)
    
    # We need to estimate the value.
    # Let's run simulations until time limit or a fixed number (e.g., 1000).
    max_sims = 2000
    
    while num_sims < max_sims:
        if time.time() - start_time > time_limit:
            break
            
        # 1. Initialize remaining capacity
        remaining = np.full(n_constraints, capacity)
        
        # 2. Get a random permutation of items
        # To make it slightly smarter than pure random, we can bias the random shuffle
        # or just use pure random to get a robust distribution.
        # Pure random might be too noisy for 100 items?
        # Let's try sorting by a noisy prize/weight ratio.
        # base_score = prizes / weights.sum(axis=0)
        # noisy_score = base_score * np.random.exponential(1.0, n_items)
        # order = np.argsort(noisy_score)[::-1]
        
        # Actually, pure random permutation is a valid way to sample the "average" case.
        # But ACO benefits from a heuristic that points to "good" items.
        # If we simulate random packings, we find out which items are "robustly good".
        
        order = np.random.permutation(n_items)
        
        packed = np.zeros(n_items, dtype=bool)
        
        # 3. Iterate through items in random order
        for idx in order:
            # Check if item fits
            # weights[:, idx] <= remaining
            if np.all(weights[:, idx] <= remaining):
                # Pack it
                remaining -= weights[:, idx]
                packed[idx] = True
                total_prize_contrib[idx] += prizes[idx]
                selection_count[idx] += 1
                
        num_sims += 1
        
    # 4. Calculate heuristic
    # Heuristic = Expected Prize Contribution = total_prize_contrib / num_sims
    # This is the average prize an item adds to the knapsack over many random packings.
    
    if num_sims == 0:
        # Fallback if no sims ran (should not happen)
        return np.ones(n_items)
        
    scores = total_prize_contrib / num_sims
    
    # Ensure positive
    # Items that never get packed get 0. 
    # The task says: "Heuristic entries that are not positive become 1e-9".
    # So 0 becomes 1e-9. This is fine.
    # However, a score of 0 means "never picked". 
    # Is it better to give them a tiny positive score or 1e-9?
    # 1e-9 is effectively 0.
    
    # We might want to amplify the difference.
    # If scores are very small, ACO pheromones might dominate.
    # Pheromone starts at 1. Heuristic is ~0.1-0.5 maybe.
    # Probability prop to P * H.
    # If H is small, P matters more.
    
    # Let's normalize or scale?
    # The task doesn't require normalization, just "larger is more attractive".
    # Using raw expected contribution is a good absolute measure.
    
    # One potential issue: if an item is very heavy, it might block others.
    # The simulation captures this (low selection frequency).
    
    # Apply a small power to sharpen?
    # The current program uses power 2.0.
    # Let's stick to the raw estimate first, maybe sqrt or log to compress?
    # Actually, linear is the most faithful to "expected value".
    
    # Ensure finite and positive
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
