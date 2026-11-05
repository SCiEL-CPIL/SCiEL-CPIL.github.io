import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    if n == 0:
        return np.array([])

    # Pre-compute neighbors as lists of indices for faster simulation
    # Using a list of lists is often faster than boolean masks for sparse graphs in Python loops
    # But we are using numpy. Let's stick to numpy operations for simulation if possible,
    # or use a hybrid.
    
    # For the simulation, we need to check if a vertex is "blocked".
    # A vertex is blocked if it is in the set OR adjacent to a vertex in the set.
    
    # Base score: Negative degree is a strong prior for MIS.
    # Low degree vertices are "cheaper" to include.
    degree = adj.sum(axis=1)
    base_scores = -degree
    
    # We will search for a better ordering by perturbing the scores.
    # This explores the space of possible greedy outcomes.
    
    best_size = 0
    best_order = np.argsort(-base_scores)  # Initial order: lowest degree first
    
    # Helper to simulate greedy MIS
    def simulate(order_arr):
        selected = np.zeros(n, dtype=bool)
        in_set = np.zeros(n, dtype=bool)
        size = 0
        # We iterate through the order. 
        # To speed up, we can maintain a "available" mask.
        # Initially all are available.
        # When we pick v, we mark v and its neighbors as unavailable.
        
        available = np.ones(n, dtype=bool)
        
        for v in order_arr:
            if available[v]:
                in_set[v] = True
                size += 1
                # Mark v and neighbors as unavailable
                available[v] = False
                # Get neighbors
                # adj[v] > 0 gives a boolean mask of neighbors
                # We need to set available[neighbors] = False
                # This is a vectorized operation
                available[adj[v] > 0] = False
        return size, in_set

    # Initial evaluation
    current_size, _ = simulate(best_order)
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    # Strategy: Simulated Annealing on the Score Vector
    # We treat the score vector as a continuous variable.
    # We want to find a score vector that yields a high-quality greedy set.
    # However, the greedy decoder is discrete.
    # Instead, we use Random Restarts with Local Refinement on the Order.
    
    # But the prompt asked for a DIFFERENT strategy.
    # Let's try: "Greedy with Lookahead"
    # When choosing the next vertex, look ahead 1-2 steps to see which choice blocks the fewest future options.
    # This is too slow for N=200 if done naively in Python.
    
    # Let's stick to the "Perturbed Scores" approach but make it smarter.
    # We will use a "Swarm" of score vectors.
    
    num_particles = 50
    particles = []
    for i in range(num_particles):
        # Initial particles: Base score + noise
        noise_scale = 0.5 * np.std(base_scores)
        scores = base_scores + rng.normal(0, noise_scale, n)
        order = np.argsort(-scores)
        size, _ = simulate(order)
        particles.append({
            'scores': scores,
            'order': order,
            'size': size
        })
        if size > best_size:
            best_size = size
            best_order = order.copy()
    
    # Optimize particles
    # Each particle tries to improve its score by swapping elements in its order?
    # No, let's keep it simple: Random Walk on Scores.
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        improved_in_round = False
        
        for p in particles:
            if time.time() > deadline:
                break
            
            # Perturb scores
            # Step size decreases over time
            step_size = 0.1 * np.std(base_scores) * temperature
            new_scores = p['scores'] + rng.normal(0, step_size, n)
            new_order = np.argsort(-new_scores)
            
            # Evaluate
            new_size, _ = simulate(new_order)
            
            # Acceptance Criteria
            if new_size > p['size']:
                p['scores'] = new_scores
                p['order'] = new_order
                p['size'] = new_size
                improved_in_round = True
                
                if new_size > best_size:
                    best_size = new_size
                    best_order = new_order.copy()
            else:
                # Metropolis Acceptance: accept worse solutions with probability exp(-delta/T)
                # Delta is the loss in size (integer). 
                # Since size is integer, delta is at least 1.
                # exp(-1/T) is very small for large T? No, T starts at 1.0.
                # exp(-1/1.0) = 0.36.
                # This allows exploration.
                if rng.random() < np.exp(-1.0 / max(temperature, 1e-6)):
                    p['scores'] = new_scores
                    p['order'] = new_order
                    p['size'] = new_size
        
        # Cooling
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
        # If no improvement in a full round, we could restart some particles
        # But for now, just keep cooling.

    # Convert best_order to scores
    # The evaluator picks vertices in descending score order.
    # So, position 0 in best_order should have the highest score.
    # We assign ranks.
    ranks = np.zeros(n, dtype=np.float64)
    for pos, v in enumerate(best_order):
        ranks[v] = float(n - pos)
        
    return ranks
