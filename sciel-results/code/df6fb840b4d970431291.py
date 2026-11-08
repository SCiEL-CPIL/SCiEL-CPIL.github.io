"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Convert to boolean matrix for efficient bitwise operations
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Helper: Evaluate the size of the independent set produced by a given order
    def greedy_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        # Iterate through vertices in the specified order
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                # Mark all neighbors as blocked
                selected[adj[v]] = True
        return count

    # Helper: Generate an initial order based on a heuristic score
    def generate_initial_order(seed=0):
        rng = np.random.RandomState(seed)
        # Heuristic: Lower degree is generally better for greedy MIS
        # Add some noise to diversify initial states
        degrees = adj.sum(axis=1)
        # Normalize degrees to [0, 1]
        max_deg = np.max(degrees) if np.max(degrees) > 0 else 1
        norm_deg = degrees / max_deg
        
        # Score = -degree + noise. We want to select low degree first.
        # So we sort by ascending (degree - noise) or descending (-degree + noise).
        # Let's use a combination of degree and a random tie-breaker.
        # score_i = -norm_deg[i] + 0.1 * rng.random()
        scores = -norm_deg + 0.1 * rng.random(n)
        return np.argsort(-scores)

    # Helper: Perturb an order by swapping two random elements
    def perturb_order(order, rng):
        new_order = order.copy()
        i = rng.randint(n)
        j = rng.randint(n)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        return new_order

    best_order = None
    best_size = -1
    
    # Multi-start Simulated Annealing
    # We run multiple independent SA chains to explore different regions of the permutation space.
    # Given N=200, each evaluation is O(N^2) ~ 40k ops.
    # In Python, a single greedy pass is fast (~1ms).
    # We can afford ~10,000-50,000 evaluations per second.
    # Budget: 140s.
    # Let's aim for 4 restarts, each running for ~35s.
    
    num_restarts = 4
    time_per_restart = 35.0
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        rng = np.random.RandomState(42 + r)
        
        # Initialize with a heuristic-based order
        current_order = generate_initial_order(seed=r)
        current_size = greedy_size(current_order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = current_order.copy()
            
        # Simulated Annealing parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.9995  # Slow cooling to allow exploration
        
        restart_start = time.time()
        
        while True:
            if time.time() > deadline - 5.0:
                break
            if time.time() - restart_start > time_per_restart:
                break
            if T < T_min:
                break
                
            # Perturb
            new_order = perturb_order(current_order, rng)
            new_size = greedy_size(new_order)
            
            # Acceptance criterion
            delta = new_size - current_size
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_order = new_order
                current_size = new_size
                
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
                    
            T = max(T_min, T * cooling)
            
    # If no order was found (shouldn't happen), fallback to degree-based
    if best_order is None:
        degrees = adj.sum(axis=1)
        best_order = np.argsort(degrees)
        
    # Convert the optimal order back to scores
    # The decoder picks vertices in descending score order.
    # So, the vertex at index 0 in best_order should have the highest score.
    # The vertex at index n-1 should have the lowest.
    # We assign scores such that rank 0 -> score 1.0, rank n-1 -> score 0.0 (or similar).
    # Actually, just assigning rank-based scores is sufficient for the decoder.
    # score[v] = n - rank[v]
    
    scores = np.zeros(n, dtype=float)
    ranks = np.zeros(n, dtype=int)
    for i, v in enumerate(best_order):
        ranks[v] = i
        
    # Higher rank index means lower priority in our order (since order[0] is first).
    # We want vertices in best_order[0] to have high scores.
    # So score = n - rank_index
    scores = n - ranks
    
    return scores
# EVOLVE-BLOCK-END
