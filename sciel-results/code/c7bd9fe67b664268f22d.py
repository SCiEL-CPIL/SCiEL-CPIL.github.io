"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    if n == 0:
        return 0
    
    # Use argsort for the order
    order = np.argsort(-scores, kind='stable')
    
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbours as forbidden
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
        
    return count


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    
    Strategy: Direct Greedy Construction with Iterative Swap Local Search.
    1. Initialize scores based on a 'blocking potential' heuristic:
       Vertices with low degree and low neighbor-degree sum are prioritized.
       This avoids picking hubs that block many others.
    2. Use a local search loop that attempts to improve the independent set size
       by swapping the priority scores of pairs of vertices.
       - First, try adjacent swaps in the current ranking (efficient hill climbing).
       - Then, try random swaps to escape local minima.
    3. Return the scores that yield the maximum found independent set size.
    """
    deadline = time.time() + 115  # Safety margin before 120s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])
    
    degree = adjacency.sum(axis=1).astype(float)
    
    # Heuristic: Minimize "cost" = degree + sum(degrees of neighbors)
    # This identifies vertices that are peripheral and whose neighbors are also peripheral.
    # We want high scores for low cost.
    neighbor_degree_sum = np.zeros(n, dtype=float)
    # For each vertex, sum the degrees of its neighbors
    # This is equivalent to (A * degree)
    neighbor_degree_sum = adjacency @ degree
    
    # Cost: Higher is worse. We want to sort by ascending cost.
    # So score = -cost.
    # Add a small tie-breaker based on index or random to ensure deterministic but varied start?
    # Deterministic is better for reproducibility, but random tie-breaker can help escape symmetry.
    # Let's use -cost + small_noise.
    cost = degree + neighbor_degree_sum
    
    # Initial scores: Negate cost so that lower cost -> higher score -> picked first.
    # Add a tiny bit of random noise to break ties deterministically but vary slightly per run?
    # Actually, the task implies a single function call. Let's use a fixed small epsilon based on index
    # to break ties consistently, or just rely on argsort stability.
    # To allow local search to move vertices, we need distinct scores or handle ties carefully.
    # Let's add a very small random noise to make scores distinct.
    np.random.seed(42) # Fixed seed for reproducibility in evaluation? 
    # The evaluator might run multiple times. Let's use a time-based seed?
    # No, "fixed greedy decoder" implies the decoder is fixed, but priority can be stochastic?
    # The task says "Write a Python priority... function". It doesn't forbid randomness.
    # However, deterministic is safer. Let's use index-based tie-breaking.
    tie_breaker = np.arange(n) * 1e-6
    base_scores = -cost + tie_breaker
    
    best_scores = base_scores.copy()
    best_count = _greedy_decode(adjacency, best_scores)
    
    # Local Search Optimization
    # We want to maximize the count returned by _greedy_decode.
    # We will iterate:
    # 1. Get current order.
    # 2. Try swapping adjacent elements in the order.
    # 3. If improvement, keep it.
    # 4. If no improvement in a full pass, try random swaps.
    # 5. Stop when time runs out.
    
    # Pre-compute neighbor lists for faster access?
    # np.where(adjacency[v]) is O(N).
    # N=200. 200 swaps * 200 ops = 40k ops per evaluation.
    # 1000 evaluations = 40M ops. Should be fine.
    
    # To speed up evaluation, we can write a slightly optimized decoder?
    # The existing _greedy_decode is fine.
    
    # Helper to get current order
    def get_order(scores):
        return np.argsort(-scores, kind='stable')
    
    # Main optimization loop
    # We track the current scores and the best scores found.
    current_scores = best_scores.copy()
    current_count = best_count
    
    # Phase 1: Adjacent Swap Hill Climbing
    # We will do multiple passes until no improvement is found or time is up.
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline:
            break
        
        order = get_order(current_scores)
        improved = False
        
        # Try swapping adjacent pairs in the order
        # i from 0 to n-2
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            v1 = order[i]
            v2 = order[i + 1]
            
            # Swap their scores
            current_scores[v1], current_scores[v2] = current_scores[v2], current_scores[v1]
            
            new_count = _greedy_decode(adjacency, current_scores)
            
            if new_count > current_count:
                current_count = new_count
                improved = True
                # Keep swap
                if current_count > best_count:
                    best_count = current_count
                    best_scores = current_scores.copy()
            else:
                # Revert swap
                current_scores[v1], current_scores[v2] = current_scores[v2], current_scores[v1]
                
        if not improved:
            break
    
    # Phase 2: Random Swap Local Search
    # If we have time left, try random swaps to escape local minima.
    # We restart from the best known state.
    if time.time() < deadline - 5:
        current_scores = best_scores.copy()
        current_count = best_count
        
        # Number of random swaps to attempt
        # Estimate how many evals we can do.
        # Each eval is ~0.001-0.005s? Let's assume 0.002s.
        # 100s left -> 50,000 swaps.
        # Let's just loop until deadline.
        
        while time.time() < deadline - 2:
            # Pick two random distinct vertices
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue
            
            # Swap
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            
            new_count = _greedy_decode(adjacency, current_scores)
            
            if new_count > current_count:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_scores = current_scores.copy()
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                
    return best_scores
# EVOLVE-BLOCK-END
