"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _greedy_decode(adjacency, order):
    """
    Run the fixed greedy decoder given a specific vertex order.
    Returns: set of selected vertices (indices)
    """
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    
    # order is a permutation of [0, n-1] representing priority (higher priority first)
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        # Mark neighbors as unavailable
        available[adjacency[v]] = False
        
    return selected

def _evaluate_order(adjacency, order):
    """
    Calculate the size of the independent set produced by a specific order.
    """
    selected = _greedy_decode(adjacency, order)
    return np.sum(selected)

def _compute_core_numbers(adjacency):
    """
    Compute the core number for each vertex in the graph.
    Core number k(v) is the highest k such that v belongs to the k-core.
    The k-core is the maximal subgraph where every vertex has degree at least k.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(int)
    # Sort vertices by degree
    order = np.argsort(degrees)
    
    # We can compute core numbers using a peeling algorithm
    # However, for n=200, a simpler iterative removal or even just using degree as a proxy 
    # might be sufficient, but core number is a better structural metric.
    
    # Efficient core number computation for small n:
    # Copy degrees
    deg = degrees.copy()
    core = np.zeros(n, dtype=int)
    
    # We process vertices in increasing order of degree
    # This is a standard algorithm:
    # 1. Sort vertices by degree
    # 2. For each vertex v in sorted order:
    #    core[v] = max(core[v], deg[v])
    #    For each neighbor u of v:
    #       if deg[u] > core[v]:
    #           deg[u] -= 1
    
    # Note: This standard algorithm requires re-sorting or a priority queue for large graphs.
    # For n=200, we can do a simpler O(n^2) approximation or just use the initial degree.
    # Actually, let's use a robust O(n^2) implementation suitable for small n.
    
    # Initialize core numbers to 0
    core = np.zeros(n, dtype=int)
    # We will simulate the peeling process
    # A simpler approach for small n:
    # For k from max_degree down to 0:
    #   Find all vertices with current degree >= k
    #   If a vertex is in this set, its core number is at least k
    
    max_deg = np.max(degrees) if n > 0 else 0
    
    # To be efficient and correct, let's use the standard algorithm but optimized for small n.
    # Since n=200, O(n^2) is fine.
    
    # We need to track "current degree" which decreases as we peel.
    # The standard algorithm sorts by degree once, but degrees change.
    # For n=200, we can just re-sort in each step or use a simple loop.
    
    # Let's use a simpler heuristic that captures similar structural info:
    # "Local Clustering Coefficient" or just "Degree of Neighbors".
    # Actually, let's stick to a robust Core Number approximation.
    
    # Algorithm:
    # 1. Sort vertices by degree.
    # 2. Iterate through sorted list.
    # 3. For vertex v, core[v] = max(core[v], current_degree[v])
    # 4. For each neighbor u of v, if current_degree[u] > core[v], decrement current_degree[u].
    #    (This is slightly inaccurate without re-sorting, but often good enough for initialization).
    
    # Let's do a more accurate O(n^2) core number calc:
    # For each k from 0 to max_deg:
    #   The k-core consists of vertices with degree >= k in the remaining graph.
    #   We can find the core number by finding the largest k such that v is in the k-core.
    
    # Given n=200, let's just compute the "weighted degree" or "neighbor degree sum" 
    # as a proxy for structural centrality, which is cheaper and often effective for MIS.
    # score = -degree(v) - alpha * sum(degrees(neighbors))
    
    # But the directive asks for a different strategy. 
    # Let's use the Core Number properly.
    
    # Re-implementation of Core Number for small n:
    # We maintain a list of vertices and their current degrees.
    # We repeatedly remove the vertex with the minimum degree.
    # The core number of a removed vertex v is its degree at the time of removal.
    
    # This is O(n^2) if we scan for min degree each time.
    remaining = np.ones(n, dtype=bool)
    current_degrees = degrees.copy()
    core = np.zeros(n, dtype=int)
    
    for _ in range(n):
        if not np.any(remaining):
            break
        # Find vertex with min degree among remaining
        # Mask out non-remaining
        masked_degrees = np.where(remaining, current_degrees, np.inf)
        v = np.argmin(masked_degrees)
        core[v] = current_degrees[v]
        
        # Remove v
        remaining[v] = False
        # Update degrees of neighbors
        neighbors = np.where(adjacency[v])[0]
        # Only update if they are still remaining
        active_neighbors = neighbors[remaining[neighbors]]
        current_degrees[active_neighbors] -= 1
        
    return core

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    
    Strategy: Simulated Annealing on Vertex Ordering.
    1. Initialize order using Core Number (descending) and Degree (ascending).
    2. Perform Simulated Annealing swaps to optimize the order.
    3. Return the scores corresponding to the best order found.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin from 120s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # 1. Compute Structural Metrics
    degrees = adjacency.sum(axis=1).astype(float)
    core_numbers = _compute_core_numbers(adjacency)
    
    # 2. Initial Order
    # We want high core number first? Or low?
    # In MIS, vertices in dense clusters (high core) are "expensive" to pick.
    # Often, picking vertices from sparse parts (low core) first is better?
    # Or is it better to pick "dominant" vertices first?
    # Let's try: Sort by Core Number (Descending), then Degree (Ascending).
    # High core = central. If we pick them, we block many. 
    # If we don't pick them, their neighbors might form a good set.
    # Standard heuristic for MIS: Low degree first.
    # Let's blend: Score_init = -core_number * 10 - degree.
    # Higher score = higher priority.
    # So we want low core and low degree to have high score?
    # Let's test: 
    # If I pick a low degree vertex, I block few neighbors. Good.
    # If I pick a high core vertex, I am in a dense area.
    
    # Let's initialize with a random permutation and let SA find the structure, 
    # OR use a heuristic initialization.
    # Heuristic: Sort by (degree + core_number) ascending? 
    # Let's use: score = -degree - 0.5 * core_number.
    # We want to pick vertices with LOW score in this metric? No, priority is descending.
    # So priority = -(-degree - 0.5 * core_number) = degree + 0.5 * core_number? 
    # No, we want LOW degree vertices to be picked FIRST.
    # So priority should be HIGH for LOW degree vertices.
    # priority = -degree - 0.5 * core_number.
    
    init_scores = -degrees - 0.5 * core_numbers
    init_order = np.argsort(-init_scores, kind='stable')
    
    current_order = init_order.copy()
    current_size = _evaluate_order(adjacency, current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # 3. Simulated Annealing
    # Temperature starts high, cools down.
    # Move: Swap two random vertices in the order.
    # Acceptance: Metropolis criterion.
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    
    # Number of steps
    # n=200. Each step is O(n) for decode? No, decode is O(n^2) worst case.
    # 200^2 = 40,000 ops. 
    # 10,000 steps * 40,000 = 400M ops. Might be too slow in Python.
    # We need to be careful.
    
    # Optimization: 
    # We can't run full decode for every step if it's too slow.
    # But n=200 is small. Let's estimate.
    # _greedy_decode loop: n iterations. Inside: boolean ops on array of size n.
    # Vectorized numpy ops are fast.
    # available[adjacency[v]] = False is a scatter operation.
    # Let's try 2000 steps first.
    
    T = T_start
    step = 0
    max_steps = 5000
    
    while step < max_steps:
        if time.time() > deadline:
            break
            
        # Pick two random indices in the order
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_size = _evaluate_order(adjacency, current_order)
        
        # Calculate delta
        delta = new_size - current_size
        
        # Accept if better or with probability exp(delta / T)
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order.copy()
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        T *= cooling_rate
        if T < T_end:
            T = T_end
            # If we've cooled down, we can increase max_steps if time permits
            # Or just break if we've done enough
            if step > 2000:
                break
                
        step += 1
        
    # Convert best order to scores
    # The score should reflect the priority.
    # If vertex v is at position p in best_order (0 is highest priority),
    # its score should be such that argsort(-scores) gives best_order.
    # So score[v] = n - 1 - position_of_v_in_order
    # Or simply, score[v] = n - rank(v)
    
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores

# EVOLVE-BLOCK-END
