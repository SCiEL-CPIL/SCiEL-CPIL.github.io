# EVOLVE-BLOCK-START
import numpy as np

def priority(adjacency):
    """
    Computes a priority score for each vertex using a static structural heuristic
    based on Core Number and Degree.
    
    Approach:
    1. Compute the core number for each vertex using the K-core decomposition algorithm.
    2. Compute the degree of each vertex.
    3. Combine these metrics to form a score where vertices with low core numbers
       and low degrees receive higher scores.
    4. This encourages the greedy decoder to pick peripheral, low-constraint vertices
       first, which is a strong heuristic for finding a large independent set.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
        
    N = adjacency.shape[0]
    
    # Convert to boolean matrix for efficient neighbor lookups
    adj_bool = adjacency.astype(bool)
    
    # 1. Compute Degrees
    degrees = adj_bool.sum(axis=1)
    
    # 2. Compute Core Numbers
    # K-core decomposition
    # We maintain a list of active vertices and their current degrees.
    # We use a bucket sort approach for O(M) complexity.
    
    max_deg = int(degrees.max()) if N > 0 else 0
    if max_deg == 0:
        return np.ones(N)
        
    # Initialize buckets
    # buckets[i] contains list of vertices with current degree i
    buckets = [[] for _ in range(max_deg + 1)]
    for i in range(N):
        buckets[int(degrees[i])].append(i)
        
    # Current degrees (will be modified)
    cur_deg = degrees.copy()
    core_num = np.zeros(N, dtype=int)
    
    # Process from highest degree down? No, standard algorithm processes from low to high?
    # Actually, the standard linear time algorithm processes vertices in increasing order of core number.
    # We can simulate this by maintaining a stack or just iterating.
    # A simpler O(N*M) approach is sufficient for N=200.
    # Let's use the O(N*M) peeling algorithm for clarity and robustness.
    
    active = np.ones(N, dtype=bool)
    core_num = np.zeros(N, dtype=int)
    
    # We need to find the max core number first or just peel iteratively.
    # Peeling:
    # While there are active vertices with degree < current_k:
    #   Remove them.
    
    # To be efficient, we can just sort by degree and process.
    # But degrees change.
    
    # Let's use a priority queue approach or simple iteration.
    # For N=200, even O(N^3) is fine.
    
    # Simple O(N^2) or O(N*M) peeling:
    # We keep a list of vertices.
    # We repeatedly find the vertex with minimum degree among active vertices.
    # Its core number is its current degree.
    # Remove it and update neighbors.
    
    active_deg = degrees.copy()
    active_mask = np.ones(N, dtype=bool)
    
    for _ in range(N):
        if not np.any(active_mask):
            break
        
        # Find active vertex with minimum degree
        masked_deg = np.where(active_mask, active_deg, np.inf)
        v = np.argmin(masked_deg)
        
        if masked_deg[v] == np.inf:
            break
            
        core_num[v] = int(masked_deg[v])
        
        # Remove v
        active_mask[v] = False
        
        # Update neighbors
        neighbors = np.where(adj_bool[v])[0]
        for u in neighbors:
            if active_mask[u]:
                active_deg[u] -= 1
                
    # 3. Combine Scores
    # We want low core_num and low degree to have HIGH priority.
    # Normalize core_num and degrees to [0, 1]
    
    max_core = core_num.max() if N > 0 else 1
    max_deg_val = degrees.max() if N > 0 else 1
    
    # Avoid division by zero
    if max_core == 0:
        norm_core = np.zeros(N)
    else:
        norm_core = core_num / max_core
        
    if max_deg_val == 0:
        norm_deg = np.zeros(N)
    else:
        norm_deg = degrees / max_deg_val
        
    # Score: Higher is better.
    # We invert the metrics.
    # Weight core number more heavily? Core number is a stronger indicator of "cliquishness".
    # Let's try equal weight first, or slight bias towards core number.
    
    score = 1.0 - (0.5 * norm_core + 0.5 * norm_deg)
    
    return score
# EVOLVE-BLOCK-END
