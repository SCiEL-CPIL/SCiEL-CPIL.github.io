"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    # Convert to boolean adjacency matrix for efficient operations
    adj = adjacency.astype(bool)
    
    # Base heuristic: Prefer low degree vertices.
    # We can refine this slightly: prefer vertices with low degree 
    # and whose neighbors also have low degree (less likely to be "wasted").
    degrees = adj.sum(axis=1).astype(float)
    
    # Calculate mean neighbor degree for a better base score
    # neighbor_deg_sum[i] = sum(deg[j] for j in neighbors of i)
    neighbor_deg_sum = adj @ degrees
    
    # Avoid division by zero
    mean_neighbor_deg = np.zeros(n)
    mask = degrees > 0
    mean_neighbor_deg[mask] = neighbor_deg_sum[mask] / degrees[mask]
    
    # Base score: Lower degree is better, lower mean neighbor degree is better.
    # Normalize to [0, 1] for stability, then invert so higher is better.
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
        
    norm_deg = normalize(degrees)
    norm_mnd = normalize(mean_neighbor_deg)
    
    # Base priority: Higher score means "better to pick early"
    # We want low degree, so -norm_deg.
    base_priority = -(0.7 * norm_deg + 0.3 * norm_mnd)
    
    # Function to simulate greedy MIS size given a priority array and an initial available mask
    # and optionally a forced first vertex
    def simulate_greedy_size(prio, initial_available, forced_vertex=None):
        available = initial_available.copy()
        selected_count = 0
        
        if forced_vertex is not None:
            # Force select forced_vertex
            if available[forced_vertex]:
                available[forced_vertex] = False
                # Remove neighbors
                available[adj[forced_vertex]] = False
                selected_count += 1
        
        while True:
            # Find max priority among available
            # Set non-available to -inf
            masked_prio = np.where(available, prio, -np.inf)
            max_idx = np.argmax(masked_prio)
            
            if not available[max_idx]:
                break
            
            # Select max_idx
            available[max_idx] = False
            available[adj[max_idx]] = False
            selected_count += 1
            
        return selected_count

    # Calculate priority for each vertex by simulating the greedy process
    # where that vertex is forced to be selected first.
    # The priority of vertex v is the size of the independent set obtained.
    
    final_priority = np.zeros(n)
    
    # Use a deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    for v in range(n):
        if time.time() > deadline:
            break
            
        # Simulate greedy with v forced first, using base_priority for subsequent picks
        size = simulate_greedy_size(base_priority, np.ones(n, dtype=bool), forced_vertex=v)
        final_priority[v] = size
        
    # If we hit the deadline, we might have some zeros. 
    # Fill remaining with a fallback heuristic (e.g., base_priority)
    # But usually 200 iterations of O(N^2) is very fast.
    
    # Optional: If we have time, we could do a second pass where we update base_priority 
    # to be final_priority and re-simulate, but the first pass is often sufficient 
    # because the "best first move" is the most critical decision.
    
    # To ensure distinct scores for tie-breaking in the decoder (though decoder logic 
    # might handle ties arbitrarily, distinct scores are safer for consistent evaluation),
    # we add a tiny noise based on index if scores are identical.
    # However, the task says "fixed greedy decoder". If scores are equal, 
    # the behavior depends on the implementation (usually first index).
    # Adding tiny noise ensures we don't accidentally rely on arbitrary tie-breaking 
    # if two vertices truly have the same simulated size.
    # But wait, if two vertices have the same simulated size, they are equally good.
    # We can add a small epsilon based on base_priority to break ties consistently 
    # in favor of the "locally better" vertex.
    
    # Combine simulated size with a tiny fraction of base priority to break ties
    # and provide a smooth gradient.
    # final_priority = final_priority + 0.001 * base_priority
    
    return final_priority
# EVOLVE-BLOCK-END
