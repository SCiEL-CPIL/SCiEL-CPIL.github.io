"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Convert to int for efficiency
    adj = adjacency.astype(np.int8)
    degrees = adj.sum(axis=1)
    
    # Selection frequency counter
    freq = np.zeros(n, dtype=np.float64)
    
    # Number of simulations - adaptive based on time
    max_sims = 200
    sim_count = 0
    
    for sim in range(max_sims):
        if time.time() > deadline:
            break
        
        # Randomized greedy MIS simulation
        selected = np.zeros(n, dtype=bool)
        available = np.ones(n, dtype=bool)
        
        # Precompute neighbor sets for fast lookup
        # Use a list of neighbors for each vertex
        # For efficiency, we'll work with the adjacency matrix directly
        
        # Greedy process: repeatedly pick a vertex from available, 
        # then remove it and its neighbors from available
        # To make it randomized, we weight by inverse degree
        
        remaining = set(range(n))
        selected_count = 0
        
        # For efficiency with n=200, we can do this in a loop
        # At each step, pick from available vertices with probability proportional to 1/(degree+1)
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Get available vertices
            avail_list = list(remaining)
            if not avail_list:
                break
            
            # Weight by inverse degree (lower degree = higher weight)
            weights = 1.0 / (degrees[avail_list] + 1.0)
            total_w = weights.sum()
            if total_w == 0:
                break
            
            # Random selection
            probs = weights / total_w
            idx = np.random.choice(len(avail_list), p=probs)
            v = avail_list[idx]
            
            # Select v
            selected[v] = True
            freq[v] += 1.0
            selected_count += 1
            
            # Remove v and its neighbors from remaining
            neighbors = np.where(adj[v] > 0)[0]
            to_remove = set(neighbors.tolist())
            to_remove.add(v)
            remaining -= to_remove
        
        sim_count += 1
    
    # If no simulations ran, fall back to static heuristic
    if sim_count == 0:
        return -degrees.astype(float)
    
    # Normalize: higher frequency = higher priority
    # The greedy decoder picks in descending order, so we want high priority = likely to be selected
    # But we also want to account for the fact that some vertices are "harder" to select
    # Use frequency as the score
    return freq
    
# EVOLVE-BLOCK-END
