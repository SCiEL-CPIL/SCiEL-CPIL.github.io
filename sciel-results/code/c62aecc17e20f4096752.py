import numpy as np
import time

def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set using a simulated
    Dynamic Min-Degree heuristic.
    
    The fixed decoder picks vertices in descending score order.
    We simulate a strong greedy MIS algorithm (Dynamic Min-Degree with
    random tie-breaking) multiple times. We assign scores based on the
    average rank (selection time) of each vertex in these simulations.
    This ensures the decoder's order closely matches the order of a
    high-quality greedy algorithm.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    deg = adjacency.sum(axis=1).astype(float)
    
    # Number of simulations
    # N=200 is small, so we can afford many.
    # 50 simulations * 200 steps * 200 updates = 2M ops. Fast.
    num_sims = 50
    
    # Accumulator for scores
    # We want higher score for earlier selection.
    # Rank 0 -> Score N. Rank N-1 -> Score 1.
    scores = np.zeros(n)
    
    for _ in range(num_sims):
        if time.time() > deadline:
            break
            
        active = np.ones(n, dtype=bool)
        active_deg = deg.copy()
        ranks = np.zeros(n)
        
        for step in range(n):
            if not active.any():
                break
                
            # Add noise to break ties and explore different orderings
            # Scale noise relative to max degree? Or fixed?
            # Fixed 1.0 is fine for small integer degrees.
            noise = np.random.normal(0, 1.0, n)
            noise[~active] = 0
            
            # We want to pick MIN degree.
            # So we minimize (active_deg + noise).
            # Inactive nodes should have +inf.
            scores_sim = active_deg + noise
            scores_sim[~active] = np.inf
            
            v = np.argmin(scores_sim)
            
            ranks[v] = step
            
            # Remove v and neighbors from active
            neighbors = np.nonzero(adjacency[v])[0]
            
            # Update degrees of neighbors that are still active
            # Note: We must check active status BEFORE setting them to False?
            # No, the degree update is for the induced subgraph of REMAINING active nodes.
            # If u is a neighbor of v, and u is active, then u loses one edge (to v).
            # So active_deg[u] -= 1.
            # Then we mark u as inactive.
            
            # We need to update degrees for neighbors that are currently active.
            # But we also mark them inactive.
            # So:
            for u in neighbors:
                if active[u]:
                    active_deg[u] -= 1
            
            active[v] = False
            active[neighbors] = False
            
        # Convert ranks to scores
        # Rank 0 (first) should have highest score.
        # Score = n - rank
        sim_scores = n - ranks
        scores += sim_scores
        
    if num_sims > 0:
        scores /= num_sims
    else:
        # Fallback if no simulations completed
        scores = -deg
        
    return scores
