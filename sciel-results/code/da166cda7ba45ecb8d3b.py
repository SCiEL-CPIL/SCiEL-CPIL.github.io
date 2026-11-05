import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Compute the principal eigenvector of the adjacency matrix using power iteration.
       This identifies the "core" of the graph. Vertices with high eigenvector centrality
       are central to the structure and likely to be part of dense subgraphs.
    2. The score is primarily based on minimizing eigenvector centrality.
       Rationale: Selecting a vertex with low eigenvector centrality removes a peripheral
       node. This is less likely to "waste" potential independent set members in the core
       because peripheral nodes are less connected to the high-value core nodes.
       In contrast, selecting a high-centrality node might block many other high-centrality
       nodes or just remove a node that doesn't contribute as much to the "spread" of the IS.
       
       Actually, for MIS, we want to pick nodes that don't block each other.
       Standard LDF (Lowest Degree First) is effective because low-degree nodes have
       fewer conflicts.
       Spectral centrality captures a global view of "conflict potential".
       A node with high eigenvector centrality is connected to many high-centrality nodes.
       Picking it blocks them.
       A node with low eigenvector centrality is connected to low-centrality nodes.
       
       Let's test if -eigenvector_centrality is a good proxy for -degree but with global context.
       
    3. Combine with degree as a tie-breaker or secondary term to ensure stability.
       Score = -alpha * eigvec[i] - beta * deg[i]
       
       We will tune alpha and beta.
       Given the stagnation in local heuristics, the spectral term should provide a different
       ordering, potentially grouping nodes by their community/periphery status.
       
    4. Power iteration details:
       - Normalize the vector at each step to prevent overflow.
       - Run for a fixed number of iterations (e.g., 100) or until convergence.
       - 100 iterations for n=200 is very fast (200x200 matrix mult is cheap).
       
    5. Safety:
       - Check time.
       - If power iteration is too slow (unlikely), fall back to degree.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # 1. Calculate degrees (for fallback and secondary term)
    deg = adj.sum(axis=1)
    
    # 2. Compute principal eigenvector via Power Iteration
    # Initialize with random positive vector or ones
    v = np.ones(n, dtype=float)
    v = v / np.linalg.norm(v)
    
    # Power iteration parameters
    # For n=200, 50 iterations is usually sufficient for convergence to high precision
    # and is extremely fast.
    max_iters = 50
    tol = 1e-8
    
    for _ in range(max_iters):
        # Check time limit
        if time.time() - start_time > 140:
            break
            
        # Matrix-vector multiplication
        v_new = adj @ v
        
        # Normalize
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            # Graph might be empty or disconnected in a way that leads to zero vector?
            # In ER graphs, it's connected with high probability.
            # If zero, keep old v or break.
            break
        v_new = v_new / norm
        
        # Check convergence
        diff = np.linalg.norm(v_new - v)
        v = v_new
        if diff < tol:
            break
            
    # Ensure v is positive (eigenvector of adjacency for connected graph should be positive)
    # If there are negative values due to numerical issues or bipartite components,
    # take absolute value or just use the raw value?
    # For connected ER graphs, Perron-Frobenius theorem ensures positive principal eigenvector.
    # If disconnected, it might have zeros.
    # We want to minimize centrality, so lower v is better.
    
    # 3. Combine scores
    # We want to prioritize LOW eigenvector centrality and LOW degree.
    # Let's weight them.
    # Eigenvector values are normalized, so their scale is ~1/sqrt(n) to 1.
    # Degrees are ~10.
    # We need to balance the scales.
    
    # Normalize eigenvector to [0, 1] range for better combination?
    # Or just use raw values with appropriate weights.
    
    # Let's try: score = -deg - 10 * v
    # Why 10? If v is ~0.07 (1/sqrt(200)), 10*0.07 = 0.7.
    # Deg is ~10.
    # So degree dominates.
    # Maybe we want spectral to dominate?
    # "Structurally peripheral" is the key.
    
    # Let's try a balance where spectral is significant.
    # score = -1.0 * v * n - 0.1 * deg
    # v * n scales v to sum ~ n? No, sum v^2 = 1.
    
    # Let's stick to a simple linear combination.
    # If we use -v, we prioritize peripheral nodes.
    # If we use -deg, we prioritize low degree nodes.
    
    # The current best (1.0150) used -deg + 0.5 * avg_nbr_deg.
    # This is a local 1-hop measure.
    # Spectral is a global measure.
    
    # Let's try: score = -deg - 5 * v
    # This adds a global penalty for being central.
    
    # Alternative: Use the eigenvector as the PRIMARY signal.
    # score = -v
    # This might be too aggressive if the eigenvector doesn't correlate well with degree.
    # But for ER graphs, high degree nodes ARE central.
    
    # Let's use a combination that respects the known good direction (low degree)
    # but adds the spectral twist.
    
    alpha = 1.0
    beta = 5.0
    
    # Normalize v to have mean 0 and std 1? No, just use it.
    # To make beta comparable to deg, we might need to scale v.
    # Max deg is ~20-30.
    # Max v is ~0.2.
    # 5 * 0.2 = 1.0.
    # So spectral term is small compared to degree.
    # Let's increase beta to 20.
    # 20 * 0.2 = 4.0.
    # Now it's comparable to degree variations.
    
    beta = 20.0
    score = -deg - beta * v
    
    # Safety deadline check
    if time.time() - start_time > 145:
        # Fallback to simple LDF if we ran out of time (unlikely)
        return -deg
        
    return score
