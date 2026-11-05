"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Safety deadline: leave 5s margin before 120s hard limit
    deadline = start_time + 115.0
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(adj[i])[0] for i in range(n)]
    
    # 1. Initial Heuristic: Hub-Avoidance / Influence Network
    # Strategy: Prioritize low-degree vertices. 
    # Vertices adjacent to high-degree hubs are "valuable" because picking them
    # removes the hub from the candidate pool without the high cost of picking the hub itself.
    # However, the hub itself should be deprioritized.
    
    deg = adj.sum(axis=1).astype(np.float64)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = adj @ deg
    
    # Average degree of neighbors (local density proxy)
    # If a vertex is connected to a hub, this is high.
    avg_neighbor_deg = neighbor_deg_sum / (deg + 1e-6)
    
    # Heuristic Score Construction:
    # We want HIGH score for:
    # - Low degree (cheap to pick)
    # - High "hub-blocking potential" (neighbors are hubs, so picking us is efficient)
    #   
    # Let's define an "influence" score where we penalize the vertex's own degree
    # but reward it if it blocks high-degree nodes.
    # 
    # score = -degree(v) + beta * (sum of neighbor degrees) / (degree(v) + 1)
    # Actually, simply -degree is often strong. 
    # Let's use a combination:
    # Base: -degree
    # Refinement: + small weight * avg_neighbor_degree (to boost leaves of hubs)
    
    # Normalize features to [0, 1] for stable weighting
    if deg.max() > deg.min():
        norm_deg = (deg - deg.min()) / (deg.max() - deg.min())
    else:
        norm_deg = np.zeros_like(deg)
        
    if avg_neighbor_deg.max() > avg_neighbor_deg.min():
        norm_avg_ndeg = (avg_neighbor_deg - avg_neighbor_deg.min()) / (avg_neighbor_deg.max() - avg_neighbor_deg.min())
    else:
        norm_avg_ndeg = np.zeros_like(avg_neighbor_deg)
    
    # We want LOW degree (so -norm_deg) and HIGH neighbor degree (so +norm_avg_ndeg)
    # Weighting: Degree is the dominant factor in greedy MIS performance usually.
    scores = -0.8 * norm_deg + 0.2 * norm_avg_ndeg
    
    # 2. Fast Simulation Function
    def simulate(s):
        """
        Simulates the greedy decoder and returns the size of the independent set.
        """
        # argsort with stable=False is faster, but we need deterministic order for ties?
        # The decoder picks in descending score. Ties are broken by index? 
        # The task says "fixed greedy decoder". Usually this implies a stable sort or index-based tie-breaking.
        # Let's assume standard np.argsort behavior (quicksort, unstable) or stable?
        # To be safe and fast, we'll use the order provided by argsort.
        order = np.argsort(-s, kind='stable') # Stable is safer for reproducibility in local search
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                # Deactivate neighbors
                if len(neighbors_list[v]) > 0:
                    active[neighbors_list[v]] = False
        return count

    # 3. Local Search (Hill Climbing with Swaps)
    # The landscape of greedy MIS is highly rugged. Small changes in priority order
    # can lead to large changes in set size. Local search is effective here.
    
    best_size = simulate(scores)
    best_scores = scores.copy()
    
    # Time budget for search
    # We have ~115s total. Simulation is O(n^2) worst case (n=200 -> 40k ops).
    # Python loop overhead is high. Let's estimate iterations.
    # One simulation ~ 0.1-0.5ms? Let's assume 0.5ms.
    # 110s / 0.5ms = 220,000 iterations. This is plenty.
    
    rng = np.random.default_rng(42)
    
    # We will perform random swaps.
    # To improve convergence, we can bias swaps towards vertices that are close in rank
    # or just random. Random is simpler and works well for N=200.
    
    while time.time() < deadline:
        # Pick two random distinct vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        # Evaluate
        new_size = simulate(scores)
        
        if new_size > best_size:
            best_size = new_size
            best_scores = scores.copy()
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
            
        # Periodic check for time to avoid overshooting
        if time.time() > deadline - 2.0:
            break

    return best_scores
# EVOLVE-BLOCK-END
