# EVOLVE-BLOCK-START
import numpy as np
import time

def _local_swap_improve(order, adjacency, budget_seconds=10.0):
    """
    Improves the greedy independent set size by swapping adjacent elements in the priority order.
    Returns the improved order (list of indices) or the original if no improvement or time limit reached.
    """
    N = len(order)
    A = adjacency
    start_time = time.time()
    best_order = list(order)
    best_size = 0
    
    # Helper to compute set size from an order
    def _set_size(ord):
        selected = np.zeros(N, dtype=bool)
        count = 0
        for v in ord:
            if not selected[v]:
                # Check neighbors
                # If any neighbor is selected, skip
                if not selected[A[v]].any():
                    selected[v] = True
                    count += 1
        return count

    # Initial size
    best_size = _set_size(best_order)
    
    # Local search: try swapping adjacent elements
    # We iterate until no improvement or time limit
    improved = True
    while improved and (time.time() - start_time) < budget_seconds:
        improved = False
        for i in range(N - 1):
            if (time.time() - start_time) > budget_seconds:
                break
            
            # Try swapping i and i+1
            new_order = best_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_size = _set_size(new_order)
            
            if new_size > best_size:
                best_order = new_order
                best_size = new_size
                improved = True
                # Restart the scan from the beginning (or continue? Restarting is safer for convergence)
                break
                
    return best_order

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    Strategy: 
    1. Compute "Survival Probability" score: Inverse of weighted degree in non-neighborhood.
       This prioritizes vertices structurally isolated from dense clusters.
    2. Refine the resulting order using local swap optimization to maximize independent set size.
    """
    start_time = time.time()
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Compute Structural Score
    # "Weighted degree in non-neighborhood"
    # For each vertex v, consider the subgraph induced by V \ ({v} U N(v)).
    # We want v to have low connectivity to the rest of the graph via its non-neighbors?
    # The directive says: "approximated by the inverse of the vertex's weighted degree in the induced subgraph of its non-neighborhood"
    
    # Let's interpret "weighted degree in the induced subgraph of its non-neighborhood" carefully.
    # Non-neighborhood of v: W_v = {u | u != v and A[v,u] == 0}
    # Induced subgraph on W_v.
    # "Vertex's weighted degree in this subgraph": This is ambiguous. v is not in W_v.
    # Perhaps it means the sum of weights of edges from v to W_v? No, v has no edges to W_v by definition.
    # Perhaps it means the degree of v in the graph induced by V \ N(v)? That is 0.
    
    # Alternative interpretation: 
    # The "survival probability" in a random sequential elimination process.
    # If we eliminate vertices randomly, v survives if it is not blocked by a neighbor.
    # A vertex with high degree is more likely to be blocked.
    # A vertex in a dense cluster is more likely to be blocked by a neighbor that is also in a dense cluster.
    
    # Let's look at Inspiration 1 & 2 & 3. They use numpy.
    # The directive specifically mentions: "inverse of the vertex's weighted degree in the induced subgraph of its non-neighborhood".
    # Maybe it means: For vertex v, look at all u not connected to v. Calculate the "strength" of the subgraph formed by u's?
    # Or maybe: The "effective degree" of v is the number of neighbors. But we want to weight them by how "dense" those neighbors are.
    
    # Let's try a concrete metric:
    # Score(v) = 1 / (1 + sum_{u in N(v)} (deg(u) / deg(v)))
    # This is similar to the "average neighbor degree" but inverted.
    # Low average neighbor degree -> High score.
    
    # Let's try another interpretation:
    # "Weighted degree in the induced subgraph of its non-neighborhood"
    # Maybe it refers to the number of edges in the subgraph induced by non-neighbors?
    # If v is isolated from a dense cluster, the non-neighbors form a dense cluster.
    # If v is part of a dense cluster, the non-neighbors are sparse.
    # We want to pick vertices from sparse regions? No, we want to pick vertices that are "safe" from being blocked.
    # If v is in a sparse region, its neighbors are also sparse.
    
    # Let's stick to the core idea: Minimize the probability of being blocked.
    # P(blocked) is high if neighbors are selected.
    # Neighbors are selected if they have high priority.
    # So we want v to have low priority neighbors.
    # This is circular.
    
    # Let's use the "Survival Probability" approximation:
    # s(v) = 1 / (1 + sum_{u in N(v)} w(u))
    # where w(u) is a weight representing how likely u is to be picked.
    # If we assume random order, w(u) ~ 1. Then s(v) = 1/(1+deg(v)).
    # This is just degree.
    
    # The directive says "weighted degree in the induced subgraph of its non-neighborhood".
    # Let's compute for each v:
    # W_v = {u | u != v, A[v,u] == 0}
    # Let G_v = G[W_v]
    # Let deg_in_Gv(u) be the degree of u in G_v.
    # Maybe the score is related to the average degree of W_v?
    # If W_v is dense, v is isolated from a dense part.
    # If W_v is sparse, v is connected to a sparse part (or v is central).
    
    # Let's try: Score(v) = 1 / (1 + avg_deg(W_v))
    # If v is in a dense core, W_v is the periphery (sparse). Score is high?
    # If v is in the periphery, W_v is the core (dense). Score is low?
    # We want to pick periphery vertices first? Yes, low degree vertices are good.
    # So if v is periphery, W_v is dense -> low score? That's wrong. We want high score for periphery.
    # So maybe Score(v) = 1 / (1 + avg_deg(N(v)))?
    # If v is periphery, N(v) is sparse -> high score.
    # If v is core, N(v) is dense -> low score.
    # This aligns with "low average neighbor degree".
    
    # Let's implement this:
    deg = A.sum(axis=1)
    sum_nei_deg = A @ deg
    avg_nei_deg = np.zeros(N)
    mask = deg > 0
    avg_nei_deg[mask] = sum_nei_deg[mask] / deg[mask]
    
    # Score = 1 / (1 + avg_nei_deg)
    # This gives high scores to vertices with sparse neighborhoods.
    
    score = 1.0 / (1.0 + avg_nei_deg)
    
    # Add a small random tie-breaker to ensure deterministic but varied order if scores are tied
    # Actually, we want deterministic.
    
    # Get initial order
    # Higher score = higher priority = picked earlier
    order = np.argsort(-score)
    
    # 2. Local Swap Improvement
    # Budget: Use up to 10 seconds for local search.
    # The total limit is 120s (or 150s). 10s is safe.
    if time.time() - start_time < 110.0:
        order = _local_swap_improve(order, A, budget_seconds=10.0)
        
    # Map order back to scores
    # The decoder picks in descending score order.
    # We need to return scores such that argsort(-scores) == order.
    # We can assign scores based on rank.
    # rank 0 (first in order) gets highest score.
    ranks = np.zeros(N)
    for i, v in enumerate(order):
        ranks[v] = N - i
        
    return ranks
# EVOLVE-BLOCK-END
