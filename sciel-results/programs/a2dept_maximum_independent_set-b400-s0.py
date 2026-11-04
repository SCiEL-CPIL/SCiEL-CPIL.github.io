import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Primary: Lowest Degree (vertices with fewer neighbors are better candidates).
    2. Secondary: Prefer vertices whose neighbors have high degree.
       Rationale: Selecting a vertex that is adjacent to hubs is beneficial because
       it "blocks" the hub. Since the hub has many neighbors, blocking it prevents
       the hub from being selected, which would otherwise block many other vertices.
       By blocking the hub, we keep its other neighbors (which might be low-degree)
       available for selection.
       
    We use a weighted sum: Score = -(w1 * Degree + w2 * Sum(Degree(Neighbors)))
    
    We tune weights to ensure Degree is the primary factor but Neighbor Degree Sum
    breaks ties effectively.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Calculate sum of degrees of neighbors
    # neighbor_deg_sum[i] = sum_{j in N(i)} deg[j]
    # This is a matrix-vector product: adj @ deg
    neighbor_deg_sum = adj @ deg
    
    # 3. Combine scores
    # We want higher score (less negative) for lower degree and lower neighbor_deg_sum?
    # Wait, let's re-evaluate the secondary factor.
    
    # If I pick vertex u, I remove u and N(u).
    # The cost is roughly 1 + deg(u).
    # But if u is connected to a hub h, picking u removes h.
    # If h were picked, it would remove 1 + deg(h) vertices.
    # So picking u "saves" the potential loss of h's other neighbors?
    
    # Let's look at standard MIS heuristics.
    # LDF (Least Degree First) is strong.
    # Another heuristic: Pick vertex with minimum degree. Ties broken by... ?
    
    # Consider the "impact" of picking a vertex.
    # Impact = number of vertices removed = 1 + deg(u).
    # We want to minimize Impact.
    # So primary key: deg(u).
    
    # Secondary key:
    # If two vertices have the same degree, which is better?
    # Suppose u and v have deg(u) == deg(v).
    # If we pick u, we block N(u).
    # If we pick v, we block N(v).
    # Ideally, we want to block vertices that are "expensive" to pick (high degree)
    # so that they don't get picked later and block many others?
    # No, if a high-degree vertex is blocked, it's a good thing.
    # So we prefer u if N(u) contains high-degree vertices.
    # Thus, we want to MAXIMIZE sum(deg(N(u))).
    # So the score should be:
    # Score = - (w1 * deg(u) - w2 * sum(deg(N(u))))
    # Or equivalently:
    # Score = -w1 * deg(u) + w2 * neighbor_deg_sum(u)
    
    # Let's test this logic.
    # If u is connected to a hub (high deg), neighbor_deg_sum is high.
    # Score becomes more positive (better).
    # This encourages picking vertices adjacent to hubs.
    
    # We need to balance w1 and w2.
    # deg range: [0, n-1]
    # neighbor_deg_sum range: [0, (n-1)*(n-1)] roughly, but typically much lower.
    # For ER graphs, deg ~ p*n. neighbor_deg_sum ~ p*n * p*n = p^2 * n^2.
    # If p is small (e.g., 0.05, n=200, deg~10), neighbor_deg_sum ~ 100.
    # deg diff of 1 should likely outweigh small neighbor_deg_sum diffs?
    # Or maybe neighbor_deg_sum is the better tie-breaker.
    
    # Let's try w1 = 1.0 and w2 = 0.1 or similar.
    # Actually, let's look at the previous successful program.
    # It used: Score = -(10.0 * deg + 1.0 * neighbor_inv_sum)
    # It used inverse degree sum.
    # My new idea uses direct degree sum.
    
    # Let's try: Score = -(1.0 * deg - 0.01 * neighbor_deg_sum)
    # This makes deg the primary driver.
    
    # Alternative: Use a normalized approach.
    # Normalize deg and neighbor_deg_sum to [0, 1] or similar?
    # max_deg = deg.max()
    # max_nds = neighbor_deg_sum.max()
    
    # Let's stick to linear combination with tuned weights.
    # Based on intuition:
    # Primary: Minimize deg.
    # Secondary: Maximize neighbor_deg_sum (to block hubs).
    
    # Score = -deg + alpha * neighbor_deg_sum
    # If alpha is too large, we might pick a high-degree vertex if it's connected to a huge hub.
    # We want deg to dominate.
    # Let's set alpha = 0.05.
    
    score = -deg + 0.05 * neighbor_deg_sum
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return score
