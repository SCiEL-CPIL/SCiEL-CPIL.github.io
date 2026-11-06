"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbours of already picked vertices)
    yields a large independent set.
    
    Strategy:
    - We want to pick low-degree vertices early to preserve options for later.
    - However, simply picking lowest degree first can be suboptimal if a low-degree
      vertex is adjacent to many other low-degree vertices (clique-like structure).
    - We use a combination of:
        1. Degree (lower is better)
        2. "Influence" or "dominance": how many low-degree neighbours it has.
           A vertex that is adjacent to many low-degree vertices should be deprioritized
           because picking it would block many good candidates.
        3. A local density measure: number of edges among its neighbours.
           If its neighbours are densely connected, picking this vertex doesn't cost
           as much in terms of blocking many independent candidates.
    
    Specifically, we compute for each vertex v:
        degree[v] = number of neighbours
        low_degree_neighbours[v] = number of neighbours with degree <= median_degree
        neighbour_edge_density[v] = (number of edges among neighbours of v) / (max(1, degree[v]*(degree[v]-1)/2))
    
    Score = -degree[v] - alpha * low_degree_neighbours[v] + beta * neighbour_edge_density[v]
    
    The idea: 
    - -degree: prefer low degree
    - -alpha * low_degree_neighbours: prefer vertices that don't block many low-degree (good) vertices
    - +beta * neighbour_edge_density: if neighbours are tightly connected, they can't all be in the independent set anyway, 
      so blocking them is less costly; thus we can afford to pick such vertices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    degrees = adj.sum(axis=1).astype(float)
    
    # Median degree for thresholding "low degree"
    med_deg = np.median(degrees)
    low_deg_mask = degrees <= med_deg
    
    # Count low-degree neighbours for each vertex
    low_deg_neighbours = (adj @ low_deg_mask).astype(float)
    
    # Compute neighbour edge density
    # For each vertex v, count edges among its neighbours
    # This is: for each v, sum over pairs (u,w) both neighbours of v of adj[u,w]
    # Efficiently: (adj @ adj) gives for each pair (u,v) the number of common neighbours,
    # but we need edges among neighbours of each vertex.
    
    # For vertex v, the number of edges among its neighbours is:
    # (1/2) * sum_{u in N(v)} sum_{w in N(v), w != u} adj[u,w]
    # = (1/2) * sum_{u in N(v)} (degree[u] - 1 - (1 if u not adjacent to v... wait, u IS adjacent to v))
    # Actually, for u in N(v), the number of neighbours of u that are also in N(v) is:
    # (adj @ adj)[v, u] - 1 (subtracting v itself if v is a neighbour of u, which it is)
    # Wait, (adj @ adj)[v, u] = number of common neighbours of v and u.
    # Since u is a neighbour of v, v is a common neighbour of v and u? No.
    # Common neighbours of v and u are vertices w such that adj[v,w] and adj[u,w].
    # So (adj @ adj)[v, u] counts w in N(v) ∩ N(u).
    # For u in N(v), the number of neighbours of u that are also in N(v) is (adj @ adj)[v, u].
    # But we want edges among N(v), i.e., pairs (u,w) in N(v) with adj[u,w].
    # Sum over u in N(v) of (number of w in N(v) with adj[u,w]) = sum over u in N(v) of (adj @ adj)[v, u]
    # But this counts each edge twice, so divide by 2.
    
    common_neighbours = adj @ adj  # (adj @ adj)[v, u] = |N(v) ∩ N(u)|
    
    # For each vertex v, sum of (adj @ adj)[v, u] over u in N(v)
    # This is: (common_neighbours * adj).sum(axis=1)
    # Because we only sum over u where adj[v, u] is True.
    
    neighbour_edge_sums = (common_neighbours * adj).sum(axis=1)
    
    # Max possible edges among neighbours of v is degree[v]*(degree[v]-1)/2
    max_possible = degrees * (degrees - 1) / 2.0
    max_possible = np.maximum(max_possible, 1.0)  # Avoid division by zero
    
    neighbour_edge_density = neighbour_edge_sums / (2.0 * max_possible)
    
    # Normalize components to have comparable scales
    deg_norm = degrees / (n if n > 0 else 1)
    ldn_norm = low_deg_neighbours / (n if n > 0 else 1)
    ned_norm = neighbour_edge_density  # Already in [0, 1]
    
    # Combine scores
    alpha = 1.0  # Weight for low-degree neighbours
    beta = 0.5   # Weight for neighbour edge density
    
    score = -deg_norm - alpha * ldn_norm + beta * ned_norm
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to simple degree-based score
        return -degrees
    
    return score.astype(float)
# EVOLVE-BLOCK-END
