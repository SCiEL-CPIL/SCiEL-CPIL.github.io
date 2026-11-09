"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Convert to float for computations
    adj = adjacency.astype(float)
    
    # Compute PageRank with damping factor 0.85
    # PageRank gives global importance based on connectivity structure
    # For MIS, we want to pick vertices that are "well-connected" in the sense
    # of being central but not too peripheral
    
    damping = 0.85
    # Initialize PageRank
    pr = np.ones(n) / n
    
    # Power iteration for PageRank
    for _ in range(50):
        if time.time() > deadline:
            break
        # Compute out-degree (for undirected graph, this is just degree)
        out_deg = adj.sum(axis=1)
        # Avoid division by zero
        out_deg_safe = np.where(out_deg > 0, out_deg, 1.0)
        
        # New PageRank
        new_pr = (1 - damping) / n + damping * (adj.T / out_deg_safe[:, np.newaxis] * pr).sum(axis=1)
        pr = new_pr
    
    # Normalize PageRank
    pr = pr / pr.sum()
    
    # Compute core numbers (k-core decomposition)
    # Core number indicates how "deep" a vertex is in the core structure
    core_numbers = _compute_core_numbers(adj)
    
    # Normalize core numbers
    max_core = core_numbers.max() if n > 0 else 1
    core_norm = core_numbers / (max_core + 1)
    
    # Compute degree centrality
    degrees = adj.sum(axis=1)
    max_deg = degrees.max() if n > 0 else 1
    deg_norm = degrees / (max_deg + 1)
    
    # Combine signals:
    # - PageRank: global importance (higher = more central)
    # - Core number: structural depth (higher = more central in core)
    # - Degree: local connectivity (we want moderate degree, not too high or low)
    # 
    # For MIS, vertices with moderate degree are often good choices:
    # - Too low degree: isolated, easy to pick but doesn't help much
    # - Too high degree: picking it removes many neighbors
    # 
    # We want to prioritize vertices that are structurally central (high PageRank,
    # high core number) but not excessively high degree.
    
    # Use a weighted combination
    # Higher PageRank and core number -> higher priority
    # Moderate degree -> higher priority (peak around median degree)
    
    median_deg = np.median(degrees) if n > 0 else 1
    # Degree penalty: penalize very high or very low degrees relative to median
    deg_penalty = 1.0 / (1.0 + np.abs(deg_norm - 0.5) * 2.0)
    
    # Combine
    weights = np.array([0.4, 0.3, 0.3])  # PageRank, core, degree-adjusted
    scores = weights[0] * pr + weights[1] * core_norm + weights[2] * deg_penalty
    
    # The greedy decoder picks in descending order, so higher score = picked first
    # Return negative scores if we want lowest first, but the task says
    # "descending score order", so higher score should be picked first
    return scores


def _compute_core_numbers(adj):
    """Compute core numbers using peeling algorithm."""
    n = len(adj)
    if n == 0:
        return np.array([])
    
    degrees = adj.sum(axis=1).astype(int).copy()
    alive = np.ones(n, dtype=bool)
    core = np.zeros(n, dtype=int)
    
    # Use a bucket approach for efficiency
    # Sort vertices by degree
    order = np.argsort(degrees)
    
    # Peeling process
    current_deg = degrees.copy()
    removed = np.zeros(n, dtype=bool)
    
    # Process vertices in order of increasing degree
    for v in order:
        if removed[v]:
            continue
        
        # Find the maximum k such that v has at least k neighbors with core >= k
        # Simplified: assign core number based on current degree at time of removal
        k = current_deg[v]
        core[v] = k
        
        # Remove v and update neighbors
        removed[v] = True
        neighbors = np.where(adj[v, :])[0]
        for u in neighbors:
            if not removed[u]:
                current_deg[u] -= 1
    
    return core
# EVOLVE-BLOCK-END
