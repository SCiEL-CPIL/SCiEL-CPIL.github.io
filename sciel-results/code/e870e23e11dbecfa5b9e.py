"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Convert to a more workable format if needed
    # adjacency is a bool matrix
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # We want to assign high priority to vertices that are likely to be in a large independent set.
    # Strategy: 
    # 1. Run a greedy MIS construction that picks low-degree vertices first (but with some randomization or tie-breaking).
    # 2. The vertices picked early in this greedy process are "safe" and should have high priority.
    # 3. However, the fixed decoder picks in descending score order. So if we give high scores to the greedy picks,
    #    the decoder will pick them in the same order, replicating the greedy MIS.
    # 4. To improve, we can use a multi-start or a more sophisticated heuristic to find a good independent set,
    #    then score vertices based on their "importance" or "centrality" within that set, or simply by the order they were picked.
    
    # A better approach: 
    # The fixed decoder is greedy by score. If we set scores such that the decoder's greedy choice matches a good MIS algorithm,
    # we get a good result.
    # A known good heuristic for MIS is: repeatedly pick the vertex with the minimum degree in the remaining graph.
    # If we assign scores inversely proportional to the degree at the time of removal in this process,
    # or simply by the order of removal (earlier = higher score), the decoder will mimic this.
    
    # Let's simulate the "Min-Degree" greedy MIS construction.
    # We'll record the order in which vertices are selected.
    # Then, we assign scores such that selected vertices have high scores, and unselected (neighbors of selected) have low scores.
    # Specifically, if vertex v is selected at step k (0-indexed), its score could be (n - k) or similar.
    # But the decoder picks in descending score. So the first picked should have the highest score.
    
    # Let's try: 
    # 1. Simulate min-degree greedy MIS.
    # 2. Record the selection order.
    # 3. Assign score = (n - index_in_selection_order) for selected vertices.
    # 4. For non-selected vertices, assign a very low score (e.g., -1) so they are never picked if avoidable.
    #    But wait, the decoder skips neighbors of picked vertices. So if we pick the greedy MIS, the decoder
    #    will also skip their neighbors. The remaining vertices are not in the greedy MIS.
    #    If we give them low scores, the decoder won't pick them, which is fine because they can't be added
    #    to the independent set without removing some already picked vertex (which the decoder won't do).
    #    Actually, the decoder picks in descending score. If we give non-MIS vertices low scores, they will be
    #    considered last. But if they are not neighbors of any picked vertex, they could be picked.
    #    However, in a maximal independent set, every non-selected vertex is a neighbor of some selected vertex.
    #    So if our greedy MIS is maximal, all non-selected vertices are neighbors of selected ones, and the decoder
    #    will skip them. So the decoder will exactly replicate the greedy MIS.
    
    # So the key is to find a good greedy MIS. Min-degree is standard. Can we do better?
    # Maybe a randomized min-degree, or a degree-based heuristic with some lookahead?
    
    # Let's implement min-degree greedy MIS with a heap for efficiency.
    # N=200 is small, so even O(N^2) is fine.
    
    def greedy_mis_min_degree(adj):
        n = len(adj)
        remaining = set(range(n))
        selected = []
        current_adj = adj.copy()
        
        while remaining:
            # Find vertex with min degree in remaining subgraph
            min_deg = float('inf')
            min_vertex = None
            for v in remaining:
                deg = current_adj[v, list(remaining)].sum()
                if deg < min_deg:
                    min_deg = deg
                    min_vertex = v
            
            selected.append(min_vertex)
            # Remove min_vertex and its neighbors from remaining
            neighbors = set(np.where(current_adj[min_vertex, :])[0])
            neighbors.add(min_vertex)
            remaining -= neighbors
            # Update current_adj (optional, since we recompute degrees each time)
            # For N=200, recomputing is fine.
        
        return selected
    
    # Run greedy MIS
    selected_order = greedy_mis_min_degree(adjacency)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return min-degree scores
        return -adjacency.sum(axis=1).astype(float)
    
    # Assign scores
    scores = np.full(n, -1000.0)  # Very low score for non-selected
    for idx, v in enumerate(selected_order):
        # Higher score for earlier selection
        scores[v] = n - idx
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
