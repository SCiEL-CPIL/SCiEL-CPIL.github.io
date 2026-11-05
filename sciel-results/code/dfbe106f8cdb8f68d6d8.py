import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Direct vertex-removal optimization on the complement graph.
    - We want to find an ordering such that the greedy decoder picks a large independent set.
    - Equivalent view: repeatedly remove a vertex; we want to remove vertices in an order
      that minimizes the total "damage" (number of vertices that become blocked/removed).
    - Use a greedy heuristic: at each step, remove the vertex v that minimizes the number
      of its currently-present neighbors (i.e., the vertex whose removal blocks the fewest
      other vertices).
    - The removal order defines the priority: vertices removed later get higher scores,
      because in the greedy decoder, vertices with higher scores are considered first.
      Actually, let's think about this more carefully.
      
    Let me reconsider: The greedy decoder picks vertices in descending score order.
    If we want vertex v to be picked early, it should have a high score.
    In the removal process, if we remove vertex v early, it means v is "not important" 
    for the independent set (its neighbors will still be available).
    
    Actually, the correct mapping is:
    - The greedy decoder selects a vertex if it's not blocked.
    - A vertex is blocked if a neighbor was selected earlier.
    - So we want vertices that are "safe to select early" to have high scores.
    - A vertex is safe to select early if it has few neighbors that would also want to be selected.
    
    Alternative approach: Use the concept of "independent set density" in the neighborhood.
    
    Let me use a different framing: 
    - Run a greedy algorithm to build an independent set, but instead of picking the 
      highest-scoring available vertex, we pick the vertex whose selection allows the 
      most remaining vertices to still be selectable.
    - This is equivalent to: pick the vertex v that minimizes the number of its neighbors 
      that are currently "unblocked" (i.e., would be blocked if we pick v).
    
    Actually, the simplest effective strategy: 
    1. Initialize with degree-based ordering (low degree first).
    2. Refine using local search on the score vector directly, where we swap scores 
       of two vertices and measure the change in greedy output.
    
    But let's try the removal-based approach properly:
    - The greedy decoder's output depends only on the relative ordering of vertices.
    - We can think of it as: process vertices one by one; if not blocked, add to IS.
    - To maximize IS size, we want to process vertices in an order that minimizes blocking.
    - Heuristic: at each step, among all unprocessed vertices, pick the one with the 
      fewest unprocessed neighbors (this minimizes the number of vertices that will be 
      blocked in the future).
    
    This is essentially the "greedy minimum degree" algorithm applied to the problem 
    of finding a good ordering. Let's implement this and then refine with local search.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = adj.sum(axis=1).astype(int)
    
    def greedy_is_size(order):
        """Compute the size of the independent set produced by the greedy decoder."""
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count
    
    # Initial ordering: sort by degree ascending
    order = np.argsort(deg, kind='stable')
    current_size = greedy_is_size(order)
    best_order = order.copy()
    best_size = current_size
    
    # Local search: try swapping pairs of vertices in the ordering
    rng = np.random.default_rng(42)
    
    # Use a multi-start local search with adaptive temperature
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 30:
            break
        
        # For restart > 0, perturb the best order found so far
        if restart == 0:
            current_order = best_order.copy()
            current_size = best_size
        else:
            # Perturb: randomly swap a few pairs
            current_order = best_order.copy()
            n_swaps = min(10, n // 10)
            for _ in range(n_swaps):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    current_order[i], current_order[j] = current_order[j], current_order[i]
            current_size = greedy_is_size(current_order)
        
        # Hill climbing with occasional restarts
        improved = True
        while improved and (time.time() - start_time) < deadline - 30:
            improved = False
            # Try random swaps
            n_attempts = 500
            for _ in range(n_attempts):
                if time.time() - start_time > deadline - 30:
                    break
                
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                
                v_i = current_order[i]
                v_j = current_order[j]
                
                # Perform swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
                new_size = greedy_is_size(current_order)
                
                if new_size > current_size:
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_order = current_order.copy()
                    improved = True
                else:
                    # Swap back
                    current_order[i], current_order[j] = current_order[j], current_order[i]
    
    # Convert best_order to scores: vertex at position k gets score (n - k)
    # So the first vertex in the order gets the highest score
    scores = np.zeros(n)
    for k, v in enumerate(best_order):
        scores[v] = n - k
    
    return scores
