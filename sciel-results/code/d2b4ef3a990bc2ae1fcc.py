import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute neighbor sets as sorted arrays for fast vectorized blocking
    neighbor_arrays = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(nei) for nei in neighbor_arrays])
    
    def greedy_mis_size_vectorized(order):
        """
        Vectorized greedy MIS size evaluation.
        order: array of vertices in priority order (first = highest priority)
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if len(neighbor_arrays[v]) > 0:
                    selected[neighbor_arrays[v]] = True
        return count
    
    def greedy_mis_size_fast(order):
        """
        Optimized greedy MIS using numpy boolean masking.
        """
        remaining = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if remaining[v]:
                count += 1
                # Block v and all its neighbors
                block = np.zeros(n, dtype=bool)
                block[v] = True
                if len(neighbor_arrays[v]) > 0:
                    block[neighbor_arrays[v]] = True
                remaining[~block] = remaining[~block] & ~block
                remaining[block] = False
        return count
    
    # Strategy: Layered removal scoring
    # Iteratively select the vertex with the best "independent potential" in the remaining graph,
    # remove it and its neighbors, and assign scores based on removal layer.
    # Higher layer = picked later = lower score.
    
    def compute_layered_scores():
        """
        Compute scores via layered removal.
        At each step, pick the vertex with minimum degree in remaining graph
        (standard MIS heuristic: low-degree vertices are good to include).
        Score = (n_layers - layer) so early-picked vertices get high scores.
        """
        remaining = np.ones(n, dtype=bool)
        remaining_count = n
        scores = np.zeros(n)
        layer = 0
        max_layer = 0
        
        while remaining_count > 0:
            # Compute degrees in remaining subgraph
            # For each remaining vertex, count neighbors that are also remaining
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Compute degree within remaining for each remaining vertex
            rem_degrees = np.zeros(n, dtype=np.int32)
            for v in rem_indices:
                if len(neighbor_arrays[v]) > 0:
                    rem_degrees[v] = np.sum(remaining[neighbor_arrays[v]])
            
            # Among remaining vertices, pick the one with minimum degree
            # Tie-break: lower original degree, then lower index for determinism
            min_deg = np.min(rem_degrees[rem_indices])
            candidates = rem_indices[rem_degrees[rem_indices] == min_deg]
            # Tie-break by original degree, then index
            cand_orig_deg = degrees[candidates]
            min_orig = np.min(cand_orig_deg)
            final_cands = candidates[cand_orig_deg == min_orig]
            v = np.min(final_cands)
            
            scores[v] = n - layer  # High score for early selection
            max_layer = max(max_layer, layer)
            layer += 1
            
            # Remove v and its neighbors from remaining
            remaining[v] = False
            remaining_count -= 1
            if len(neighbor_arrays[v]) > 0:
                for u in neighbor_arrays[v]:
                    if remaining[u]:
                        remaining[u] = False
                        remaining_count -= 1
        
        return scores, max_layer
    
    # Initial order from layered removal
    initial_scores, _ = compute_layered_scores()
    initial_order = np.argsort(-initial_scores).astype(int)
    
    current_order = initial_order
    current_size = greedy_mis_size_vectorized(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    # Local Search: Simulated Annealing on the permutation
    # With vectorized evaluation, we can do many more iterations
    
    num_restarts = 8
    steps_per_restart = 50000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            order = initial_order.copy()
        else:
            # Perturb the best order found so far
            order = best_order.copy()
            # Perform a few random swaps to escape local optimum
            for _ in range(15):
                i, j = rng.integers(0, n, 2)
                order[i], order[j] = order[j], order[i]
            
        current_size = greedy_mis_size_vectorized(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order.copy()
            
        temp = 2.0
        cooling = 0.9997
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size_vectorized(order)
            
            delta = current_size - new_size  # Positive if we got worse
            
            if delta < 0:
                # Improvement
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order.copy()
            elif delta == 0:
                # Neutral
                current_size = new_size
            else:
                # Worse
                # Accept with probability exp(-delta / temp)
                if rng.random() < np.exp(-delta / temp):
                    current_size = new_size
                else:
                    # Revert
                    order[i], order[j] = order[j], order[i]
                    
            temp *= cooling
            if temp < 0.01:
                break
                
    # Convert best_order to scores
    # The decoder sorts by descending score.
    # So the first element in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
