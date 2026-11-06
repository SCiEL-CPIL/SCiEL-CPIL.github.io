"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_mis(adjacency, order):
    """Run greedy MIS with given vertex order (descending priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency[v]] = False
    return count

def priority(adjacency):
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Pre-compute static features
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency.dot(degrees)
    
    # 1. Generate a diverse pool of candidate score vectors
    candidates = []
    
    # Base: Min Degree
    candidates.append(-degrees)
    
    # Weighted variations: -deg + lambda * neighbor_deg_sum
    # Positive lambda rewards high-neighbor-degree (bad for MIS usually, but maybe context dependent?)
    # Negative lambda penalizes high-neighbor-degree (good: prefer nodes with low-degree neighbors)
    for lam in [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]:
        candidates.append(-degrees + lam * neighbor_deg_sum)
        
    # Structural variations: 
    # -deg^2 (super-min-degree)
    candidates.append(-degrees**2)
    
    # -deg / (1 + neighbor_deg_sum) ?
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_score = -degrees / (1.0 + neighbor_deg_sum)
    candidates.append(ratio_score)
    
    # 2. 1-Step Lookahead Candidate
    # For each vertex v, score = 1 + estimated MIS size of (G - N[v])
    # Using Min-Degree greedy as the estimator for the subgraph.
    if time.time() < deadline - 30:
        lookahead_scores = np.zeros(n)
        for v in range(n):
            mask = np.ones(n, dtype=bool)
            mask[v] = False
            mask[adjacency[v]] = False
            
            # Get available indices
            avail_indices = np.where(mask)[0]
            if len(avail_indices) == 0:
                lookahead_scores[v] = 1.0
                continue
            
            # Compute degrees in subgraph for available nodes
            # deg_sub[v] = sum(adjacency[v, mask])
            degs = np.sum(adjacency[:, mask], axis=1)
            degs = degs[avail_indices]
            
            # Sort by degree (ascending)
            sorted_indices = avail_indices[np.argsort(degs)]
            
            # Simulate greedy on this order
            avail_sim = np.zeros(n, dtype=bool)
            avail_sim[mask] = True
            cnt = 0
            for u in sorted_indices:
                if avail_sim[u]:
                    cnt += 1
                    avail_sim[u] = False
                    avail_sim[adjacency[u]] = False
            
            lookahead_scores[v] = 1.0 + cnt
        
        candidates.append(lookahead_scores)
        
    # 3. Evaluate all candidates and pick the best one
    best_size = -1
    best_score = candidates[0]
    
    for score in candidates:
        if time.time() > deadline - 10:
            break
        order = np.argsort(-score)
        size = _greedy_mis(adjacency, order)
        if size > best_size:
            best_size = size
            best_score = score
            
    # 4. Local Search Refinement (Adjacent Swaps)
    # Start from the best order found above
    best_order = np.argsort(-best_score)
    
    # Optimization: Pre-compute neighbor masks for faster updates if possible?
    # For N=200, _greedy_mis is fast enough (~0.1ms per call).
    # We have ~10s left. 10s / 0.1ms = 100,000 calls.
    # N=200. One pass is 199 swaps. 
    # We can do many passes.
    
    # To speed up, we can try to be smart about swaps, but simple hill climbing is robust.
    # Let's add a "restart" mechanism if we get stuck? No, just multiple passes.
    
    # Use a slightly larger safety margin for local search to ensure we return
    ls_deadline = deadline - 2
    
    # Multi-pass local search
    # We will iterate until no improvement is found in a full pass, or time runs out.
    # To avoid infinite loops on plateaus, we can limit total passes or use random restarts if stuck.
    # Given the time budget, simple multi-pass is sufficient.
    
    for _pass in range(20):  # Max 20 passes
        if time.time() > ls_deadline:
            break
            
        improved = False
        for i in range(n - 1):
            if time.time() > ls_deadline:
                break
                
            # Swap i and i+1
            # Instead of copying the whole array, we can simulate the swap effect?
            # _greedy_mis takes an array. Copying 200 ints is cheap.
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_size = _greedy_mis(adjacency, new_order)
            
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True
            # We do NOT revert if worse, we just don't update best_order.
            # This is standard First-Improve or Best-Improve? 
            # This is First-Improve within the pass.
            # Actually, this loop structure updates best_order immediately if better.
            # This is a valid local search.
        
        if not improved:
            # No improvement in this pass. 
            # We could try a "shakeup" (random perturbation) to escape local minima, 
            # but for greedy MIS, simple swaps often suffice.
            # Let's try a few random swaps to escape local minima if time permits.
            if time.time() < ls_deadline - 1:
                # Random restarts within local search
                for _ in range(10):
                    if time.time() > ls_deadline - 1:
                        break
                    # Randomly swap two distant elements
                    idx1 = np.random.randint(0, n)
                    idx2 = np.random.randint(0, n)
                    if idx1 == idx2:
                        continue
                    new_order = best_order.copy()
                    new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
                    new_size = _greedy_mis(adjacency, new_order)
                    if new_size > best_size:
                        best_size = new_size
                        best_order = new_order
                        improved = True # Force another pass? Or just accept?
                        # Let's just accept and break out of random loop to do a proper pass
                        break
            
            if not improved:
                break # No improvement even with random kicks, stop.

    # 5. Convert best_order back to scores
    # The greedy decoder picks vertices in descending score order.
    # Vertex at index 0 in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
# EVOLVE-BLOCK-END
