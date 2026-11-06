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

def _fast_greedy_size(adjacency, available_mask):
    """
    Estimate MIS size of the subgraph defined by available_mask 
    using a simple Min-Degree heuristic.
    """
    n = adjacency.shape[0]
    avail = available_mask.copy()
    avail_indices = np.where(avail)[0]
    if len(avail_indices) == 0:
        return 0
        
    # Compute degrees in subgraph
    degs = np.sum(adjacency[:, avail], axis=1)
    degs = degs[avail_indices]
    
    # Sort available indices by degree (ascending)
    sorted_indices = avail_indices[np.argsort(degs)]
    
    # Greedy simulation on this order
    avail_sim = np.zeros(n, dtype=bool)
    avail_sim[avail] = True
    
    cnt = 0
    for v in sorted_indices:
        if avail_sim[v]:
            cnt += 1
            avail_sim[v] = False
            avail_sim[adjacency[v]] = False
            
    return cnt

def priority(adjacency):
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Pre-compute static features
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency.dot(degrees)
    
    # 1. Generate a diverse pool of candidate score vectors
    candidates = []
    
    # Base: Min Degree
    candidates.append(-degrees)
    
    # Weighted variations: -deg + lambda * neighbor_deg_sum
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
    if time.time() < deadline - 60:
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
        if time.time() > deadline - 50:
            break
        order = np.argsort(-score)
        size = _greedy_mis(adjacency, order)
        if size > best_size:
            best_size = size
            best_score = score
            
    # 4. Local Search Refinement (Adjacent Swaps + Random Kicks)
    # Start from the best order found above
    best_order = np.argsort(-best_score)
    
    # Use a slightly larger safety margin for local search to ensure we return
    ls_deadline = deadline - 5
    
    # Multi-pass local search with random kicks to escape local minima
    # We will iterate until no improvement is found in a full pass, or time runs out.
    
    # First, do a few passes of pure adjacent swaps to settle
    for _pass in range(10):
        if time.time() > ls_deadline:
            break
            
        improved = False
        for i in range(n - 1):
            if time.time() > ls_deadline:
                break
                
            # Swap i and i+1
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_size = _greedy_mis(adjacency, new_order)
            
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True
        
        if not improved:
            # No improvement in this pass. 
            # Try random kicks to escape local minima.
            if time.time() < ls_deadline - 2:
                rng = np.random.default_rng()
                kick_improved = False
                for _ in range(50):
                    if time.time() > ls_deadline - 2:
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
                        kick_improved = True
                        break # Restart adjacent swap passes
                
                if kick_improved:
                    continue # Go back to adjacent swaps
                    
            break # No improvement even with random kicks, stop.

    # 5. Convert best_order back to scores
    # The greedy decoder picks vertices in descending score order.
    # Vertex at index 0 in best_order should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
# EVOLVE-BLOCK-END
