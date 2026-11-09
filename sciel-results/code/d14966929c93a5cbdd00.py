"""Strategy: Fast greedy with multi-start local search on vertex ordering.
The key insight is that the greedy decoder is very sensitive to the ordering.
We use a fast baseline (Min-Degree) and then refine the order using a
local search that swaps adjacent or random vertices to improve the count.
We also try to directly optimize the independent set via a "repair" step
that attempts to swap vertices in/out to increase the set size.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_decode(adjacency, order):
    """
    Run the fixed greedy decoder given a specific vertex order (permutation).
    Returns: (count, selected_mask)
    """
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        available[adjacency[v]] = False
        
    return count, selected

def _evaluate_scores(adjacency, scores):
    """
    Evaluate a set of scores by sorting them and running the greedy decoder.
    Returns: count
    """
    order = np.argsort(-scores, kind='stable')
    count, _ = _greedy_decode(adjacency, order)
    return count

def _compute_core_numbers(adjacency):
    """
    Compute the core number for each vertex in the graph.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(int)
    
    deg = degrees.copy()
    core = np.zeros(n, dtype=int)
    
    remaining = np.ones(n, dtype=bool)
    
    for _ in range(n):
        if not np.any(remaining):
            break
        masked_degrees = np.where(remaining, deg, np.inf)
        v = np.argmin(masked_degrees)
        core[v] = deg[v]
        
        remaining[v] = False
        neighbors = np.where(adjacency[v])[0]
        active_neighbors = neighbors[remaining[neighbors]]
        if len(active_neighbors) > 0:
            deg[active_neighbors] -= 1
            
    return core

def _compute_spectral_partition(adjacency):
    """
    Compute spectral partition using the Fiedler vector (second-smallest eigenvector
    of the graph Laplacian). Returns:
    - partition: array of 0/1 indicating which side of the cut each vertex is on
    - fiedler_vec: the eigenvector itself
    """
    n = adjacency.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), np.zeros(n)
    
    # Compute Laplacian L = D - A
    degrees = adjacency.sum(axis=1).astype(float)
    L = np.diag(degrees) - adjacency.astype(float)
    
    # For numerical stability, use eigh for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback: return degree-based partition
        degrees_int = adjacency.sum(axis=1).astype(int)
        median_deg = np.median(degrees_int)
        partition = (degrees_int > median_deg).astype(int)
        return partition, np.zeros(n)
    
    # Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
    # Sort eigenvalues and take the second one
    idx = np.argsort(eigenvalues)
    fiedler_vec = eigenvectors[:, idx[1]]
    
    # Partition based on sign of Fiedler vector
    partition = (fiedler_vec >= 0).astype(int)
    
    return partition, fiedler_vec

def _baseline_scores(adjacency):
    """
    Generate several baseline score vectors.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees
    coreness = _compute_core_numbers(adjacency).astype(float)
    
    scores_list = []
    
    # 1. Min Degree (classic)
    scores_list.append(-degrees)
    
    # 2. Min Neighbor Degree Sum
    scores_list.append(-neighbor_deg_sum)
    
    # 3. Min Coreness
    scores_list.append(-coreness)
    
    # 4. Spectral partition: prioritize vertices in the smaller/sparser partition
    partition, fiedler_vec = _compute_spectral_partition(adjacency)
    # Count partition sizes
    part0_count = np.sum(partition == 0)
    part1_count = np.sum(partition == 1)
    
    # Prioritize the smaller partition (they are more constrained)
    # Within each partition, use degree as tiebreaker
    if part0_count <= part1_count:
        # Partition 0 is smaller or equal, prioritize it
        spectral_score = np.where(partition == 0, -degrees, degrees)
    else:
        # Partition 1 is smaller, prioritize it
        spectral_score = np.where(partition == 1, -degrees, degrees)
    scores_list.append(spectral_score)
    
    # 5. Combination of spectral and degree
    def norm(x):
        return x / (x.max() + 1e-9)
        
    d_n = norm(degrees)
    nds_n = norm(neighbor_deg_sum)
    c_n = norm(coreness)
    
    # Heuristic: low degree and low coreness are good
    scores_list.append(-(0.5 * d_n + 0.5 * c_n))
    scores_list.append(-(0.7 * d_n + 0.3 * nds_n))
    scores_list.append(-(0.3 * d_n + 0.7 * nds_n))
    scores_list.append(-(d_n + nds_n + c_n))
    
    # 6. Spectral + degree combination
    # Vertices in the sparser partition with low degree are most valuable
    if part0_count <= part1_count:
        in_small_part = (partition == 0).astype(float)
    else:
        in_small_part = (partition == 1).astype(float)
    
    scores_list.append(-(0.6 * in_small_part + 0.4 * d_n))
    scores_list.append(-(0.4 * in_small_part + 0.6 * d_n))
    
    # 7. Random
    scores_list.append(np.random.randn(n))
    
    return scores_list

def _local_search_order(adjacency, initial_order, deadline):
    """
    Improve the order via local search (swaps).
    We focus on swaps that might unblock a better vertex or block a worse one.
    """
    n = adjacency.shape[0]
    current_order = initial_order.copy()
    current_count, _ = _greedy_decode(adjacency, current_order)
    
    best_order = current_order.copy()
    best_count = current_count
    
    # Precompute adjacency rows for speed
    adj_rows = [adjacency[i] for i in range(n)]
    
    step = 0
    no_improve_count = 0
    max_no_improve = 500
    T = 1.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        # Choose a move: swap two random positions
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_count, _ = _greedy_decode(adjacency, current_order)
        
        delta = new_count - current_count
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_order = current_order.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_count += 1
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
        step += 1
        
    return best_order, best_count

def _local_search_set(adjacency, initial_set, deadline):
    """
    Directly improve the independent set.
    """
    n = adjacency.shape[0]
    current_set = initial_set.copy()
    current_size = np.sum(current_set)
    
    best_set = current_set.copy()
    best_size = current_size
    
    # Precompute neighbors
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    while time.time() < deadline:
        # Recompute blocked for simplicity (n=200 is small)
        blocked = np.zeros(n, dtype=bool)
        blocked[current_set] = True
        for v in np.where(current_set)[0]:
            blocked[neighbors[v]] = True
            
        candidates = np.where(~blocked)[0]
        
        if len(candidates) > 0:
            # Pick the one with min degree among candidates
            cand_degrees = [len(neighbors[c]) for c in candidates]
            min_idx = np.argmin(cand_degrees)
            v = candidates[min_idx]
            
            # Add v
            current_set[v] = True
            current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
                
        else:
            # No candidates to add. Try a swap?
            if np.random.random() < 0.1:
                set_vertices = np.where(current_set)[0]
                if len(set_vertices) > 0:
                    u = set_vertices[np.random.randint(len(set_vertices))]
                    current_set[u] = False
                    current_size -= 1
                    # Now try to add as many as possible greedily
                    blocked = np.zeros(n, dtype=bool)
                    blocked[current_set] = True
                    for v in np.where(current_set)[0]:
                        blocked[neighbors[v]] = True
                    candidates = np.where(~blocked)[0]
                    if len(candidates) > 0:
                        cand_degrees = [len(neighbors[c]) for c in candidates]
                        min_idx = np.argmin(cand_degrees)
                        v = candidates[min_idx]
                        current_set[v] = True
                        current_size += 1
                        
                        # Try to add another
                        blocked = np.zeros(n, dtype=bool)
                        blocked[current_set] = True
                        for v in np.where(current_set)[0]:
                            blocked[neighbors[v]] = True
                        candidates = np.where(~blocked)[0]
                        if len(candidates) > 0:
                            cand_degrees = [len(neighbors[c]) for c in candidates]
                            min_idx = np.argmin(cand_degrees)
                            v = candidates[min_idx]
                            current_set[v] = True
                            current_size += 1
                            
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set.copy()
            else:
                break
                
    return best_set, best_size

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # 1. Generate baselines (now includes spectral partition heuristics)
    baseline_scores = _baseline_scores(adjacency)
    
    best_count = -1
    best_order = None
    
    # Evaluate baselines
    for scores in baseline_scores:
        order = np.argsort(-scores, kind='stable')
        count, _ = _greedy_decode(adjacency, order)
        if count > best_count:
            best_count = count
            best_order = order.copy()
            
    # 2. Local Search on Order
    # Run multiple restarts if time permits
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Start from best_order or a perturbed version
        if r == 0:
            init_order = best_order.copy()
        else:
            # Perturb best_order
            init_order = best_order.copy()
            for _ in range(10):
                i, j = np.random.choice(n, 2, replace=False)
                init_order[i], init_order[j] = init_order[j], init_order[i]
                
        # Run local search with a time slice
        slice_deadline = min(deadline - 10.0, time.time() + 20.0)
        ls_order, ls_count = _local_search_order(adjacency, init_order, slice_deadline)
        
        if ls_count > best_count:
            best_count = ls_count
            best_order = ls_order.copy()
            
    # 3. Local Search on Set (Polishing)
    if time.time() < deadline - 5.0:
        _, init_set = _greedy_decode(adjacency, best_order)
        ls_set, ls_count = _local_search_set(adjacency, init_set, deadline - 1.0)
        
        if ls_count > best_count:
            # Convert set to scores
            scores = np.zeros(n)
            scores[ls_set] = 1.0
            return scores
            
    # 4. Convert best_order to scores
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores

# EVOLVE-BLOCK-END
