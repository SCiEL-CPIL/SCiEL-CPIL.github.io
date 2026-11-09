"""Strategy: Fast greedy with multi-start local search on vertex ordering, enhanced with
spectral partitioning to identify global structure. We compute the Fiedler vector (second
eigenvector of the Laplacian) to find a natural bipartition, then use that to bias the
scoring so that vertices on the 'sparse' side of the cut are prioritized. We combine this
with local degree/coreness heuristics and refine via simulated annealing on the ordering.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh

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
    Compute the Fiedler vector to get a soft partition of the graph.
    Returns a vector where values indicate which 'side' of the cut a vertex is on.
    """
    n = adjacency.shape[0]
    if n < 3:
        return np.zeros(n)
    
    # Build sparse Laplacian
    L = sparse_laplacian(csr_matrix(adjacency))
    
    try:
        # Find the second smallest eigenvalue and eigenvector
        # k=2 because the smallest eigenvalue is 0 (constant vector)
        vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
        # The eigenvector corresponding to the second smallest eigenvalue
        # eigsh returns eigenvalues in ascending order when using which='LM' with sigma=0?
        # Actually, with sigma=0, it finds eigenvalues closest to 0.
        # Let's just take the one that is not constant.
        
        # Find the index of the eigenvalue that is not zero (or smallest non-zero)
        # For a connected graph, only the first is 0.
        # If disconnected, there might be multiple zeros.
        # We want the Fiedler vector, which is the eigenvector of the smallest positive eigenvalue.
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        # Skip the first one (should be ~0)
        if len(idx) > 1:
            fiedler_idx = idx[1]
        else:
            fiedler_idx = idx[0]
            
        fiedler = vecs[:, fiedler_idx]
        
        # Normalize so that mean is 0 and std is 1
        fiedler = (fiedler - fiedler.mean()) / (fiedler.std() + 1e-9)
        
        return fiedler
        
    except Exception:
        # Fallback: random partition
        return np.random.randn(n)

def _baseline_scores(adjacency, spectral_vec):
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
    
    # 4. Spectral-based: Prioritize vertices on the 'sparse' side of the cut
    # The Fiedler vector has positive and negative values.
    # We want to pick vertices that are 'isolated' in the spectral sense.
    # Typically, vertices with extreme Fiedler values are on the boundary.
    # We want to avoid boundary vertices? Or pick one side?
    # Let's try: score = -|fiedler| (pick vertices near the 'center' of the partition?)
    # Or: score = fiedler (pick one side) or -fiedler (pick the other side).
    # Actually, for MIS, we want a set of non-adjacent vertices.
    # A good heuristic is to pick vertices that have low degree and are on the 'sparse' side.
    
    # Normalize features
    def norm(x):
        return x / (x.max() + 1e-9)
        
    d_n = norm(degrees)
    nds_n = norm(neighbor_deg_sum)
    c_n = norm(coreness)
    
    # Spectral score: vertices with large negative Fiedler values are on one side,
    # large positive on the other. Let's try prioritizing the side with lower average degree.
    
    # Split vertices by sign of Fiedler
    pos_mask = spectral_vec > 0
    neg_mask = spectral_vec < 0
    
    if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
        avg_deg_pos = degrees[pos_mask].mean()
        avg_deg_neg = degrees[neg_mask].mean()
        
        # Prioritize the side with lower average degree
        if avg_deg_pos < avg_deg_neg:
            # Prefer positive side
            spec_score = spectral_vec
        else:
            # Prefer negative side
            spec_score = -spectral_vec
    else:
        spec_score = spectral_vec
    
    spec_n = norm(spec_score)
    
    # Combinations
    scores_list.append(-(0.5 * d_n + 0.5 * c_n))
    scores_list.append(-(0.7 * d_n + 0.3 * nds_n))
    scores_list.append(-(0.3 * d_n + 0.7 * nds_n))
    scores_list.append(-(d_n + nds_n + c_n))
    
    # Spectral combinations
    scores_list.append(-(0.5 * d_n + 0.5 * spec_n))
    scores_list.append(-(0.3 * d_n + 0.7 * spec_n))
    scores_list.append(-(0.7 * d_n + 0.3 * spec_n))
    scores_list.append(-(d_n + spec_n))
    scores_list.append(-(c_n + spec_n))
    
    # Random
    scores_list.append(np.random.randn(n))
    
    return scores_list

def _local_search_order(adjacency, initial_order, deadline):
    """
    Improve the order via local search (swaps) using simulated annealing.
    """
    n = adjacency.shape[0]
    current_order = initial_order.copy()
    current_count, _ = _greedy_decode(adjacency, current_order)
    
    best_order = current_order.copy()
    best_count = current_count
    
    adj_rows = [adjacency[i] for i in range(n)]
    
    no_improve_count = 0
    max_no_improve = 800
    T = 1.0
    cooling_rate = 0.998
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        i, j = np.random.choice(n, 2, replace=False)
        
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
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_count += 1
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return best_order, best_count

def _local_search_set(adjacency, initial_set, deadline):
    """
    Directly improve the independent set by adding/removing vertices.
    """
    n = adjacency.shape[0]
    current_set = initial_set.copy()
    current_size = np.sum(current_set)
    
    best_set = current_set.copy()
    best_size = current_size
    
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    while time.time() < deadline:
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
            if np.random.random() < 0.1:
                set_vertices = np.where(current_set)[0]
                if len(set_vertices) > 0:
                    u = set_vertices[np.random.randint(len(set_vertices))]
                    current_set[u] = False
                    current_size -= 1
                    
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
    
    # 1. Compute spectral partition
    spectral_vec = _compute_spectral_partition(adjacency)
    
    # 2. Generate baselines
    baseline_scores = _baseline_scores(adjacency, spectral_vec)
    
    best_count = -1
    best_order = None
    
    # Evaluate baselines
    for scores in baseline_scores:
        order = np.argsort(-scores, kind='stable')
        count, _ = _greedy_decode(adjacency, order)
        if count > best_count:
            best_count = count
            best_order = order.copy()
            
    # 3. Local Search on Order
    restarts = 6
    for r in range(restarts):
        if time.time() > deadline - 10.0:
            break
            
        if r == 0:
            init_order = best_order.copy()
        else:
            init_order = best_order.copy()
            for _ in range(15):
                i, j = np.random.choice(n, 2, replace=False)
                init_order[i], init_order[j] = init_order[j], init_order[i]
                
        slice_deadline = min(deadline - 10.0, time.time() + 15.0)
        ls_order, ls_count = _local_search_order(adjacency, init_order, slice_deadline)
        
        if ls_count > best_count:
            best_count = ls_count
            best_order = ls_order.copy()
            
    # 4. Local Search on Set (Polishing)
    if time.time() < deadline - 5.0:
        _, init_set = _greedy_decode(adjacency, best_order)
        ls_set, ls_count = _local_search_set(adjacency, init_set, deadline - 1.0)
        
        if ls_count > best_count:
            scores = np.zeros(n)
            scores[ls_set] = 1.0
            return scores
            
    # 5. Convert best_order to scores
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores

# EVOLVE-BLOCK-END
