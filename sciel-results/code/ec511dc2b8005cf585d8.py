"""Combined program for Maximum Independent Set greedy scoring.
Uses dynamic heuristics and spectral methods for initial orders,
with fast incremental local search (swap and insertion moves).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority, ascending)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    return count

def _greedy_size_incremental(A, order, pos1, pos2):
    """Fast evaluation of a single swap at positions pos1 and pos2 in the order.
    Only re-simulates from min(pos1, pos2) onward.
    """
    n = len(order)
    lo = min(pos1, pos2)
    if lo >= n:
        return -1

    available = np.ones(n, dtype=bool)
    count = 0
    for i in range(lo):
        v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False

    v1, v2 = order[pos1], order[pos2]
    for i in range(lo, n):
        if i == pos1:
            v = v2
        elif i == pos2:
            v = v1
        else:
            v = order[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False
    return count

def _greedy_size_insertion(A, order, old_pos, new_pos):
    """Evaluate moving element from old_pos to new_pos.
    Re-simulates from min(old_pos, new_pos) onward.
    """
    n = len(order)
    lo = min(old_pos, new_pos)
    if lo >= n:
        return -1

    # Build the modified order
    elem = order[old_pos]
    if old_pos < new_pos:
        # Remove from old_pos, insert at new_pos
        # Elements from old_pos+1 to new_pos shift left by 1
        modified = list(order)
        del modified[old_pos]
        modified.insert(new_pos, elem)
    else:
        # Remove from old_pos, insert at new_pos
        # Elements from new_pos to old_pos-1 shift right by 1
        modified = list(order)
        del modified[old_pos]
        modified.insert(new_pos, elem)

    available = np.ones(n, dtype=bool)
    count = 0
    for i in range(lo):
        v = modified[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False

    for i in range(lo, n):
        v = modified[i]
        if available[v]:
            count += 1
            available[v] = False
            available[A[v]] = False
    return count

def _dynamic_min_degree_order(A, n):
    """Generate order using dynamic min-degree heuristic."""
    available = np.ones(n, dtype=bool)
    order = np.zeros(n, dtype=int)
    deg = A.sum(axis=1).astype(int)
    
    for pos in range(n):
        if not available.any():
            break
        cur_deg = np.where(available, deg, np.inf)
        v = np.argmin(cur_deg)
        order[pos] = v
        available[v] = False
        neighbors = np.where(A[v])[0]
        deg[neighbors] -= 1
        
    return order

def _dynamic_max_neighbor_degree_order(A, n):
    """Generate order preferring vertices with low-degree neighbors."""
    available = np.ones(n, dtype=bool)
    order = np.zeros(n, dtype=int)
    deg = A.sum(axis=1).astype(int)
    
    for pos in range(n):
        if not available.any():
            break
        scores = np.full(n, np.inf)
        avail_idx = np.where(available)[0]
        for v in avail_idx:
            nbrs = np.where(A[v])[0]
            nbrs_avail = nbrs[available[nbrs]]
            if len(nbrs_avail) > 0:
                avg_nbr_deg = deg[nbrs_avail].mean()
            else:
                avg_nbr_deg = 0.0
            scores[v] = deg[v] + 0.5 * avg_nbr_deg
            
        v = np.argmin(scores)
        order[pos] = v
        available[v] = False
        neighbors = np.where(A[v])[0]
        deg[neighbors] -= 1
        
    return order

def _spectral_order(A, n, reverse=False):
    """Generate order based on spectral centrality."""
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        idx = np.argmax(eigvals)
        vec = np.real(eigvecs[:, idx])
        vec = vec - vec.min()
        vec = vec / (vec.max() + 1e-9)
        
        if reverse:
            return np.argsort(vec, kind='stable')
        else:
            return np.argsort(-vec, kind='stable')
    except:
        return np.arange(n)

def _spectral_scores_power(A, n, rng, num_iters=30):
    """Spectral scores via power iteration on modified adjacency."""
    alpha = 0.8
    M = np.eye(n) - alpha * A.astype(float)
    x = rng.random(n)
    x = x / (np.linalg.norm(x) + 1e-10)
    
    for _ in range(num_iters):
        x_new = M @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            break
        x = x_new / norm
    
    deg = A.sum(axis=1).astype(float)
    deg_norm = deg / (deg.max() + 1e-10)
    x_mean = x.mean()
    x_std = x.std() + 1e-10
    x_norm = (x - x_mean) / x_std
    combined = deg_norm - 0.3 * x_norm
    return combined

def _core_order(A, deg, n):
    """Generate order based on core decomposition (peel lowest degree first)."""
    order = np.zeros(n, dtype=int)
    remaining = set(range(n))
    deg_copy = deg.copy().astype(int)
    
    for pos in range(n):
        if not remaining:
            break
        min_deg = min(deg_copy[v] for v in remaining)
        candidates = [v for v in remaining if deg_copy[v] == min_deg]
        v = random.choice(candidates)
        order[pos] = v
        remaining.remove(v)
        for u in np.where(A[v])[0]:
            if u in remaining:
                deg_copy[u] -= 1
    return order

def _generate_heuristic_orders(A, deg, n, rng):
    """Generate a diverse set of candidate orders."""
    orders = []
    
    # 1. Static Ascending Degree
    orders.append(np.argsort(deg, kind='stable'))
    
    # 2. Static Ascending Degree with Noise
    noise = rng.random(n) * 0.5
    orders.append(np.argsort(deg + noise, kind='stable'))
    
    # 3. Dynamic Min-Degree
    orders.append(_dynamic_min_degree_order(A, n))
    
    # 4. Dynamic Max-Neighbor-Degree
    orders.append(_dynamic_max_neighbor_degree_order(A, n))
    
    # 5. Spectral (Low Centrality first)
    orders.append(_spectral_order(A, n, reverse=True))
    
    # 6. Spectral (High Centrality first)
    orders.append(_spectral_order(A, n, reverse=False))
    
    # 7. Random
    orders.append(rng.permutation(n))
    
    # 8. Core Number Order
    orders.append(_core_order(A, deg, n))
    
    # 9. Spectral power iteration based
    spectral_scores = _spectral_scores_power(A, n, rng, num_iters=20)
    orders.append(np.argsort(spectral_scores, kind='stable'))
    
    # 10. 2-hop neighborhood size
    two_hop = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) == 0:
            two_hop[v] = 0
        else:
            two_hop_nbrs = set()
            for u in nbrs:
                two_hop_nbrs.update(np.where(A[u])[0])
            two_hop_nbrs.discard(v)
            two_hop_nbrs.difference_update(nbrs)
            two_hop[v] = len(two_hop_nbrs)
    orders.append(np.argsort(two_hop, kind='stable'))
    
    # 11. Weighted random with degree bias
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    orders.append(perm)
    
    # 12. Perturbed degree order
    base_order = np.argsort(deg, kind='stable')
    perturbed = base_order.copy()
    for _ in range(n // 10):
        i, j = rng.integers(0, n, size=2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    orders.append(perturbed)
    
    # 13. Neighbor clustering score: prefer vertices whose neighbors are densely connected
    #     (picking such vertices blocks fewer potential future picks)
    neighbor_clustering = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) < 2:
            neighbor_clustering[v] = 0
        else:
            sub_A = A[np.ix_(nbrs, nbrs)]
            edges = sub_A.sum() / 2
            possible = len(nbrs) * (len(nbrs) - 1) / 2
            neighbor_clustering[v] = edges / possible if possible > 0 else 0
    
    # Combine: low degree + high neighbor clustering is good
    deg_norm = deg / (deg.max() + 1e-10)
    cluster_norm = neighbor_clustering / (neighbor_clustering.max() + 1e-10)
    combined_score = deg_norm - 0.5 * cluster_norm
    orders.append(np.argsort(combined_score, kind='stable'))
    
    # 14. Mean neighbor degree signal (from top-performing inspiration)
    mean_nbr_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            mean_nbr_deg[v] = deg[nbrs].mean()
        else:
            mean_nbr_deg[v] = 0
    
    mean_nbr_norm = mean_nbr_deg / (mean_nbr_deg.max() + 1e-10)
    deg_norm2 = deg / (deg.max() + 1e-10)
    combined2 = deg_norm2 + 0.3 * mean_nbr_norm
    orders.append(np.argsort(combined2, kind='stable'))
    
    return orders

def _local_search(A, start_order, n, rng, deadline):
    """Perform local search with swap and insertion moves using incremental evaluation.
    Uses adaptive move ordering and early termination.
    """
    current_order = start_order.copy()
    current_size = _greedy_size(A, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    max_no_improve = 200  # Reduced to allow more restarts
    
    # Track recent moves for bias
    recent_positions = {}
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        # Choose move type: swap (65%) or insertion (35%)
        if rng.random() < 0.65:
            # Swap move with position bias toward recent active positions
            if recent_positions and rng.random() < 0.3:
                # Sample from recent positions
                pos_list = list(recent_positions.keys())
                i = pos_list[rng.integers(0, len(pos_list))]
                j = rng.integers(0, n)
            else:
                i = rng.integers(0, n)
                j = rng.integers(0, n)
            
            if i == j:
                continue
            
            new_size = _greedy_size_incremental(A, current_order, i, j)
            
            if new_size > current_size:
                current_order[i], current_order[j] = current_order[j], current_order[i]
                current_size = new_size
                no_improve = 0
                # Update recent positions
                recent_positions[i] = time.time()
                recent_positions[j] = time.time()
                # Prune old entries
                cutoff = time.time() - 2.0
                for k in list(recent_positions.keys()):
                    if recent_positions[k] < cutoff:
                        del recent_positions[k]
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve += 1
        else:
            # Insertion move
            old_pos = rng.integers(0, n)
            new_pos = rng.integers(0, n)
            if old_pos == new_pos:
                continue
            
            new_size = _greedy_size_insertion(A, current_order, old_pos, new_pos)
            
            if new_size > current_size:
                # Apply the move
                elem = current_order[old_pos]
                if old_pos < new_pos:
                    current_order = np.delete(current_order, old_pos)
                    current_order = np.insert(current_order, new_pos, elem)
                else:
                    current_order = np.delete(current_order, old_pos)
                    current_order = np.insert(current_order, new_pos, elem)
                current_size = new_size
                no_improve = 0
                recent_positions[old_pos] = time.time()
                recent_positions[new_pos] = time.time()
                cutoff = time.time() - 2.0
                for k in list(recent_positions.keys()):
                    if recent_positions[k] < cutoff:
                        del recent_positions[k]
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve += 1

    return best_order, best_size

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    deg = A.sum(axis=1).astype(float)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse candidate orders
    candidates = _generate_heuristic_orders(A, deg, n, rng)
    
    best_size = -1
    best_order = None
    top_candidates = []
    
    # Evaluate all initial candidates
    for order in candidates:
        if time.time() > deadline - 40:
            break
        size = _greedy_size(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
        top_candidates.append((size, order))
        
    if best_order is None:
        best_order = np.argsort(deg, kind='stable')
        best_size = _greedy_size(A, best_order)

    # Sort candidates by size to pick the best ones for local search
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Local search from the best candidates
    ls_deadline = deadline - 5.0
    
    # 1. Local search from the absolute best
    best_order, best_size = _local_search(A, best_order, n, rng, ls_deadline)
    
    # 2. If time permits, local search from the 2nd best
    if len(top_candidates) > 1 and time.time() < ls_deadline - 12:
        _, second_best = top_candidates[1]
        o2, s2 = _local_search(A, second_best, n, rng, ls_deadline)
        if s2 > best_size:
            best_size = s2
            best_order = o2

    # 3. If time permits, local search from the 3rd best
    if len(top_candidates) > 2 and time.time() < ls_deadline - 8:
        _, third_best = top_candidates[2]
        o3, s3 = _local_search(A, third_best, n, rng, ls_deadline)
        if s3 > best_size:
            best_size = s3
            best_order = o3

    # 4. If time permits, local search from the 4th best
    if len(top_candidates) > 3 and time.time() < ls_deadline - 5:
        _, fourth_best = top_candidates[3]
        o4, s4 = _local_search(A, fourth_best, n, rng, ls_deadline)
        if s4 > best_size:
            best_size = s4
            best_order = o4

    # Convert best_order to priority vector
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
