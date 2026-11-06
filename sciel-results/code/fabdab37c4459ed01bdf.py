"""Improved program for Maximum Independent Set greedy scoring.
Uses faster vectorized greedy evaluation, better local search moves,
and more diverse heuristic initializations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _greedy_size(A_bool, neighbor_lists, order):
    """Compute independent set size given a vertex order.
    
    A_bool: (n, n) bool adjacency matrix
    neighbor_lists: list of numpy arrays of neighbor indices
    order: array of vertex indices in selection order (first = highest priority)
    Returns: size of the greedy independent set.
    """
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    
    for v in order:
        if not available[v]:
            continue
        count += 1
        available[v] = False
        nbrs = neighbor_lists[v]
        if len(nbrs) > 0:
            available[nbrs] = False
    
    return count


def _compute_neighbor_lists(A_bool):
    """Precompute neighbor lists as list of numpy arrays."""
    n = A_bool.shape[0]
    neighbor_lists = []
    for v in range(n):
        nbrs = np.where(A_bool[v])[0]
        neighbor_lists.append(nbrs)
    return neighbor_lists


def _heuristic_min_nbr_degree_sum(A, deg, n):
    """Order vertices by (own degree + sum of neighbor degrees) ascending."""
    nbr_deg_sum = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            nbr_deg_sum[v] = deg[nbrs].sum()
    score = deg + nbr_deg_sum
    return np.argsort(score, kind='stable')


def _heuristic_min_avg_nbr_degree(A, deg, n):
    """Order by (own degree + avg neighbor degree) ascending."""
    avg_nbr_deg = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            avg_nbr_deg[v] = deg[nbrs].mean()
    score = deg + 0.5 * avg_nbr_deg
    return np.argsort(score, kind='stable')


def _core_order(A, deg, n, rng):
    """Core decomposition: repeatedly remove lowest-degree vertex."""
    order = np.zeros(n, dtype=int)
    remaining = set(range(n))
    deg_copy = deg.copy().astype(int)
    
    for pos in range(n):
        if not remaining:
            break
        min_deg = min(deg_copy[v] for v in remaining)
        candidates = [v for v in remaining if deg_copy[v] == min_deg]
        v = rng.choice(candidates)
        order[pos] = v
        remaining.remove(v)
        for u in np.where(A[v])[0]:
            if u in remaining:
                deg_copy[u] -= 1
    
    return order


def _spectral_order(A, n, rng, num_iters=40):
    """Spectral-based ordering using power iteration."""
    alpha = 0.7
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
    combined = deg_norm - 0.4 * x_norm
    return np.argsort(combined, kind='stable')


def _generate_candidates(A, deg, n, rng):
    """Generate diverse candidate orderings."""
    candidates = []
    candidates.append(np.argsort(deg, kind='stable'))
    noise = rng.random(n) * 0.3
    candidates.append(np.argsort(deg + noise, kind='stable'))
    candidates.append(_heuristic_min_nbr_degree_sum(A, deg, n))
    candidates.append(_heuristic_min_avg_nbr_degree(A, deg, n))
    candidates.append(_core_order(A, deg, n, rng))
    candidates.append(_spectral_order(A, n, rng))
    
    weights = np.exp(-deg / max(deg.max(), 1))
    weights /= weights.sum()
    perm = rng.choice(n, size=n, replace=False, p=weights)
    candidates.append(perm)
    candidates.append(rng.permutation(n))
    
    base = np.argsort(deg, kind='stable')
    perturbed = base.copy()
    for _ in range(n // 5):
        i, j = rng.integers(0, n, size=2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    candidates.append(perturbed)
    
    nbr_deg_sum = np.zeros(n)
    for v in range(n):
        nbrs = np.where(A[v])[0]
        if len(nbrs) > 0:
            nbr_deg_sum[v] = deg[nbrs].sum()
    combo = 0.5 * deg + 0.3 * nbr_deg_sum + 0.2 * rng.random(n)
    candidates.append(np.argsort(combo, kind='stable'))
    
    return candidates


def _do_insertion_move(order, i, j, n):
    """Perform insertion: remove element at position i, insert at position j.
    Returns new order array."""
    elem = order[i]
    remaining = np.delete(order, i)
    if i < j:
        insert_pos = j - 1
    else:
        insert_pos = j
    insert_pos = max(0, min(insert_pos, n - 1))
    return np.insert(remaining, insert_pos, elem)


def _local_search_insertion(A, neighbor_lists, start_order, n, rng, deadline,
                             max_no_improve=300):
    """Local search using insertion moves."""
    current_order = start_order.copy()
    current_size = _greedy_size(A, neighbor_lists, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        new_order = _do_insertion_move(current_order, i, j, n)
        new_size = _greedy_size(A, neighbor_lists, new_order)
        
        if new_size > current_size:
            current_order = new_order
            current_size = new_size
            no_improve = 0
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            no_improve += 1
    
    return best_order, best_size


def _local_search_swap(A, neighbor_lists, start_order, n, rng, deadline,
                       max_no_improve=300):
    """Local search using swap moves."""
    current_order = start_order.copy()
    current_size = _greedy_size(A, neighbor_lists, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        new_order = current_order.copy()
        new_order[i], new_order[j] = current_order[j], current_order[i]
        
        new_size = _greedy_size(A, neighbor_lists, new_order)
        
        if new_size > current_size:
            current_order = new_order
            current_size = new_size
            no_improve = 0
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            no_improve += 1
    
    return best_order, best_size


def _simulated_annealing(A, neighbor_lists, start_order, n, rng, deadline,
                         max_no_improve=500):
    """Simulated annealing with insertion moves."""
    current_order = start_order.copy()
    current_size = _greedy_size(A, neighbor_lists, current_order)
    best_order = current_order.copy()
    best_size = current_size
    no_improve = 0
    
    T_start = 5.0
    T_end = 0.1
    iter_count = 0
    total_iters = 8000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        T = T_start * (T_end / T_start) ** (iter_count / total_iters)
        iter_count += 1
        
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        new_order = _do_insertion_move(current_order, i, j, n)
        new_size = _greedy_size(A, neighbor_lists, new_order)
        delta = new_size - current_size
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_order = new_order
            current_size = new_size
            no_improve = 0
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            no_improve += 1
    
    return best_order, best_size


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    deg = A.sum(axis=1).astype(float)
    
    rng = np.random.default_rng(42)
    
    neighbor_lists = _compute_neighbor_lists(A)
    candidates = _generate_candidates(A, deg, n, rng)
    
    best_size = -1
    best_order = None
    
    for order in candidates:
        if time.time() > deadline - 30:
            break
        size = _greedy_size(A, neighbor_lists, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
    
    if best_order is None:
        best_order = np.argsort(deg, kind='stable')
        best_size = _greedy_size(A, neighbor_lists, best_order)
    
    ls_deadline = deadline - 5.0
    
    # Phase 1: Insertion local search
    best_order, best_size = _local_search_insertion(
        A, neighbor_lists, best_order, n, rng, ls_deadline, max_no_improve=200)
    
    # Phase 2: Other diverse starts
    if time.time() < ls_deadline - 10:
        other_starts = [candidates[0], candidates[2], candidates[4]]
        for start in other_starts:
            if time.time() > ls_deadline - 5:
                break
            o, s = _local_search_insertion(
                A, neighbor_lists, start, n, rng, ls_deadline, max_no_improve=100)
            if s > best_size:
                best_size = s
                best_order = o
    
    # Phase 3: Simulated annealing
    if time.time() < ls_deadline - 5:
        o, s = _simulated_annealing(
            A, neighbor_lists, best_order, n, rng, ls_deadline, max_no_improve=300)
        if s > best_size:
            best_size = s
            best_order = o
    
    # Phase 4: Swap polish
    if time.time() < deadline - 2:
        o, s = _local_search_swap(
            A, neighbor_lists, best_order, n, rng, deadline - 2, max_no_improve=100)
        if s > best_size:
            best_size = s
            best_order = o
    
    # Convert to priority vector
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
    
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
