"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _build_conflict_graph(values):
    """Build adjacency list and conflict count array for the conflict graph.
    Edge (i,j) exists if values[i]+values[j] is also in values.
    Returns (adj, conflict_count) where adj[i] is set of neighbors and
    conflict_count[i] is the number of edges incident to i.
    """
    val_set = set(values)
    n = len(values)
    adj = [set() for _ in range(n)]
    # Use numpy for faster sum checking
    vals_arr = np.array(values)
    for i in range(n):
        # Check which j > i satisfy values[i] + values[j] in val_set
        # We can vectorize: for each j, check if values[i]+values[j] is in val_set
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i].add(j)
                adj[j].add(i)
    
    conflict_count = [len(adj[i]) for i in range(n)]
    return adj, conflict_count

def _greedy_mis(adj, n, rng, conflict_count):
    """Greedy Maximum Independent Set with degree-based selection.
    Pick vertex with lowest current degree among remaining vertices.
    """
    in_set = [False] * n
    remaining = set(range(n))
    # Track current degrees
    cur_deg = conflict_count[:]
    
    while remaining:
        # Find vertex with minimum current degree
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            # Count actual neighbors still in remaining
            deg = sum(1 for nb in adj[v] if nb in remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
        
        in_set[best_v] = True
        remaining.discard(best_v)
        # Remove neighbors of best_v from remaining
        for nb in list(adj[best_v]):
            if nb in remaining:
                remaining.discard(nb)
    
    return in_set

def _local_search_swap(in_set, adj, n, rng, max_iters, time_limit):
    """Local search with swap moves: remove one vertex and add another.
    This can escape local optima better than single-flip moves.
    """
    best_in_set = in_set[:]
    best_size = sum(in_set)
    cur_in_set = in_set[:]
    cur_size = best_size
    
    start_time = time.time()
    iters = 0
    
    while iters < max_iters:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Try a swap: remove a random in-set vertex, add a random not-in-set vertex
        in_indices = [i for i in range(n) if cur_in_set[i]]
        out_indices = [i for i in range(n) if not cur_in_set[i]]
        
        if not in_indices or not out_indices:
            break
        
        v_remove = rng.choice(in_indices)
        v_add = rng.choice(out_indices)
        
        # Tentatively swap
        cur_in_set[v_remove] = False
        cur_in_set[v_add] = True
        
        # Check if valid (no conflicts)
        valid = True
        for nb in adj[v_add]:
            if cur_in_set[nb]:
                valid = False
                break
        
        if valid:
            new_size = cur_size - 1 + 1  # same size, but different composition
            # Actually we want to try to increase size, so let's also try just adding
            # For now, accept if valid (keeps size same, may enable future improvements)
            # Better: try to add a vertex that doesn't conflict
            pass
        else:
            # Revert
            cur_in_set[v_remove] = True
            cur_in_set[v_add] = False
        
        # Also try single add: add a random out vertex if no conflicts
        if out_indices:
            v_try = rng.choice(out_indices)
            if not cur_in_set[v_try]:
                conflicts = any(cur_in_set[nb] for nb in adj[v_try])
                if not conflicts:
                    cur_in_set[v_try] = True
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_in_set = cur_in_set[:]
        
        # Try single remove if we have conflicts (shouldn't happen if we maintain validity)
        iters += 1
    
    return best_in_set

def _simulated_annealing(values, adj, n, rng, max_iters, t_start, t_end, time_limit):
    """Simulated annealing on the conflict graph.
    State: boolean array in_set.
    Moves: flip one vertex (add or remove).
    Objective: maximize |in_set| subject to no conflicts within in_set.
    """
    # Start with greedy solution
    conflict_count = [len(adj[i]) for i in range(n)]
    in_set = _greedy_mis(adj, n, rng, conflict_count)
    
    # Track current conflicts for fast evaluation
    def count_conflicts(in_set):
        cnt = 0
        for i in range(n):
            if in_set[i]:
                for j in adj[i]:
                    if j > i and in_set[j]:
                        cnt += 1
        return cnt
    
    def objective(in_set):
        size = sum(in_set)
        conf = count_conflicts(in_set)
        if conf > 0:
            return size - 10000 * conf
        return size
    
    cur_obj = objective(in_set)
    best_in_set = in_set[:]
    best_obj = cur_obj
    
    start_time = time.time()
    
    for it in range(max_iters):
        # Time check
        if time.time() - start_time > time_limit:
            break
        
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Random move: flip a random vertex
        v = rng.randint(0, n - 1)
        in_set[v] = not in_set[v]
        
        new_obj = objective(in_set)
        delta = new_obj - cur_obj
        
        if delta >= 0 or rng.random() < pow(2.718281828, delta / max(t, 1e-10)):
            cur_obj = new_obj
            if cur_obj > best_obj:
                best_obj = cur_obj
                best_in_set = in_set[:]
        else:
            in_set[v] = not in_set[v]  # revert
    
    return best_in_set

def _solve_instance(values, time_budget, seed, deadline):
    """Solve one instance: find max sum-free subset."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    
    # Build conflict graph
    adj, conflict_count = _build_conflict_graph(values)
    
    # Try multiple strategies and keep the best
    best_subset = []
    best_size = 0
    
    # Strategy 1: Greedy MIS
    greedy = _greedy_mis(adj, n, rng, conflict_count)
    size = sum(greedy)
    if size > best_size:
        best_size = size
        best_subset = [values[i] for i in range(n) if greedy[i]]
    
    # Strategy 2: Local search with swaps
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 3:
        ls_iters = int(remaining_time * 5000)
        ls_iters = max(ls_iters, 500)
        ls_iters = min(ls_iters, 500000)
        ls_result = _local_search_swap(greedy, adj, n, rng, ls_iters, remaining_time * 0.4)
        size = sum(ls_result)
        if size > best_size:
            best_size = size
            best_subset = [values[i] for i in range(n) if ls_result[i]]
    
    # Strategy 3: Simulated annealing with remaining time
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 3:
        sa_iters = int(remaining_time * 2000)
        sa_iters = max(sa_iters, 500)
        sa_iters = min(sa_iters, 300000)
        sa_result = _simulated_annealing(values, adj, n, rng, sa_iters, 10.0, 0.01, remaining_time * 0.5)
        size = sum(sa_result)
        if size > best_size:
            best_size = size
            best_subset = [values[i] for i in range(n) if sa_result[i]]
    
    # Strategy 4: Try picking all odds/evens as baseline
    odds = [values[i] for i in range(n) if values[i] % 2 != 0]
    evens = [values[i] for i in range(n) if values[i] % 2 == 0]
    if len(odds) > best_size:
        best_size = len(odds)
        best_subset = odds
    if len(evens) > best_size:
        best_size = len(evens)
        best_subset = evens
    
    return best_subset

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    results = []
    for i, values in enumerate(instances):
        # Allocate time proportionally but with minimum
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5  # 5s safety margin
        if remaining < 0:
            remaining = 0.1
        # Allocate more time to larger instances
        weight = len(values) / sum(len(v) for v in instances)
        time_budget = max(remaining * weight, 1.0)
        
        subset = _solve_instance(values, time_budget, 801 + i, deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
