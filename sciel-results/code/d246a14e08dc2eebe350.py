"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Track unassigned predecessor count
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if not eligible and n > 0:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        # Sort eligible by priority descending
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed_in_this_pass = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                # Place task
                current_load += times[task]
                eligible.discard(task)
                placed_in_this_pass = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed_in_this_pass:
            # Overflow to next station
            station_count += 1
            current_load = 0
            # Check for impossible case
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count


def _compute_critical_path(times, predecessors, successors):
    """Compute longest path from each task to any sink."""
    n = len(times)
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    return cp


def _count_transitive(start, adj_map):
    """Count transitive successors/predecessors."""
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for s in adj_map[node]:
            if s not in visited:
                visited.add(s)
                stack.append(s)
    return len(visited)


def _local_search(times, predecessors, cycle_time, initial_priority, max_iters=2000, deadline=None):
    """
    Local search with multiple move types: swap, single perturbation, and 2-opt style moves.
    Uses simulated annealing acceptance to escape local optima.
    Returns best priority found.
    """
    n = len(times)
    best_priority = initial_priority.copy()
    best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
    
    rng = np.random.RandomState(42)
    current_priority = best_priority.copy()
    current_score = best_score
    
    # Simulated annealing parameters
    T = 2.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    it = 0
    while it < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose move type
        move_type = rng.randint(0, 3)
        new_priority = current_priority.copy()
        
        if move_type == 0:
            # Swap two priorities
            i, j = rng.choice(n, 2, replace=False)
            new_priority[i], new_priority[j] = current_priority[j], current_priority[i]
        elif move_type == 1:
            # Perturb one priority
            i = rng.randint(n)
            new_priority[i] = current_priority[i] + rng.normal(0, 1)
        else:
            # Swap a random pair and perturb one
            i, j = rng.choice(n, 2, replace=False)
            new_priority[i], new_priority[j] = current_priority[j], current_priority[i]
            k = rng.randint(n)
            new_priority[k] = current_priority[k] + rng.normal(0, 0.5)
        
        score = _greedy_balance(times, predecessors, cycle_time, new_priority)
        
        # Accept or reject based on simulated annealing
        if score < current_score or rng.random() < np.exp(-(score - current_score) / T):
            current_priority = new_priority
            current_score = score
            if score < best_score:
                best_priority = new_priority.copy()
                best_score = score
        
        T = max(T * cooling_rate, min_T)
        it += 1
    
    return best_priority


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Precompute useful quantities
    # Critical path
    cp = _compute_critical_path(times, predecessors, successors)
    
    # Number of direct successors
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of direct predecessors
    num_direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Transitive successor counts (how many tasks are downstream)
    trans_succ = np.array([_count_transitive(i, successors) for i in range(n)], dtype=float)
    
    # Transitive predecessor counts (how many tasks are upstream)
    trans_pred = np.array([_count_transitive(i, {i: preds for i, preds in enumerate(predecessors)}) for i in range(n)], dtype=float)
    
    # Candidate priority vectors
    candidates = []
    
    # 1. LPT (longest processing time)
    candidates.append(times.copy())
    # 2. Critical path
    candidates.append(cp.copy())
    # 3. Time + critical path
    candidates.append(times + cp)
    # 4. Most direct successors
    candidates.append(num_direct_succ.copy())
    # 5. Fewest direct predecessors
    candidates.append(-num_direct_pred.copy())
    # 6. Time + successors
    candidates.append(times + num_direct_succ)
    # 7. Critical path + successors
    candidates.append(cp + num_direct_succ)
    # 8. Time - predecessors
    candidates.append(times - num_direct_pred)
    # 9. (time * successors) / (preds + 1) - weighted
    candidates.append(times * num_direct_succ / (num_direct_pred + 1))
    # 10. cp * num_direct_succ
    candidates.append(cp * num_direct_succ)
    # 11. Transitive successors (bottleneck tasks)
    candidates.append(trans_succ.copy())
    # 12. Time + transitive successors
    candidates.append(times + trans_succ)
    # 13. Critical path + transitive successors
    candidates.append(cp + trans_succ)
    # 14. Fewest transitive predecessors (start early)
    candidates.append(-trans_pred.copy())
    # 15. Time * transitive successors
    candidates.append(times * trans_succ)
    # 16. Combined: time + 0.5*cp + 0.5*trans_succ
    candidates.append(times + 0.5 * cp + 0.5 * trans_succ)
    # 17. cp * (1 + trans_succ)
    candidates.append(cp * (1 + trans_succ))
    
    # Evaluate each candidate and find the best
    best_score = float('inf')
    best_priority = candidates[0]
    
    for pri in candidates:
        if time.time() > deadline:
            break
        n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
        if n_stations < best_score:
            best_score = n_stations
            best_priority = pri.copy()
        if best_score == 1:
            break
    
    # Run local search from top 3 candidates with different seeds
    if time.time() < deadline:
        # Sort candidates by their scores
        candidate_scores = []
        for pri in candidates:
            if time.time() > deadline:
                break
            n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
            candidate_scores.append((n_stations, pri))
        
        candidate_scores.sort(key=lambda x: x[0])
        
        # Run local search from top 3 candidates
        for rank, (cs, cp) in enumerate(candidate_scores[:3]):
            if time.time() > deadline:
                break
            if cs >= best_score and rank > 0:
                continue  # Skip if not better than current best
            
            # Use different random seed for each restart
            seed = 42 + rank * 1000
            rng = np.random.RandomState(seed)
            
            # Run local search with this seed
            best_priority_ls = _local_search(
                times, predecessors, cycle_time, 
                cp.copy(), 
                max_iters=3000,
                deadline=deadline
            )
            
            ls_score = _greedy_balance(times, predecessors, cycle_time, best_priority_ls)
            if ls_score < best_score:
                best_score = ls_score
                best_priority = best_priority_ls
        
        # If we still have time, do another round from the best
        if time.time() < deadline and best_score > 1:
            best_priority = _local_search(
                times, predecessors, cycle_time, 
                best_priority, 
                max_iters=3000,
                deadline=deadline
            )
    
    return best_priority
# EVOLVE-BLOCK-END
