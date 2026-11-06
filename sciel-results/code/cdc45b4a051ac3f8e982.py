"""Strong edge colourings: exact B&B MIS extraction with bitset pruning."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict
import math

_DEADLINE_FRAC = 0.88  # use 88% of the 150s budget for safety

def _bounded_degree_graph(n, d, seed):
    """Reproduce the verifier's graph generation exactly."""
    rng = _random_mod.Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        e = (min(u, v), max(u, v))
        if u != v and degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_conflict_bitmask(edges, adj, n):
    """Build conflict as bitmasks for fast operations. Returns list of ints."""
    m = len(edges)
    conflict = [0] * m
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                continue
            if c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    return conflict

def _popcount(x):
    return bin(x).count('1')

def _greedy_mis(conflict, remaining_mask, m):
    """Greedy MIS: pick vertex with min degree in remaining, remove it and its neighbors."""
    matching = []
    remaining = remaining_mask
    while remaining:
        # Find vertex with minimum degree in remaining subgraph
        best_v = -1
        best_deg = m + 1
        tmp = remaining
        while tmp:
            v = (tmp & (-tmp)).bit_length() - 1
            tmp &= tmp - 1
            deg = _popcount(conflict[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        matching.append(best_v)
        remaining &= ~(conflict[best_v] | (1 << best_v))
    return matching

def _branch_and_bound_mis(remaining_mask, conflict, m, deadline, best_size, best_set):
    """
    Branch and bound to find a large independent set in the conflict graph.
    Uses bitset-based pruning.
    Returns (best_size, best_set).
    """
    if time.time() > deadline:
        return (best_size, best_set)
    
    if remaining_mask == 0:
        if len(best_set) > best_size:
            best_size = len(best_set)
            best_set = best_set[:]
        return (best_size, best_set)
    
    # Upper bound: popcount of remaining
    ub = _popcount(remaining_mask)
    if len(best_set) + ub <= best_size:
        return (best_size, best_set)
    
    # Pick a vertex to branch on: highest degree in remaining for better pruning
    best_v = -1
    best_deg = -1
    tmp = remaining_mask
    while tmp:
        v = (tmp & (-tmp)).bit_length() - 1
        tmp &= tmp - 1
        deg = _popcount(conflict[v] & remaining_mask)
        if deg > best_deg:
            best_deg = deg
            best_v = v
    
    if best_v < 0:
        return (best_size, best_set)
    
    # Branch 1: include best_v
    new_remaining = remaining_mask & ~(conflict[best_v] | (1 << best_v))
    best_set.append(best_v)
    best_size, best_set = _branch_and_bound_mis(new_remaining, conflict, m, deadline, best_size, best_set)
    best_set.pop()
    
    # Branch 2: exclude best_v
    new_remaining = remaining_mask & ~(1 << best_v)
    best_size, best_set = _branch_and_bound_mis(new_remaining, conflict, m, deadline, best_size, best_set)
    
    return (best_size, best_set)

def _extract_mis(conflict, m, remaining_mask, deadline, seed=0):
    """Extract a large induced matching from remaining edges using B&B MIS with greedy fallback."""
    # First try greedy to get a lower bound
    greedy_match = _greedy_mis(conflict, remaining_mask, m)
    best_size = len(greedy_match)
    best_set = greedy_match[:]
    
    # Try B&B with a time-limited search
    bb_deadline = deadline - 0.05
    if time.time() < bb_deadline:
        best_size, best_set = _branch_and_bound_mis(remaining_mask, conflict, m, bb_deadline, best_size, best_set)
    
    return best_set

def _greedy_coloring(conflict, m, order=None):
    """Greedy coloring using bitmask conflict. Returns list of colours."""
    if order is None:
        order = list(range(m))
    colour = [-1] * m
    neighbour_colours = [set() for _ in range(m)]
    
    for idx in order:
        c = 0
        while c in neighbour_colours[idx]:
            c += 1
        colour[idx] = c
        cm = conflict[idx]
        for j in range(m):
            if cm & (1 << j):
                neighbour_colours[j].add(c)
    
    return colour

def _try_recolor(colour, idx, new_color, conflict, m):
    """Check if we can assign new_color to edge idx."""
    cm = conflict[idx]
    for j in range(m):
        if cm & (1 << j):
            if colour[j] == new_color:
                return False
    return True

def _local_improve(colour, conflict, m, max_iters=300):
    """Try to reduce the number of colours by recoloring edges."""
    if m == 0:
        return colour
    
    for _ in range(max_iters):
        improved = False
        current_max = max(colour) + 1
        
        # Try to recolor each edge to a smaller colour
        for idx in range(m):
            current_color = colour[idx]
            for new_c in range(current_color):
                if _try_recolor(colour, idx, new_c, conflict, m):
                    colour[idx] = new_c
                    improved = True
                    break
        
        # Try to reduce the overall max colour
        current_max = max(colour) + 1
        if current_max > 0:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    if _try_recolor(colour, idx, new_c, conflict, m):
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
        
        if not improved:
            break
    
    return colour

def _color_merge_improve(colour, conflict, m, max_iters=50):
    """Try to merge color classes by recoloring all edges of one color to another."""
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        
        for target_c in range(current_max - 1, -1, -1):
            if target_c >= current_max:
                continue
            edges_in_color = [i for i in range(m) if colour[i] == target_c]
            if not edges_in_color:
                continue
            
            for new_c in range(target_c):
                can_move = True
                for idx in edges_in_color:
                    cm = conflict[idx]
                    for j in range(m):
                        if cm & (1 << j):
                            if colour[j] == new_c:
                                can_move = False
                                break
                    if not can_move:
                        break
                if can_move:
                    for idx in edges_in_color:
                        colour[idx] = new_c
                    improved = True
                    break
            
            if improved:
                current_max = max(colour) + 1
                break
        
        if not improved:
            break
    
    return colour

def _solve_with_mis_extraction(conflict, m, deadline, seed=0):
    """
    Solve by repeatedly extracting large induced matchings via B&B MIS.
    Each MIS becomes one color class.
    """
    colour = [-1] * m
    remaining_mask = (1 << m) - 1
    color_idx = 0
    
    while remaining_mask:
        if time.time() > deadline:
            # Fallback: greedy assign remaining
            tmp = remaining_mask
            while tmp:
                e = (tmp & (-tmp)).bit_length() - 1
                tmp &= tmp - 1
                used = set()
                cm = conflict[e]
                for j in range(m):
                    if cm & (1 << j):
                        if colour[j] != -1:
                            used.add(colour[j])
                c = 0
                while c in used:
                    c += 1
                colour[e] = c
            break
        
        # Extract MIS from remaining
        matching = _extract_mis(conflict, m, remaining_mask, deadline, seed=seed)
        
        if not matching:
            tmp = remaining_mask
            e = (tmp & (-tmp)).bit_length() - 1
            matching = [e]
        
        for e in matching:
            colour[e] = color_idx
            remaining_mask &= ~(1 << e)
        
        color_idx += 1
    
    return colour

def _simulated_annealing(colour, conflict, m, deadline, rng=None):
    """
    Simulated annealing to reduce number of colours.
    """
    if rng is None:
        rng = _random_mod.Random()
    
    if m == 0:
        return colour
    
    colour = colour[:]
    current_max = max(colour) + 1
    best_colour = colour[:]
    best_max = current_max
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min and time.time() < deadline:
        improved = False
        for _ in range(m * 3):
            if time.time() > deadline:
                break
            
            idx = rng.randint(0, m - 1)
            current_c = colour[idx]
            
            max_c_to_try = min(current_max + 2, m)
            new_c = rng.randint(0, max_c_to_try - 1)
            if new_c == current_c:
                continue
            
            if not _try_recolor(colour, idx, new_c, conflict, m):
                continue
            
            colour[idx] = new_c
            new_max = max(colour) + 1
            
            if new_max <= current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_colour = colour[:]
                improved = True
            else:
                delta = new_max - current_max
                if rng.random() < math.exp(-delta / T):
                    current_max = new_max
                    improved = True
                else:
                    colour[idx] = current_c
        
        T *= cooling
        if not improved and T < 0.5:
            break
    
    return best_colour

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance, returning a list of colours for edges in sorted order."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    deadline = t0 + time_budget
    
    # Precompute conflict as bitmask
    conflict = _build_conflict_bitmask(edges, adj, n)
    
    best_colour = None
    best_count = float('inf')
    
    def _update_best(colour):
        nonlocal best_colour, best_count
        if colour is not None and all(c >= 0 for c in colour):
            count = max(colour) + 1
            if count < best_count:
                best_count = count
                best_colour = colour[:]
    
    # Strategy 1: B&B MIS extraction (primary) - multiple seeds
    mis_deadline = t0 + time_budget * 0.40
    for seed in range(40):
        if time.time() > mis_deadline:
            break
        colour = _solve_with_mis_extraction(conflict, m, mis_deadline, seed=seed)
        colour = _local_improve(colour, conflict, m, max_iters=200)
        colour = _color_merge_improve(colour, conflict, m, max_iters=50)
        _update_best(colour)
    
    # Strategy 2: Greedy coloring with different orderings
    if time.time() < t0 + time_budget * 0.55:
        # Largest first
        edge_deg = [len(adj[u]) + len(adj[v]) for u, v in edges]
        order = sorted(range(m), key=lambda i: -edge_deg[i])
        colour = _greedy_coloring(conflict, m, order)
        colour = _local_improve(colour, conflict, m, max_iters=200)
        colour = _color_merge_improve(colour, conflict, m, max_iters=50)
        _update_best(colour)
        
        # Randomized orderings
        rng = _random_mod.Random(99)
        for _ in range(10):
            if time.time() > t0 + time_budget * 0.55:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour = _greedy_coloring(conflict, m, order)
            colour = _local_improve(colour, conflict, m, max_iters=150)
            colour = _color_merge_improve(colour, conflict, m, max_iters=40)
            _update_best(colour)
    
    # Strategy 3: Simulated annealing on best solution
    if best_colour is not None and time.time() < t0 + time_budget * 0.85:
        sa_deadline = t0 + time_budget * 0.85
        rng = _random_mod.Random(123)
        for attempt in range(8):
            if time.time() > sa_deadline:
                break
            sa_colour = _simulated_annealing(best_colour, conflict, m, sa_deadline, rng)
            _update_best(sa_colour)
    
    # Strategy 4: More MIS extraction attempts with remaining time
    if time.time() < t0 + time_budget * 0.95:
        remaining_deadline = t0 + time_budget * 0.95
        for seed in range(60, 120):
            if time.time() > remaining_deadline:
                break
            colour = _solve_with_mis_extraction(conflict, m, remaining_deadline, seed=seed)
            colour = _local_improve(colour, conflict, m, max_iters=150)
            colour = _color_merge_improve(colour, conflict, m, max_iters=40)
            _update_best(colour)
    
    if best_colour is None:
        best_colour = _greedy_coloring(conflict, m)
    
    return best_colour

def construct():
    t_start = time.time()
    total_budget = 150.0 * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    # Allocate budget: smaller instances get less time
    budget_fracs = [0.22, 0.30, 0.48]
    
    for i, spec in enumerate(specs):
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        adj = _build_adj(edges, n)
        
        elapsed = time.time() - t_start
        remaining_total = total_budget - elapsed
        if remaining_total < 2.0:
            # Not enough time; use a very fast fallback
            conflict = _build_conflict_bitmask(edges, adj, n)
            m = len(edges)
            colour = _greedy_coloring(conflict, m)
            results.append(colour)
            continue
        
        # Allocate budget based on fraction of remaining
        instance_budget = max(8.0, remaining_total * budget_fracs[i])
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
