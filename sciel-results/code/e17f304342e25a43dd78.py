"""Strong edge colourings: maximum clique (independent set) branch-and-bound on conflict graph."""
import time
import random as _random_mod
from collections import defaultdict

_DEADLINE_FRAC = 0.88

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

def _build_conflict(edges, adj, n):
    """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
    m = len(edges)
    conflict = [set() for _ in range(m)]
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
                continue
            if c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                conflict[i].add(j)
                conflict[j].add(i)
    return conflict

def _greedy_coloring(conflict, order):
    """Greedy coloring in the given order. Returns list of colours."""
    m = len(conflict)
    colour = [-1] * m
    for idx in order:
        used = set()
        for j in conflict[idx]:
            if colour[j] != -1:
                used.add(colour[j])
        c = 0
        while c in used:
            c += 1
        colour[idx] = c
    return colour

def _count_colors(colour):
    if not colour:
        return 0
    return max(colour) + 1

def _try_recolor(colour, idx, new_color, conflict):
    for j in conflict[idx]:
        if colour[j] == new_color:
            return False
    return True

def _local_improve(colour, conflict, max_iters=300):
    """Try to reduce the number of colours by recoloring edges."""
    m = len(colour)
    if m == 0:
        return colour
    current_max = max(colour) + 1
    for _ in range(max_iters):
        improved = False
        for idx in range(m):
            current_color = colour[idx]
            for new_c in range(current_color):
                if _try_recolor(colour, idx, new_c, conflict):
                    colour[idx] = new_c
                    improved = True
                    break
        if current_max > 0:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    if _try_recolor(colour, idx, new_c, conflict):
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
        if not improved:
            break
    return colour

def _color_merge_improve(colour, conflict, max_iters=80):
    """Try to merge color classes."""
    m = len(colour)
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
                    for j in conflict[idx]:
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

def _dsatur_ordering(conflict):
    """DSATUR heuristic ordering for greedy coloring."""
    m = len(conflict)
    if m == 0:
        return []
    conf_deg = [len(conflict[i]) for i in range(m)]
    uncolored = set(range(m))
    order = []
    colors = [-1] * m
    if uncolored:
        first = max(uncolored, key=lambda i: conf_deg[i])
        order.append(first)
        uncolored.remove(first)
        colors[first] = 0
    while uncolored:
        best_edge = None
        best_sat = -1
        best_deg = -1
        for i in uncolored:
            sat_colors = set()
            for j in conflict[i]:
                if colors[j] != -1:
                    sat_colors.add(colors[j])
            sat = len(sat_colors)
            if sat > best_sat or (sat == best_sat and conf_deg[i] > best_deg):
                best_sat = sat
                best_deg = conf_deg[i]
                best_edge = i
        order.append(best_edge)
        uncolored.remove(best_edge)
        used = set()
        for j in conflict[best_edge]:
            if colors[j] != -1:
                used.add(colors[j])
        c = 0
        while c in used:
            c += 1
        colors[best_edge] = c
    return order

def _largest_first_ordering(conflict):
    m = len(conflict)
    return sorted(range(m), key=lambda i: -len(conflict[i]))

def _max_independent_set(conflict, candidates, deadline):
    """
    Find a maximum independent set in the conflict graph restricted to candidates.
    Uses branch-and-bound with greedy upper bound.
    Returns the set of node indices in the MIS.
    """
    if not candidates:
        return set()
    if time.time() > deadline:
        return {max(candidates, key=lambda x: len(conflict[x]))}
    
    # Greedy upper bound: repeatedly pick the node with fewest remaining conflicts
    def _greedy_bound(cands):
        if not cands:
            return 0
        cands = list(cands)
        count = 0
        remaining = set(cands)
        while remaining:
            # Pick node with min degree in remaining
            best = min(remaining, key=lambda x: sum(1 for y in conflict[x] if y in remaining))
            count += 1
            remaining.discard(best)
            remaining -= conflict[best]
        return count
    
    best_set = set()
    best_size = 0
    
    def _branch(cands, current_set):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if not cands:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        # Upper bound
        ub = len(current_set) + _greedy_bound(cands)
        if ub <= best_size:
            return
        # Choose pivot: node with max degree in cands
        pivot = max(cands, key=lambda x: sum(1 for y in conflict[x] if y in cands))
        # Branch: include pivot or exclude pivot
        # Include pivot
        new_cands = cands - conflict[pivot] - {pivot}
        _branch(new_cands, current_set | {pivot})
        # Exclude pivot
        cands2 = cands - {pivot}
        _branch(cands2, current_set)
    
    _branch(set(candidates), set())
    return best_set

def _solve_with_mis_extraction(conflict, deadline, max_colors_hint=100):
    """
    Solve by repeatedly extracting maximum independent sets (induced matchings).
    Each MIS becomes one color class.
    """
    m = len(conflict)
    colour = [-1] * m
    remaining = set(range(m))
    color_idx = 0
    
    while remaining:
        if time.time() > deadline:
            # Fallback: greedy for remaining
            for e in sorted(remaining):
                used = set()
                for c in conflict[e]:
                    if colour[c] != -1:
                        used.add(colour[c])
                c = 0
                while c in used:
                    c += 1
                colour[e] = c
            break
        
        # Find MIS in remaining
        mis = _max_independent_set(conflict, remaining, deadline)
        
        if not mis:
            # Pick one node
            mis = {min(remaining)}
        
        for e in mis:
            colour[e] = color_idx
            remaining.discard(e)
        
        color_idx += 1
        if color_idx > max_colors_hint:
            break
    
    return colour

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    conflict = _build_conflict(edges, adj, n)
    
    best_colour = None
    best_count = float('inf')
    
    def _update_best(colour):
        nonlocal best_colour, best_count
        if colour is not None and all(c >= 0 for c in colour):
            count = _count_colors(colour)
            if count < best_count:
                best_count = count
                best_colour = colour[:]
    
    # Strategy 1: MIS extraction with branch-and-bound
    mis_deadline = t0 + time_budget * 0.5
    colour = _solve_with_mis_extraction(conflict, mis_deadline)
    colour = _local_improve(colour, conflict, max_iters=200)
    colour = _color_merge_improve(colour, conflict, max_iters=50)
    _update_best(colour)
    
    # Strategy 2: DSATUR greedy + local improvement
    if time.time() - t0 < t0 + time_budget * 0.7:
        order = _dsatur_ordering(conflict)
        colour = _greedy_coloring(conflict, order)
        colour = _local_improve(colour, conflict, max_iters=300)
        colour = _color_merge_improve(colour, conflict, max_iters=80)
        _update_best(colour)
    
    # Strategy 3: Largest first greedy + local improvement
    if time.time() - t0 < t0 + time_budget * 0.8:
        order = _largest_first_ordering(conflict)
        colour = _greedy_coloring(conflict, order)
        colour = _local_improve(colour, conflict, max_iters=300)
        colour = _color_merge_improve(colour, conflict, max_iters=80)
        _update_best(colour)
    
    # Strategy 4: Randomized restarts
    rng = _random_mod.Random(123)
    remaining_time = time_budget - (time.time() - t0)
    max_random = max(5, int(remaining_time / 0.2))
    for attempt in range(max_random):
        if time.time() - t0 > t0 + time_budget:
            break
        order = list(range(m))
        rng.shuffle(order)
        colour = _greedy_coloring(conflict, order)
        colour = _local_improve(colour, conflict, max_iters=150)
        colour = _color_merge_improve(colour, conflict, max_iters=40)
        _update_best(colour)
    
    if best_colour is None:
        best_colour = _greedy_coloring(conflict, list(range(m)))
    
    return best_colour

def construct():
    t_start = time.time()
    total_budget = 150.0 * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    budget_fracs = [0.30, 0.30, 0.40]
    
    for i, spec in enumerate(specs):
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        adj = _build_adj(edges, n)
        
        elapsed = time.time() - t_start
        remaining_total = total_budget - elapsed
        if remaining_total < 2.0:
            conflict = _build_conflict(edges, adj, n)
            colour = _greedy_coloring(conflict, list(range(len(edges))))
            results.append(colour)
            continue
        
        instance_budget = max(10.0, remaining_total * budget_fracs[i])
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
