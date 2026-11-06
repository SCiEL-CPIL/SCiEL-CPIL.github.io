"""Strong edge colourings: exact backtracking with pruning on conflict graph."""
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

def _local_improve(colour, conflict, max_iters=200):
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
                ok = True
                for j in conflict[idx]:
                    if colour[j] == new_c:
                        ok = False
                        break
                if ok:
                    colour[idx] = new_c
                    improved = True
                    break
        if current_max > 0:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    ok = True
                    for j in conflict[idx]:
                        if colour[j] == new_c:
                            ok = False
                            break
                    if ok:
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
        if not improved:
            break
    return colour

def _color_merge_improve(colour, conflict, max_iters=50):
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

def _clique_lower_bound(conflict, candidates):
    """Find a clique in the conflict graph among candidates (lower bound on chromatic number)."""
    if not candidates:
        return 0
    # Greedy clique: repeatedly pick node with max degree in remaining
    remaining = set(candidates)
    clique = 0
    while remaining:
        best = max(remaining, key=lambda x: sum(1 for y in conflict[x] if y in remaining))
        clique += 1
        remaining.discard(best)
        remaining &= conflict[best]
    return clique

def _exact_backtracking(conflict, deadline, initial_bound=None):
    """
    Exact backtracking to find minimum chromatic number of conflict graph.
    Uses colouring approach: assign colours 0..k-1 to nodes, check validity.
    Returns (chromatic_number, colour_array) or (None, None) if timeout.
    """
    m = len(conflict)
    if m == 0:
        return 0, []
    
    # Get a good initial upper bound
    order = _dsatur_ordering(conflict)
    upper = _count_colors(_greedy_coloring(conflict, order))
    
    if initial_bound is not None and initial_bound < upper:
        upper = initial_bound
    
    # Lower bound from clique
    lower = _clique_lower_bound(conflict, set(range(m)))
    
    best_colour = None
    best_k = upper
    
    # Try to find solution with k colours, starting from lower bound
    # We'll do binary search between lower and upper
    
    def _try_color_k(k, deadline):
        """Try to colour the graph with exactly k colours. Returns colour array or None."""
        colour = [-1] * m
        # Order nodes by degree descending for better pruning
        node_order = sorted(range(m), key=lambda i: -len(conflict[i]))
        
        def _backtrack(pos, deadline):
            if time.time() > deadline:
                return False
            if pos == m:
                return True
            
            node = node_order[pos]
            # Try each colour
            for c in range(k):
                # Check if colour c is available
                ok = True
                for j in conflict[node]:
                    if colour[j] == c:
                        ok = False
                        break
                if ok:
                    colour[node] = c
                    if _backtrack(pos + 1, deadline):
                        return True
                    colour[node] = -1
            return False
        
        if _backtrack(0, deadline):
            return colour
        return None
    
    # First try with lower bound
    if lower < upper:
        test_deadline = time.time() + 5.0
        result = _try_color_k(lower, test_deadline)
        if result is not None:
            return lower, result
    
    # Now try values between lower+1 and upper-1
    for k in range(lower + 1, upper):
        if time.time() > deadline:
            break
        test_deadline = min(deadline, time.time() + 10.0)
        result = _try_color_k(k, test_deadline)
        if result is not None:
            return k, result
    
    # Fallback: return upper bound solution
    colour = _greedy_coloring(conflict, order)
    return upper, colour

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
    
    # Strategy 1: DSATUR greedy + local improvement (fast baseline)
    order = _dsatur_ordering(conflict)
    colour = _greedy_coloring(conflict, order)
    colour = _local_improve(colour, conflict, max_iters=200)
    colour = _color_merge_improve(colour, conflict, max_iters=50)
    _update_best(colour)
    
    # Strategy 2: Largest first greedy + local improvement
    if time.time() - t0 < t0 + time_budget * 0.3:
        order2 = _largest_first_ordering(conflict)
        colour2 = _greedy_coloring(conflict, order2)
        colour2 = _local_improve(colour2, conflict, max_iters=200)
        colour2 = _color_merge_improve(colour2, conflict, max_iters=50)
        _update_best(colour2)
    
    # Strategy 3: Exact backtracking with pruning
    remaining_time = time_budget - (time.time() - t0)
    if remaining_time > 5.0:
        exact_deadline = t0 + time_budget * 0.95
        initial_bound = best_count if best_count < float('inf') else None
        k, exact_colour = _exact_backtracking(conflict, exact_deadline, initial_bound)
        if exact_colour is not None and all(c >= 0 for c in exact_colour):
            _update_best(exact_colour)
    
    # Strategy 4: Randomized restarts with local improvement
    rng = _random_mod.Random(42)
    remaining_time = time_budget - (time.time() - t0)
    max_random = max(3, int(remaining_time / 1.0))
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
