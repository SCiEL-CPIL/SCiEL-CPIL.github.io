"""Combined program: random regular graph generation + local search via edge swaps."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _verify_girth_fast_adj_list(adj_list, n, g):
    """Fast girth check using adjacency lists.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _build_adj_from_edges(edges, n):
    """Build adjacency list from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return [list(s) for s in adj]

def _random_regular_graph_3(n, rng):
    """Generate a random 3-regular graph on n vertices.
    Returns (n, edges) or None if generation fails."""
    if n < 4 or n % 2 != 0:
        return None
    # Use configuration model with random matching
    # Each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    
    # Randomly match stubs
    edges_set = set()
    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        attempts += 1
        rng.shuffle(stubs)
        edges_set = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges_set:
                valid = False
                break
            edges_set.add(edge)
        if valid:
            # Check degree
            degree = [0] * n
            for u, v in edges_set:
                degree[u] += 1
                degree[v] += 1
            if all(d == 3 for d in degree):
                return n, list(edges_set)
    return None

def _edge_swap_improve(adj, n, g, rng, time_budget):
    """Try to improve girth via edge swaps.
    Returns (n, edges) if improved, else None."""
    edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
    
    # Find all cycles shorter than g
    def find_short_cycle(adj, n, g):
        """Find a cycle of length < g, or return None."""
        for start in range(n):
            dist = {}
            parent = {}
            dq = deque()
            dq.append((start, -1))
            dist[(start, -1)] = 0
            while dq:
                v, prev = dq.popleft()
                d = dist[(v, prev)]
                if d >= g - 1:
                    continue
                for w in adj[v]:
                    if w == prev:
                        continue
                    if w == start and d + 1 >= 3:
                        # Found a cycle of length d+1
                        # Reconstruct cycle
                        cycle = [v]
                        cur = v
                        while cur != start:
                            cur = parent.get((cur, prev), -1)
                            if cur == -1:
                                break
                            cycle.append(cur)
                        cycle.reverse()
                        return cycle
                    state = (w, v)
                    if state not in dist:
                        dist[state] = d + 1
                        parent[state] = prev
                        dq.append(state)
        return None
    
    # Try edge swaps to eliminate short cycles
    deadline = _time.time() + time_budget
    current_edges = set(edges)
    
    while _time.time() < deadline:
        # Find a short cycle
        cycle = find_short_cycle(adj, n, g)
        if cycle is None:
            return n, list(current_edges)
        
        # Try to break the cycle via edge swap
        # For each edge in the cycle, try swapping with a non-cycle edge
        cycle_edges = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            cycle_edges.add((min(u, v), max(u, v)))
        
        # Find candidate edges to swap
        # An edge (a,b) can be swapped with (c,d) if:
        # - (a,b) and (c,d) are both in current_edges
        # - a != c, a != d, b != c, b != d
        # - (a,c) and (b,d) or (a,d) and (b,c) are not in current_edges
        # - After swap, degrees remain 3
        
        # Get all edges
        all_edges = list(current_edges)
        rng.shuffle(all_edges)
        
        found_swap = False
        for e1 in all_edges:
            if _time.time() > deadline:
                break
            a, b = e1
            for e2 in all_edges:
                if e2 == e1:
                    continue
                c, d = e2
                # Check if swap is valid
                if a in (c, d) or b in (c, d):
                    continue
                # Try both matchings
                for new_e1, new_e2 in [((a, c), (b, d)), ((a, d), (b, c))]:
                    new_e1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                    new_e2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                    if new_e1 in current_edges or new_e2 in current_edges:
                        continue
                    if new_e1 == new_e2:
                        continue
                    # Perform swap
                    old_edges = (e1, e2)
                    current_edges.discard(e1)
                    current_edges.discard(e2)
                    current_edges.add(new_e1)
                    current_edges.add(new_e2)
                    
                    # Update adjacency
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    adj[a].append(new_e1[0] if new_e1[1] == a else new_e1[1])
                    adj[new_e1[0] if new_e1[1] == a else new_e1[1]].append(a)
                    adj[b].append(new_e2[0] if new_e2[1] == b else new_e2[1])
                    adj[new_e2[0] if new_e2[1] == b else new_e2[1]].append(b)
                    
                    found_swap = True
                    break
                if found_swap:
                    break
            if found_swap:
                break
        
        if not found_swap:
            # No valid swap found, try a different approach
            break
    
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    if k != 3:
        # For k != 3, fall back to a simple construction
        # Use a random regular graph
        n = 2 * (k ** (g // 2 - 1))
        if n % 2 != 0:
            n += 1
        rng = _random.Random(42)
        result = _random_regular_graph_3(n, rng) if k == 3 else None
        if result is not None:
            return result
        # Fallback: return a trivial graph
        n = k * (g - 1)
        edges = []
        for i in range(n // 2):
            edges.append((i, n - 1 - i))
        return n, edges
    
    # k=3, target girth g (typically 16)
    if g % 2 == 0:
        moore = 2 * (2 ** (g // 2) - 1)
    else:
        moore = 2 * (2 ** (g // 2 - 1) - 1) + 2
    
    # Try random regular graphs with local search
    best_result = None
    best_n = float('inf')
    
    # Start from Moore bound and increase
    for n_try in range(moore, moore + 200, 2):  # n must be even for 3-regular
        if _time.time() > deadline - 10:
            break
        if n_try >= best_n:
            continue
        
        # Try multiple random seeds
        for seed in range(5):
            if _time.time() > deadline - 5:
                break
            rng = _random.Random(seed * 1000 + n_try)
            result = _random_regular_graph_3(n_try, rng)
            if result is None:
                continue
            n_r, edges = result
            adj = _build_adj_from_edges(edges, n_r)
            
            # Check girth
            if _verify_girth_fast_adj_list(adj, n_r, g):
                best_n = n_r
                best_result = (n_r, edges)
                return n_r, edges
            
            # Try local search to improve
            time_left = deadline - _time.time() - 5
            if time_left > 2:
                improved = _edge_swap_improve(adj, n_r, g, rng, min(time_left, 10))
                if improved is not None:
                    n_imp, edges_imp = improved
                    adj_imp = _build_adj_from_edges(edges_imp, n_imp)
                    if _verify_girth_fast_adj_list(adj_imp, n_imp, g):
                        best_n = n_imp
                        best_result = (n_imp, edges_imp)
                        return n_imp, edges_imp
    
    # Fallback: use a known small graph or return a trivial construction
    if best_result is not None:
        return best_result
    
    # Last resort: return a graph that may not satisfy girth but is valid 3-regular
    n = max(4, moore)
    if n % 2 != 0:
        n += 1
    rng = _random.Random(12345)
    result = _random_regular_graph_3(n, rng)
    if result is not None:
        return result
    
    # Absolute fallback
    n = 6
    edges = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]
    return n, edges
# EVOLVE-BLOCK-END
