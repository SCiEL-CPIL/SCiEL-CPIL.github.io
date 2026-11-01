"""
Construct a k-regular graph with girth at least g.
Combined approach:
1. Fast circulant C_n(1,a,b) search with algebraic pre-filters (refined).
2. Randomized greedy tree-extension (BFS layering from seed tree).
3. Voltage lift over Z_m (Petersen base).
4. LUW fallback.
"""
import time
import random
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Systematic circulant C_n(1,a,b) search
        result = _find_circulant_systematic(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: Randomized greedy tree-extension
        if not check_deadline():
            result2 = _greedy_tree_extension(3, 16, deadline)
            if result2 is not None:
                return result2

        # Strategy 3: Voltage lift over Z_m
        if not check_deadline():
            result3 = _voltage_lift_search(3, 16, deadline)
            if result3 is not None:
                return result3

        # Strategy 4: LUW fallback
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _find_circulant_systematic(k, g, deadline):
    """
    Systematic search for C_n(1,a,b) with girth >= g.
    Vertex-transitive: check girth from vertex 0 only.
    Algebraic pre-filters for 4-cycles and 6-cycles.
    """
    min_n = 512
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        half = n // 2
        # Heuristic: try a near n/3 first
        a_candidates = list(range(max(3, n // 4), half))
        a_candidates.sort(key=lambda a: abs(a - n // 3))

        for a in a_candidates:
            if time.time() > deadline - 3:
                return None

            two_a = (2 * a) % n
            if two_a in (0, 1, n - 1):
                continue

            for b in range(a + 1, half):
                if time.time() > deadline - 3:
                    return None

                if _has_4cycle_fast(n, a, b):
                    continue

                if _has_6cycle_fast(n, a, b):
                    continue

                if _girth_check_circ(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle_fast(n, a, b):
    """Check if C_n(1,a,b) has a 4-cycle."""
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    gens_set = set(gens)
    
    for i in range(6):
        g1 = gens[i]
        for j in range(6):
            g2 = gens[j]
            if g2 == (n - g1) % n:
                continue
            s12 = (g1 + g2) % n
            for k in range(6):
                g3 = gens[k]
                if g3 == (n - g2) % n:
                    continue
                s123 = (s12 + g3) % n
                g4 = (n - s123) % n
                if g4 == (n - g3) % n:
                    continue
                if g4 in gens_set:
                    v1 = g1
                    v2 = s12
                    v3 = s123
                    if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                        return True
    return False


def _has_6cycle_fast(n, a, b):
    """Check if C_n(1,a,b) has a 6-cycle using BFS to depth 3."""
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    
    dist = {0: 0}
    parent = {0: -1}
    level = [0]
    
    for depth in range(3):
        next_level = []
        for u in level:
            for g in gens:
                v = (u + g) % n
                if v == 0 and depth < 2:
                    continue
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    # Check for 6-cycles: two vertices at depth 2 sharing a neighbor at depth 3
    depth2_vertices = [v for v in dist if dist[v] == 2]
    depth3_vertices = [v for v in dist if dist[v] == 3]
    
    for v3 in depth3_vertices:
        neighbors_at_depth2 = set()
        for g in gens:
            w = (v3 + g) % n
            if w in dist and dist[w] == 2:
                neighbors_at_depth2.add(w)
        
        if len(neighbors_at_depth2) >= 2:
            # Check if these depth-2 vertices have different parents at depth 1
            parents = set()
            for w in neighbors_at_depth2:
                p = parent[w]
                parents.add(p)
            if len(parents) >= 2:
                return True
            elif len(parents) == 1:
                # Same parent: 0->p->w1->v3->w2->p->0 is a 5-cycle
                # Check if this is actually a valid 5-cycle
                p = parent[list(neighbors_at_depth2)[0]]
                w1 = list(neighbors_at_depth2)[0]
                w2 = list(neighbors_at_depth2)[1]
                # Verify all 5 vertices distinct: 0, p, w1, v3, w2
                if p != 0 and w1 != 0 and w2 != 0 and w1 != w2 and p != w1 and p != w2:
                    return True
    
    return False


def _girth_check_circ(n, a, b, g):
    """
    Non-backtracking BFS from vertex 0 in C_n(1,a,b).
    Returns True if girth >= g.
    """
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    
    best_depth = {}
    queue = deque()
    
    start = (0, -1)
    best_depth[start] = 0
    queue.append(start)
    
    max_len = g - 1
    
    while queue:
        u, prev = queue.popleft()
        d = best_depth[(u, prev)]
        
        if d >= max_len:
            continue
        
        for g_step in gens:
            v = (u + g_step) % n
            if v == prev:
                continue
            
            ns = (v, u)
            nd = d + 1
            
            if nd > max_len:
                continue
            
            if ns not in best_depth or nd < best_depth[ns]:
                best_depth[ns] = nd
                
                if v == 0 and u != -1:
                    if nd < g:
                        return False
                else:
                    queue.append(ns)
    
    return True


def _greedy_tree_extension(k, g, deadline):
    """
    Randomized greedy tree-extension algorithm.
    Start from a seed 3-regular tree (BFS layering), then incrementally
    add vertices by attaching to existing leaves via random edges.
    Any resulting cycle shorter than g is resolved by pruning and re-routing.
    """
    random.seed(42)
    
    # Start with a 3-regular tree of depth 7 (since g=16, we need no cycle < 16)
    # A 3-regular tree of depth 7 has 1 + 3*(1 + 3 + 9 + 27 + 81 + 243 + 729) = 1 + 3*1093 = 3280 vertices
    # That's too many. Let's start with a smaller seed and grow.
    
    # Actually, for girth 16, we need the graph to have no cycle shorter than 16.
    # Start with a tree (no cycles at all), then add edges carefully.
    
    # Strategy: build a 3-regular graph by starting with a tree and adding edges
    # that don't create short cycles.
    
    # Use a BFS-based approach:
    # 1. Start with vertex 0.
    # 2. Grow a 3-regular tree to some depth.
    # 3. Add edges between leaves at sufficient distance.
    
    # For girth 16, two vertices can be connected by an edge only if their
    # distance in the current graph is at least 7 (so that the resulting cycle
    # has length at least 2*7 = 14... wait, that's not quite right).
    
    # Actually, if we have a tree and add an edge between u and v, the cycle
    # created has length dist(u,v) + 1. We need this to be >= 16, so
    # dist(u,v) >= 15.
    
    # Let's try a different approach: use a "cage construction" heuristic.
    # Build a 3-regular graph with n vertices, trying to maximize girth.
    
    # For n around 512-936, use a randomized approach:
    # 1. Start with a random 3-regular graph on n vertices.
    # 2. Check girth. If girth < 16, try to fix by edge swaps.
    
    # Actually, let's try a more systematic approach:
    # Use the "random regular graph" approach with girth checking.
    
    for n in [512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 536,
              538, 540, 542, 544, 546, 548, 550]:
        if time.time() > deadline - 5:
            break
        
        # Try to construct a 3-regular graph on n vertices with girth >= 16
        result = _random_regular_graph_with_girth(n, 3, 16, deadline)
        if result is not None:
            return result
    
    return None


def _random_regular_graph_with_girth(n, k, g, deadline):
    """
    Try to construct a k-regular graph on n vertices with girth >= g
    using random edge swaps.
    """
    if n % 2 == 0 and n * k % 2 != 0:
        return None
    
    # Start with a random k-regular graph
    adj = _random_regular_graph(n, k)
    if adj is None:
        return None
    
    # Check girth
    if _girth_check_graph(adj, n, g):
        edges = _adj_to_edges(adj, n)
        return n, edges
    
    # Try edge swaps to improve girth
    max_swaps = 5000
    for _ in range(max_swaps):
        if time.time() > deadline - 5:
            break
        
        # Find a short cycle and try to break it
        short_cycle = _find_short_cycle(adj, n, g)
        if short_cycle is None:
            # No short cycle found
            edges = _adj_to_edges(adj, n)
            return n, edges
        
        # Try to break the cycle by edge swaps
        improved = False
        for i in range(len(short_cycle) - 1):
            u = short_cycle[i]
            v = short_cycle[(i + 1) % len(short_cycle)]
            
            # Try to swap edge (u,v) with some other edge
            for w in adj[u]:
                if w == v:
                    continue
                # Find a neighbor x of w such that (v,x) is not an edge
                for x in adj[w]:
                    if x == u:
                        continue
                    if x not in adj[v]:
                        # Swap: remove (u,v) and (w,x), add (u,x) and (v,w)
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[w].remove(x)
                        adj[x].remove(w)
                        
                        adj[u].append(x)
                        adj[x].append(u)
                        adj[v].append(w)
                        adj[w].append(v)
                        
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Final check
    if _girth_check_graph(adj, n, g):
        edges = _adj_to_edges(adj, n)
        return n, edges
    
    return None


def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    edges = set()
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            continue
        if (u, v) in edges or (v, u) in edges:
            continue
        adj[u].append(v)
        adj[v].append(u)
        edges.add((u, v))
        edges.add((v, u))
    
    # Check if all vertices have degree k
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    return adj


def _girth_check_graph(adj, n, g):
    """Check if graph has girth >= g using BFS from each vertex."""
    for src in range(n):
        if _bfs_short_cycle(adj, n, src, g):
            return False
    return True


def _bfs_short_cycle(adj, n, src, g):
    """BFS from src to find if there's a cycle shorter than g."""
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    
    max_depth = g // 2
    
    while queue:
        u = queue.popleft()
        depth = dist[u]
        
        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue
        
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True
    
    return False


def _find_short_cycle(adj, n, g):
    """Find a short cycle in the graph. Returns list of vertices or None."""
    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if parent[v] != u and parent[u] != v:
                        cycle_len = dist[u] + 1 + dist[v]
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = [u, v]
                            # Trace back from u to LCA
                            u_path = []
                            x = u
                            while x != -1:
                                u_path.append(x)
                                x = parent[x]
                            
                            v_path = []
                            x = v
                            while x != -1:
                                v_path.append(x)
                                x = parent[x]
                            
                            # Find LCA
                            u_set = set(u_path)
                            lca = None
                            for node in v_path:
                                if node in u_set:
                                    lca = node
                                    break
                            
                            if lca is not None:
                                # Build cycle
                                cycle = []
                                x = u
                                while x != lca:
                                    cycle.append(x)
                                    x = parent[x]
                                cycle.append(lca)
                                x = v
                                while x != lca:
                                    cycle.append(x)
                                    x = parent[x]
                                cycle.reverse()
                                return cycle
    return None


def _adj_to_edges(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _voltage_lift_search(k, g, deadline):
    """
    Voltage lift of the Petersen graph over Z_m.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    best_result = None
    best_n = float('inf')
    
    for m in range(51, 150):
        if time.time() > deadline - 5:
            break
        
        for trial in range(8):
            if time.time() > deadline - 3:
                break
            
            voltages = {}
            for idx, (u, v) in enumerate(petersen_edges):
                if trial == 0:
                    vol = (idx * 7 + 3) % m
                elif trial == 1:
                    vol = (u * 3 + v * 7 + idx) % m
                elif trial == 2:
                    vol = (idx * idx + 2 * idx + 1) % m
                elif trial == 3:
                    vol = (u * v + idx * 3) % m
                elif trial == 4:
                    vol = (idx * 13 + 11) % m
                elif trial == 5:
                    vol = (u * 11 + v * 17 + 1) % m
                elif trial == 6:
                    vol = (idx * 19 + u * 5) % m
                else:
                    seed = (idx * 2654435761 + trial * 12345) % (2**31)
                    vol = (seed * 1103515245 + 12345) % (2**31) % m
                
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
            
            if _lift_girth_check_fast(petersen_adj, voltages, m, g):
                n = 10 * m
                if n < best_n:
                    edges = []
                    for u, v in petersen_edges:
                        vol = voltages[(u, v)]
                        for i in range(m):
                            a = u * m + i
                            b = v * m + ((i + vol) % m)
                            if a < b:
                                edges.append((a, b))
                            else:
                                edges.append((b, a))
                    if len(edges) == 15 * m:
                        best_n = n
                        best_result = (n, edges)
                    break
    
    return best_result


def _lift_girth_check_fast(petersen_adj, voltages, m, g):
    """Check if voltage lift has girth >= g."""
    for start in range(10):
        if _lift_bfs_from_fast(start, petersen_adj, voltages, m, g):
            return False
    return True


def _lift_bfs_from_fast(start, petersen_adj, voltages, m, g):
    """Non-backtracking BFS from 'start' tracking voltage."""
    max_len = g - 1
    visited = set()
    queue = deque()
    
    start_state = (start, -1, 0)
    visited.add(start_state)
    queue.append((start, -1, 0, 0))
    
    while queue:
        u, prev, vol, depth = queue.popleft()
        
        if depth >= max_len:
            continue
        
        for v in petersen_adj[u]:
            if v == prev:
                continue
            
            new_vol = (vol + voltages[(u, v)]) % m
            new_state = (v, u, new_vol)
            
            if new_state in visited:
                continue
            
            visited.add(new_state)
            new_depth = depth + 1
            
            if v == start and new_vol == 0 and new_depth >= 2:
                if new_depth < g:
                    return True
            
            if new_depth < max_len:
                queue.append((v, u, new_vol, new_depth))
    
    return False


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _verify_girth_fast(n, edges, g):
    """Fast girth verification using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(adj, n, src, g):
            return False

    return True

# EVOLVE-BLOCK-END
