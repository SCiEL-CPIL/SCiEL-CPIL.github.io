"""Construct a K4-free graph with no independent 6-set, maximizing n.

Strategy (structurally different from seed):
1. Start with a known-valid base (Paley(17) or circulant).
2. GREEDY EDGE-ACCUMULATION: for each pair of non-adjacent vertices, try
   adding the edge; keep it only if the graph remains K4-free AND alpha <= 5.
   This builds a maximal valid graph for the given n.
3. GROWTH: to increase n, add a new vertex with a structured neighborhood
   chosen to minimize new K4s and new independent-6-sets, then run a
   MAX-FLOW edge-deletion repair (Dinic's algorithm) to remove a minimum
   set of edges that destroys all K4s while keeping alpha <= 5.
4. Repeat until time budget is exhausted.
"""

import time
import random
import sys

_START = time.time()
DEADLINE = 140.0  # safety margin before 150s kill


def _time_left():
    return DEADLINE - (time.time() - _START)


def construct():
    global _START
    _START = time.time()

    # --- Phase 1: Build a valid base ---
    adj, n = _build_base()
    if adj is None:
        # Absolute fallback
        adj, n = _paley_graph(17)

    # --- Phase 2: Greedy edge accumulation to make the graph maximal ---
    adj = _greedy_edge_accumulate(adj, n)

    # --- Phase 3: Grow vertex count ---
    adj, n = _grow_with_flow_repair(adj, n)

    # --- Phase 4: Final greedy edge accumulation on the grown graph ---
    adj = _greedy_edge_accumulate(adj, n)

    # Final validation
    if not _is_valid(adj, n):
        # Try to repair
        adj = _repair_flow(adj, n)
        if adj is None or not _is_valid(adj, n):
            # Fall back to base
            adj2, n2 = _paley_graph(17)
            adj2 = _greedy_edge_accumulate(adj2, n2)
            return {'n': n2, 'edges': _adj_to_edges(adj2, n2)}

    return {'n': n, 'edges': _adj_to_edges(adj, n)}


def _paley_graph(q):
    """Paley graph on q vertices (q prime, q ≡ 1 mod 4). Returns (adj, n)."""
    adj = [set() for _ in range(q)]
    qr = set()
    for x in range(1, q):
        if pow(x, (q - 1) // 2, q) == 1:
            qr.add(x)
    for i in range(q):
        for j in range(i + 1, q):
            d = (j - i) % q
            if d in qr:
                adj[i].add(j)
                adj[j].add(i)
    return adj, q


def _build_base():
    """Try several base constructions, return the best valid one."""
    # Try Paley(17)
    adj, n = _paley_graph(17)
    if _is_valid(adj, n):
        return adj, n

    # Try circulant constructions
    for n_try in range(25, 37):
        if _time_left() < 20:
            break
        adj = _try_circulant(n_try)
        if adj is not None:
            return adj, n_try

    # Fallback
    adj, n = _paley_graph(17)
    return adj, n


def _try_circulant(n):
    """Try to find a valid circulant graph on n vertices."""
    rng = random.Random(n * 7 + 13)
    candidates = []
    for k in [3, 4, 5, 6, 7, 8]:
        if 2 * k < n:
            candidates.append(set(range(1, k + 1)))
    for k in [2, 3, 4, 5]:
        d = set()
        m = k
        while m <= n // 2:
            d.add(m)
            m += k
        if d:
            candidates.append(d)
    for size in [n // 4, n // 3, n // 2 - 2]:
        for _ in range(5):
            d = set(rng.sample(range(1, n // 2 + 1), min(size, n // 2)))
            candidates.append(d)

    for D in candidates:
        D = {d % n for d in D if d % n != 0}
        D = {min(d, n - d) for d in D}
        if not D:
            continue
        adj = [set() for _ in range(n)]
        for i in range(n):
            for d in D:
                j = (i + d) % n
                if j > i:
                    adj[i].add(j)
                    adj[j].add(i)
        if _is_valid(adj, n):
            return adj
    return None


def _greedy_edge_accumulate(adj, n):
    """Add edges greedily: for each non-edge, add it if graph stays valid.
    
    Key insight: we add edges to REDUCE the independence number (helping alpha<=5)
    while maintaining K4-freeness. This makes the graph denser in a controlled way.
    """
    adj = [set(s) for s in adj]
    rng = random.Random(42)
    
    # Build list of non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
    
    # Shuffle for diversity
    rng.shuffle(non_edges)
    
    for (u, v) in non_edges:
        if _time_left() < 5:
            break
        if v in adj[u]:
            continue  # already added by a previous step
        
        # Try adding edge (u, v)
        adj[u].add(v)
        adj[v].add(u)
        
        if _is_valid(adj, n):
            pass  # keep the edge
        else:
            # Remove the edge
            adj[u].discard(v)
            adj[v].discard(u)
    
    return adj


def _grow_with_flow_repair(adj, n):
    """Grow the graph by adding vertices, using flow-based K4 repair."""
    rng = random.Random(999)
    
    for target_n in range(n + 1, 41):
        if _time_left() < 8:
            break
        
        success = False
        for attempt in range(40):
            if _time_left() < 4:
                break
            
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            
            # Choose neighborhood for new vertex
            # Strategy: connect to a random subset, biased toward vertices
            # that are "central" (high degree) to avoid creating independent sets
            degs = [len(adj[i]) for i in range(n)]
            avg_deg = sum(degs) / max(n, 1)
            
            # Connect to ~30-50% of vertices, preferring higher-degree ones
            candidates = list(range(n))
            # Weight by degree
            weights = [degs[i] + 1 for i in range(n)]
            total_w = sum(weights)
            num_connect = max(1, int(new_n * 0.35))
            
            # Weighted random sample
            connected = set()
            while len(connected) < num_connect and candidates:
                r = rng.random() * total_w
                cum = 0
                for idx, i in enumerate(candidates):
                    cum += weights[idx]
                    if r <= cum:
                        connected.add(i)
                        candidates.pop(idx)
                        total_w -= weights[idx]
                        break
            
            for i in connected:
                new_adj[new_n - 1].add(i)
                new_adj[i].add(new_n - 1)
            
            # Check validity; if invalid, try flow-based repair
            if _is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                success = True
                break
            else:
                # Try flow-based repair: remove minimum edges to kill K4s
                repaired = _repair_flow(new_adj, new_n)
                if repaired is not None and _is_valid(repaired, new_n):
                    adj = repaired
                    n = new_n
                    success = True
                    break
        
        if not success:
            break
    
    return adj, n


def _repair_flow(adj, n):
    """Repair graph by removing a minimum set of edges to destroy all K4s,
    while maintaining alpha <= 5.
    
    Uses a max-flow formulation:
    - Source -> K4 nodes (capacity 1)
    - K4 nodes -> edge nodes (capacity 1) for each edge in the K4
    - Edge nodes -> sink (capacity 1)
    
    The min-cut gives a minimum set of edges whose removal destroys all K4s.
    """
    adj = [set(s) for s in adj]
    
    # Find all K4s
    k4s = _find_all_k4s(adj, n)
    if not k4s:
        # No K4s; check alpha
        if not _has_ind6(adj, n):
            return adj
        # Need to add edges to break independent sets
        return _repair_alpha(adj, n)
    
    # Build flow network
    # Nodes: source, sink, one per K4, one per edge
    edge_list = []
    edge_id = {}
    for i in range(n):
        for j in adj[i]:
            if i < j:
                eid = len(edge_list)
                edge_list.append((i, j))
                edge_id[(i, j)] = eid
    
    num_edges = len(edge_list)
    num_k4s = len(k4s)
    
    # Node indexing:
    # 0 = source
    # 1..num_k4s = K4 nodes
    # num_k4s+1 .. num_k4s+num_edges = edge nodes
    # num_k4s+num_edges+1 = sink
    source = 0
    sink = num_k4s + num_edges + 1
    total_nodes = sink + 1
    
    # Build adjacency for Dinic
    # graph[node] = list of (neighbor, capacity, reverse_index)
    graph = [[] for _ in range(total_nodes)]
    
    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])
    
    # Source -> K4 nodes
    for ki in range(num_k4s):
        add_edge(source, 1 + ki, 1)
    
    # K4 nodes -> edge nodes
    for ki, k4 in enumerate(k4s):
        a, b, c, d = k4
        pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        for (u, v) in pairs:
            if u > v:
                u, v = v, u
            if (u, v) in edge_id:
                eid = edge_id[(u, v)]
                add_edge(1 + ki, 1 + num_k4s + eid, 1)
    
    # Edge nodes -> sink
    for ei in range(num_edges):
        add_edge(1 + num_k4s + ei, sink, 1)
    
    # Run max flow
    flow = _dinic(graph, source, sink, total_nodes)
    
    # Find min cut: edges from source-side to sink-side
    # BFS from source in residual graph
    visited = [False] * total_nodes
    visited[source] = True
    queue = [source]
    while queue:
        u = queue.pop(0)
        for (v, cap, _) in graph[u]:
            if cap > 0 and not visited[v]:
                visited[v] = True
                queue.append(v)
    
    # Edges to remove: edge nodes on sink side that had flow
    # Actually, the min cut edges are those from source-side to sink-side
    # For edge nodes: if edge node is on sink side, the edge should be removed
    edges_to_remove = set()
    for ei in range(num_edges):
        node = 1 + num_k4s + ei
        if not visited[node]:
            # This edge node is on sink side -> remove this edge
            u, v = edge_list[ei]
            edges_to_remove.add((u, v))
    
    # Remove edges
    for (u, v) in edges_to_remove:
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Now check alpha; if still has independent 6-set, add edges
    if _has_ind6(adj, n):
        result = _repair_alpha(adj, n)
        return result
    
    if _is_valid(adj, n):
        return adj
    
    return None


def _repair_alpha(adj, n):
    """Repair alpha violation by adding edges within independent 6-sets."""
    adj = [set(s) for s in adj]
    rng = random.Random(777)
    
    for _ in range(200):
        if _time_left() < 2:
            break
        if not _has_ind6(adj, n):
            break
        ind_set = _find_ind6_vertices(adj, n)
        if ind_set is None:
            break
        # Add a random edge within the independent set
        u, v = ind_set[rng.randrange(len(ind_set))]
        adj[u].add(v)
        adj[v].add(u)
        # This might create K4s; if so, we'll need to remove some edges
        if _has_k4(adj, n):
            # Remove the edge we just added and try a different one
            adj[u].discard(v)
            adj[v].discard(u)
            # Try a different pair
            for _ in range(10):
                u2, v2 = ind_set[rng.randrange(len(ind_set))]
                adj[u2].add(v2)
                adj[v2].add(u2)
                if not _has_k4(adj, n):
                    break
                adj[u2].discard(v2)
                adj[v2].discard(u2)
    
    if _is_valid(adj, n):
        return adj
    return None


def _dinic(graph, source, sink, total_nodes):
    """Dinic's max-flow algorithm. Returns max flow value."""
    n = total_nodes
    flow = 0
    INF = float('inf')
    
    while True:
        # BFS to build level graph
        level = [-1] * n
        level[source] = 0
        queue = [source]
        while queue:
            u = queue.pop(0)
            for (v, cap, _) in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        
        if level[sink] < 0:
            break
        
        # DFS to find blocking flow
        it = [0] * n
        
        def dfs(u, f):
            if u == sink:
                return f
            while it[u] < len(graph[u]):
                v, cap, rev = graph[u][it[u]]
                if cap > 0 and level[v] == level[u] + 1:
                    d = dfs(v, min(f, cap))
                    if d > 0:
                        graph[u][it[u]][1] -= d
                        graph[v][rev][1] += d
                        return d
                it[u] += 1
            return 0
        
        while True:
            f = dfs(source, INF)
            if f == 0:
                break
            flow += f
    
    return flow


def _find_all_k4s(adj, n, limit=500):
    """Find all K4s (up to limit). Returns list of (a,b,c,d) tuples."""
    bits = [0] * n
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        bits[i] = b
    
    k4s = []
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            common = bits[a] & bits[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            c_list = []
            c = common
            while c:
                low = c & (-c)
                c -= low
                c_list.append(low.bit_length() - 1)
            for idx, c in enumerate(c_list):
                for d in c_list[idx + 1:]:
                    if (bits[c] >> d) & 1:
                        k4s.append((a, b, c, d))
                        if len(k4s) >= limit:
                            return k4s
    return k4s


def _find_ind6_vertices(adj, n):
    """Find an independent set of size 6. Returns list of (u,v) pairs or None."""
    all_bits = (1 << n) - 1
    non_adj = [0] * n
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        non_adj[i] = all_bits ^ b ^ (1 << i)
    
    def backtrack(candidates, current):
        if len(current) >= 6:
            return current[:]
        if not candidates:
            return None
        low = candidates & (-candidates)
        v = low.bit_length() - 1
        rest = candidates & ~low
        new_cands = rest & non_adj[v]
        result = backtrack(new_cands, current + [v])
        if result is not None:
            return result
        return backtrack(rest, current)
    
    vertices = backtrack(all_bits, [])
    if vertices is None:
        return None
    return [(vertices[i], vertices[j]) for i in range(len(vertices)) for j in range(i + 1, len(vertices))]


def _has_k4(adj, n):
    """Check for K4. Returns True if found."""
    bits = [0] * n
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        bits[i] = b
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            common = bits[a] & bits[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            c_list = []
            c = common
            while c:
                low = c & (-c)
                c -= low
                c_list.append(low.bit_length() - 1)
            for idx, c in enumerate(c_list):
                for d in c_list[idx + 1:]:
                    if (bits[c] >> d) & 1:
                        return True
    return False


def _has_ind6(adj, n):
    """Check for independent set of size 6."""
    all_bits = (1 << n) - 1
    non_adj = [0] * n
    for i in range(n):
        b = 0
        for j in adj[i]:
            b |= (1 << j)
        non_adj[i] = all_bits ^ b ^ (1 << i)
    
    def backtrack(candidates, current_size):
        if current_size >= 6:
            return True
        if not candidates:
            return False
        # Pruning: if we can't reach size 6
        if current_size + _popcount(candidates) < 6:
            return False
        low = candidates & (-candidates)
        v = low.bit_length() - 1
        rest = candidates & ~low
        if backtrack(rest & non_adj[v], current_size + 1):
            return True
        if backtrack(rest, current_size):
            return True
        return False
    
    return backtrack(all_bits, 0)


def _popcount(x):
    """Count set bits in integer."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _is_valid(adj, n):
    """Check K4-free and no independent 6-set."""
    if _has_k4(adj, n):
        return False
    if _has_ind6(adj, n):
        return False
    return True


def _adj_to_edges(adj, n):
    """Convert adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
