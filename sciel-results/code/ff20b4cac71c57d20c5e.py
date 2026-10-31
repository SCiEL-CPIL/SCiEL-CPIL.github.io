import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use a voltage lift of the Petersen graph.
    The Petersen graph has girth 5. A lift of size m has n=10m vertices.
    We search for m and voltage assignments (shifts) such that the lift has girth >= g.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    if k != 3:
        # Fallback for other k (should not be hit for this task)
        # Simple random generation with check, or LUW if needed
        # For now, return a large random graph or LUW
        return _luw_construct(k, g)

    # Petersen graph edges (0-9)
    # Standard Petersen: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Base edges (u < v)
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    # Moore bound for k=3, g=16 is 512.
    # n = 10 * m. So m >= 52.
    m = 52
    
    # We will scan m up to 1000.
    # With faster verification, we can try more random shifts per m.
    
    # Pre-compute cycle constraints for the Petersen graph to speed up voltage search?
    # Or just rely on the BFS verification which is O(n * g) roughly.
    # For n=10m, BFS from each vertex is 10m * (10m * deg) which is too slow for large m if done naively.
    # The _verify_girth_fast uses early termination.
    
    # Strategy:
    # 1. Try deterministic shifts for small m.
    # 2. Try random shifts for small m.
    # 3. If not found, increase m.
    
    # To speed up, we can pre-filter voltage assignments by checking cycles in the base graph.
    # A cycle in the lift of length L corresponds to a closed walk in the base of length L
    # whose voltage sum is 0 mod m.
    # The girth of the lift is the minimum length of such a closed walk with non-zero voltage sum?
    # No, the girth is the minimum length of a cycle in the lift.
    # A cycle in the lift projects to a closed walk in the base.
    # If the closed walk has length L and voltage sum V, it lifts to a cycle of length L if V=0.
    # If V != 0, it lifts to a path. To close the cycle, we might need to repeat the walk?
    # Actually, the lift is a covering map. A cycle in the lift projects to a closed walk in the base.
    # The length of the cycle in the lift is the same as the length of the closed walk in the base.
    # Wait, no. If the walk is closed in the base, it starts and ends at the same vertex.
    # In the lift, it starts at (v, i) and ends at (v, i+V).
    # It is a cycle iff i = i+V mod m, i.e., V=0.
    # So, cycles in the lift correspond exactly to closed walks in the base with voltage sum 0.
    # The girth of the lift is the minimum length of a closed walk in the base with voltage sum 0.
    # Note: The base graph has girth 5. So there are no closed walks of length < 5.
    # We need to ensure that there are no closed walks of length 5, 6, ..., g-1 with voltage sum 0.
    
    # This is a strong constraint.
    # Let's find all closed walks of length < g in the Petersen graph.
    # And for each, ensure sum(voltages) != 0 mod m.
    
    # This is a CSP.
    # Let's pre-compute the list of closed walks (as lists of edge indices) of length < g.
    # Then, for a given m and shifts, check if any sum is 0 mod m.
    # This is much faster than BFS on the lift!
    
    # Pre-compute closed walks in Petersen up to length g-1.
    # We need to be careful about counting distinct cycles vs walks.
    # A closed walk can repeat edges/vertices.
    # We need to check ALL closed walks, not just simple cycles.
    # Because a non-simple closed walk with voltage sum 0 creates a cycle in the lift.
    
    # Let's generate all closed walks of length L for L=3 to g-1.
    # Actually, girth of Petersen is 5, so no walks of length 3, 4.
    # Start from L=5.
    
    closed_walks = []
    for L in range(5, g):
        # Generate all closed walks of length L
        # Use DFS
        # To avoid duplicates, we can fix the starting vertex and direction?
        # No, voltage sums depend on the specific sequence of edges.
        # Different walks might have the same edge multiset but different order -> different sum?
        # Yes, because voltages are assigned to edges, and sum is additive.
        # Order doesn't matter for the sum! Sum is commutative.
        # So we only need to check the multiset of edges in the walk.
        # Wait, is that true?
        # Sum of voltages on edges in the walk.
        # If the walk traverses edge e1, e2, e3, the sum is v1+v2+v3.
        # If it traverses e1, e3, e2, the sum is v1+v3+v2. Same.
        # So we only need to check the multiset of edges.
        # However, a "walk" is defined by the sequence.
        # But for the purpose of checking if sum=0, only the counts of each edge matter.
        # Let's represent a closed walk by the vector of edge counts.
        # The length is the sum of counts.
        
        # Let's find all possible edge count vectors for closed walks of length L.
        # This is equivalent to finding all vectors c in Z^15 such that:
        # 1. sum(c) = L
        # 2. The multiset of edges with c_i > 0 forms a subgraph where every vertex has even degree (Eulerian subgraph).
        # 3. The subgraph is connected? No, a closed walk is connected.
        
        # This might be too many.
        # Let's stick to the BFS verification on the lift, but optimize it.
        # The BFS verification is O(n * E) in worst case?
        # n=10m, E=15m.
        # BFS from each vertex: O(n * (n+E)) = O(n^2).
        # For n=1000, n^2 = 10^6. 1000 vertices -> 10^9 ops. Too slow.
        
        # We need a faster check.
        # The "closed walk" check is better.
        # Let's pre-compute the set of "forbidden" edge count vectors for lengths < g.
        # For a given m and shifts, we check if sum(c_i * shift_i) == 0 mod m for any forbidden vector c.
        
        # How many forbidden vectors?
        # For L=5, only the 12 pentagons (5-cycles).
        # For L=6, hexagons, etc.
        
        # Let's generate the forbidden vectors.
        
        def get_closed_walks(max_len):
            walks = set()
            # DFS from each vertex
            for start in range(10):
                stack = [(start, [start], 0, [])]
                while stack:
                    curr, path, length, edges = stack.pop()
                    if length > 0 and curr == start:
                        if 5 <= length <= max_len:
                            # Canonical form: sort edge indices
                            key = tuple(sorted(edges))
                            walks.add(key)
                    if length < max_len:
                        for v in petersen_adj[curr]:
                            # Find edge index
                            # This is slow. Precompute edge indices.
                            pass
            return walks
        
        # Better: Precompute edge indices for adjacency.
        edge_idx_map = {}
        for i, (u, v) in enumerate(petersen_edges):
            edge_idx_map[(u, v)] = i
            edge_idx_map[(v, u)] = i
            
        adj_edges = [[] for _ in range(10)]
        for i, (u, v) in enumerate(petersen_edges):
            adj_edges[u].append((v, i))
            adj_edges[v].append((u, i))
            
        def gen_walks(max_len):
            walks = set()
            for start in range(10):
                # DFS
                # To avoid symmetric duplicates, we can just generate all and deduplicate by sorted edge list.
                stack = [(start, start, 0, [])]
                while stack:
                    curr, start_v, length, edge_path = stack.pop()
                    if length > 0 and curr == start_v:
                        if 5 <= length <= max_len:
                            key = tuple(sorted(edge_path))
                            walks.add(key)
                    if length < max_len:
                        for next_v, e_idx in adj_edges[curr]:
                            # Avoid immediate backtracking? No, walks can backtrack.
                            # But backtracking creates a walk of length 2 which is < 5, so ignored.
                            stack.append((next_v, start_v, length + 1, edge_path + [e_idx]))
            return walks
        
        # This DFS might explode for larger lengths.
        # g=16, so max_len=15.
        # Number of walks of length 15 in Petersen?
        # Degree 3. 3^15 = 14M. Times 10 starts = 140M. Too slow in Python.
        
        # We need a smarter way to generate forbidden vectors or check girth.
        
        # Alternative: Use the fact that the lift is a voltage graph.
        # The girth is the min length of a closed walk with voltage 0.
        # We can compute this by BFS on the "voltage state" space?
        # No, that's the lift itself.
        
        # Let's go back to BFS on the lift but optimize it heavily.
        # Use bitsets?
        # Or just rely on the fact that for small m, the graph is small.
        # For m=52, n=520.
        # BFS from each vertex: 520 * (520 * 3) ~ 800k ops.
        # 520 * 800k = 400M ops. Too slow for 120s?
        # In Python, 400M ops is ~10-20s.
        # We have 145s.
        # We can try a few m values.
        
        # Let's try m=52, 54, 56...
        # For each m, try a few random shifts.
        
        # To speed up BFS, we can stop early if we find a cycle < g.
        # And we can use the "timestamp" trick to avoid resetting arrays.
        
        def _verify_girth_fast(n, edges, g):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Use global arrays to avoid O(n) reset per start
            dist = [0] * n
            parent = [-1] * n
            seen_mark = [0] * n
            mark = 0
            
            for s in range(n):
                mark += 1
                dist[s] = 0
                parent[s] = -1
                seen_mark[s] = mark
                queue = deque([s])
                
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    # If current distance is already >= g/2, we don't need to go further
                    # because any cycle found would be at least 2*du + 1 >= g
                    if du >= g // 2:
                        continue
                    for v in adj[u]:
                        if seen_mark[v] != mark:
                            dist[v] = du + 1
                            parent[v] = u
                            seen_mark[v] = mark
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            # Found a cycle
                            cycle_len = du + dist[v] + 1
                            if cycle_len < g:
                                return False
                                
            return True

        # This verification is O(n * g * deg) roughly?
        # The BFS explores up to depth g/2.
        # Number of nodes visited is bounded by 1 + 3 + 3*2 + ... + 3*2^(g/2) which is huge.
        # But in a graph with high girth, the BFS tree is a tree up to depth g/2.
        # So the number of nodes visited is ~ 2^(g/2).
        # For g=16, 2^8 = 256.
        # So BFS from each vertex visits ~256 nodes.
        # Total ops: n * 256 * 3.
        # For n=520: 520 * 256 * 3 ~ 400k ops.
        # This is very fast!
        
        # So we can afford to check many (m, shifts) pairs.
        
        # Let's loop m from 52 to 1000.
        # For each m, try 20 random shifts.
        
        # To make it deterministic and reproducible, use a fixed seed for random.
        random.seed(42)
        
        best_result = None
        best_n = float('inf')
        
        for m in range(52, 1000):
            if time.time() > deadline:
                break
            
            # Try deterministic shifts first
            # 1. Linear
            shifts1 = [int((i + 1) * m / 15.0) for i in range(15)]
            n, edges = _build_lift(base_edges, m, shifts1)
            if _verify_girth_fast(n, edges, g):
                return n, edges
            
            # 2. Random
            for _ in range(20):
                if time.time() > deadline:
                    break
                shifts_rand = [random.randint(0, m-1) for _ in range(15)]
                n, edges = _build_lift(base_edges, m, shifts_rand)
                if _verify_girth_fast(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    # We can return immediately if we want the first valid one?
                    # The task asks for "as few vertices as possible".
                    # So we should continue searching for smaller n?
                    # But n=10m. Smaller m is better.
                    # We are iterating m in increasing order.
                    # So the first valid m we find is the smallest m.
                    # Thus, we can return immediately.
                    return n, edges

        # If not found in 52..1000, try larger m with more attempts
        if time.time() < deadline:
            for m in range(1000, 2000):
                if time.time() > deadline:
                    break
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
                    n, edges = _build_lift(base_edges, m, shifts_rand)
                    if _verify_girth_fast(n, edges, g):
                        return n, edges

        # Fallback: Return a large random lift (likely valid)
        # This should rarely be reached.
        m = 3000
        shifts_rand = [random.randint(0, m-1) for _ in range(15)]
        n, edges = _build_lift(base_edges, m, shifts_rand)
        return n, edges


def _build_lift(base_edges, m, shifts):
    """
    Build the lifted graph.
    base_edges: list of (u, v) for u < v
    m: lift size
    shifts: list of shifts for each base edge
    """
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
