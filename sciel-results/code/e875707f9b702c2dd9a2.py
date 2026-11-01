"""Construct a 3-regular graph of girth >= 16 with as few vertices as possible."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    deadline = time.time() + 145.0  # 145s to be safe under 150s limit
    best_result = None
    best_n = float('inf')

    # Moore bound for 3-regular girth 16:
    # For even girth g=2m: n >= 2 * sum_{i=0}^{m-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*255 = 510
    # Actually for cage problem, Moore bound for (3,16): n >= 2*(1+2+4+8+16+32+64+128) = 2*255 = 510
    # But the known lower bound is 512. Let's use 512 as a practical lower bound.
    moore_bound = 512

    # Strategy: Random lifting of a base graph
    # We'll try to construct a 3-regular bipartite graph with girth >= 16
    # using random lifts of smaller graphs.

    # Approach: Generate random 3-regular bipartite graphs of size 2m
    # and check girth. Use non-backtracking walk to verify girth.

    # For efficiency, we'll use the following strategy:
    # 1. Start with a random 3-regular bipartite graph
    # 2. Check if girth >= 16 using BFS/non-backtracking
    # 3. If not, try to repair or generate new ones

    # A more efficient approach: use the "random lift" technique
    # Take a small base graph and lift it randomly

    # Actually, let's use a direct approach: generate random 3-regular
    # bipartite graphs and test girth.

    def generate_random_3_reg_bipartite(n_half, rng):
        """Generate a random 3-regular bipartite graph with n_half vertices on each side."""
        n = 2 * n_half
        # Each left vertex connects to 3 right vertices
        # Use configuration model
        left_ports = [(i, j) for i in range(n_half) for j in range(3)]
        right_ports = [(n_half + i, j) for i in range(n_half) for j in range(3)]

        rng.shuffle(left_ports)
        rng.shuffle(right_ports)

        edges = []
        seen = set()
        for (u, _), (v, _) in zip(left_ports, right_ports):
            if (u, v) not in seen and u != v:
                edges.append((u, v))
                seen.add((u, v))

        # Check if we got exactly 3*n_half edges (no duplicates)
        if len(edges) != 3 * n_half:
            return None

        return edges

    def check_girth_bipartite(n_half, edges, min_girth, deadline):
        """Check if a 3-regular bipartite graph has girth >= min_girth.
        Uses non-backtracking walk from each vertex.
        Returns True if girth >= min_girth, False otherwise."""
        n = 2 * n_half
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return False

        # Non-backtracking walk to find shortest cycle
        # For each directed edge, do BFS avoiding backtracking
        # A cycle of length L exists iff there's a non-backtracking walk
        # of length L-1 from (u,v) to (v,u) for some edge (u,v)

        # Actually, simpler: BFS from each vertex, tracking parent to avoid backtracking
        # But that's O(n^2) which is too slow for n=936

        # Better: use the fact that for bipartite graphs, girth is even.
        # We need to check no cycle of length 4, 6, 8, 10, 12, 14.

        # Efficient method: for each pair of vertices at distance 2,
        # check if they share a common neighbor (cycle of length 4)
        # For longer cycles, use matrix powers or BFS with depth limit.

        # For girth >= 16, we need to verify no cycles of length 4,6,8,10,12,14.
        # This is expensive. Let's use a smarter approach.

        # Use the "girth via BFS from each vertex up to depth g/2 - 1"
        # If any two neighbors of a vertex at depth < g/2 - 1 are connected
        # by a path of length < g - 2d, we have a short cycle.

        # Actually, the standard method: for each vertex, do BFS up to depth g//2 - 1.
        # If we find a vertex at depth d that also appears at depth d' < d in the same
        # BFS tree, we have a cycle. But this doesn't quite work directly.

        # Let me use a different approach: count the number of non-backtracking
        # walks of length L from each edge. If for L = girth-1, we find a walk
        # from (u,v) back to (v,u), there's a cycle of length girth.

        # For practical purposes with n <= ~1000 and girth 16, let's do:
        # BFS from each vertex up to depth 7 (g/2 - 1 = 7), tracking the path.
        # If we revisit a vertex at a different depth, we have a cycle.

        # But this is O(n * 3^7) = O(n * 2187) which for n=936 is about 2M operations.
        # That's feasible.

        for start in range(n):
            if time.time() > deadline:
                return None  # timeout, unknown
            # BFS from start, tracking depth and parent
            visited = {}  # vertex -> (depth, parent)
            queue = deque()
            queue.append((start, 0, -1))
            visited[start] = (0, -1)

            while queue:
                if time.time() > deadline:
                    return None
                v, depth, parent = queue.popleft()

                if depth >= min_girth // 2 - 1:
                    continue

                for w in adj[v]:
                    if w == parent:
                        continue  # no backtracking
                    if w in visited:
                        # Found a cycle
                        # Cycle length = depth + 1 + visited[w][0]
                        # But we need to be careful: this is the cycle length
                        # if the path from start to v and start to w are disjoint
                        # except for start.
                        cycle_len = depth + 1 + visited[w][0]
                        if cycle_len < min_girth:
                            return False
                    else:
                        visited[w] = (depth + 1, v)
                        queue.append((w, depth + 1, v))

        return True

    # Let's use a more efficient girth check
    def check_girth_fast(n_half, edges, min_girth, deadline):
        """Fast girth check for 3-regular bipartite graph."""
        n = 2 * n_half
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for i in range(n):
            if len(adj[i]) != 3:
                return False

        # Check for cycles of length 4: for each vertex, check if any two
        # neighbors share another neighbor
        for v in range(n):
            if time.time() > deadline:
                return None
            nbrs = adj[v]
            for i in range(3):
                for j in range(i + 1, 3):
                    # Check if nbrs[i] and nbrs[j] share a common neighbor
                    # (other than v)
                    common = set(adj[nbrs[i]]) & set(adj[nbrs[j]])
                    if len(common) > 0:
                        return False

        # For longer cycles (6, 8, 10, 12, 14), use BFS with depth limit
        # from each vertex
        for start in range(n):
            if time.time() > deadline:
                return None
            # BFS up to depth min_girth//2 - 1 = 7
            # Track: for each vertex, the shortest distance and the path
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])

            while queue:
                if time.time() > deadline:
                    return None
                v = queue.popleft()
                if dist[v] >= min_girth // 2 - 1:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    else:
                        # Found a cycle
                        # The cycle goes: start -> ... -> w -> v -> ... -> start
                        # Actually, we need to find the cycle length
                        # w is already visited at dist[w], and we're trying to
                        # reach it from v at dist[v]+1
                        # The cycle length is dist[v] + 1 + dist[w]
                        # if the paths from start to v and start to w are
                        # internally disjoint. But they might not be.
                        #
                        # A simpler check: if dist[v] + 1 + dist[w] < min_girth
                        # and the LCA of v and w is start, then we have a cycle
                        # of that length.
                        #
                        # For safety, let's just check if dist[v] + dist[w] + 1 < min_girth
                        # This is a sufficient condition for a short cycle.
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < min_girth:
                            return False

        return True

    # Actually, the above girth check is not quite right. Let me use a cleaner approach.
    # For a 3-regular bipartite graph, the girth is the length of the shortest cycle.
    # We can check this by, for each directed edge (u,v), doing a non-backtracking
    # BFS to see if we can return to (u,v) in less than girth steps.

    def check_girth_nonbacktracking(n_half, edges, min_girth, deadline):
        """Check girth >= min_girth using non-backtracking walks."""
        n = 2 * n_half
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for i in range(n):
            if len(adj[i]) != 3:
                return False

        # For each directed edge (u,v), do BFS in the non-backtracking graph
        # to see if we can reach (v,u) in less than min_girth-1 steps.
        # The non-backtracking graph has states (u,v) where u!=v and u~v.
        # From (u,v), we can go to (v,w) for w in adj[v], w != u.

        # This is expensive: n*3 states, and BFS from each is O(n*3*2) per step.
        # For n=936, that's about 2800 states, and BFS depth up to 14.
        # Total: 2800 * 2800 * 14 ≈ 110M operations. Too slow.

        # Alternative: use the fact that for 3-regular graphs, we can use
        # the "girth via counting" approach or just do BFS from each vertex
        # with a depth limit, being careful about the cycle detection.

        # Let me use a simpler but correct approach:
        # For each vertex s, do BFS up to depth d = min_girth//2 - 1.
        # Record for each vertex the distance from s.
        # If we find an edge (u,v) where dist[u] + dist[v] + 1 < min_girth
        # and u and v are in the BFS tree of s, we need to check if the
        # paths from s to u and s to v are internally disjoint.
        #
        # Actually, the cleanest correct method:
        # For each pair of vertices (u,v) with u < v, check if there's a path
        # of length < min_girth - 1 between them (avoiding the direct edge if it exists).
        # This is too expensive.

        # Let me just use the BFS approach with careful cycle detection:
        # From each vertex s, do BFS. When we encounter a vertex w that's already
        # visited, the cycle length is dist[v] + 1 + dist[w] where v is the current
        # vertex and w is the already-visited neighbor. But we need to verify that
        # the two paths from s are internally disjoint.

        # For a 3-regular graph, the number of vertices at distance i from s is
        # at most 1 + 3*(2^i - 1) for i < girth/2. For girth 16, up to distance 7:
        # 1, 3, 6, 12, 24, 48, 96, 192. Total < 400. So BFS is fast.

        for s in range(n):
            if time.time() > deadline:
                return None
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])

            while queue:
                if time.time() > deadline:
                    return None
                v = queue.popleft()
                if dist[v] >= min_girth // 2 - 1:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    else:
                        # Found a potential cycle
                        # Cycle length: dist[v] + 1 + dist[w]
                        # But only if the paths from s to v and s to w
                        # are internally disjoint (i.e., their LCA is s)
                        #
                        # To check: walk up from v and w to find LCA
                        # If LCA is s, then cycle length = dist[v] + dist[w] + 1
                        # Otherwise, the cycle is shorter and we should have
                        # found it earlier.

                        # For simplicity, if dist[v] + dist[w] + 1 < min_girth,
                        # it's likely a short cycle. Let's verify by checking
                        # if s is on the path.

                        # Actually, in a BFS tree, if we find a back edge (v,w)
                        # where both v and w are in the tree, the cycle is:
                        # s -> ... -> v -> w -> ... -> s
                        # The length is dist[v] + 1 + dist[w] if the paths
                        # from s to v and s to w only meet at s.

                        # Let's check by finding the LCA of v and w
                        a, b = v, w
                        da, db = dist[a], dist[b]
                        # Walk up the deeper one
                        while da > db:
                            a = parent[a]
                            da -= 1
                        while db > da:
                            b = parent[b]
                            db -= 1
                        while a != b:
                            a = parent[a]
                            b = parent[b]
                        # a is the LCA
                        if a == s:
                            cycle_len = dist[v] + dist[w] + 1
                            if cycle_len < min_girth:
                                return False
                        else:
                            # The cycle doesn't go through s, so it's shorter
                            # and should have been detected when we started
                            # BFS from a vertex on that cycle.
                            # But to be safe, let's still check
                            cycle_len = dist[v] + dist[w] + 1 - 2 * dist[a]
                            if cycle_len < min_girth:
                                return False

        return True

    # Try different sizes
    # Start from smaller sizes and go up
    # The known best is 936, so let's try sizes around there
    # n_half = n/2, so n_half = 468 for n=936

    rng = random.Random(42)

    # Try various sizes
    # For efficiency, let's try sizes that are multiples of small numbers
    # to make it easier to construct 3-regular bipartite graphs

    # Actually, let's use a smarter construction:
    # Start with a known small girth-16 graph (if we can construct one)
    # and try to find smaller ones.

    # For now, let's just do random generation with girth checking.
    # We'll try sizes from 512/2=256 up to 936/2=468

    # But generating random 3-regular bipartite graphs and checking girth
    # is very slow. Let's use a different approach.

    # Approach: Use the "random lift" of a smaller graph.
    # Take a base graph with girth >= g and lift it.

    # Actually, let's try a completely different approach:
    # Use the known construction from Lazebnik-Ustimenko-Woldar but
    # with a smaller field.

    # For k=3, q must be prime or 4. q=3 is the smallest prime.
    # The LUW graph with q=3 has girth d+5 for odd d.
    # For girth 16, we need d+5 >= 16, so d >= 11.
    # The number of vertices is 2 * 3^d.
    # For d=11: 2 * 3^11 = 2 * 177147 = 354294. Way too large.

    # So the LUW construction doesn't help for small graphs.

    # Let me try a different strategy: use simulated annealing or
    # local search to find a 3-regular graph with girth >= 16.

    # Start with a random 3-regular bipartite graph and try to
    # increase the girth by edge swaps.

    def try_local_search(n_half, max_iterations, deadline):
        """Try to find a 3-regular bipartite graph with girth >= 16
        using local search (edge swaps)."""
        n = 2 * n_half

        # Generate initial random 3-regular bipartite graph
        for attempt in range(100):
            if time.time() > deadline:
                break

            # Configuration model
            left_ports = [(i, j) for i in range(n_half) for j in range(3)]
            right_ports = [(n_half + i, j) for i in range(n_half) for j in range(3)]

            rng.shuffle(left_ports)
            rng.shuffle(right_ports)

            edges = set()
            valid = True
            for (u, _), (v, _) in zip(left_ports, right_ports):
                if u == v or (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))

            if not valid or len(edges) != 3 * n_half:
                continue

            edge_list = list(edges)

            # Check initial girth
            girth_ok = check_girth_nonbacktracking(n_half, edge_list, 16, deadline)
            if girth_ok is True:
                return edge_list
            if girth_ok is None:
                break

            # Local search: try edge swaps to increase girth
            # An edge swap: remove edges (a,b) and (c,d), add (a,d) and (c,b)
            # where a,c are on the same side and b,d are on the other side.

            best_girth = 0
            for iteration in range(max_iterations):
                if time.time() > deadline:
                    break

                # Pick two random edges to swap
                i, j = rng.sample(range(len(edge_list)), 2)
                a1, b1 = edge_list[i]
                a2, b2 = edge_list[j]

                # a1, a2 should be on the same side
                if a1 < n_half and a2 < n_half:
                    new_edges = [(a1, b2), (a2, b1)]
                elif a1 >= n_half and a2 >= n_half:
                    new_edges = [(a1, b2), (a2, b1)]
                elif b1 < n_half and b2 < n_half:
                    new_edges = [(a1, b2), (a2, b1)]
                elif b1 >= n_half and b2 >= n_half:
                    new_edges = [(a1, b2), (a2, b1)]
                else:
                    continue

                # Check no loops or duplicates
                if any(u == v for u, v in new_edges):
                    continue
                if any((u, v) in edges or (v, u) in edges for u, v in new_edges):
                    continue

                # Try the swap
                edges.discard((a1, b1))
                edges.discard((a2, b2))
                edges.add((a1, b2))
                edges.add((a2, b1))
                edge_list = list(edges)

                girth_ok = check_girth_nonbacktracking(n_half, edge_list, 16, deadline)
                if girth_ok is True:
                    return edge_list
                if girth_ok is None:
                    break

                # Revert if not improved (simplified: always revert)
                edges.discard((a1, b2))
                edges.discard((a2, b1))
                edges.add((a1, b1))
                edges.add((a2, b2))
                edge_list = list(edges)

        return None

    # Try different sizes
    # Start from smaller sizes
    for n_half in [256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300,
                    304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348,
                    352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396,
                    400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444,
                    448, 452, 456, 460, 464, 468]:
        if time.time() > deadline - 10:
            break

        # Try local search
        result = try_local_search(n_half, 5000, deadline)
        if result is not None:
            n = 2 * n_half
            if n < best_n:
                best_n = n
                best_result = (n, result)

        if best_n == moore_bound:
            break

    if best_result is not None:
        return best_result

    # Fallback: return a valid graph (even if large)
    # Use the LUW construction with q=3
    q = 3
    d = 11  # girth = d + 5 = 16
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
                y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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

# EVOLVE-BLOCK-END
