import random
import time
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 110

    # Moore bound for (3,16) is 512. Known best is 936.
    # We'll try Petersen lifts with various voltage group sizes.

    def petersen_lift(n_lift, seed=42):
        """Construct a cubic graph by lifting Petersen graph with Z_n_lift voltage assignment.
        Returns (n, edges) if valid, else None."""
        rng = random.Random(seed)
        n = 10 * n_lift

        # Petersen graph edges (standard labeling)
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]

        # Assign random voltages (elements of Z_n_lift) to each edge
        # Use a symmetric voltage assignment to ensure the lift is regular
        voltages = {}
        for u, v in petersen_edges:
            voltages[(u, v)] = rng.randint(0, n_lift - 1)
            voltages[(v, u)] = (n_lift - voltages[(u, v)]) % n_lift

        # Build the lifted graph
        # Vertex (i, a) where i is Petersen vertex (0..9), a is voltage (0..n_lift-1)
        # Edge between (u, a) and (v, (a + voltages[(u,v)]) % n_lift)
        edges = []
        edge_set = set()
        for u, v in petersen_edges:
            vol = voltages[(u, v)]
            for a in range(n_lift):
                b = (a + vol) % n_lift
                u_idx = u * n_lift + a
                v_idx = v * n_lift + b
                if u_idx == v_idx:
                    return None  # loop
                key = (min(u_idx, v_idx), max(u_idx, v_idx))
                if key in edge_set:
                    return None  # multi-edge
                edge_set.add(key)
                edges.append(key)

        # Check degree: each vertex should have degree 3
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != 3:
                return None

        # Check connectivity
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            curr = queue.popleft()
            for nei in adj[curr]:
                if not visited[nei]:
                    visited[nei] = True
                    count += 1
                    queue.append(nei)
        if count != n:
            return None

        return n, edges

    def check_girth(n, edges, target_girth):
        """Check if graph has girth >= target_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # For each vertex, do BFS and check that no vertex is reached at distance < target_girth/2
        # More precisely: girth >= g iff no two vertices at distance < g/2 have a common neighbor
        # Actually, simpler: BFS from each vertex, if we find a vertex at distance d where
        # the BFS tree would create a cycle of length < g, we fail.
        # Most efficient: for each edge (u,v), check shortest path from u to v avoiding (u,v).
        # If any such path has length < g-1, girth < g.

        # Faster approach: BFS from each vertex up to depth (g-1)//2
        # If any vertex is reached twice (via different parents), we have a cycle.
        # Actually, let's use the standard approach:
        # girth = min over all edges (u,v) of (shortest path from u to v in G-(u,v)) + 1

        half = (target_girth - 1) // 2  # We only need to check up to this depth

        for start in range(n):
            # BFS from start, tracking parent to detect cycles
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            found_cycle = False
            while queue:
                curr = queue.popleft()
                if dist[curr] >= half:
                    continue
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        parent[nei] = curr
                        queue.append(nei)
                    elif nei != parent[curr] and curr != parent[nei]:
                        # Found a cycle: curr -> ... -> start -> ... -> nei -> curr
                        # Cycle length = dist[curr] + dist[nei] + 1
                        cycle_len = dist[curr] + dist[nei] + 1
                        if cycle_len < target_girth:
                            return False
            # Also need to check edges not in BFS tree
            # Actually the above catches all cycles through start.
            # But we need cycles not through start too.
            # The per-edge BFS is more reliable.

        # Do per-edge check for certainty
        for u, v in edges:
            # BFS from u to v avoiding direct edge (u,v)
            dist = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                if dist[curr] >= target_girth - 1:
                    break
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        # Skip the direct edge u-v
                        if curr == u and nei == v:
                            continue
                        dist[nei] = dist[curr] + 1
                        if nei == v:
                            if dist[v] + 1 < target_girth:
                                return False
                        queue.append(nei)
        return True

    # Strategy: Try Petersen lifts with increasing n_lift
    # n = 10 * n_lift
    # We want n >= 512 (Moore bound) and ideally close to 936.
    # n_lift = 94 gives n = 940, n_lift = 93 gives n = 930, etc.
    # But n must be even, and 10*n_lift is always even.
    # Try n_lift values that give n near 936: 93->930, 94->940, 95->950

    # First, try a few small lifts to verify the method works, then scale up
    # Actually, let's just try n_lift values directly.
    # The girth of a Petersen lift depends on the voltage assignment.
    # With random voltages, girth is typically around 10-14 for n~1000.
    # We need girth >= 16, which is harder.

    # Alternative: Use a more structured voltage assignment.
    # The key insight: if the voltage assignment is "generic enough",
    # the girth of the lift equals the girth of the base graph (Petersen has girth 5).
    # That's not helpful. We need the lift to have LARGER girth than the base.

    # Actually, for a voltage graph, cycles in the lift correspond to closed walks
    # in the base whose voltage sum is 0 mod n_lift. The girth of the lift is
    # the length of the shortest closed walk in the base with voltage sum 0.
    # Petersen has girth 5, so the shortest closed walk is length 5.
    # We need no closed walk of length < 16 to have voltage sum 0.
    # This means: for every closed walk of length 5,6,...,15 in Petersen,
    # the voltage sum should be non-zero mod n_lift.

    # Number of closed walks of length L in Petersen grows, but for L <= 15 it's manageable.
    # We can enumerate all closed walks of length up to 15 and check voltage sums.

    def enumerate_closed_walks(max_len):
        """Enumerate all closed walks in Petersen graph up to max_len.
        Returns list of (length, sequence_of_vertices)."""
        petersen_adj = {
            0: [1, 4, 5],
            1: [0, 2, 6],
            2: [1, 3, 7],
            3: [2, 4, 8],
            4: [3, 0, 9],
            5: [0, 7, 8],
            6: [1, 9, 8],
            7: [2, 5, 9],
            8: [3, 6, 5],
            9: [4, 6, 7]
        }
        walks = []
        # Start from each vertex, do DFS up to max_len, track if we return to start
        for start in range(10):
            stack = [(start, [start])]
            while stack:
                curr, path = stack.pop()
                if len(path) > max_len:
                    continue
                if len(path) >= 5 and curr == start:
                    # Closed walk of length len(path)-1
                    walks.append((len(path) - 1, path))
                    # Don't extend further from a closed walk (would create repeats)
                    continue
                for nei in petersen_adj[curr]:
                    # Avoid backtracking (non-backtracking walks)
                    if len(path) >= 2 and nei == path[-2]:
                        continue
                    stack.append((nei, path + [nei]))
        return walks

    # Get all non-backtracking closed walks of length 5 to 15 in Petersen
    closed_walks = enumerate_closed_walks(15)
    # Filter to only lengths 5-15 (girth of Petersen is 5, so no shorter closed non-backtracking walks)
    relevant_walks = [(L, w) for L, w in closed_walks if 5 <= L <= 15]

    # Now we need to assign voltages to the 15 edges of Petersen such that
    # for every relevant closed walk, the voltage sum is non-zero mod n_lift.
    # This is a constraint satisfaction problem.

    # For a closed walk v0->v1->...->vk=v0, the voltage sum is:
    # sum of voltages along the walk, where voltage of edge (u,v) in direction u->v is voltages[(u,v)]
    # Since voltages[(v,u)] = -voltages[(u,v)] mod n_lift, the sum is well-defined.

    # Let's try random voltage assignments and check.
    # For n_lift ~ 94, we have 15 variables in Z_94.
    # Number of constraints: number of relevant closed walks.

    # Count them
    print(f"Number of relevant closed walks: {len(relevant_walks)}", file=sys.stderr)

    # For each closed walk, compute the net voltage contribution of each edge.
    # A closed walk uses each edge some number of times (with sign depending on direction).
    # The voltage sum = sum over edges (net_count_edge * voltage_edge) mod n_lift

    # Build a system: for each walk, a linear equation in the 15 edge voltages.
    # We want all sums to be non-zero.

    # This is hard to solve directly. Instead, try random assignments.

    # Edge list for Petersen (indexed 0-14)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # For each closed walk, compute coefficient vector (how many times each edge is used, with sign)
    walk_coeffs = []
    for L, w in relevant_walks:
        coeffs = [0] * 15
        for i in range(L):
            u = w[i]
            v = w[i + 1]
            ei = edge_index[(u, v)]
            # Determine direction sign
            if pet_edges[ei][0] == u:
                coeffs[ei] += 1
            else:
                coeffs[ei] -= 1
        walk_coeffs.append(coeffs)

    # Now try random voltage assignments
    def try_voltages(n_lift, seed):
        rng = random.Random(seed)
        # Try multiple random assignments
        for attempt in range(500):
            vols = [rng.randint(1, n_lift - 1) for _ in range(15)]
            # Check all walks
            valid = True
            for coeffs in walk_coeffs:
                s = sum(c * vols[i] for i, c in enumerate(coeffs)) % n_lift
                if s == 0:
                    valid = False
                    break
            if valid:
                return vols
        return None

    # Try different n_lift values
    # n = 10 * n_lift, want n >= 512 so n_lift >= 52
    # Known best n = 936, so n_lift = 93.6 -> not integer.
    # Try n_lift = 93 (n=930), 94 (n=940), 95 (n=950), etc.
    # Also try smaller: 52 (n=520), 60 (n=600), 70 (n=700), 80 (n=800)

    # Start with smaller n_lift (easier to satisfy constraints, smaller n)
    n_lift_candidates = [52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 93, 94, 95, 96, 98, 100]

    for n_lift in n_lift_candidates:
        if time.time() > deadline - 15:
            break
        n = 10 * n_lift
        # Try to find valid voltage assignment
        vols = try_voltages(n_lift, seed=n_lift)
        if vols is None:
            continue

        # Build the lifted graph
        voltages = {}
        for i, (u, v) in enumerate(pet_edges):
            voltages[(u, v)] = vols[i]
            voltages[(v, u)] = (n_lift - vols[i]) % n_lift

        edges = []
        edge_set = set()
        for u, v in pet_edges:
            vol = voltages[(u, v)]
            for a in range(n_lift):
                b = (a + vol) % n_lift
                u_idx = u * n_lift + a
                v_idx = v * n_lift + b
                if u_idx == v_idx:
                    break
                key = (min(u_idx, v_idx), max(u_idx, v_idx))
                if key in edge_set:
                    break
                edge_set.add(key)
                edges.append(key)
            else:
                continue
            break  # Invalid
        else:
            # Check degree
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            if all(len(adj[i]) == 3 for i in range(n)):
                # Check connectivity
                visited = [False] * n
                queue = deque([0])
                visited[0] = True
                count = 1
                while queue:
                    curr = queue.popleft()
                    for nei in adj[curr]:
                        if not visited[nei]:
                            visited[nei] = True
                            count += 1
                            queue.append(nei)
                if count == n:
                    # Verify girth
                    if check_girth(n, edges, g):
                        return n, edges

    # Fallback: LUW construction
    def luw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
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

    return luw_fallback()
