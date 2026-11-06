"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque, defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def hopcroft_karp(left, right, adj):
        """Maximum bipartite matching. adj: dict left->list of right. Returns (matchL, matchR)."""
        matchL = {u: -1 for u in left}
        matchR = {v: -1 for v in right}

        def bfs():
            dist = {}
            q = deque()
            for u in left:
                if matchL[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj.get(u, ()):
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist.get(mu, -1) == -1:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return dist, found

        def dfs(u):
            for v in adj.get(u, ()):
                mu = matchR[v]
                if mu == -1 or (dist.get(mu, -1) == dist.get(u, -1) + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = -1
            return False

        while True:
            dist, found = bfs()
            if not found:
                break
            for u in left:
                if matchL[u] == -1:
                    dfs(u)
        return matchL, matchR

    def decompose(edges):
        if not edges:
            return []
        remaining = set(canonical(e) for e in edges)
        pieces = []
        
        # Store cycles as lists of vertices
        # Single edges are stored as [u, v]
        
        def build_adj(rem):
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def extract_2factor_cycles(rem):
            """Find a large 2-factor (edge-disjoint cycles) via max bipartite matching."""
            adj = build_adj(rem)
            if not adj:
                return []
            vertices = sorted(adj.keys())
            left = vertices
            right = vertices
            bip = {u: [] for u in left}
            for u in left:
                for v in adj.get(u, ()):
                    if v != u:
                        bip[u].append(v)
            
            if not any(bip[u] for u in left):
                return []

            matchL, matchR = hopcroft_karp(left, right, bip)
            
            visited = set()
            cycles = []
            for u in left:
                if u in visited or matchL[u] == -1:
                    continue
                cyc = []
                cur = u
                while cur not in visited:
                    visited.add(cur)
                    cyc.append(cur)
                    nxt = matchL[cur]
                    if nxt == -1:
                        break
                    cur = nxt
                if len(cyc) >= 3:
                    cycles.append(cyc)
            return cycles

        def cycle_edges(cyc):
            es = set()
            m = len(cyc)
            for i in range(m):
                es.add(canonical((cyc[i], cyc[(i + 1) % m])))
            return es

        def find_path_in_rem(u, v, rem, adj):
            """Find a simple path from u to v in the graph formed by rem."""
            if u == v:
                return []
            visited = {u}
            parent = {u: None}
            q = deque([u])
            while q:
                node = q.popleft()
                if node == v:
                    # Reconstruct path
                    path = []
                    cur = v
                    while cur != u:
                        path.append((cur, parent[cur]))
                        cur = parent[cur]
                    path.reverse()
                    return path
                for nbr in adj.get(node, ()):
                    if nbr not in visited:
                        visited.add(nbr)
                        parent[nbr] = node
                        q.append(nbr)
            return None

        def merge_path_into_cycle(cycle, path, rem):
            """
            cycle: list of vertices [v0, v1, ..., vk-1]
            path: list of edges [(u1, v1), (v1, u2), ...] from u_start to v_end
            Returns new cycle vertices.
            """
            if not path:
                return cycle
            
            u_start = path[0][0]
            v_end = path[-1][1]
            
            # Find indices of u_start and v_end in cycle
            idx_u = -1
            idx_v = -1
            for i, node in enumerate(cycle):
                if node == u_start:
                    idx_u = i
                if node == v_end:
                    idx_v = i
            
            if idx_u == -1 or idx_v == -1:
                return None # Should not happen if path connects vertices in cycle

            # We have two arcs in the cycle between u_start and v_end.
            # Arc 1: idx_u -> ... -> idx_v
            # Arc 2: idx_v -> ... -> idx_u (wrapping around)
            
            # We can combine path with Arc 1 or Arc 2.
            # To maximize cycle length, we should pick the longer arc? 
            # Actually, we just need to form *a* valid cycle.
            # Let's pick Arc 1.
            
            # Path vertices: u_start, ..., v_end
            path_vertices = [u_start]
            for e in path:
                path_vertices.append(e[1])
            
            # Cycle vertices between u_start and v_end (forward direction)
            arc1_vertices = []
            cur = idx_u
            while True:
                arc1_vertices.append(cycle[cur])
                if cur == idx_v:
                    break
                cur = (cur + 1) % len(cycle)
            
            # New cycle: path_vertices + arc1_vertices[1:] (since path ends at v_end, and arc1 starts at u_start, we need to connect them)
            # Wait, path is u_start -> ... -> v_end.
            # Arc1 is u_start -> ... -> v_end.
            # If we combine them, we get a "theta" graph. We must pick one arc to close the loop.
            # The new cycle is: Path (u_start to v_end) + Arc2 (v_end to u_start).
            
            # Let's construct Arc2: v_end -> ... -> u_start
            arc2_vertices = []
            cur = idx_v
            while True:
                arc2_vertices.append(cycle[cur])
                if cur == idx_u:
                    break
                cur = (cur + 1) % len(cycle)
            
            # New cycle vertices: path_vertices + arc2_vertices[1:]
            new_cycle = path_vertices + arc2_vertices[1:]
            return new_cycle

        def find_cycle_in_rem(rem, adj):
            """Find any simple cycle in rem."""
            visited = set()
            for start in sorted(adj.keys()):
                if start in visited:
                    continue
                # DFS
                stack = [(start, [start])]
                while stack:
                    node, path = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    for nbr in sorted(adj.get(node, ())):
                        if nbr == start and len(path) >= 2:
                            return path
                        if nbr not in visited:
                            stack.append((nbr, path + [nbr]))
            return None

        # Phase 1: 2-factor
        if time.time() - t0 < DEADLINE - 5:
            cyc2 = extract_2factor_cycles(remaining)
            for c in cyc2:
                ces = cycle_edges(c)
                if ces <= remaining:
                    pieces.append(list(c))
                    remaining -= ces

        # Phase 2: Merge paths into existing cycles
        # We maintain a list of cycles in 'pieces'.
        # We repeatedly try to find a path in 'remaining' that connects two vertices of an existing cycle.
        
        # Optimization: Keep track of vertices in each cycle
        # Rebuild this structure if pieces change
        
        while remaining and time.time() - t0 < DEADLINE - 5:
            adj = build_adj(remaining)
            merged = False
            
            # Iterate over cycles in pieces
            # We need to identify which pieces are cycles (len > 2)
            # Note: Single edges are len 2.
            
            for i, piece in enumerate(pieces):
                if len(piece) < 3:
                    continue
                
                cycle = piece
                cycle_vertices = set(cycle)
                
                # For each pair of vertices in the cycle, try to find a path in remaining
                # To speed up, we can just BFS from each vertex in the cycle and see if we hit another vertex in the cycle
                for u in cycle:
                    # BFS from u in remaining graph
                    visited = {u}
                    parent = {u: None}
                    q = deque([u])
                    found_v = None
                    while q:
                        node = q.popleft()
                        for nbr in adj.get(node, ()):
                            if nbr not in visited:
                                visited.add(nbr)
                                parent[nbr] = node
                                if nbr in cycle_vertices and nbr != u:
                                    found_v = nbr
                                    q = deque() # Break out
                                    break
                        if found_v:
                            break
                    
                    if found_v:
                        # Reconstruct path
                        path_edges = []
                        cur = found_v
                        while cur != u:
                            path_edges.append((cur, parent[cur]))
                            cur = parent[cur]
                        path_edges.reverse()
                        
                        # Merge
                        new_cycle = merge_path_into_cycle(cycle, path_edges, remaining)
                        if new_cycle:
                            # Remove path edges from remaining
                            for e in path_edges:
                                ce = canonical(e)
                                if ce in remaining:
                                    remaining.remove(ce)
                            
                            # Replace piece
                            pieces[i] = new_cycle
                            merged = True
                            break
                if merged:
                    break
            
            if merged:
                continue
            
            # If no merge, try to find a new cycle in remaining
            cyc = find_cycle_in_rem(remaining, adj)
            if cyc and len(cyc) >= 3:
                ces = cycle_edges(cyc)
                if ces <= remaining:
                    pieces.append(list(cyc))
                    remaining -= ces
                    continue
            
            # If no cycle found, take a single edge
            if remaining:
                e = next(iter(remaining))
                remaining.remove(e)
                pieces.append([e[0], e[1]])
                continue
            
            break

        # Phase 3: Final cleanup (should be empty)
        for e in sorted(remaining):
            pieces.append(list(e))
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
