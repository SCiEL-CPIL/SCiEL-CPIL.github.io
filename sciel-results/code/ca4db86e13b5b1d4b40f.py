"""Program that decomposes graphs into few cycles and edges using maximum 2-factor extraction."""
import time
from collections import deque
from random import Random


def construct():
    start_time = time.time()
    deadline = start_time + 115

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hopcroft_karp(adj, left_nodes, right_nodes):
        """Maximum bipartite matching via Hopcroft-Karp."""
        pair_u = {u: -1 for u in left_nodes}
        pair_v = {v: -1 for v in right_nodes}
        dist = {}

        def bfs():
            queue = deque()
            found = False
            for u in left_nodes:
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    pu = pair_v.get(v, -1)
                    if pu == -1:
                        found = True
                    elif dist.get(pu, float('inf')) == float('inf'):
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in adj.get(u, []):
                pu = pair_v.get(v, -1)
                if pu == -1 or (dist.get(pu, float('inf')) == dist.get(u, float('inf')) + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False

        matching = 0
        while bfs():
            for u in left_nodes:
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u, pair_v, matching

    def find_max_2factor(adj, n, remaining):
        """Find a maximum 2-factor in the remaining graph using bipartite matching.
        
        The 2-factor is a set of vertex-disjoint cycles. We model this as a maximum
        matching in the bipartite double cover of the graph.
        """
        # Build bipartite graph: left = vertices, right = vertices
        # Edge u_left -> v_right exists if (u,v) is in remaining and u != v
        # A matching gives us a set of directed edges forming cycles
        # We need to avoid 2-cycles (u->v and v->u)
        
        # Actually, the standard reduction: a 2-factor in G corresponds to a
        # perfect matching in the bipartite graph B where B has V x V and
        # edge (u_left, v_right) exists iff {u,v} in E and u != v.
        # But this can give 2-cycles. To handle this, we use the standard
        # maximum 2-matching algorithm.
        
        # Simpler approach: just find a maximum matching, extract cycles,
        # discard 2-cycles.
        
        left_nodes = list(range(n))
        right_nodes = list(range(n))
        
        bip_adj = {}
        for u in range(n):
            bip_adj[u] = []
            for v in adj.get(u, []):
                e = (min(u, v), max(u, v))
                if e in remaining and u != v:
                    bip_adj[u].append(v)
        
        pair_u, pair_v, matching = hopcroft_karp(bip_adj, left_nodes, right_nodes)
        
        if matching == 0:
            return []
        
        # Extract cycles from the matching
        # pair_u[u] = v means u is matched to v (edge u->v in the 2-factor)
        # Build the functional graph: u -> pair_u[u]
        visited = set()
        cycles = []
        
        for u in range(n):
            if u in visited or pair_u[u] == -1:
                continue
            # Trace the cycle
            path = [u]
            cur = pair_u[u]
            visited.add(u)
            while cur != u and cur not in visited and cur != -1:
                path.append(cur)
                visited.add(cur)
                cur = pair_u[cur] if cur in pair_u else -1
            if cur == u and len(path) >= 3:
                cycles.append(path)
        
        return cycles

    def find_longest_cycle(adj, n, remaining, rng, max_depth=80):
        """Find a long cycle in the remaining graph using randomized DFS."""
        best = None
        
        # Try multiple random starts and orderings
        for start in range(n):
            if len(adj.get(start, [])) < 2:
                continue
            
            for attempt in range(5):
                visited = set()
                parent = {}
                found = None
                
                def dfs(node, depth):
                    nonlocal found
                    if found is not None:
                        return
                    if depth > max_depth:
                        return
                    visited.add(node)
                    
                    neighbors = [nb for nb in adj.get(node, [])
                                if (min(node, nb), max(node, nb)) in remaining]
                    if attempt > 0:
                        rng.shuffle(neighbors)
                    
                    for neighbor in neighbors:
                        if neighbor == start and len(visited) >= 3:
                            # Found a cycle
                            cycle = [start]
                            cur = node
                            while cur != start:
                                cycle.append(cur)
                                cur = parent[cur]
                            found = cycle
                            return
                        if neighbor not in visited:
                            parent[neighbor] = node
                            dfs(neighbor, depth + 1)
                            if found is not None:
                                return
                
                dfs(start, 0)
                
                if found is not None:
                    if best is None or len(found) > len(best):
                        best = found
                    if len(found) >= 10:  # Good enough, stop early
                        return best
        
        return best

    def decompose(edges, n):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        remaining = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            remaining.add((min(u, v), max(u, v)))

        pieces = []
        rng = Random(42)
        max_iterations = 200
        iter_count = 0
        
        while iter_count < max_iterations:
            iter_count += 1
            if time.time() > deadline - 10:
                break
            if not remaining:
                break
            
            # Try to find a maximum 2-factor
            cycles = find_max_2factor(adj, n, remaining)
            
            if cycles:
                # Use the 2-factor cycles
                for cycle in cycles:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        remaining.discard(e)
                    pieces.append(cycle)
            else:
                # No 2-factor found, try to find a single long cycle
                cycle = find_longest_cycle(adj, n, remaining, rng)
                if cycle is None:
                    break
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                pieces.append(cycle)
        
        # Remaining edges become individual pieces
        for u, v in sorted(remaining):
            pieces.append([u, v])
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
