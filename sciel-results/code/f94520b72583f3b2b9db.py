# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import heapq

def construct():
    t_start = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    def min_t_join(n, edges, odd_vertices):
        """
        Finds a minimum T-join for the set of odd_vertices.
        Returns the set of edges in the T-join.
        """
        if not odd_vertices:
            return set()
        
        # 1. Compute shortest path distances between all pairs of odd vertices
        # and store the paths
        # Dijkstra from each odd vertex
        dist = {u: {v: float('inf') for v in odd_vertices} for u in odd_vertices}
        parent = {u: {v: None for v in odd_vertices} for u in odd_vertices}
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for src in odd_vertices:
            dist[src][src] = 0
            pq = [(0, src)]
            visited = set()
            while pq:
                d, u = heapq.heappop(pq)
                if u in visited:
                    continue
                visited.add(u)
                if d > dist[src][u]:
                    continue
                for v in adj[u]:
                    if v in visited:
                        continue
                    nd = d + 1
                    if nd < dist[src][v]:
                        dist[src][v] = nd
                        parent[src][v] = u
                        heapq.heappush(pq, (nd, v))
                        
        # 2. Minimum Weight Perfect Matching on the complete graph of odd_vertices
        # Using DP over subsets
        k = len(odd_vertices)
        # Map odd_vertices to indices 0..k-1
        idx_map = {v: i for i, v in enumerate(odd_vertices)}
        
        # dp[mask] = min cost to match vertices in mask
        # mask is a bitmask of size k
        # Only valid if number of set bits is even
        size = 1 << k
        dp = [float('inf')] * size
        dp[0] = 0
        # parent for reconstruction: store the pair matched
        match_parent = [None] * size
        
        for mask in range(size):
            if dp[mask] == float('inf'):
                continue
            # Find the first unset bit
            # Actually, we iterate over masks with even number of bits
            # To ensure we don't skip, we can just iterate all masks
            # But we need to pick the smallest index not in mask to avoid permutations
            # If mask is 0, we are done.
            # We need to add two vertices.
            
            # Find smallest i not in mask
            i = 0
            while i < k and (mask & (1 << i)):
                i += 1
            if i == k:
                continue
            
            # Try to match i with j > i
            for j in range(i + 1, k):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << i) | (1 << j)
                cost = dist[odd_vertices[i]][odd_vertices[j]]
                if cost == float('inf'):
                    continue
                if dp[mask] + cost < dp[new_mask]:
                    dp[new_mask] = dp[mask] + cost
                    match_parent[new_mask] = (i, j, mask)
        
        full_mask = (1 << k) - 1
        if dp[full_mask] == float('inf'):
            # Graph might be disconnected? T-join doesn't exist?
            # But G is connected, so T-join exists.
            return set()
            
        # 3. Reconstruct the paths and take symmetric difference
        edges_in_join = set()
        mask = full_mask
        while mask != 0:
            i, j, prev_mask = match_parent[mask]
            u = odd_vertices[i]
            v = odd_vertices[j]
            
            # Reconstruct path from u to v
            # parent[src][v] gives previous node
            # We need path from u to v.
            # Dijkstra was run from src=u.
            # So parent[u][v] is the path.
            curr = v
            path_edges = []
            while curr != u:
                prev = parent[u][curr]
                if prev is None:
                    break
                edge = tuple(sorted((curr, prev)))
                path_edges.append(edge)
                curr = prev
            
            # Toggle edges in edges_in_join
            for e in path_edges:
                if e in edges_in_join:
                    edges_in_join.remove(e)
                else:
                    edges_in_join.add(e)
            
            mask = prev_mask
            
        return edges_in_join

    def decompose_eulerian(n, edges):
        """
        Decompose an Eulerian graph (all even degrees) into simple cycles.
        Returns list of cycles (list of vertices).
        """
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        edge_set = set(edges)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        cycles = []
        # To ensure we cover all edges, we process components
        # But Hierholzer on a component gives one circuit.
        # We decompose that circuit into simple cycles.
        
        # We need to track used edges to remove them
        # But Hierholzer uses a specific traversal.
        # Alternative: Just find any cycle, remove it, repeat.
        # This is simpler and guarantees simple cycles.
        
        remaining_edges = set(edges)
        
        while remaining_edges:
            # Find a vertex with non-zero degree in remaining_edges
            # We can just pick any edge
            u, v = next(iter(remaining_edges))
            
            # Find a cycle starting from u
            # DFS/BFS
            # We want a simple cycle.
            # We can use DFS to find a cycle containing u.
            # Since the graph is Eulerian, a cycle must exist.
            
            # Simple approach: DFS from u, keep track of path.
            # If we hit a visited node, we found a cycle.
            # But we need to ensure we find a cycle that uses edges from remaining_edges.
            
            # Let's use a standard cycle finding algorithm on the subgraph induced by remaining_edges.
            # We can just do DFS from u.
            
            visited = set()
            parent = {}
            found_cycle = None
            
            def dfs(curr):
                nonlocal found_cycle
                visited.add(curr)
                for nxt in adj[curr]:
                    if tuple(sorted((curr, nxt))) not in remaining_edges:
                        continue
                    if nxt not in visited:
                        parent[nxt] = curr
                        if dfs(nxt):
                            return True
                    else:
                        if nxt == u:
                            # Cycle found: u -> ... -> curr -> u
                            # But we need to extract the cycle from the tree path
                            # The cycle is the path from u to curr in the tree + edge (curr, u)
                            # Wait, DFS tree path from u to curr is stored in parent.
                            # But we need the sequence of vertices.
                            pass
                        else:
                            # Back edge (curr, nxt) where nxt is visited and not parent
                            # Cycle: nxt -> ... -> curr -> nxt
                            # Extract path from nxt to curr in tree
                            path = [curr]
                            c = curr
                            while c != nxt:
                                c = parent[c]
                                if c == -1:
                                    return False
                                path.append(c)
                            path.append(nxt)
                            # path is [curr, ..., nxt]
                            # Cycle is [nxt, ..., curr, nxt]
                            # Reverse path to get nxt -> ... -> curr
                            path.reverse()
                            # Now path is [nxt, ..., curr]
                            # Add nxt at end? No, cycle is path + [nxt]
                            # But we need distinct vertices.
                            # If len(path) >= 3, it's a valid cycle.
                            if len(path) >= 3:
                                found_cycle = path + [nxt]
                                return True
                return False

            # We need to be careful with the DFS to not get stuck in deep recursion
            # n is small (30), so recursion is fine.
            
            # Start DFS from u
            # We need to ensure we find a cycle.
            # If the component is a single edge? No, Eulerian means even degrees.
            # Smallest cycle is length 3.
            
            # To ensure we find a cycle, we can just pick an edge (u,v) and try to find a path from v to u in the graph excluding (u,v)?
            # No, simpler: Just run DFS from u.
            
            # Reset visited for this component?
            # No, we just need to find ONE cycle.
            
            # Let's try a different strategy for finding a cycle in a Eulerian graph:
            # Pick any edge (u,v). Remove it.
            # Find a path from u to v in the remaining graph.
            # This path + (u,v) forms a cycle.
            # This is guaranteed to work in an Eulerian graph (connected component).
            
            u, v = next(iter(remaining_edges))
            remaining_edges.remove((u, v))
            
            # BFS/DFS from u to v in remaining_edges
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue and not found:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    if tuple(sorted((curr, nxt))) not in remaining_edges:
                        continue
                    if nxt not in parent:
                        parent[nxt] = curr
                        if nxt == v:
                            found = True
                            break
                        queue.append(nxt)
            
            if found:
                # Reconstruct path
                path = [v]
                curr = v
                while curr != u:
                    curr = parent[curr]
                    path.append(curr)
                path.reverse() # u -> ... -> v
                cycle = path + [u]
                # Mark edges in cycle as used
                for i in range(len(cycle) - 1):
                    e = tuple(sorted((cycle[i], cycle[i+1])))
                    remaining_edges.remove(e)
                cycles.append(cycle)
            else:
                # Should not happen if graph is Eulerian and connected
                # But if not connected, we picked an edge from one component.
                # The path must exist.
                pass

        return cycles

    def decompose(edges, n):
        edge_set = set(edges)
        degrees = defaultdict(int)
        for u, v in edge_set:
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in range(n) if degrees[v] % 2 != 0]
        
        # Find T-Join
        t_join_edges = min_t_join(n, edge_set, odd_vertices)
        
        # Single edges
        pieces = [list(e) for e in t_join_edges]
        
        # Remaining edges
        remaining_edges = edge_set - t_join_edges
        
        # Decompose remaining into cycles
        cycles = decompose_eulerian(n, remaining_edges)
        pieces.extend(cycles)
        
        return pieces

    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        result = decompose(edges, n)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
