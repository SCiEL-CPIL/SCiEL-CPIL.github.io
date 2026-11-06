"""Optimal path decomposition using Minimum Weight Perfect Matching on Odd Vertices."""
import time
import sys
from collections import defaultdict, deque
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def all_pairs_shortest_paths(n, adj):
        """Compute shortest paths between all pairs using Dijkstra or BFS (unweighted)."""
        # Since it's an unweighted graph, BFS is sufficient.
        dist = [[-1] * n for _ in range(n)]
        prev = [[-1] * n for _ in range(n)]
        
        for s in range(n):
            dist[s][s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[s][v] == -1:
                        dist[s][v] = dist[s][u] + 1
                        prev[s][v] = u
                        queue.append(v)
        return dist, prev

    def reconstruct_path(prev, s, t):
        """Reconstruct path from s to t using prev table."""
        if prev[s][t] == -1 and s != t:
            return []
        path = []
        cur = t
        while cur != -1:
            path.append(cur)
            if cur == s:
                break
            cur = prev[s][cur]
        path.reverse()
        return path

    def min_weight_perfect_matching(odd_vertices, dist):
        """
        Find min weight perfect matching on odd_vertices.
        odd_vertices: list of vertex indices.
        dist: distance matrix.
        Returns list of pairs (u, v) and total weight.
        Uses DP with bitmask.
        """
        k = len(odd_vertices)
        if k == 0:
            return [], 0
        
        # Map odd vertices to indices 0..k-1 for DP
        # dp[mask] = min cost to match vertices in mask
        # mask is a bitmask of length k.
        # We iterate over masks with even number of bits.
        
        size = 1 << k
        dp = [float('inf')] * size
        parent = [(-1, -1)] * size  # (i, j) matched
        dp[0] = 0
        
        # Precompute weights between odd vertices
        w = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i+1, k):
                u = odd_vertices[i]
                v = odd_vertices[j]
                w[i][j] = dist[u][v]
                w[j][i] = dist[u][v]
        
        for mask in range(size):
            if dp[mask] == float('inf'):
                continue
            # Find first unset bit
            # Actually, standard approach: iterate masks, add next available pair.
            # It's better to iterate by number of set bits.
            # Or simply: for each mask, if it's not full, pick the first unset bit i,
            # and try to match it with every other unset bit j.
            
            # Find first unset bit
            i = -1
            for b in range(k):
                if not (mask & (1 << b)):
                    i = b
                    break
            if i == -1:
                continue
            
            # Try to match i with j
            for j in range(i + 1, k):
                if not (mask & (1 << j)):
                    new_mask = mask | (1 << i) | (1 << j)
                    new_cost = dp[mask] + w[i][j]
                    if new_cost < dp[new_mask]:
                        dp[new_mask] = new_cost
                        parent[new_mask] = (i, j)
        
        # Backtrack
        pairs = []
        mask = size - 1
        while mask != 0:
            i, j = parent[mask]
            pairs.append((odd_vertices[i], odd_vertices[j]))
            mask ^= (1 << i) | (1 << j)
            
        return pairs, dp[size - 1]

    def find_eulerian_circuit(adj_mult, start):
        """
        Find Eulerian circuit in a multigraph.
        adj_mult: dict of list of neighbors.
        We need to handle multi-edges.
        We'll use a stack-based Hierholzer's algorithm.
        We need to track which edges are used.
        Since we are building a multigraph from G + Matching, we can represent edges uniquely.
        However, for simplicity, we can just use the adjacency lists and pop edges.
        To ensure we return the circuit, we'll store the path.
        """
        # We need a mutable adjacency list where we can remove edges.
        # adj_mult is defaultdict(list).
        # We'll copy it to avoid modifying the original if needed, but we don't reuse it.
        adj_work = defaultdict(list)
        for u, neighbors in adj_mult.items():
            adj_work[u] = list(neighbors)
        
        path = []
        stack = [start]
        while stack:
            u = stack[-1]
            if adj_work[u]:
                v = adj_work[u].pop()
                stack.append(v)
            else:
                path.append(stack.pop())
        
        path.reverse()
        return path

    def decompose_graph(edges, n):
        adj = get_adjacency(edges)
        
        # 1. Find odd vertices
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        
        if not odd_vertices:
            # Eulerian graph, can be decomposed into 1 cycle (or 1 path if we allow cycles as paths? 
            # The task asks for "paths". A cycle is not a simple path unless we cut it.
            # But Gallai's conjecture is about paths. If the graph is Eulerian, we can find an Eulerian circuit.
            # We can split it into 1 path by cutting an edge? No, that leaves 2 endpoints.
            # Actually, if all degrees are even, we can decompose into 1 closed walk.
            # The task requires "simple paths".
            # If the graph is a single cycle, it's 1 path (if we break it) or 0?
            # Usually, "path" implies open.
            # If we have an Eulerian circuit, we can pick any edge and break the cycle into a path.
            # This path will cover all edges. So 1 path is sufficient.
            # Let's find an Eulerian circuit and return it as a single path (removing the last vertex which is the start).
            # Wait, a cycle v0-v1-...-v0 is not a simple path.
            # We can return [v0, v1, ..., v_{k-1}]. This covers all edges.
            # Is it simple? Yes, if the cycle is simple.
            # Eulerian circuit in a simple graph is not necessarily a simple cycle.
            # But we can just return the sequence of vertices from the Eulerian circuit, dropping the last one (which is the start).
            # This forms a trail that covers all edges. Is it a simple path?
            # No, it might repeat vertices.
            # The task requires "simple" paths.
            # If the graph is Eulerian, it might not be decomposable into 1 *simple* path.
            # However, for the specific graphs in the task (random connected), they are likely not Eulerian.
            # If they are, we might need more paths.
            # But let's assume the matching approach handles this. If odd_vertices is empty, matching is empty.
            # We just need to find a decomposition.
            # If the graph is Eulerian, we can find an Eulerian circuit.
            # We can split it into simple paths?
            # Actually, if the graph is Eulerian, the minimum number of *simple* paths to cover edges is not necessarily 1.
            # Example: Figure 8 graph. Eulerian, but needs 2 simple paths to cover all edges without repeating vertices?
            # No, Figure 8: two cycles sharing a vertex.
            # Path 1: Cycle 1. Path 2: Cycle 2.
            # So 2 paths.
            # The matching approach yields 0 added edges.
            # We need a fallback for Eulerian graphs.
            # For now, let's assume odd_vertices is not empty for these random graphs.
            # If it is empty, we'll use a greedy approach or just return the Eulerian circuit as a "path" (which might violate simplicity if it repeats vertices, but let's check the graphs).
            # Given n=20, p=0.3, it's very likely to have odd vertices.
            
            # If no odd vertices, we can try to find a Hamiltonian path? No, too hard.
            # We'll just use the greedy method as fallback.
            pass

        if len(odd_vertices) % 2 != 0:
            # Should not happen in a connected graph
            return None

        if not odd_vertices:
            # Fallback for Eulerian
            # Find Eulerian circuit
            adj_mult = defaultdict(list)
            for u, v in edges:
                adj_mult[u].append(v)
                adj_mult[v].append(u)
            start = next(iter(adj_mult))
            circuit = find_eulerian_circuit(adj_mult, start)
            # The circuit is a list of vertices.
            # We need to convert this to simple paths.
            # This is complex. For random graphs, this case is unlikely.
            # Let's just return the circuit as a single path (ignoring simplicity constraint for this rare case, 
            # or better, use the greedy method).
            # Let's use greedy for this case to ensure validity.
            return greedy_decompose(edges, adj, n)

        # 2. Compute shortest paths
        dist, prev = all_pairs_shortest_paths(n, adj)
        
        # 3. Min weight perfect matching
        pairs, weight = min_weight_perfect_matching(odd_vertices, dist)
        
        # 4. Construct augmented multigraph
        # We need to add the edges of the shortest paths for each pair in the matching.
        # The shortest path is a sequence of vertices. We add the edges in that sequence.
        adj_mult = defaultdict(list)
        for u, v in edges:
            adj_mult[u].append(v)
            adj_mult[v].append(u)
            
        # Track added edges to identify where to split
        # We need to distinguish original edges from added edges in the Eulerian circuit.
        # We can store edges as (u, v, is_added).
        # But Hierholzer's on adjacency lists doesn't easily support edge attributes if we just pop neighbors.
        # Alternative:
        # 1. Find Eulerian circuit in the multigraph.
        # 2. The circuit is a sequence of vertices.
        # 3. We know the set of "added" edges.
        # 4. We traverse the circuit. Whenever we traverse an "added" edge, we cut the path.
        
        # To do this, we need to know which edges are added.
        # Let's create a set of added edges.
        added_edges = set()
        for u, v in pairs:
            path = reconstruct_path(prev, u, v)
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                added_edges.add(e)
                # Add to multigraph
                adj_mult[path[i]].append(path[i+1])
                adj_mult[path[i+1]].append(path[i])
        
        # Find Eulerian circuit
        # We need to handle multi-edges properly.
        # The adjacency lists in adj_mult now contain duplicates for added edges if they overlap with original edges?
        # Yes, if a shortest path uses an original edge, that edge appears twice in the multigraph (once from G, once from M).
        # Hierholzer's will traverse both.
        
        start = odd_vertices[0]
        circuit = find_eulerian_circuit(adj_mult, start)
        
        # Now, split the circuit.
        # The circuit is a list of vertices: [v0, v1, ..., vk, v0]
        # We iterate through the edges (vi, vi+1).
        # If the edge is in added_edges, it's a cut point.
        # We start a new path.
        
        paths = []
        current_path = []
        
        # We need to be careful: the added edge might be traversed in either direction.
        # Also, an added edge might be an original edge.
        # If an edge is in added_edges, it was part of a shortest path in the matching.
        # We cut at these edges.
        
        for i in range(len(circuit) - 1):
            u = circuit[i]
            v = circuit[i+1]
            e = tuple(sorted((u, v)))
            
            if not current_path:
                current_path.append(u)
            
            # If this edge is an added edge, we finish the current path at u?
            # No, the added edge connects two segments.
            # The logic: The Eulerian circuit in G+M consists of paths from G connected by edges from M.
            # So, if we encounter an edge from M, we are leaving a "path segment" of G.
            # So we should close the current path.
            
            if e in added_edges:
                # The edge (u,v) is an added edge.
                # The current path ends at u.
                # We start a new path at v.
                if len(current_path) > 1:
                    paths.append(current_path)
                current_path = [v]
            else:
                # This is an original edge.
                # Extend current path.
                # But wait, what if the previous edge was also original?
                # We just append v.
                current_path.append(v)
        
        # After the loop, we have a current_path.
        # The circuit ends at v0 (start).
        # The last edge was (circuit[-2], circuit[-1]=v0).
        # If the last edge was original, current_path ends at v0.
        # If the last edge was added, current_path is [v0] (started at v0, cut immediately? No).
        
        # Let's trace:
        # Circuit: v0 -> v1 -> ... -> vk -> v0
        # If (v_{k-1}, v0) is added:
        #   At i=k-1: e is added.
        #   current_path was built up to v_{k-1}.
        #   We append current_path to paths.
        #   current_path becomes [v0].
        # End of loop.
        # We have a dangling current_path [v0].
        # We should discard it if it has length 1.
        
        if len(current_path) > 1:
            paths.append(current_path)
        
        # Verify paths
        # The paths should cover all original edges.
        # They should be simple.
        # They should be edge-disjoint.
        
        # There is a risk that the "added" edge is also an original edge.
        # In that case, it is in added_edges.
        # We cut there.
        # This is correct.
        
        # However, if the shortest path between u and v in the matching uses multiple edges,
        # all those edges are in added_edges.
        # We will cut at *each* of them?
        # No, that would break the path into many small pieces.
        # The matching adds a *path* of edges.
        # In the multigraph, these edges are distinct from the original edges (even if they are the same physical edge, they are parallel edges in the multigraph).
        # But in our `added_edges` set, we just store the tuple (u,v).
        # If the shortest path is u-x-v, then (u,x) and (x,v) are in added_edges.
        # When we traverse the Eulerian circuit, we will encounter (u,x) and cut.
        # Then we start a new path at x.
        # Then we encounter (x,v) and cut.
        # This effectively removes the entire matching path from the "original" path segments.
        # This is correct! The segments between added edges are exactly the paths in G.
        
        # One issue: What if the added edge is (u,v) and we traverse it from v to u?
        # `e = tuple(sorted((u,v)))` handles this.
        
        # Check validity
        if not validate_paths(paths, edges, n):
            # Fallback to greedy
            return greedy_decompose(edges, adj, n)
            
        return paths

    def validate_paths(paths, edges, n):
        all_edges = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(set(p)) != len(p):
                return False
            for i in range(len(p)-1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in all_edges:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == all_edges

    def greedy_decompose(edges, adj, n):
        # Simple greedy fallback
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        while remaining:
            # Find a vertex with highest degree in remaining
            degs = defaultdict(int)
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            if not degs:
                break
            start = max(degs.keys(), key=lambda x: degs[x])
            path = [start]
            current = start
            while True:
                next_v = None
                for v in adj[current]:
                    e = tuple(sorted((current, v)))
                    if e in remaining and v not in path:
                        next_v = v
                        break
                if next_v is None:
                    break
                path.append(next_v)
                remaining.remove(tuple(sorted((current, next_v))))
                current = next_v
            if len(path) > 1:
                paths.append(path)
            else:
                # Single vertex, pick any remaining edge incident to it
                for e in list(remaining):
                    if start in e:
                        other = e[0] if e[1] == start else e[1]
                        paths.append([start, other])
                        remaining.remove(e)
                        break
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        if time.time() - start_time > deadline - 10:
            # Fallback
            results.append(greedy_decompose(edges, adj, n))
            continue
            
        paths = decompose_graph(edges, n)
        if paths is None:
            paths = greedy_decompose(edges, adj, n)
            
        results.append(paths)

    return results
