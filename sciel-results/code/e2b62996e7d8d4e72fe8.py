"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
from random import Random

def construct():
    _start = time.time()
    _DEADLINE = 145.0  # 5s safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def greedy_path_cover(edges):
        """
        Greedy edge-disjoint path cover:
        1. Build adjacency with edge IDs.
        2. Repeatedly find a longest simple path in the remaining edge graph
           using a heuristic (start from a vertex with highest remaining degree,
           then do a greedy walk preferring neighbors with higher remaining degree).
        3. Remove used edges, repeat until all edges are covered.
        """
        if not edges:
            return []

        # Map each undirected edge to an index
        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v)) if u <= v else edge_set.add((v, u))

        adj = defaultdict(list)  # adj[u] = list of (v, edge_key)
        for (u, v) in edge_set:
            adj[u].append((v, (u, v)))
            adj[v].append((u, (u, v)))

        used_edges = set()
        paths = []

        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining:
                break

            # Build remaining adjacency
            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)

            # Choose start vertex: highest degree in remaining graph
            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None:
                break

            # Greedy longest path: walk preferring unvisited neighbors with high remaining degree
            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors:
                    break
                # Prefer neighbor with highest remaining degree (excluding used edges)
                # Count how many remaining edges each neighbor has to unvisited vertices
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor

            # Convert path to edge list and mark as used
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)

            # Verify all edges in path are actually in remaining
            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                # Fallback: just take one edge
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])

        return paths

    def eulerian_decomposition(edges):
        """
        Structural algorithm:
        1. Find odd-degree vertices.
        2. Pair them and find shortest paths between pairs.
        3. Add those paths as "virtual edges" to make the graph Eulerian.
        4. Find an Eulerian circuit in the augmented multigraph.
        5. Break the circuit at the virtual edges to get the original paths.
        This yields at most |odd|/2 + 1 paths per connected component.
        """
        if not edges:
            return []

        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Find odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]

        # Helper to find Eulerian circuit
        def find_eulerian_circuit(edge_list, adj_map):
            if not edge_list:
                return []
            
            edge_count = defaultdict(int)
            for (u, v) in edge_list:
                key = (u, v) if u <= v else (v, u)
                edge_count[key] += 1

            hier_adj = defaultdict(list)
            for (u, v) in edge_list:
                key = (u, v) if u <= v else (v, u)
                hier_adj[u].append((v, key))
                hier_adj[v].append((u, key))

            used_count = defaultdict(int)
            start = next(iter(adj_map))
            stack = [start]
            circuit = []
            while stack:
                v = stack[-1]
                found = False
                for (nb, key) in hier_adj[v]:
                    if used_count[key] < edge_count[key]:
                        used_count[key] += 1
                        stack.append(nb)
                        found = True
                        break
                if not found:
                    circuit.append(stack.pop())
            circuit.reverse()
            return circuit

        # Helper for BFS shortest path
        def bfs_shortest_path(src, dst, adj_map):
            if src == dst:
                return []
            visited = {src}
            parent = {src: None}
            q = deque([src])
            while q:
                cur = q.popleft()
                for nb in adj_map[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = cur
                        if nb == dst:
                            path = []
                            node = dst
                            while node is not None:
                                path.append(node)
                                node = parent[node]
                            path.reverse()
                            return path
                        q.append(nb)
            return None

        if not odd_vertices:
            # Already Eulerian
            circuit = find_eulerian_circuit(edges, adj)
            if circuit:
                return [circuit]
            return greedy_path_cover(edges)

        # Pair odd vertices
        # For better results, we could use a matching algorithm, but greedy pairing is fast
        pairs = []
        for i in range(0, len(odd_vertices) - 1, 2):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))

        # Find shortest paths for each pair
        virtual_edges = []
        for (a, b) in pairs:
            sp = bfs_shortest_path(a, b, adj)
            if sp and len(sp) >= 2:
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)

        # Build augmented edge list (original + virtual)
        # Note: virtual edges might overlap with original edges, creating multi-edges
        # We handle this by tracking counts in Hierholzer's
        
        all_edges_for_euler = list(edges) + virtual_edges
        
        # We need to be careful: if a virtual edge is the same as an original edge, 
        # the count increases. Hierholzer's with counts handles this.
        
        circuit = find_eulerian_circuit(all_edges_for_euler, adj)
        
        # Now break the circuit at virtual edges
        virtual_set = set(virtual_edges)
        
        paths = []
        current_path = [circuit[0]]
        for i in range(1, len(circuit)):
            u, v = circuit[i - 1], circuit[i]
            key = (u, v) if u <= v else (v, u)
            current_path.append(v)
            if key in virtual_set:
                # This edge is virtual - cut here
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = [v]

        if len(current_path) >= 2:
            paths.append(current_path)

        # Filter out trivial paths (single vertex)
        paths = [p for p in paths if len(p) >= 2]
        
        # Verify validity
        orig_edge_set = set()
        for (u, v) in edges:
            orig_edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        valid = True
        for p in paths:
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in orig_edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

        if valid and covered == orig_edge_set:
            return paths
        else:
            return greedy_path_cover(edges)

    def local_search_improve(paths, edges, adj, time_limit=5.0):
        """
        Local search to reduce the number of paths.
        Moves:
        1. Merge two paths P1, P2 if there is an edge (u, v) where u is an endpoint of P1 
           and v is an endpoint of P2 (or vice versa). The new path is P1 + [v] + reversed(P2) 
           (or similar). We must ensure the new path is simple (no repeated vertices).
        2. (More complex) Re-route: Remove an edge from a path, creating two paths, 
           then try to merge them with other paths in a way that reduces total count or 
           allows further merges. This is hard to implement simply. 
           Instead, let's stick to merging and maybe "swapping" endpoints if possible?
           
        Let's implement Merging.
        """
        start_time = time.time()
        best_paths = [list(p) for p in paths]
        best_score = len(best_paths)
        
        # Precompute endpoints for quick lookup
        # We'll maintain the list of paths and try to merge any pair.
        
        def is_simple_path(path):
            return len(set(path)) == len(path)
            
        def get_edges(path):
            return set((min(path[i], path[i+1]), max(path[i], path[i+1])) for i in range(len(path)-1))
            
        # To make merging efficient, we can try to merge path i and path j.
        # Endpoints of i: i[0], i[-1]
        # Endpoints of j: j[0], j[-1]
        # Possible connections:
        # i[-1] -> j[0] : new path = i + j[1:]
        # i[0] -> j[0] : new path = reversed(i) + j[1:] (if i[0] connected to j[0])
        # i[-1] -> j[-1] : new path = i + reversed(j[1:])
        # i[0] -> j[-1] : new path = reversed(i) + reversed(j[1:])
        
        # We need to check if the connecting edge exists in the graph AND is not used by other paths.
        # Wait, the paths are edge-disjoint. If we merge, we use the connecting edge.
        # The connecting edge must be in the graph.
        
        # Optimization: Keep a set of all edges in the current cover? 
        # Actually, all edges in the graph are covered. If we merge, we don't change the set of covered edges, 
        # we just change how they are grouped. 
        # BUT, the connecting edge (u,v) must be an edge in the graph.
        # Is it possible that the edge (u,v) is already used in path i or path j?
        # If u is an endpoint of i, the edge incident to u in i is unique (unless i is a cycle, but paths are simple).
        # So (u,v) cannot be in i unless v is the other endpoint of i (i is a single edge).
        # If (u,v) is in i, then v must be the other endpoint. Then merging i and j via (u,v) would imply v is also an endpoint of j?
        # If i=[u,v] and j=[v,w], merging via (u,v) is invalid because (u,v) is used in i.
        # We must ensure the connecting edge is NOT in the union of edges of P1 and P2.
        # Actually, if the connecting edge is in P1 or P2, the new path would repeat an edge, violating edge-disjointness of the single path? 
        # No, a path is a sequence of vertices. If we repeat an edge, it's not a simple path in terms of edges? 
        # The task says "Paths must be simple". Usually simple path means no repeated vertices.
        # If we repeat a vertex, we repeat an edge (in a simple graph).
        # So checking for simple path (no repeated vertices) is sufficient.
        
        while time.time() - start_time < time_limit:
            improved = False
            n_paths = len(best_paths)
            
            # Try all pairs
            for i in range(n_paths):
                if time.time() - start_time > time_limit:
                    return best_paths
                for j in range(i + 1, n_paths):
                    p1 = best_paths[i]
                    p2 = best_paths[j]
                    
                    # Check 4 connection possibilities
                    # 1. p1_end to p2_start
                    # 2. p1_start to p2_start
                    # 3. p1_end to p2_end
                    # 4. p1_start to p2_end
                    
                    candidates = []
                    
                    # 1. p1[-1] to p2[0]
                    u, v = p1[-1], p2[0]
                    if (u, v) in adj: # (u,v) is in graph
                        new_path = p1 + p2[1:]
                        if is_simple_path(new_path):
                            # Check if edge (u,v) is already in p1 or p2? 
                            # If new_path is simple, no edge is repeated. 
                            # So (u,v) is not in p1 or p2.
                            candidates.append(new_path)
                            
                    # 2. p1[0] to p2[0]
                    u, v = p1[0], p2[0]
                    if (u, v) in adj:
                        new_path = list(reversed(p1)) + p2[1:]
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                            
                    # 3. p1[-1] to p2[-1]
                    u, v = p1[-1], p2[-1]
                    if (u, v) in adj:
                        new_path = p1 + list(reversed(p2[1:]))
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                            
                    # 4. p1[0] to p2[-1]
                    u, v = p1[0], p2[-1]
                    if (u, v) in adj:
                        new_path = list(reversed(p1)) + list(reversed(p2[1:]))
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                            
                    if candidates:
                        # Pick one (e.g., the one that results in the "longest" path? Or just the first?)
                        # Any valid merge reduces count by 1.
                        merged = candidates[0]
                        
                        # Replace p1 and p2 with merged
                        # Remove j, then i (or adjust indices)
                        # It's easier to rebuild the list
                        new_paths = []
                        for k in range(n_paths):
                            if k == i or k == j:
                                continue
                            new_paths.append(best_paths[k])
                        new_paths.append(merged)
                        
                        best_paths = new_paths
                        improved = True
                        break # Restart pair search with new list
                if improved:
                    break
                    
            if not improved:
                # No merges possible. Try to "shuffle" to enable merges?
                # A simple shuffle: pick a random path, reverse it. This doesn't change validity but changes endpoints.
                # Endpoints change! Reversing a path swaps its start and end.
                # This might enable a merge that wasn't possible before.
                if len(best_paths) > 0:
                    idx = random.randint(0, len(best_paths) - 1)
                    best_paths[idx] = list(reversed(best_paths[idx]))
                else:
                    break

        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 10:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        adj = build_adjacency(edges)
        
        # 1. Get initial solution using Eulerian decomposition (strong structural start)
        try:
            paths = eulerian_decomposition(edges)
        except Exception:
            paths = greedy_path_cover(edges)
            
        # Fallback if Eulerian failed
        if not paths:
            paths = greedy_path_cover(edges)

        # Validate initial solution
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        valid = True
        for p in paths:
            if len(p) < 2:
                valid = False
                break
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

        if not valid or covered != edge_set:
            paths = greedy_path_cover(edges)
            # Re-validate greedy
            covered2 = set()
            valid2 = True
            for p in paths:
                if len(p) < 2:
                    valid2 = False
                    break
                if len(set(p)) != len(p):
                    valid2 = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid2 = False
                        break
                    if key in covered2:
                        valid2 = False
                        break
                    covered2.add(key)
                if not valid2:
                    break
            if not valid2 or covered2 != edge_set:
                # Last resort: each edge as its own path
                paths = [[list(e)] for e in edges]

        # 2. Improve with Local Search
        # Allocate remaining time / 3 for each graph? 
        # Or just a fixed time per graph if time permits.
        remaining_time = _DEADLINE - time.time()
        graphs_left = 3 - (len(results)) # rough estimate
        time_per_graph = max(1.0, remaining_time / max(1, graphs_left))
        
        try:
            improved_paths = local_search_improve(paths, edges, adj, time_limit=time_per_graph)
            # Validate improved paths
            covered3 = set()
            valid3 = True
            for p in improved_paths:
                if len(p) < 2:
                    valid3 = False
                    break
                if len(set(p)) != len(p):
                    valid3 = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid3 = False
                        break
                    if key in covered3:
                        valid3 = False
                        break
                    covered3.add(key)
                if not valid3:
                    break
                    
            if valid3 and covered3 == edge_set:
                paths = improved_paths
        except Exception:
            pass # Keep original paths

        results.append(paths)

    return results
# EVOLVE-BLOCK-END
