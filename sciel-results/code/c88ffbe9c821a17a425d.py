"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    # Deadline guard: 120s hard limit, keep safety margin
    start_time = time.time()
    DEADLINE = start_time + 115  # 5s safety margin before 120s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def path_cover(edges, n):
        """Find edge-disjoint path cover minimizing number of paths.
        
        Uses a greedy approach:
        1. Build adjacency from edges
        2. Repeatedly find longest simple path in remaining edges
        3. This gives a valid path cover (edge-disjoint, covers all edges)
        
        For better results, use a flow-based approach to get minimum path cover.
        """
        if not edges:
            return []
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            edge_set.add((v, u))
        
        # Try to find minimum path cover using a matching-based approach
        # Minimum path cover in undirected graph can be found via:
        # - If graph has a Hamiltonian path, 1 path suffices
        # - Otherwise, use greedy longest path decomposition
        
        # First try: check if graph has a Hamiltonian path (rare for these sizes)
        # For n up to 30, we can try a heuristic
        
        # Greedy longest path approach
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            # Find the longest simple path in the subgraph of remaining edges
            # Use a heuristic: start from a leaf or low-degree vertex, extend greedily
            
            # Build subgraph from remaining edges
            sub_adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find vertices with edges in remaining subgraph
            active_verts = [v for v in range(n) if sub_adj[v]]
            
            if not active_verts:
                break
            
            # Start from a vertex with minimum degree in subgraph (prefer leaves)
            start = min(active_verts, key=lambda v: len(sub_adj[v]))
            
            # Greedy extend: at each step, go to neighbor with fewest options
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Find best next vertex
                candidates = [nb for nb in sub_adj[current] if nb not in visited]
                if not candidates:
                    break
                
                # Choose candidate that leads to longest continuation
                # Heuristic: pick neighbor with minimum degree (to avoid dead ends)
                best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                
                # Remove edges used so far in this path
                for i in range(len(path) - 1):
                    e1 = (path[i], path[i+1])
                    e2 = (path[i+1], path[i])
                    remaining_edges.discard(e1)
                    remaining_edges.discard(e2)
                    sub_adj[path[i]].discard(path[i+1])
                    sub_adj[path[i+1]].discard(path[i])
                
                path.append(best_next)
                visited.add(best_next)
                current = best_next
            
            # Remove edges in the final path
            for i in range(len(path) - 1):
                e1 = (path[i], path[i+1])
                e2 = (path[i+1], path[i])
                remaining_edges.discard(e1)
                remaining_edges.discard(e2)
                sub_adj[path[i]].discard(path[i+1])
                sub_adj[path[i+1]].discard(path[i])
            
            if len(path) > 1:
                paths.append(path)
        
        # Verify all edges are covered
        covered = set()
        for path in paths:
            for i in range(len(path) - 1):
                covered.add(tuple(sorted((path[i], path[i+1]))))
        
        # If some edges are missed (shouldn't happen), add them as single-edge paths
        for e in edges:
            if e not in covered:
                paths.append(list(e))
        
        return paths

    def path_cover_flow(edges, n):
        """Use a better algorithm: find minimum path cover via edge coloring / matching.
        
        For an undirected graph, the minimum number of edge-disjoint paths that cover all edges
        equals ceil(odd_degree_vertices / 2) if the graph is connected, but this is a lower bound.
        
        Actually, the minimum path cover in terms of number of paths is related to the
        number of odd-degree vertices. Each path has exactly 2 endpoints (or 0 if it's a cycle).
        A path that is not a cycle contributes 2 to the odd-degree count at its endpoints.
        
        The minimum number of paths needed is at least ceil(#odd_degree_vertices / 2).
        
        For a better algorithm, we can use the fact that:
        - If all vertices have even degree, we can decompose into cycles (0 paths if we allow cycles, 
          but we need paths, so we break cycles into paths)
        - Actually, for path cover (not cycle cover), we need to think differently.
        
        Let me use a simpler but effective heuristic:
        1. Pair up odd-degree vertices
        2. Add "dummy" edges to make all degrees even
        3. Find an Eulerian circuit in the augmented graph
        4. Break the circuit into paths at the dummy edge locations
        """
        if not edges:
            return []
        
        # Compute degrees
        degree = {i: 0 for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        
        if not odd_verts:
            # All even: find Eulerian circuit, then break into paths
            # But we still need at least 1 path if there are edges
            # Actually, if all degrees are even and graph is connected, there's an Eulerian circuit.
            # We can break it into 1 path by cutting at one point.
            # But the minimum might be 1 if there's a Hamiltonian path, or more.
            # For now, use the Eulerian circuit approach.
            pass
        
        # Use a simpler greedy approach with backtracking for small graphs
        # Given time constraints, use the greedy longest path method
        return path_cover(edges, n)

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Return what we have, pad with trivial solutions
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        
        # Use the greedy path cover
        paths = path_cover(edges, n)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
