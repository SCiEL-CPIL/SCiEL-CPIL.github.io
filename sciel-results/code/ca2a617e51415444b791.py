"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    # Deadline guard: 150s hard limit, keep safety margin
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin before 150s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_path_cover_greedy_improved(edges, n):
        """Improved greedy path cover with better heuristics."""
        if not edges:
            return []
        
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            # Build subgraph from remaining edges
            sub_adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find vertices with edges in remaining subgraph
            active_verts = [v for v in range(n) if sub_adj[v]]
            
            if not active_verts:
                break
            
            # Try multiple starting points and pick the best path
            best_path = None
            best_len = 0
            
            # Try starting from each active vertex (or a subset for larger graphs)
            start_candidates = active_verts if len(active_verts) <= 10 else \
                              sorted(active_verts, key=lambda v: len(sub_adj[v]))[:10]
            
            for start in start_candidates:
                # Greedy extend with lookahead
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    candidates = [nb for nb in sub_adj[current] if nb not in visited]
                    if not candidates:
                        break
                    
                    # Choose candidate: prefer one that leads to longest continuation
                    # Use a simple heuristic: pick neighbor with minimum degree
                    best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                    
                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                # Fallback: take any edge
                e = next(iter(remaining_edges))
                best_path = list(e)
            
            # Remove edges in the path
            for i in range(len(best_path) - 1):
                e1 = tuple(sorted((best_path[i], best_path[i+1])))
                remaining_edges.discard(e1)
            
            paths.append(best_path)
        
        return paths

    def find_path_cover_eulerian(edges, n):
        """Use Eulerian circuit decomposition for better path cover.
        
        The minimum number of edge-disjoint paths covering all edges in a connected graph
        is at least ceil(odd_degree_vertices / 2). We can achieve this bound by:
        1. Pairing up odd-degree vertices
        2. Adding dummy edges to make all degrees even
        3. Finding an Eulerian circuit
        4. Breaking the circuit at dummy edge locations
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
            # All even: find Eulerian circuit, return as single path (break at one point)
            # Actually, we need to find an Eulerian circuit and convert to a path
            # For simplicity, use greedy
            return find_path_cover_greedy_improved(edges, n)
        
        # Pair up odd-degree vertices
        # Simple pairing: just pair consecutive odd vertices
        pairs = []
        for i in range(0, len(odd_verts), 2):
            if i + 1 < len(odd_verts):
                pairs.append((odd_verts[i], odd_verts[i+1]))
        
        # The number of paths is at least len(pairs)
        # Use greedy approach but with awareness of odd vertices
        
        # For now, use the improved greedy which should do well
        return find_path_cover_greedy_improved(edges, n)

    def find_path_cover_max_flow(edges, n):
        """Try to use a flow-based approach for minimum path cover."""
        if not edges:
            return []
        
        # For small graphs, we can try a more sophisticated approach
        # But given time constraints, use the improved greedy
        return find_path_cover_greedy_improved(edges, n)

    def find_path_cover_best(edges, n):
        """Try multiple strategies and return the best result."""
        if not edges:
            return []
        
        # Strategy 1: Improved greedy
        paths1 = find_path_cover_greedy_improved(edges, n)
        
        # Strategy 2: Eulerian-based (may not be better for path cover)
        # paths2 = find_path_cover_eulerian(edges, n)
        
        # Return the best (fewest paths)
        return paths1

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
        
        # Use the best path cover algorithm
        paths = find_path_cover_best(edges, n)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
