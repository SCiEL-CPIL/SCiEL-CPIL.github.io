"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_eulerian_trails(edges):
        """Find minimum number of edge-disjoint paths covering all edges.
        Uses the fact that a graph has an Eulerian trail iff all vertices have even degree
        (or exactly two have odd degree). We add virtual edges to pair up odd-degree vertices,
        find an Eulerian circuit, then split at virtual edges."""
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        # If no odd vertices, we can find one Eulerian circuit
        if not odd_vertices:
            # Find Eulerian circuit using Hierholzer's algorithm
            circuit = find_eulerian_circuit(adj)
            return [circuit]
        
        # Pair up odd vertices greedily (they come in pairs)
        # Add virtual edges between pairs
        n = max(adj.keys()) + 1
        virtual_edges = []
        for i in range(0, len(odd_vertices), 2):
            u, v = odd_vertices[i], odd_vertices[i+1]
            virtual_edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        # Find Eulerian circuit on augmented graph
        circuit = find_eulerian_circuit(adj)
        
        # Split circuit at virtual edges to get paths
        paths = []
        current_path = []
        for i in range(len(circuit)):
            u = circuit[i]
            v = circuit[(i+1) % len(circuit)]
            edge = tuple(sorted((u, v)))
            current_path.append(u)
            if edge in set(tuple(sorted(e)) for e in virtual_edges):
                # This is a virtual edge, end current path
                if current_path:
                    paths.append(current_path[:])
                current_path = []
        
        if current_path:
            paths.append(current_path)
        
        # Remove duplicate consecutive vertices and ensure paths are simple
        cleaned_paths = []
        for path in paths:
            # Remove consecutive duplicates
            cleaned = []
            for v in path:
                if not cleaned or cleaned[-1] != v:
                    cleaned.append(v)
            if len(cleaned) >= 2:
                cleaned_paths.append(cleaned)
        
        return cleaned_paths
    
    def find_eulerian_circuit(adj):
        """Find an Eulerian circuit using Hierholzer's algorithm."""
        # Work with a copy of adjacency
        adj_copy = {v: list(neighbors) for v, neighbors in adj.items()}
        
        # Find a vertex with non-empty adjacency
        start = None
        for v in adj_copy:
            if adj_copy[v]:
                start = v
                break
        if start is None:
            return []
        
        # Hierholzer's algorithm
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                next_v = adj_copy[v].pop()
                stack.append(next_v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback to trivial solution if out of time
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        
        edges = instance(*spec)
        paths = find_eulerian_trails(edges)
        
        # Verify and clean up paths
        valid_paths = []
        edge_set = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            # Check if path uses only valid edges and is simple
            path_edges = []
            valid = True
            seen_in_path = set()
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                edge = tuple(sorted((u, v)))
                if edge not in edge_set:
                    valid = False
                    break
                if u in seen_in_path or v in seen_in_path:
                    valid = False
                    break
                seen_in_path.add(u)
                seen_in_path.add(v)
                path_edges.append(edge)
            
            if not valid:
                continue
            
            # Check edge-disjointness
            if any(e in used_edges for e in path_edges):
                continue
            
            used_edges.update(path_edges)
            valid_paths.append(path)
        
        # If we didn't cover all edges, add remaining edges as single-edge paths
        remaining_edges = edge_set - used_edges
        for u, v in remaining_edges:
            valid_paths.append([u, v])
        
        results.append(valid_paths)
    
    return results
# EVOLVE-BLOCK-END
