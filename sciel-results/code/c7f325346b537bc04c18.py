import sys
from collections import defaultdict

def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(n, edges):
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find vertices with odd degree
        odd_vertices = [u for u in range(n) if len(adj[u]) % 2 != 0]
        
        # Pair up odd vertices
        virtual_edges = []
        for i in range(0, len(odd_vertices), 2):
            if i + 1 < len(odd_vertices):
                u, v = odd_vertices[i], odd_vertices[i + 1]
                virtual_edges.append((u, v))
                # Add to adj for Eulerian circuit finding
                adj[u].append(v)
                adj[v].append(u)
        
        # Find Eulerian Circuit using Hierholzer's
        # We need to distinguish real edges from virtual ones
        # Let's use a set of edges to track usage
        # Since we added virtual edges to adj, we need to know which are which
        # Better: Reconstruct adj with edge IDs or just track used edges
        
        # Let's use a standard Hierholzer's with edge tracking
        # We need to ensure we use each edge exactly once
        # For simple graphs, we can just remove edges as we visit them
        
        # Reset adj to original + virtual
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in virtual_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Map edge to a unique identifier to handle potential duplicates in adj list?
        # In this graph, no multi-edges. So (u,v) is unique.
        
        circuit = []
        
        def hierholzer(start):
            stack = [start]
            path = []
            while stack:
                v = stack[-1]
                if adj[v]:
                    next_v = adj[v].pop()
                    # Remove the reverse edge
                    # Since it's undirected, we need to remove v from next_v's list
                    if v in adj[next_v]:
                        adj[next_v].remove(v)
                    stack.append(next_v)
                else:
                    path.append(stack.pop())
            return path

        # Graph is connected, so start at any vertex with edges
        start_node = None
        for u in range(n):
            if adj[u]:
                start_node = u
                break
        
        if start_node is None:
            return []
            
        circuit = hierholzer(start_node)
        # circuit is in reverse order
        circuit.reverse()
        
        # circuit is [v0, v1, ..., vk, v0]
        # Edges are (circuit[i], circuit[i+1])
        
        # Identify virtual edges in the circuit
        virtual_set = {tuple(sorted(e)) for e in virtual_edges}
        
        paths = []
        current_path = [circuit[0]]
        
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            current_path.append(v)
            edge = tuple(sorted((u, v)))
            if edge in virtual_set:
                # End of path
                # The path is current_path
                # Note: current_path ends at v, which is the start of the next virtual edge?
                # The virtual edge is (u,v). The path ends at v.
                # The next path will start at v?
                # Wait, the circuit continues from v.
                # So the next path starts at v.
                # But the virtual edge is "removed" from the path.
                # So the path is circuit[0]...v.
                paths.append(current_path)
                current_path = [v] # Start next path at v
                
        # Add the last path
        if len(current_path) > 1:
            paths.append(current_path)
            
        # If there are no virtual edges (Eulerian graph), we have one big path (cycle)
        # We need to split it into ceil(m/2) paths
        if not virtual_edges:
            # circuit is a cycle of length m
            # We split into segments of length 2
            # Vertices: c[0], c[1], c[2] ...
            # Path 1: c[0]-c[1]-c[2]
            # Path 2: c[3]-c[4]-c[5]
            # ...
            new_paths = []
            i = 0
            m = len(circuit) - 1
            while i < m:
                if i + 2 < len(circuit):
                    # Path of length 2
                    p = [circuit[i], circuit[i+1], circuit[i+2]]
                    new_paths.append(p)
                    i += 2
                else:
                    # Remaining edges
                    # i to m
                    p = circuit[i:m+1]
                    new_paths.append(p)
                    i = m
            paths = new_paths
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_graph(n, edges)
        results.append(paths)
        
    return results
