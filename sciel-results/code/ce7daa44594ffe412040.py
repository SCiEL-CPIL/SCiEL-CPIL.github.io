"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict

def construct():
    # Generate the three fixed instance graphs
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    graphs = [instance(*spec) for spec in specs]
    
    def decompose_paths(edges):
        """
        Decompose a connected graph into edge-disjoint paths covering all edges.
        Uses a greedy algorithm with matching-based merging to minimize path count.
        """
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find all odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        # Start with each edge as a separate path
        # Each path is represented as a list of vertices
        # We'll use a more efficient representation: paths as (start, end, vertices_set, edge_set)
        
        # Initialize: each edge is a path
        # path_id -> (vertices_list, start, end)
        # But we need to track which edges are used
        
        # Let's use a different approach: 
        # 1. Find an Eulerian trail in each connected component after adding dummy edges
        #    to make all degrees even (pair up odd vertices)
        # 2. Or use the standard algorithm:
        #    - Find a trail decomposition
        
        # Standard algorithm for path decomposition:
        # 1. While there are edges:
        #    a. Find any edge not yet used
        #    b. Follow a trail from that edge as long as possible
        #    c. This gives a path
        #    d. Remove those edges
        # This is simple but doesn't minimize path count.
        
        # Better approach:
        # The minimum number of paths needed to cover all edges in a connected graph
        # is max(1, odd_degree_count / 2) if we can pair odd vertices optimally.
        
        # Algorithm:
        # 1. Add dummy edges to pair up all odd-degree vertices, making the graph Eulerian
        # 2. Find an Eulerian circuit
        # 3. The dummy edges split the circuit into paths
        
        # But we don't want to actually add edges; we just need to know how to split.
        
        # Let's use a greedy approach that works well in practice:
        # 1. Start with all edges as separate paths
        # 2. Repeatedly merge paths when possible (when the end of one matches the start of another
        #    and the merged path is simple)
        # 3. Use matching to find optimal merges
        
        # Simpler greedy: 
        # 1. Find a spanning tree or use DFS to create trails
        # 2. Decompose into trails that cover all edges
        
        # Let's implement the standard "trail decomposition" algorithm:
        # 1. While edges remain:
        #    a. Pick any unused edge (u, v)
        #    b. Start a trail from u, follow unused edges greedily until we can't
        #    c. This gives a maximal trail
        #    d. Record this trail as a path
        #    e. Mark all edges in the trail as used
        
        # This gives a valid decomposition but may not be minimal.
        # For better results, we can use the pairing approach.
        
        # Let's use the pairing approach:
        # 1. Identify all odd-degree vertices
        # 2. Pair them up (greedily or optimally)
        # 3. The pairing defines how to split an Eulerian circuit into paths
        
        # First, let's find an Eulerian trail decomposition by:
        # 1. Adding dummy edges to make all degrees even
        # 2. Finding an Eulerian circuit
        # 3. Removing dummy edges to get paths
        
        # Step 1: Create a multigraph with original edges + dummy edges
        # Dummy edges connect pairs of odd vertices
        
        # Pair up odd vertices greedily
        odd_list = list(odd_vertices)
        dummy_edges = []
        for i in range(0, len(odd_list), 2):
            if i + 1 < len(odd_list):
                dummy_edges.append((odd_list[i], odd_list[i+1]))
        
        # Build the multigraph
        mult_adj = defaultdict(list)
        edge_list = list(edges)
        for u, v in edge_list:
            mult_adj[u].append(v)
            mult_adj[v].append(u)
        
        # Add dummy edges
        dummy_set = set()
        for u, v in dummy_edges:
            mult_adj[u].append(v)
            mult_adj[v].append(u)
            dummy_set.add(tuple(sorted((u, v))))
        
        # Find Eulerian circuit using Hierholzer's algorithm
        def find_eulerian_circuit(adj, start):
            """Find an Eulerian circuit in a connected graph."""
            # Use Hierholzer's algorithm
            stack = [start]
            circuit = []
            # Use a copy of adj since we'll modify it
            adj_copy = defaultdict(list)
            for node in adj:
                adj_copy[node] = list(adj[node])
            
            while stack:
                v = stack[-1]
                if adj_copy[v]:
                    w = adj_copy[v].pop()
                    stack.append(w)
                else:
                    circuit.append(stack.pop())
            
            circuit.reverse()
            return circuit
        
        # Find a connected component with edges
        visited = set()
        def find_component(start):
            stack = [start]
            visited.add(start)
            component = []
            while stack:
                v = stack.pop()
                component.append(v)
                for w in adj.get(v, []):
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
            return component
        
        # Find the start vertex (one with edges)
        start_vertex = None
        for v in mult_adj:
            if mult_adj[v]:
                start_vertex = v
                break
        
        if start_vertex is None:
            return []
        
        # Find Eulerian circuit
        circuit = find_eulerian_circuit(mult_adj, start_vertex)
        
        # The circuit is a list of vertices: [v0, v1, v2, ..., vk] where (vi, vi+1) are edges
        # Now we need to split this into paths at the dummy edges
        
        # Extract the edges from the circuit
        circuit_edges = []
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            circuit_edges.append((u, v))
        
        # Now split the circuit into segments separated by dummy edges
        # A dummy edge is one that's in dummy_set
        # The paths are the segments between consecutive dummy edges
        
        paths = []
        current_path = [circuit[0]]
        
        for i in range(len(circuit_edges)):
            u, v = circuit_edges[i]
            current_path.append(v)
            edge_key = tuple(sorted((u, v)))
            if edge_key in dummy_set:
                # End of a path
                if len(current_path) > 1:
                    paths.append(list(current_path))
                current_path = [v]
        
        # Add the last path if it has edges
        if len(current_path) > 1:
            paths.append(list(current_path))
        
        # Verify: each path should be simple and use only original edges
        # Filter out any invalid paths
        valid_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            # Check if path is simple (no repeated vertices)
            if len(set(path)) != len(path):
                continue
            # Check if all edges in path are in original edge set
            edge_set = set(edges)
            valid = True
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in edge_set:
                    valid = False
                    break
            if valid:
                valid_paths.append(path)
        
        # Check if we covered all edges
        # If not, fall back to simple decomposition
        used_edges = set()
        for path in valid_paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                used_edges.add(e)
        
        if used_edges != set(edges):
            # Fall back to simple trail decomposition
            valid_paths = []
            remaining = set(edges)
            adj_remaining = defaultdict(list)
            for u, v in remaining:
                adj_remaining[u].append(v)
                adj_remaining[v].append(u)
            
            while remaining:
                # Pick any edge
                u, v = next(iter(remaining))
                path = [u, v]
                remaining.remove(tuple(sorted((u, v))))
                adj_remaining[u].remove(v)
                adj_remaining[v].remove(u)
                
                current = v
                prev = u
                while adj_remaining[current]:
                    # Find next vertex
                    next_v = adj_remaining[current][0]
                    path.append(next_v)
                    e = tuple(sorted((current, next_v)))
                    remaining.discard(e)
                    adj_remaining[current].remove(next_v)
                    adj_remaining[next_v].remove(current)
                    prev, current = current, next_v
                
                valid_paths.append(path)
        
        return valid_paths
    
    results = [decompose_paths(g) for g in graphs]
    return results
# EVOLVE-BLOCK-END
