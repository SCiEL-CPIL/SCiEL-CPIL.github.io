"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Solve edge-disjoint path cover for three fixed random connected graphs.
    Uses minimum-cost flow formulation:
    - Each undirected edge becomes two directed arcs (u->v, v->u) with capacity 1, cost 1.
    - Node balances: b(v) = (deg(v) mod 2) * sign, where odd-degree nodes are endpoints.
      Actually for path cover: sum of out - sum of in = b(v) for each node.
      For a path cover, odd-degree vertices must be endpoints (start or end of some path).
      The number of paths = (number of odd-degree vertices) / 2.
      Wait, that's for a single path or Eulerian trails. For path cover of all edges:
      
    Actually, the minimum number of edge-disjoint paths covering all edges in a graph
    is at least ceil(number_of_odd_vertices / 2). This is because each path has exactly
    two endpoints, and only odd-degree vertices can be endpoints in a path decomposition
    where all edges are covered.
    
    More precisely: In any path decomposition, a vertex v is an endpoint of some path
    if and only if the number of times paths "start or end" at v is odd, which relates
    to deg(v) mod 2. Specifically, the minimum number of paths is exactly 
    max(1, number_of_odd_vertices // 2) if the graph is connected and has edges.
    
    Wait, that's not quite right either. Let me think more carefully.
    
    For a connected graph, the minimum number of edge-disjoint paths that cover all edges
    equals half the number of odd-degree vertices (if there are any), or 1 if all degrees
    are even (Eulerian circuit can be split into one path... no, it's one cycle, but we need
    paths). Actually for Eulerian (all even degrees), we can have 1 path if we break the
    cycle at any point, but the path would need to start and end at the same vertex which
    isn't allowed for a simple path unless it's a single vertex. 
    
    Hmm, let me reconsider. A path is a sequence of distinct vertices where consecutive
    vertices are connected by edges. So a path has a start and end vertex (which can be 
    different, or if it's just one vertex, start=end).
    
    The key theorem: The minimum number of edge-disjoint paths covering all edges of a
    connected graph G is:
    - If G has 0 edges: 0 paths
    - If G has edges: max(1, |{v : deg(v) is odd}| / 2)
    
    Wait, this is actually a known result. The minimum number of paths in an edge-disjoint
    path decomposition is ceil(#odd_vertices / 2) if there are odd vertices, and 1 if 
    all degrees are even and there's at least one edge.
    
    Actually I recall now: it's exactly #odd_vertices / 2 if #odd_vertices > 0, else 1 (for
    non-empty graph). This is because you can always add dummy edges to make all degrees even,
    find an Eulerian circuit, then remove the dummy edges to get paths.
    
    So the minimum number of paths is determined solely by the number of odd-degree vertices!
    
    Let me verify: For the three graphs:
    - n=20, p=0.3, seed=1
    - n=25, p=0.3, seed=2  
    - n=30, p=0.25, seed=3
    
    The seed program uses all edges as separate paths, giving total = sum of |E| for each graph.
    The minimum is sum of max(1, odd_count/2) for each graph.
    
    So I just need to:
    1. Generate each graph
    2. Count odd-degree vertices
    3. Compute min_paths = max(1, odd_count // 2) if edges exist, else 0
    4. Actually construct the paths
    
    To construct the paths, I can:
    1. Add dummy edges between odd-degree vertices to make all degrees even
    2. Find an Eulerian circuit
    3. Remove dummy edges to split into paths
    
    Let me implement this.
    """
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_eulerian_circuit(adj, start):
        """Find Eulerian circuit using Hierholzer's algorithm."""
        # adj: dict mapping node to list of (neighbor, edge_index)
        # We need to track used edges
        used = [False] * len(adj_edges)
        circuit = []
        
        def dfs(v):
            while adj[v]:
                # Find next unused edge
                while adj[v] and used[adj[v][-1][1]]:
                    adj[v].pop()
                if not adj[v]:
                    break
                w, ei = adj[v].pop()
                used[ei] = True
                dfs(w)
                circuit.append(v)
        
        dfs(start)
        circuit.reverse()
        return circuit
    
    def decompose_into_paths(n, edge_list):
        """Decompose edge list into minimum number of edge-disjoint paths."""
        if not edge_list:
            return []
        
        # Build degree count and adjacency
        degree = [0] * n
        adj = [[] for _ in range(n)]
        
        for i, (u, v) in enumerate(edge_list):
            degree[u] += 1
            degree[v] += 1
            adj[u].append((v, i))
            adj[v].append((u, i))
        
        # Find odd-degree vertices
        odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]
        
        if len(odd_vertices) == 0:
            # Eulerian graph: one circuit, which we convert to one path
            # Find a start vertex with edges
            start = next((v for v in range(n) if degree[v] > 0), 0)
            
            # Hierholzer's algorithm
            adj_copy = [list(lst) for lst in adj]
            used = [False] * len(edge_list)
            stack = [start]
            circuit = []
            
            while stack:
                v = stack[-1]
                # Find next unused edge from v
                found = False
                while adj_copy[v]:
                    w, ei = adj_copy[v][-1]
                    if not used[ei]:
                        adj_copy[v].pop()
                        used[ei] = True
                        stack.append(w)
                        found = True
                        break
                    else:
                        adj_copy[v].pop()
                if not found:
                    circuit.append(stack.pop())
            
            circuit.reverse()
            # circuit is a cycle: [v0, v1, ..., vk, v0]
            # Convert to path by removing the last vertex (which equals first)
            if len(circuit) > 1 and circuit[0] == circuit[-1]:
                path = circuit[:-1]
            else:
                path = circuit
            return [path]
        
        else:
            # Pair up odd vertices and add dummy edges
            # This makes all degrees even
            dummy_edges = []
            for i in range(0, len(odd_vertices), 2):
                u, v = odd_vertices[i], odd_vertices[i + 1]
                dummy_edges.append((u, v))
            
            # Build combined adjacency with real and dummy edges
            all_edges = list(edge_list) + dummy_edges
            adj2 = [[] for _ in range(n)]
            for i, (u, v) in enumerate(all_edges):
                adj2[u].append((v, i))
                adj2[v].append((u, i))
            
            # Find Eulerian circuit in the augmented graph
            start = next((v for v in range(n) if degree[v] > 0 or any(adj2[v])), 0)
            
            used = [False] * len(all_edges)
            stack = [start]
            circuit = []
            
            while stack:
                v = stack[-1]
                found = False
                while adj2[v]:
                    w, ei = adj2[v][-1]
                    if not used[ei]:
                        adj2[v].pop()
                        used[ei] = True
                        stack.append(w)
                        found = True
                        break
                    else:
                        adj2[v].pop()
                if not found:
                    circuit.append(stack.pop())
            
            circuit.reverse()
            # circuit is a cycle in the augmented graph
            # Remove the last vertex (equals first) to get a path
            if len(circuit) > 1 and circuit[0] == circuit[-1]:
                path = circuit[:-1]
            else:
                path = circuit
            
            # Now split the path at dummy edges to get the actual paths
            # A dummy edge is one that's in dummy_edges
            dummy_set = set()
            for u, v in dummy_edges:
                dummy_set.add((u, v))
                dummy_set.add((v, u))
            
            paths = []
            current_path = [path[0]]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                current_path.append(v)
                if (u, v) in dummy_set:
                    # End of a real path
                    if len(current_path) > 1:
                        paths.append(current_path)
                    current_path = [v]
            
            if len(current_path) > 1:
                paths.append(current_path)
            
            # Filter out any paths that are just single vertices (shouldn't happen)
            paths = [p for p in paths if len(p) > 1]
            
            return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_into_paths(n, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
