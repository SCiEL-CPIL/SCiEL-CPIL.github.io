"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """Construct a 12-vertex graph minimizing distinct simple cycle lengths.
    
    Strategy: Search for a graph with value 1 (single cycle length).
    If not found, fall back to two K_{3,3} components (value 2).
    """
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    # First, try to find a graph with only 5-cycles (value 1)
    # The Petersen graph on 10 vertices has this property.
    # For 12 vertices, try to find a 3-regular triangle-free graph with girth 5.
    
    # Known: Petersen graph has 10 vertices, 3-regular, girth 5, only 5-cycles.
    # Try adding 2 vertices to Petersen while maintaining constraints.
    
    # Petersen graph edges (standard labeling)
    petersen_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # outer 5-cycle
        [5, 7], [7, 9], [9, 6], [6, 8], [8, 5],  # inner star
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9]   # spokes
    ]
    
    # Try to extend Petersen to 12 vertices by adding vertices 10, 11
    # Each new vertex needs degree >= 3, connected to existing vertices
    # such that no new cycle lengths are introduced (only 5-cycles allowed)
    
    # This is very constrained. Let's do a targeted search.
    # For each pair of connections for vertex 10 (3 edges) and vertex 11 (3 edges),
    # check if the resulting graph has only 5-cycles.
    
    # Actually, let's first verify: is value 1 even possible for 12 vertices?
    # By the (3,5)-cage being Petersen (10 vertices), any 3-regular girth-5 graph
    # on more than 10 vertices must have additional cycles.
    # 
    # Let's try a different approach: search over all 3-regular triangle-free graphs
    # on 12 vertices and find one with minimum distinct cycle lengths.
    # This is expensive, so let's be smart about it.
    
    # Alternative: try the "dodecahedron-like" structure or other known graphs.
    
    # Let's try: two disjoint Petersen-like structures won't work (need 20 vertices).
    
    # Try a single connected component. One candidate: the 12-vertex 3-regular 
    # graph formed by two 5-cycles with cross connections (but this may have triangles).
    
    # Let me try a specific construction:
    # Vertices 0-4: outer cycle, 5-9: inner cycle, 10-11: extra
    # This is getting complex. Let's do a bounded search.
    
    # Search strategy: try random 3-regular triangle-free graphs on 12 vertices
    # and check their cycle length spectrum.
    
    import random
    random.seed(42)
    
    def get_cycle_lengths(adj, n):
        """Find all simple cycle lengths in the graph."""
        cycle_lengths = set()
        
        def dfs(start, current, visited, path):
            if time.time() > deadline:
                return
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 4:
                    cycle_lengths.add(len(path))
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        
        for start in range(n):
            visited = {start}
            dfs(start, start, visited, [start])
        
        return cycle_lengths
    
    def is_valid_3regular_triangle_free(adj, n):
        """Check if graph is 3-regular, triangle-free, and has a cycle."""
        for i in range(n):
            if len(adj[i]) != 3:
                return False
        # Check triangles
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        return False
        # Check has cycle
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                has_cycle = False
                while stack:
                    v = stack.pop()
                    for u in adj[v]:
                        if not visited[u]:
                            visited[u] = True
                            stack.append(u)
                        elif u != i:  # rough check
                            has_cycle = True
                # More careful: just check if any vertex has degree >= 2 in the component
                # For 3-regular, if n >= 4, there's always a cycle
        return True
    
    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    best_edges = None
    best_cycle_count = float('inf')
    
    # Try random 3-regular graphs
    n = 12
    attempts = 0
    max_attempts = 500
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate random 3-regular graph
        # Use configuration model
        half_edges = []
        for i in range(n):
            for _ in range(3):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        edges = []
        for i in range(0, n * 3, 2):
            u, v = half_edges[i], half_edges[i+1]
            if u != v:
                edges.append((u, v))
        
        # Check for duplicates and loops
        if len(edges) != n * 3 // 2:
            continue
        
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u,v), max(u,v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        adj = build_adj(edges, n)
        
        if not is_valid_3regular_triangle_free(adj, n):
            continue
        
        # Get cycle lengths
        cl = get_cycle_lengths(adj, n)
        if len(cl) < best_cycle_count:
            best_cycle_count = len(cl)
            best_edges = [list(e) for e in edges]
            
            if best_cycle_count == 1:
                break
    
    # If we found a value-1 graph, return it
    if best_edges is not None and best_cycle_count == 1:
        # Verify and return
        adj = build_adj(best_edges, n)
        for e in best_edges:
            u, v = e[0], e[1]
            if not isinstance(u, int) or isinstance(u, bool):
                raise ValueError(f"Edge endpoint {u} is not a pure int")
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"Edge endpoint {v} is not a pure int")
            if u == v:
                raise ValueError("Loop detected")
        for i in range(n):
            if len(adj[i]) < 3:
                raise ValueError(f"Vertex {i} has degree {len(adj[i])} < 3")
        return [[int(e[0]), int(e[1])] for e in best_edges]
    
    # Fallback: two K_{3,3} components (value 2)
    edges = []
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([int(u), int(v)])
    
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([int(u), int(v)])
    
    # Verification pass
    if time.time() < deadline:
        adj = [set() for _ in range(n)]
        for e in edges:
            u, v = e[0], e[1]
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) < 3:
                raise ValueError(f"Vertex {i} has degree {len(adj[i])} < 3")
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        raise ValueError(f"Triangle found: {i}-{j}-{k}")
        if len(edges) != 18:
            raise ValueError(f"Expected 18 edges, got {len(edges)}")
    
    return [[int(e[0]), int(e[1])] for e in edges]

# EVOLVE-BLOCK-END
