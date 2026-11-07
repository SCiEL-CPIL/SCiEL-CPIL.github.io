"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Construct a 12-vertex graph minimizing distinct simple cycle lengths.
    
    Strategy: Two disjoint K_{3,3} components.
    - 12 vertices: {0,1,2,3,4,5} and {6,7,8,9,10,11}
    - Each vertex has degree exactly 3
    - Bipartite (girth 4, triangle-free)
    - Cycle lengths present: {4, 6}
    - Distinct cycle length count: 2
    """
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([int(u), int(v)])
    
    # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([int(u), int(v)])
    
    # Verification pass: check all invariants before returning
    if time.time() < deadline:
        n = 12
        # Build adjacency
        adj = [set() for _ in range(n)]
        for e in edges:
            u, v = e[0], e[1]
            if not isinstance(u, int) or isinstance(u, bool):
                raise ValueError(f"Edge endpoint {u} is not a pure int")
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"Edge endpoint {v} is not a pure int")
            if u == v:
                raise ValueError("Loop detected")
            if v in adj[u]:
                raise ValueError("Duplicate edge detected")
            adj[u].add(v)
            adj[v].add(u)
        
        # Check minimum degree >= 3
        for i in range(n):
            if len(adj[i]) < 3:
                raise ValueError(f"Vertex {i} has degree {len(adj[i])} < 3")
        
        # Check no triangles (girth >= 4)
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        raise ValueError(f"Triangle found: {i}-{j}-{k}")
        
        # Check graph contains at least one cycle (it does: K_{3,3} has cycles)
        # and verify edge count
        if len(edges) != 18:
            raise ValueError(f"Expected 18 edges, got {len(edges)}")
    
    # Ensure return type is list of lists of ints
    return [[int(e[0]), int(e[1])] for e in edges]

# EVOLVE-BLOCK-END
