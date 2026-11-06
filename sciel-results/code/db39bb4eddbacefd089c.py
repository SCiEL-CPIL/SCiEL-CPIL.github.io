import time
import itertools

def construct():
    """
    Returns a list of 91 edges [u, v] on vertices 0..30 minimizing the number of C4s.
    Uses the polarity graph of order 5 as a seed (90 edges, 0 C4s) and adds
    the single edge that creates the fewest C4s.
    """
    q = 5
    # Generate vertices of PG(2, q)
    # Representation: (x, y, z) normalized such that first non-zero is 1
    vertices = []
    # Case 1: x = 1
    for i in range(q):
        for j in range(q):
            vertices.append((1, i, j))
    # Case 2: x = 0, y = 1
    for k in range(q):
        vertices.append((0, 1, k))
    # Case 3: x = 0, y = 0, z = 1
    vertices.append((0, 0, 1))
    
    n = len(vertices) # 31
    
    # Map vertex tuple to index
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build adjacency list for the polarity graph
    # u ~ v if dot(u, v) == 0 mod q
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                
    # We have 90 edges.
    # Find the non-edge that adds the fewest C4s.
    # A C4 using new edge (u, v) is formed by u-x-y-v where x in N(u), y in N(v), x~y, x!=y.
    # Count is number of edges in G between N(u)\{v} and N(v)\{u}.
    
    best_cost = float('inf')
    best_edge = None
    
    # Precompute neighbor lists as lists for faster iteration if needed, but sets are fine for small n
    # n=31, degree ~ 10.
    
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue # Already an edge
            
            # N(u) \ {v} and N(v) \ {u}
            # Since u, v are not adjacent, v not in N(u) and u not in N(v)
            A = adj[u]
            B = adj[v]
            
            cost = 0
            # Count edges between A and B
            # Iterate over smaller set? Sizes are similar (~10)
            for x in A:
                # Check neighbors of x in B
                # adj[x] is a set
                # We want y in B such that y in adj[x]
                # intersection = adj[x] & B
                # But we must exclude x itself?
                # If x in B, then x is in A and B.
                # The edge (x, x) is not in the graph.
                # So intersection of adj[x] and B counts y in B such that x~y.
                # If y=x, x~x is false, so it's not counted.
                # So len(adj[x] & B) is correct.
                cost += len(adj[x] & B)
                
            if cost < best_cost:
                best_cost = cost
                best_edge = [u, v]
                
    edges.append(best_edge)
    
    # Ensure exactly 91 edges
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    # Ensure distinct edges
    assert len(set(tuple(sorted(e)) for e in edges)) == 91, "Edges are not distinct"
    
    return edges
