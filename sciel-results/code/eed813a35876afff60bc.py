import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 150  # Seconds
    safety_margin = 5  # Seconds
    
    # Adjacency as list of sets for fast neighbor lookups
    adj = [set() for _ in range(n)]
    edges = []
    
    def would_create_c6(u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 containing (u,v) would be u-a-b-c-d-v-u (length 5 path from u to v not using edge u-v).
        Equivalently, there's a path of length 4 from u to v in the graph without edge (u,v).
        """
        # Find all vertices at distance 1 from u (neighbors of u)
        # and all vertices at distance 1 from v (neighbors of v)
        # A C6: u - a - b - c - d - v where a,b,c,d are distinct from u,v
        # This means there's a path of length 4 from u to v.
        # Path: u -> a -> b -> c -> d -> v
        # So a is neighbor of u, d is neighbor of v, and there's a path of length 2 from a to d
        # through b (a-b-c-d, so b and c are intermediate)
        
        # For each neighbor a of u (a != v), and each neighbor d of v (d != u):
        # check if there's a path a -> b -> c -> d where b,c not in {u,v} and b != c
        # This is: exists b in N(a) \ {u, v, d} such that exists c in N(b) ∩ N(d) \ {u, v}
        
        for a in adj[u]:
            if a == v:
                continue
            for d in adj[v]:
                if d == u:
                    continue
                # Check if there's a path a-b-c-d with b,c distinct from u,v
                # b must be in N(a) and not in {u, v, d}
                # c must be in N(b) ∩ N(d) and not in {u, v}
                for b in adj[a]:
                    if b == u or b == v or b == d:
                        continue
                    # Check if N(b) and N(d) share a vertex c not in {u, v}
                    # Use set intersection
                    common = adj[b] & adj[d]
                    if common:
                        # Check if any common vertex is not u or v
                        for c in common:
                            if c != u and c != v:
                                return True
        return False
    
    def is_c6_free():
        """Verify the entire graph is C6-free."""
        # Check all pairs of non-adjacent vertices
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    continue
                # Check if there's a path of length 4 from u to v
                # u-a-b-c-d-v
                for a in adj[u]:
                    for b in adj[a]:
                        if b == u or b == v:
                            continue
                        for c in adj[b]:
                            if c == u or c == v or c == b:
                                continue
                            if c in adj[v]:
                                # Found a C6
                                return False
        return True
    
    # Greedy: try to add edges in random order
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() - start_time > deadline - safety_margin:
            break
        if v in adj[u]:
            continue
        if not would_create_c6(u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Try to add more edges with different orderings
    attempts = 0
    while time.time() - start_time < deadline - safety_margin:
        attempts += 1
        random.seed(42 + attempts)
        random.shuffle(all_pairs)
        
        improved = False
        for u, v in all_pairs:
            if time.time() - start_time > deadline - safety_margin:
                break
            if v in adj[u]:
                continue
            if not would_create_c6(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                improved = True
        
        if not improved:
            break
    
    return edges
# EVOLVE-BLOCK-END
