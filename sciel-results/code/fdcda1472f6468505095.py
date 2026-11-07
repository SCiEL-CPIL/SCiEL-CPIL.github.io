"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

P = {'n': 15}

def construct():
    n = P['n']
    # Start from K_{7,8} (56 edges) and greedily try to add edges.
    # A 3-edge path s-u-v-t violates if deg(s) == deg(t).
    # We maintain a valid graph and only add an edge (u,v) if no violation is created.
    
    # Initial: K_{7,8} with partition A=0..6, B=7..14
    a_size = 7
    b_size = 8
    partition_a = list(range(0, a_size))
    partition_b = list(range(a_size, n))
    
    # Build adjacency set for quick lookup
    adj = [set() for _ in range(n)]
    edges = set()
    
    for u in partition_a:
        for v in partition_b:
            edges.add((min(u,v), max(u,v)))
            adj[u].add(v)
            adj[v].add(u)
    
    # Degrees
    deg = [len(adj[i]) for i in range(n)]
    
    # Candidate edges to try: all non-edges
    candidate_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges:
                candidate_edges.append((i, j))
    
    # Shuffle for deterministic but varied exploration
    random.seed(42)
    random.shuffle(candidate_edges)
    
    def is_valid_with_edge(u, v):
        """Check if adding edge (u,v) keeps the graph valid."""
        # New degrees
        new_deg = deg[:]
        new_deg[u] += 1
        new_deg[v] += 1
        
        # New adjacencies
        # For every 3-edge path s-x-y-t where (x,y) is an edge in the NEW graph,
        # we need new_deg[s] != new_deg[t] for s != t.
        
        # The new edge is (u,v). Paths that become NEW or CHANGED:
        # 1. Paths where the middle edge is (u,v): s-u-v-t or s-v-u-t
        #    For s-u-v-t: s is neighbor of u (in old graph, since u-v is new), t is neighbor of v
        #    Actually, s must be adjacent to u in the new graph, t adjacent to v in new graph.
        #    Since u-v is new, s can be any neighbor of u (old) and t any neighbor of v (old),
        #    plus s can be v? No, s-u-v-t means s-u, u-v, v-t are edges. s != t.
        #    s is neighbor of u, t is neighbor of v. s != t.
        #    Also s could be v? No, s-u-v-t: s adjacent to u, v adjacent to t. If s=v, then v-u-v-t which is not a simple path (repeats v).
        #    So s in adj[u] (old), t in adj[v] (old), s != t.
        
        # Check paths with middle edge (u,v):
        for s in adj[u]:
            for t in adj[v]:
                if s != t:
                    if new_deg[s] == new_deg[t]:
                        return False
        
        # 2. Paths where the middle edge is an old edge (x,y), but the new edge (u,v) creates
        #    new 3-edge paths? No, a 3-edge path is s-x-y-t. The edges are s-x, x-y, y-t.
        #    Adding (u,v) doesn't change whether s-x, x-y, y-t exist unless one of them is (u,v).
        #    But wait: adding (u,v) could create NEW paths where one of the outer edges is (u,v).
        #    E.g., path u-x-y-t where u-x is the new edge? No, u-x must be an edge. If x != v, u-x is not new.
        #    Actually, the new edge (u,v) can serve as:
        #    - The middle edge: covered above (s-u-v-t)
        #    - An outer edge: e.g., u-x-y-t where u-x is the new edge? Only if x=v, so u-v-y-t.
        #      This is a path where the first edge is (u,v), middle is (v,y), last is (y,t).
        #      So: u-v-y-t. u is endpoint, t is endpoint. Need new_deg[u] != new_deg[t].
        #      u is neighbor of v (now), y is neighbor of v (old), t is neighbor of y (old).
        #      Wait, the path is u-v-y-t. Edges: (u,v), (v,y), (y,t).
        #      So for each y in adj[v] (old), for each t in adj[y] (old), t != u:
        #      Check new_deg[u] != new_deg[t].
        
        for y in adj[v]:
            for t in adj[y]:
                if t != u:
                    if new_deg[u] == new_deg[t]:
                        return False
        
        # Similarly, v-u-y-t: v is endpoint, t is endpoint.
        for y in adj[u]:
            for t in adj[y]:
                if t != v:
                    if new_deg[v] == new_deg[t]:
                        return False
        
        return True
    
    added = 0
    for (u, v) in candidate_edges:
        if is_valid_with_edge(u, v):
            edges.add((min(u,v), max(u,v)))
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            added += 1
    
    # Convert to list of [u,v]
    result = [[u, v] for (u, v) in edges]
    return result
# EVOLVE-BLOCK-END
