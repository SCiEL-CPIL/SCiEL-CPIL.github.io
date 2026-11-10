"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 75  # safety margin before 90s limit
    
    n = 15
    # Start with K_{7,8}: partition into A (7 vertices) and B (8 vertices)
    A = list(range(7))       # vertices 0-6
    B = list(range(7, 15))   # vertices 7-14
    
    # Build edge set
    edges = set()
    for a in A:
        for b in B:
            edges.add(frozenset([a, b]))
    
    def get_degrees(edge_set):
        degs = [0] * n
        for e in edge_set:
            u, v = tuple(e)
            degs[u] += 1
            degs[v] += 1
        return degs
    
    def is_valid(edge_set):
        degs = get_degrees(edge_set)
        # For every 3-edge path s-u-v-t (s != t), check deg(s) != deg(t)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for e in edge_set:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge (u,v), for each s in adj[u] (s != v), for each t in adj[v] (t != u):
        # path s-u-v-t, check s != t and deg(s) != deg(t)
        for e in edge_set:
            u, v = tuple(e)
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if degs[s] == degs[t]:
                        return False
        return True
    
    # Current state
    current_edges = edges
    if not is_valid(current_edges):
        # Should not happen for K_{7,8} with 7 != 8
        pass
    
    # Try to add edges greedily
    # Candidate edges: within A and within B
    candidates = []
    for a1, a2 in itertools.combinations(A, 2):
        candidates.append(frozenset([a1, a2]))
    for b1, b2 in itertools.combinations(B, 2):
        candidates.append(frozenset([b1, b2]))
    
    # Sort candidates by some heuristic - try to add them
    # We'll do multiple passes
    for iteration in range(20):
        if time.time() > deadline:
            break
        added = False
        # Shuffle candidates for diversity
        import random
        random.seed(iteration)
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if cand in current_edges:
                continue
            new_edges = current_edges | {cand}
            if is_valid(new_edges):
                current_edges = new_edges
                added = True
                # Remove from candidates
                candidates.remove(cand)
    
    # Convert to list of [u, v] with u < v
    result = []
    for e in current_edges:
        u, v = tuple(e)
        if u > v:
            u, v = v, u
        result.append([int(u), int(v)])
    
    # Sort for determinism
    result.sort()
    
    return result
# EVOLVE-BLOCK-END
