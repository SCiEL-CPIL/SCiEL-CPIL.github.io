import time
import random
import itertools

# Precompute Q3 edge sets on vertices 0..7
def generate_q3_edge_sets():
    # Q3 vertices are 0..7. Edges connect i, j if i xor j is power of 2.
    # But we need all isomorphic copies? No, just the structure on 0..7.
    # The set of edges of Q3 on 0..7 is fixed if we fix the labeling.
    # But we are checking if the induced subgraph CONTAINS a Q3.
    # So we need to check if the induced subgraph (on 8 vertices) has a subgraph isomorphic to Q3.
    # This is equivalent to checking if the edge set of the induced subgraph
    # contains any of the 840 edge sets corresponding to Q3 labelings.
    
    # Standard Q3 edges on 0..7 (binary 000..111)
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor == 1 or xor == 2 or xor == 4:
                base_edges.append((i, j))
                
    # Generate all permutations of 0..7
    # For each perm, map base_edges to permuted edges
    q3_sets = set()
    for perm in itertools.permutations(range(8)):
        # perm[i] is the vertex in the graph corresponding to Q3 vertex i
        # Wait, we want to map Q3 vertices to graph vertices.
        # Let's say Q3 vertex i is mapped to graph vertex perm[i].
        # Edge (i, j) in Q3 becomes (perm[i], perm[j]) in graph.
        # We want the set of edges in the graph (on 0..7) that form a Q3.
        e_set = frozenset(
            (min(perm[i], perm[j]), max(perm[i], perm[j])) 
            for (i, j) in base_edges
        )
        q3_sets.add(e_set)
        
    return list(q3_sets)

Q3_EDGE_SETS = generate_q3_edge_sets()

def has_q3(adj, n=16):
    # adj is list of ints (bitmasks)
    # Iterate through all 8-vertex subsets
    for vertices in itertools.combinations(range(n), 8):
        # Build induced edge set on these vertices
        # Map vertices to 0..7
        v_map = {v: i for i, v in enumerate(vertices)}
        
        induced_edges = set()
        for i in range(8):
            u = vertices[i]
            # Check neighbors in vertices
            mask = adj[u]
            for j in range(i+1, 8):
                v = vertices[j]
                if mask & (1 << v):
                    induced_edges.add((i, j))
                    
        # Check if any Q3 edge set is a subset of induced_edges
        # induced_edges is a set of (i,j) with i<j
        # Q3_EDGE_SETS are frozensets of (i,j) with i<j
        for q3_set in Q3_EDGE_SETS:
            if q3_set <= induced_edges:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    # Start with K3,13
    edges = set()
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
    best_edges = set(edges)
    
    # Candidates: all edges not in best_edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidates = [e for e in all_possible if e not in best_edges]
    
    # Local search
    while time.time() < deadline:
        random.shuffle(candidates)
        improved = False
        for e in candidates:
            if time.time() > deadline:
                break
                
            u, v = e
            # Try adding e
            best_edges.add(e)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3(adj, n):
                # Remove e
                best_edges.remove(e)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Keep e
                candidates.remove(e)
                improved = True
                
        if not improved:
            # No more edges can be added in one pass
            # Try a different strategy or break?
            # Maybe restart with a different seed?
            # For now, break to save time, or try random removals?
            # Let's just break.
            break
            
    return [list(e) for e in best_edges]
