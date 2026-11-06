import itertools
import time
import random

def construct():
    n = 18
    # Paley graph of order 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Build adjacency for Paley(17)
    adj = [[False] * 17 for _ in range(17)]
    edges_paley = []
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                adj[u][v] = True
                adj[v][u] = True
                edges_paley.append([u, v])
                
    # Precompute triangles in Paley(17)
    triangles = []
    for u in range(17):
        for v in range(u + 1, 17):
            if adj[u][v]:
                for w in range(v + 1, 17):
                    if adj[u][w] and adj[v][w]:
                        triangles.append((u, v, w))
                        
    # Precompute all independent sets in Paley(17)
    # A set is independent if no two vertices are connected
    independent_sets = []
    # Max independent set size is 5.
    # We can enumerate all subsets of size <= 5 and check independence.
    # Or just generate all subsets and filter.
    # 2^17 = 131072 subsets. This is fast.
    for mask in range(1 << 17):
        verts = [i for i in range(17) if (mask >> i) & 1]
        is_indep = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if adj[verts[i]][verts[j]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            independent_sets.append(set(verts))
            
    # Sort independent sets by size descending
    independent_sets.sort(key=len, reverse=True)
    
    # We need to find a set A (neighbors of 17) such that:
    # 1. A is triangle-free in Paley(17) (so G is K4-free)
    # 2. There is no TF set S' in Paley(17) of size > 7 such that S' intersect A is independent.
    #    If such S' exists, MaxTF with 17 is > 8.
    #    We want MaxTF with 17 <= 8.
    #    So we require that for all TF sets S' of size >= 8, S' intersect A is NOT independent.
    
    # Let's find all TF sets of size 8 in Paley(17)
    tf_sets_8 = []
    for combo in itertools.combinations(range(17), 8):
        # Check if triangle-free
        v_set = set(combo)
        is_tf = True
        for t in triangles:
            if t[0] in v_set and t[1] in v_set and t[2] in v_set:
                is_tf = False
                break
        if is_tf:
            tf_sets_8.append(set(combo))
            
    # Also check size 9 just in case?
    # If MaxTF(Paley) is 8, there are no TF sets of size 9.
    # The current score 1.333 implies MaxTF(Paley)=8.
    
    # We want to pick A such that for every T in tf_sets_8, T intersect A is not independent.
    # A must be triangle-free.
    
    # Heuristic: Pick A randomly from triangle-free sets.
    # Triangle-free sets in Paley(17) are abundant.
    
    # Let's generate a pool of candidate A's.
    # A must be triangle-free.
    # We can just pick random subsets and check if they are triangle-free.
    
    random.seed(42)
    start_time = time.time()
    deadline = 140.0
    
    best_A = None
    found = False
    
    # To speed up checking "T intersect A is not independent":
    # Precompute for each T in tf_sets_8, the list of independent subsets of T?
    # No, just check if T intersect A is in the list of independent sets.
    # We have independent_sets list. We can make a set of frozensets for O(1) lookup.
    indep_frozensets = {frozenset(s) for s in independent_sets}
    
    # Candidate A generation:
    # Try small A first? Or random?
    # If A is empty, T intersect A is empty (independent). So A cannot be empty.
    # If A has 1 vertex, T intersect A has 1 vertex (independent). So A must have size >= 2?
    # Actually, if A has size 1, say {v}, and T contains v, then T intersect A = {v} which is independent.
    # So we need T intersect A to have an edge.
    # This implies T intersect A must have size >= 2 and contain an edge.
    
    # Let's try random triangle-free sets A.
    
    for _ in range(20000):
        if time.time() > deadline:
            break
            
        # Generate random triangle-free set A
        # Pick random vertices, ensure no triangle.
        # Simpler: Pick random subset, check if triangle-free.
        # Or build A by adding vertices one by one if no triangle created.
        
        A = set()
        # Try to build A of size ~4 or 5
        candidates = list(range(17))
        random.shuffle(candidates)
        
        for v in candidates:
            # Check if adding v creates a triangle in A
            creates_tri = False
            for u in A:
                if adj[u][v]:
                    for w in A:
                        if w != u and adj[u][w] and adj[v][w]:
                            creates_tri = True
                            break
                if creates_tri:
                    break
            if not creates_tri:
                A.add(v)
                
        if len(A) < 2:
            continue
            
        # Check if A hits all tf_sets_8 with an edge
        valid = True
        for T in tf_sets_8:
            inter = T.intersection(A)
            # Check if inter is independent
            if frozenset(inter) in indep_frozensets:
                valid = False
                break
                
        if valid:
            best_A = A
            found = True
            break
            
    if found and best_A is not None:
        edges = edges_paley[:]
        for v in best_A:
            edges.append([17, v])
        return edges
    else:
        # Fallback: Use the old strategy or just Paley + isolated
        # If we fail to find A, we return Paley + isolated (Score 1.33)
        edges = edges_paley[:]
        return edges
