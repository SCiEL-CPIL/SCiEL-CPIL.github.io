import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Build complement adjacency (for Bron-Kerbosch on complement)
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and (min(u,v), max(u,v)) not in edge_set:
                comp_adj[u].add(v)
                comp_adj[v].add(u)

    # Enumerate maximal bicliques using Bron-Kerbosch on complement graph
    # A biclique (L, R) in G is a set of vertices where all L-R pairs are edges in G
    # This means no L-R pairs are edges in complement
    # So L ∪ R must be a clique in complement (no complement edges between L and R)
    # Actually, we need: for all l in L, r in R: (l,r) is edge in G
    # Equivalently: for all l in L, r in R: (l,r) is NOT edge in complement
    # So within L∪R, the only complement edges allowed are within L or within R
    # This is equivalent to finding a partition of a clique in complement into two parts
    # where no cross edges exist in complement
    
    # Better approach: enumerate all maximal bicliques directly
    # A biclique is maximal if we can't add any vertex to L or R
    
    def find_maximal_bicliques():
        """Find all maximal bicliques using a modified Bron-Kerbosch approach."""
        bicliques = []
        seen = set()
        
        # For each pair of vertices (u,v) that form an edge, try to grow a biclique
        # containing u in L and v in R
        edge_list = list(edge_set)
        
        for idx, (u, v) in enumerate(edge_list):
            if time.time() > deadline:
                break
            
            # Start with L={u}, R={v}
            # Try all ways to assign other vertices to L, R, or neither
            # But this is exponential. Instead, use a greedy expansion approach.
            
            # For each remaining vertex w, determine if it can go to L, R, or neither
            # w can go to L if w connects to all in R (and w != any in R)
            # w can go to R if w connects to all in L (and w != any in L)
            
            # Use backtracking with pruning
            def expand(L, R, remaining_vertices):
                if time.time() > deadline:
                    return
                
                # Check if this biclique is maximal
                is_maximal = True
                for w in remaining_vertices:
                    can_L = all((min(w, r), max(w, r)) in edge_set for r in R) and w not in R
                    can_R = all((min(w, l), max(w, l)) in edge_set for l in L) and w not in L
                    if can_L or can_R:
                        is_maximal = False
                        break
                
                if is_maximal:
                    # Normalize: sort L and R, ensure canonical form
                    L_sorted = tuple(sorted(L))
                    R_sorted = tuple(sorted(R))
                    # Canonical form: ensure min(L) < min(R) or some other rule
                    # Actually, (L,R) and (R,L) are different bicliques
                    key = (L_sorted, R_sorted)
                    if key not in seen:
                        seen.add(key)
                        bicliques.append((set(L), set(R)))
                
                # Try adding each remaining vertex to L or R
                for w in list(remaining_vertices):
                    if time.time() > deadline:
                        return
                    
                    new_remaining = remaining_vertices - {w}
                    
                    # Try adding w to L
                    can_L = w not in R and all((min(w, r), max(w, r)) in edge_set for r in R)
                    if can_L:
                        expand(L | {w}, R, new_remaining)
                    
                    # Try adding w to R
                    can_R = w not in L and all((min(w, l), max(w, l)) in edge_set for l in L)
                    if can_R:
                        expand(L, R | {w}, new_remaining)
            
            # Start with L={u}, R={v}, and all other vertices as candidates
            initial_remaining = set(range(n)) - {u, v}
            expand({u}, {v}, initial_remaining)
        
        return bicliques

    # Greedy partition (fallback)
    def greedy_partition():
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_biclique = None
            best_size = 0
            
            for u in range(n):
                if time.time() > deadline:
                    break
                u_remaining_neighbors = set()
                for v in adj[u]:
                    if (min(u,v), max(u,v)) in remaining:
                        u_remaining_neighbors.add(v)
                
                if not u_remaining_neighbors:
                    continue
                
                L = {u}
                R = set(u_remaining_neighbors)
                
                changed = True
                while changed:
                    changed = False
                    new_L = set()
                    for v in range(n):
                        if v not in L and v not in R:
                            if all((min(v,r), max(v,r)) in remaining for r in R):
                                new_L.add(v)
                    if new_L:
                        L = L | new_L
                        changed = True
                    
                    new_R = set()
                    for v in range(n):
                        if v not in L and v not in R:
                            if all((min(v,l), max(v,l)) in remaining for l in L):
                                new_R.add(v)
                    if new_R:
                        R = R | new_R
                        changed = True
                
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
            
            if best_biclique is None:
                e = next(iter(remaining))
                best_biclique = ({e[0]}, {e[1]})
            
            L, R = best_biclique
            valid_L = set()
            for l in L:
                if all((min(l,r), max(l,r)) in remaining for r in R):
                    valid_L.add(l)
            
            if not valid_L:
                e = next(iter(remaining))
                valid_L = {e[0]}
                R = {e[1]}
            
            for l in valid_L:
                for r in R:
                    remaining.discard((min(l,r), max(l,r)))
            
            bicliques.append((list(valid_L), list(R)))
        
        return bicliques

    # Try exact approach first
    exact_result = None
    try:
        # First try greedy to get a baseline
        greedy_bicliques = greedy_partition()
        greedy_count = len(greedy_bicliques)
        
        # Now try to find a better solution using local search
        # Start with greedy solution and try to merge/replace bicliques
        
        # For now, just use greedy as the answer if exact takes too long
        # But let's try a limited exact search
        
        # Actually, let's use a different strategy:
        # 1. Get greedy solution
        # 2. Try to reduce by finding larger bicliques that cover multiple greedy bicliques
        
        best_solution = greedy_bicliques
        best_count = greedy_count
        
        # Local search: try replacing subsets of bicliques with fewer bicliques
        # This is complex, so let's just return the greedy solution for now
        # but with a slight improvement: try multiple random orderings
        
        import random
        random.seed(42)
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            remaining = set(edge_set)
            bicliques = []
            
            while remaining:
                if time.time() > deadline - 2:
                    break
                
                best_biclique = None
                best_size = 0
                
                # Randomize vertex order
                vertices = list(range(n))
                random.shuffle(vertices)
                
                for u in vertices:
                    if time.time() > deadline - 2:
                        break
                    u_remaining_neighbors = set()
                    for v in adj[u]:
                        if (min(u,v), max(u,v)) in remaining:
                            u_remaining_neighbors.add(v)
                    
                    if not u_remaining_neighbors:
                        continue
                    
                    L = {u}
                    R = set(u_remaining_neighbors)
                    
                    changed = True
                    while changed:
                        changed = False
                        new_L = set()
                        for v in range(n):
                            if v not in L and v not in R:
                                if all((min(v,r), max(v,r)) in remaining for r in R):
                                    new_L.add(v)
                        if new_L:
                            L = L | new_L
                            changed = True
                        
                        new_R = set()
                        for v in range(n):
                            if v not in L and v not in R:
                                if all((min(v,l), max(v,l)) in remaining for l in L):
                                    new_R.add(v)
                        if new_R:
                            R = R | new_R
                            changed = True
                    
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_biclique = (L, R)
                
                if best_biclique is None:
                    e = next(iter(remaining))
                    best_biclique = ({e[0]}, {e[1]})
                
                L, R = best_biclique
                valid_L = set()
                for l in L:
                    if all((min(l,r), max(l,r)) in remaining for r in R):
                        valid_L.add(l)
                
                if not valid_L:
                    e = next(iter(remaining))
                    valid_L = {e[0]}
                    R = {e[1]}
                
                for l in valid_L:
                    for r in R:
                        remaining.discard((min(l,r), max(l,r)))
                
                bicliques.append((list(valid_L), list(R)))
            
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_solution = bicliques
            
            if best_count <= 3:
                break
        
        exact_result = best_solution
    
    except Exception:
        exact_result = greedy_partition()

    # Format result
    result = []
    for L, R in exact_result:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        result.append([L, R])
    
    return result
