import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    total_edges = len(edge_set)
    
    def make_edge(u, v):
        return (min(u, v), max(u, v))
    
    def expand_biclique(L, R, remaining):
        """Expand biclique (L, R) to fixpoint in remaining edges."""
        changed = True
        while changed:
            changed = False
            # Expand L: find vertices connected to all in R
            new_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for r in R:
                    if make_edge(w, r) not in remaining:
                        valid = False
                        break
                if valid:
                    new_L.add(w)
            if new_L:
                L |= new_L
                changed = True
            
            # Expand R: find vertices connected to all in L
            new_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for l in L:
                    if make_edge(w, l) not in remaining:
                        valid = False
                        break
                if valid:
                    new_R.add(w)
            if new_R:
                R |= new_R
                changed = True
        
        return L, R
    
    def find_max_biclique(remaining):
        """Find the largest biclique in the subgraph defined by remaining edges."""
        if not remaining:
            return None
        
        best_biclique = None
        best_size = 0
        
        # Try each edge as a seed
        for (u, v) in remaining:
            L, R = expand_biclique({u}, {v}, remaining)
            
            # Verify it's a valid biclique in remaining_edges
            valid = True
            for l in L:
                for r in R:
                    if l == r:
                        continue
                    if make_edge(l, r) not in remaining:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(L) * len(R) > best_size:
                best_size = len(L) * len(R)
                best_biclique = (L, R)
        
        return best_biclique
    
    def greedy_partition(remaining):
        """Greedy: repeatedly find largest biclique, remove its edges."""
        bicliques = []
        remaining = set(remaining)
        
        while remaining:
            if time.time() > deadline:
                break
            
            bc = find_max_biclique(remaining)
            
            if bc is None:
                # Fallback: single edge bicliques
                for (u, v) in remaining:
                    bicliques.append((frozenset([u]), frozenset([v])))
                remaining = set()
                break
            
            L, R = bc
            bicliques.append((frozenset(L), frozenset(R)))
            
            # Remove all edges between L and R from remaining
            to_remove = set()
            for l in L:
                for r in R:
                    if l != r:
                        e = make_edge(l, r)
                        if e in remaining:
                            to_remove.add(e)
            remaining -= to_remove
        
        return bicliques, remaining
    
    def try_merge(bicliques):
        """Try to merge pairs of bicliques that form a valid larger biclique."""
        improved = True
        current = list(bicliques)
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    L1, R1 = current[i]
                    L2, R2 = current[j]
                    
                    # Try merging: new L = L1 ∪ L2, new R = R1 ∪ R2
                    newL = L1 | L2
                    newR = R1 | R2
                    
                    # Check: no vertex in both newL and newR
                    if newL & newR:
                        continue
                    
                    # Check all cross edges exist in graph
                    valid = True
                    for l in newL:
                        for r in newR:
                            if l == r:
                                continue
                            if make_edge(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Merge is valid
                        current[i] = (newL, newR)
                        current.pop(j)
                        improved = True
                        break
                if improved:
                    break
        
        return current
    
    def local_search(bicliques, max_iterations=50):
        """Local search: try swapping vertices between bicliques to improve."""
        current = list(bicliques)
        best_count = len(current)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try moving a vertex from one biclique to another
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                L_i, R_i = current[i]
                for j in range(len(current)):
                    if i == j:
                        continue
                    L_j, R_j = current[j]
                    
                    # Try moving a vertex from L_i to L_j
                    for v in list(L_i):
                        new_L_i = L_i - {v}
                        new_L_j = L_j | {v}
                        
                        # Check if new bicliques are valid
                        valid_i = True
                        for l in new_L_i:
                            for r in R_i:
                                if l == r:
                                    continue
                                if make_edge(l, r) not in edge_set:
                                    valid_i = False
                                    break
                            if not valid_i:
                                break
                        
                        valid_j = True
                        for l in new_L_j:
                            for r in R_j:
                                if l == r:
                                    continue
                                if make_edge(l, r) not in edge_set:
                                    valid_j = False
                                    break
                            if not valid_j:
                                break
                        
                        if valid_i and valid_j and new_L_i and new_L_j:
                            # Check if this reduces total bicliques (merge opportunity)
                            # For now, just accept if valid
                            current[i] = (new_L_i, R_i)
                            current[j] = (new_L_j, R_j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    # Run greedy
    bicliques, leftover = greedy_partition(edge_set)
    
    # If greedy didn't cover all, add leftover as single edges
    if leftover:
        for (u, v) in leftover:
            bicliques.append((frozenset([u]), frozenset([v])))
    
    # Try to improve: merge bicliques
    bicliques = try_merge(bicliques)
    
    # Try local search
    bicliques = local_search(bicliques, max_iterations=30)
    
    # Convert to required format
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    # Validate
    for bc in result:
        if not isinstance(bc, list) or len(bc) != 2:
            break
        if not isinstance(bc[0], list) or not isinstance(bc[1], list):
            break
        if len(bc[0]) == 0 or len(bc[1]) == 0:
            break
        if set(bc[0]) & set(bc[1]):
            break
    
    return result
