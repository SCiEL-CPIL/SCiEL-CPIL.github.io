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
    
    def find_max_biclique(remaining, adj, n, strategy='expand'):
        """Find a large biclique in the subgraph defined by remaining edges."""
        if not remaining:
            return None
        
        best_biclique = None
        best_size = 0
        
        if strategy == 'expand':
            # Try each edge as a seed, expand to fixpoint
            for (u, v) in remaining:
                L = {u}
                R = {v}
                
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
                
                # Verify
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
        
        elif strategy == 'common_neighbors':
            # For each vertex, try it as a singleton left side
            for u in range(n):
                # Find all vertices connected to u in remaining
                neighbors = set()
                for v in range(n):
                    if v != u and make_edge(u, v) in remaining:
                        neighbors.add(v)
                if not neighbors:
                    continue
                
                # Try subsets of neighbors as right side
                # Start with all neighbors, then remove those that break biclique property
                R = set(neighbors)
                L = {u}
                
                # Now expand: add vertices that connect to all in R
                changed = True
                while changed:
                    changed = False
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
        
        elif strategy == 'degree_heuristic':
            # Pick high-degree vertices first
            deg = {v: len([e for e in remaining if v in e]) for v in range(n)}
            candidates = sorted(range(n), key=lambda v: -deg[v])
            
            for u in candidates[:20]:  # Limit to top 20
                neighbors = set()
                for v in range(n):
                    if v != u and make_edge(u, v) in remaining:
                        neighbors.add(v)
                if not neighbors:
                    continue
                
                L = {u}
                R = set(neighbors)
                
                changed = True
                while changed:
                    changed = False
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
    
    def greedy_partition(remaining, adj, n, strategies):
        """Greedy: repeatedly find largest biclique using given strategies, remove its edges."""
        bicliques = []
        remaining = set(remaining)
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_bc = None
            best_size = 0
            
            for strategy in strategies:
                bc = find_max_biclique(remaining, adj, n, strategy)
                if bc is not None:
                    L, R = bc
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_bc = bc
            
            if best_bc is None:
                # Fallback: single edge bicliques
                for (u, v) in remaining:
                    bicliques.append((frozenset([u]), frozenset([v])))
                remaining = set()
                break
            
            L, R = best_bc
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
    
    def try_merge(bicliques, edge_set):
        """Try to merge pairs of bicliques. Returns merged list or None if no merge possible."""
        n_bics = len(bicliques)
        for i in range(n_bics):
            for j in range(i + 1, n_bics):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try all 4 ways to merge:
                # 1. L1∪L2, R1∪R2
                # 2. L1∪R2, R1∪L2  (swap sides)
                # 3. L1∪L2, R1  (absorb L2 into L1, keep R1) - but need R2 subset of R1
                # 4. L1, R1∪R2  (absorb R2 into R1, keep L1) - but need L2 subset of L1
                
                candidates = []
                
                # Option 1: Union both sides
                newL = L1 | L2
                newR = R1 | R2
                if not (newL & newR):
                    candidates.append((newL, newR))
                
                # Option 2: Cross merge
                newL = L1 | R2
                newR = R1 | L2
                if not (newL & newR):
                    candidates.append((newL, newR))
                
                # Option 3: L2 subset of L1, merge R
                if L2 <= L1:
                    newR = R1 | R2
                    if not (L1 & newR):
                        candidates.append((L1, newR))
                
                # Option 4: R2 subset of R1, merge L
                if R2 <= R1:
                    newL = L1 | L2
                    if not (newL & R1):
                        candidates.append((newL, R1))
                
                # Option 5: L1 subset of L2, merge R
                if L1 <= L2:
                    newR = R1 | R2
                    if not (L2 & newR):
                        candidates.append((L2, newR))
                
                # Option 6: R1 subset of R2, merge L
                if R1 <= R2:
                    newL = L1 | L2
                    if not (newL & R2):
                        candidates.append((newL, R2))
                
                for newL, newR in candidates:
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue
                    
                    # Check all cross edges exist in original graph
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
                        # This merge is valid
                        result = list(bicliques)
                        result.pop(j)
                        result.pop(i)
                        result.append((frozenset(newL), frozenset(newR)))
                        return result
        
        return None
    
    def local_search_improve(bicliques, edge_set, adj, n):
        """Try to improve by moving vertices between bicliques."""
        if time.time() > deadline:
            return bicliques
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = list(bicliques)
            
            # Try merging pairs
            merged = try_merge(current, edge_set)
            if merged is not None:
                current = merged
                improved = True
                continue
            
            # Try splitting a biclique and recombining
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                L, R = current[i]
                
                # Try moving one vertex from L to R or vice versa
                for v in list(L):
                    newL = L - {v}
                    newR = R | {v}
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue
                    
                    # Check if this is still a valid biclique in the graph
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
                        # Check if the moved vertex's edges to old L are covered elsewhere
                        # Actually, we need to ensure the edge partition is still valid
                        # This is complex, so just do the merge-based improvement
                        pass
                
                for v in list(R):
                    newL = L | {v}
                    newR = R - {v}
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue
                    
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
                        pass
        
        return current
    
    # Run multiple greedy strategies
    best_result = None
    best_count = total_edges + 1
    
    strategies_list = [
        ['expand'],
        ['common_neighbors'],
        ['degree_heuristic'],
        ['expand', 'common_neighbors'],
        ['expand', 'degree_heuristic'],
        ['common_neighbors', 'degree_heuristic'],
        ['expand', 'common_neighbors', 'degree_heuristic'],
    ]
    
    for strategies in strategies_list:
        if time.time() > deadline:
            break
        
        bicliques, leftover = greedy_partition(edge_set, adj, n, strategies)
        
        if leftover:
            for (u, v) in leftover:
                bicliques.append((frozenset([u]), frozenset([v])))
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques
        
        # Try to improve via merging
        improved = True
        while improved and time.time() < deadline:
            improved = False
            merged = try_merge(best_result, edge_set)
            if merged is not None:
                best_result = merged
                best_count = len(merged)
                improved = True
    
    # Final conversion to required format
    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])
    
    # Validate the result
    covered_edges = set()
    for bc in result:
        if not isinstance(bc, list) or len(bc) != 2:
            break
        left, right = bc
        if not isinstance(left, list) or not isinstance(right, list):
            break
        if len(left) == 0 or len(right) == 0:
            break
        if set(left) & set(right):
            break
        for l in left:
            if not isinstance(l, int) or isinstance(l, bool):
                break
            for r in right:
                if not isinstance(r, int) or isinstance(r, bool):
                    break
                if l == r:
                    break
                e = make_edge(l, r)
                if e not in edge_set:
                    break
                if e in covered_edges:
                    break
                covered_edges.add(e)
    
    # Ensure all edges are covered exactly once
    if covered_edges != edge_set:
        # Fallback: use the seed approach
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([[u], neighbors])
    
    return result
