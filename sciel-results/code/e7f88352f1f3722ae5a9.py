import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add(_edge_key(u, v))
    
    all_mask = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_vertices(mask):
        return [i for i in range(n) if mask & (1 << i)]
    
    def find_maximal_bicliques_from_vertex(u, remaining_mask):
        """Find all maximal bicliques containing vertex u in the remaining graph."""
        bicliques = []
        # For vertex u, try all subsets of its neighbors as the other side
        # This is exponential, so we use a greedy approach instead
        # Find the maximal biclique with u on the left
        neighbors = remaining_mask & adj_mask[u]
        if neighbors == 0:
            return bicliques
        
        # Greedy expansion: start with u, keep adding vertices
        L = 1 << u
        R = 0
        # Find common neighbors of all in R (initially just u's neighbors)
        common = neighbors
        
        # Build maximal biclique by iteratively adding vertices
        changed = True
        while changed:
            changed = False
            # Try adding to R: vertex v can join R if connected to all in L
            cand_R = common
            # Remove vertices already in L
            cand_R &= ~L
            if cand_R:
                # Pick the vertex with highest degree in common
                best_v = -1
                best_deg = -1
                temp = cand_R
                while temp:
                    v = temp & (-temp)
                    temp ^= v
                    deg = popcount(common & adj_mask[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                R |= (1 << best_v)
                # Update common: intersection of neighbors of all in R
                common = all_mask
                temp = R
                while temp:
                    v = temp & (-temp)
                    temp ^= v
                    common &= adj_mask[v]
                common &= remaining_mask
                changed = True
            
            # Try adding to L: vertex w can join L if connected to all in R
            if R:
                cand_L = common
                cand_L &= ~R
                if cand_L:
                    best_w = -1
                    best_deg = -1
                    temp = cand_L
                    while temp:
                        w = temp & (-temp)
                        temp ^= w
                        deg = popcount(common & adj_mask[w])
                        if deg > best_deg:
                            best_deg = deg
                            best_w = w
                    L |= (1 << best_w)
                    # Update common
                    common = all_mask
                    temp = R
                    while temp:
                        v = temp & (-temp)
                        temp ^= v
                        common &= adj_mask[v]
                    temp = L
                    while temp:
                        w = temp & (-temp)
                        temp ^= w
                        common &= adj_mask[w]
                    common &= remaining_mask
                    changed = True
        
        if L and R:
            bicliques.append((L, R))
        
        return bicliques
    
    def greedy_biclique_cover(seed):
        random.seed(seed)
        remaining = edge_set
        rem_adj_mask = [0] * n
        for u, v in remaining:
            rem_adj_mask[u] |= (1 << v)
            rem_adj_mask[v] |= (1 << u)
        
        bicliques = []
        
        while remaining:
            if time.time() > deadline - 5.0:
                break
            
            # Find the best biclique to extract
            best_L = 0
            best_R = 0
            best_score = -1
            
            # Sample edges to try (or all if few)
            edge_list = list(remaining)
            random.shuffle(edge_list)
            
            # Limit candidates for speed
            max_candidates = min(len(edge_list), 50)
            
            for u, v in edge_list[:max_candidates]:
                # Expand from this edge
                L = 1 << u
                R = 1 << v
                
                # Iterative expansion
                changed = True
                while changed:
                    changed = False
                    
                    # Find candidates for R: connected to all in L
                    if L:
                        common_R = all_mask
                        temp = L
                        while temp:
                            w = temp & (-temp)
                            temp ^= w
                            common_R &= rem_adj_mask[w]
                        cand_R = common_R & ~L & ~R
                        if cand_R:
                            # Add all candidates at once for maximal
                            R |= cand_R
                            changed = True
                    
                    # Find candidates for L: connected to all in R
                    if R:
                        common_L = all_mask
                        temp = R
                        while temp:
                            w = temp & (-temp)
                            temp ^= w
                            common_L &= rem_adj_mask[w]
                        cand_L = common_L & ~L & ~R
                        if cand_L:
                            L |= cand_L
                            changed = True
                
                score = popcount(L) * popcount(R)
                if score > best_score:
                    best_score = score
                    best_L = L
                    best_R = R
            
            if best_L == 0 or best_R == 0:
                # Fallback: take first edge
                u, v = next(iter(remaining))
                bicliques.append(([u], [v]))
                remaining.remove(_edge_key(u, v))
                rem_adj_mask[u] &= ~(1 << v)
                rem_adj_mask[v] &= ~(1 << u)
            else:
                L_list = get_vertices(best_L)
                R_list = get_vertices(best_R)
                bicliques.append((L_list, R_list))
                
                # Remove all edges between L and R from remaining
                for u in L_list:
                    for v in R_list:
                        e = _edge_key(u, v)
                        if e in remaining:
                            remaining.remove(e)
                            rem_adj_mask[u] &= ~(1 << v)
                            rem_adj_mask[v] &= ~(1 << u)
        
        return bicliques
    
    def local_search(bicliques, max_iters=50):
        """Try to improve by merging or reassigning edges."""
        if time.time() > deadline - 3.0:
            return bicliques
        
        bicliques = [(list(L), list(R)) for L, R in bicliques]
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline - 3.0:
            improved = False
            iter_count += 1
            
            # Try merging pairs
            for i in range(len(bicliques)):
                if time.time() > deadline - 3.0:
                    return bicliques
                for j in range(i + 1, len(bicliques)):
                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]
                    
                    # Check if we can merge: no edges between Li-Rj or Lj-Ri in original
                    can_merge = True
                    for u in Li:
                        for v in Rj:
                            if (1 << v) & adj_mask[u]:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        for u in Lj:
                            for v in Ri:
                                if (1 << v) & adj_mask[u]:
                                    can_merge = False
                                    break
                                if not can_merge:
                                    break
                    
                    if can_merge:
                        new_L = list(set(Li) | set(Lj))
                        new_R = list(set(Ri) | set(Rj))
                        bicliques[i] = (new_L, new_R)
                        bicliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
            
            # Try moving a vertex from one biclique's L to another's L
            if not improved and len(bicliques) >= 2:
                for i in range(len(bicliques)):
                    if time.time() > deadline - 3.0:
                        return bicliques
                    for j in range(len(bicliques)):
                        if i == j:
                            continue
                        Li, Ri = bicliques[i]
                        Lj, Rj = bicliques[j]
                        
                        for u in list(Li):
                            # Check if u can move to Lj
                            # u must be connected to all in Rj
                            can_move = True
                            for v in Rj:
                                if not ((1 << v) & adj_mask[u]):
                                    can_move = False
                                    break
                            if can_move:
                                # Also check that removing u from Li doesn't break biclique
                                # Li without u must still be complete bipartite with Ri
                                # This is always true since we're removing from L
                                new_Li = [x for x in Li if x != u]
                                if not new_Li:
                                    continue  # Can't have empty L
                                
                                # Check if all remaining Li vertices are connected to all Ri
                                valid = True
                                for x in new_Li:
                                    for v in Ri:
                                        if not ((1 << v) & adj_mask[x]):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                
                                if valid:
                                    bicliques[i] = (new_Li, list(Ri))
                                    bicliques[j] = (Lj + [u], list(Rj))
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        return bicliques
    
    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts with different seeds
    num_restarts = 200
    for s in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
        b = greedy_biclique_cover(s)
        b = local_search(b, max_iters=30)
        if len(b) < best_count:
            best_count = len(b)
            best_bicliques = b
    
    if best_bicliques is None:
        return [[[u], [v]] for u, v in edges]
    
    # Final validation
    covered = set()
    valid = True
    for L, R in best_bicliques:
        if not L or not R:
            valid = False
            break
        if set(L) & set(R):
            valid = False
            break
        for u in L:
            for v in R:
                k = _edge_key(u, v)
                if k not in edge_set:
                    valid = False
                    break
                if k in covered:
                    valid = False
                    break
                covered.add(k)
    
    if not valid or covered != edge_set:
        return [[[u], [v]] for u, v in edges]
    
    result = []
    for L, R in best_bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
    
    return result
