import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 55
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    # Precompute: for each vertex, bitmask of its neighbors
    # A biclique (L, R) is valid iff for all l in L, r in R: edge exists
    # i.e., adj_mask[l] & R_mask == R_mask for all l in L (and symmetrically)
    
    def is_biclique(L_mask, R_mask):
        """Check if (L_mask, R_mask) forms a valid biclique in the host graph."""
        if L_mask == 0 or R_mask == 0:
            return False
        # For every vertex in L, all vertices in R must be neighbors
        lm = L_mask
        while lm:
            low = lm & (-lm)
            v = low.bit_length() - 1
            if (adj_mask[v] & R_mask) != R_mask:
                return False
            lm ^= low
        return True
    
    def get_cross_edges(L_mask, R_mask):
        """Return the set of edges between L and R."""
        result = set()
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            rm = R_mask
            while rm:
                low_r = rm & (-rm)
                v = low_r.bit_length() - 1
                result.add((min(u, v), max(u, v)))
                rm ^= low_r
            lm ^= low
        return result
    
    def max_biclique_containing(edge, remaining_mask):
        """Find a maximal biclique containing the given edge, restricted to remaining edges."""
        u, v = edge
        L_mask = 1 << u
        R_mask = 1 << v
        
        changed = True
        while changed:
            changed = False
            # Try to add vertex x to L
            for x in range(n):
                if x == u or x == v:
                    continue
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue
                # x can join L if all r in R are neighbors of x AND all (x,r) edges are in remaining
                new_L_mask = L_mask | (1 << x)
                # Check: for all r in R, (x,r) is an edge and in remaining
                ok = True
                rm = R_mask
                while rm:
                    low_r = rm & (-rm)
                    r = low_r.bit_length() - 1
                    if not (adj_mask[x] >> r) & 1:
                        ok = False
                        break
                    e = (min(x, r), max(x, r))
                    if e not in remaining_mask:
                        ok = False
                        break
                    rm ^= low_r
                if ok:
                    L_mask = new_L_mask
                    changed = True
            
            # Try to add vertex y to R
            for y in range(n):
                if y == u or y == v:
                    continue
                if (L_mask >> y) & 1 or (R_mask >> y) & 1:
                    continue
                ok = True
                lm = L_mask
                while lm:
                    low_l = lm & (-lm)
                    l = low_l.bit_length() - 1
                    if not (adj_mask[y] >> l) & 1:
                        ok = False
                        break
                    e = (min(l, y), max(l, y))
                    if e not in remaining_mask:
                        ok = False
                        break
                    lm ^= low_l
                if ok:
                    R_mask |= (1 << y)
                    changed = True
        
        return L_mask, R_mask
    
    # Greedy initialization with bitmask-based biclique finding
    def greedy_init():
        remaining = set(all_edges)
        bicliques = []
        while remaining:
            best_L = 0
            best_R = 0
            best_count = 0
            
            # Sample edges to find good bicliques
            sample = list(remaining)
            random.shuffle(sample)
            for e in sample[:15]:
                L_mask, R_mask = max_biclique_containing(e, remaining)
                cross = get_cross_edges(L_mask, R_mask)
                count = len(cross & remaining)
                if count > best_count:
                    best_count = count
                    best_L = L_mask
                    best_R = R_mask
            
            if best_L == 0:
                # Fallback: single edge
                e = next(iter(remaining))
                best_L = 1 << e[0]
                best_R = 1 << e[1]
            
            # Extract vertex lists
            L_list = [i for i in range(n) if (best_L >> i) & 1]
            R_list = [i for i in range(n) if (best_R >> i) & 1]
            bicliques.append([L_list, R_list])
            
            # Remove covered edges
            cross = get_cross_edges(best_L, best_R)
            remaining -= cross
        
        return bicliques
    
    # Try to merge two bicliques into one
    def try_merge_pair(bi, bj, bicliques):
        Li, Ri = bicliques[bi]
        Lj, Rj = bicliques[bj]
        
        # Compute edge sets
        ei = get_cross_edges(0, 0)  # placeholder
        # Actually compute from lists
        ei = set()
        for u in Li:
            for v in Ri:
                ei.add((min(u, v), max(u, v)))
        ej = set()
        for u in Lj:
            for v in Rj:
                ej.add((min(u, v), max(u, v)))
        
        union = ei | ej
        
        # Check if union forms a biclique
        # Find L, R masks
        vertices = set()
        for u, v in union:
            vertices.add(u)
            vertices.add(v)
        
        # Bipartite check on the subgraph
        color = {}
        sub_adj = {v: set() for v in vertices}
        for u, v in union:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
        
        for start in vertices:
            if start not in color:
                color[start] = 0
                stack = [start]
                while stack:
                    curr = stack.pop()
                    for nb in sub_adj[curr]:
                        if nb not in color:
                            color[nb] = 1 - color[curr]
                            stack.append(nb)
                        elif color[nb] == color[curr]:
                            return None
        
        L = [v for v in vertices if color[v] == 0]
        R = [v for v in vertices if color[v] == 1]
        
        # Verify completeness
        L_set = set(L)
        R_set = set(R)
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in union:
                    return None
        
        # Verify no within-side edges in union
        for u in L:
            for w in L:
                if u < w and (u, w) in union:
                    return None
        for v in R:
            for w in R:
                if v < w and (v, w) in union:
                    return None
        
        new_bicliques = [b for k, b in enumerate(bicliques) if k != bi and k != bj]
        new_bicliques.append([L, R])
        return new_bicliques
    
    def local_search(bicliques):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, nb):
                    result = try_merge_pair(i, j, bicliques)
                    if result is not None:
                        bicliques = result
                        improved = True
                        break
                if improved:
                    break
        return bicliques
    
    # Also try: split a biclique and reassign edges to other bicliques (2-opt style)
    # For now, keep it simple: greedy + merge
    
    best_bicliques = None
    best_count = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        bicliques = greedy_init()
        bicliques = local_search(bicliques)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    if best_bicliques is None:
        best_bicliques = [[[u], [v]] for u, v in all_edges]
    
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result

# EVOLVE-BLOCK-END
