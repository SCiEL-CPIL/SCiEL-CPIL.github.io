import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency as bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def popcount(x):
        return bin(x).count('1')
    
    def bits_to_list(mask):
        result = []
        m = mask
        while m:
            lsb = m & -m
            result.append(lsb.bit_length() - 1)
            m -= lsb
        return result
    
    def greedy_cover(seed, mode=0):
        random.seed(seed)
        uncovered = [0] * n
        for u, v in edges:
            uncovered[u] |= (1 << v)
            uncovered[v] |= (1 << u)
        
        bicliques = []
        remaining_edges = len(edges)
        
        while remaining_edges > 0:
            # Find vertex with highest uncovered degree
            best_u = -1
            best_deg = -1
            for u in range(n):
                deg = popcount(uncovered[u])
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
            
            if best_u == -1:
                break
            
            v_mask = uncovered[best_u]
            neighbors = bits_to_list(v_mask)
            if not neighbors:
                break
            
            v = random.choice(neighbors)
            
            L = {best_u}
            R = {v}
            
            # Expand with better scoring
            changed = True
            while changed:
                changed = False
                
                # Try to add to L: candidate must connect to ALL of R
                best_u_prime = -1
                best_score = -1
                for u_prime in range(n):
                    if u_prime in L:
                        continue
                    # Must be connected to all v in R
                    valid = True
                    for v_r in R:
                        if not (uncovered[u_prime] & (1 << v_r)):
                            valid = False
                            break
                    if not valid:
                        continue
                    # Score: number of uncovered edges this vertex has to vertices NOT in R
                    # These are edges that would be "newly covered" by adding this vertex
                    # Actually, all edges from u_prime to R are covered by this biclique
                    # The score should be: how many total uncovered edges does u_prime have?
                    # Higher degree = more edges covered
                    deg = popcount(uncovered[u_prime])
                    if deg > best_score:
                        best_score = deg
                        best_u_prime = u_prime
                
                if best_u_prime != -1:
                    L.add(best_u_prime)
                    changed = True
                    continue
                
                # Try to add to R: candidate must connect to ALL of L
                best_v_prime = -1
                best_score = -1
                for v_prime in range(n):
                    if v_prime in R:
                        continue
                    valid = True
                    for u_l in L:
                        if not (uncovered[u_l] & (1 << v_prime)):
                            valid = False
                            break
                    if not valid:
                        continue
                    deg = popcount(uncovered[v_prime])
                    if deg > best_score:
                        best_score = deg
                        best_v_prime = v_prime
                
                if best_v_prime != -1:
                    R.add(best_v_prime)
                    changed = True
                    continue
            
            bicliques.append([list(L), list(R)])
            
            # Remove covered edges
            for u in L:
                for v in R:
                    uncovered[u] &= ~(1 << v)
                    uncovered[v] &= ~(1 << u)
                    remaining_edges -= 1
                    
        return bicliques
    
    def try_merge(bicliques):
        """Try to merge compatible bicliques."""
        if len(bicliques) <= 1:
            return bicliques
        
        # Build edge set for quick lookup
        edge_set = set()
        for u, v in edges:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
        
        # For each pair, check if they can be merged
        # Two bicliques can be merged if their union forms a valid biclique
        # i.e., all cross pairs in the merged set must be edges
        merged = True
        result = list(bicliques)
        
        while merged:
            merged = False
            i = 0
            while i < len(result):
                j = i + 1
                while j < len(result):
                    L1, R1 = result[i]
                    L2, R2 = result[j]
                    
                    # Try merging: new_L = L1 ∪ L2, new_R = R1 ∪ R2
                    # But this might not be a valid biclique
                    # Alternative: new_L = L1, new_R = R1 ∪ R2 (if valid)
                    # Or new_L = L1 ∪ L2, new_R = R2 (if valid)
                    
                    # Check if L1 × (R1 ∪ R2) is a biclique
                    # i.e., every u in L1 is connected to every v in R1 ∪ R2
                    valid1 = True
                    for u in L1:
                        for v in R2:
                            if u < v:
                                if (u, v) not in edge_set:
                                    valid1 = False
                                    break
                            else:
                                if (v, u) not in edge_set:
                                    valid1 = False
                                    break
                        if not valid1:
                            break
                    
                    if valid1:
                        # Merge: keep L1, extend R
                        result[i] = [L1, list(set(R1) | set(R2))]
                        result.pop(j)
                        merged = True
                        break
                    
                    # Check if (L1 ∪ L2) × R2 is a biclique
                    valid2 = True
                    for u in L2:
                        for v in R2:
                            if u < v:
                                if (u, v) not in edge_set:
                                    valid2 = False
                                    break
                            else:
                                if (v, u) not in edge_set:
                                    valid2 = False
                                    break
                        if not valid2:
                            break
                    # Also need L1 × R2 to be valid (already checked above)
                    if valid2 and valid1:
                        # Check L1 × R2 and L2 × R2
                        # We already know L1 × R2 is valid from valid1
                        # And L2 × R2 is valid from valid2
                        # So (L1 ∪ L2) × R2 is valid
                        # But we also need (L1 ∪ L2) × R1 to be valid
                        valid3 = True
                        for u in L2:
                            for v in R1:
                                if u < v:
                                    if (u, v) not in edge_set:
                                        valid3 = False
                                        break
                                else:
                                    if (v, u) not in edge_set:
                                        valid3 = False
                                        break
                            if not valid3:
                                break
                        if valid3:
                            # Full merge: (L1 ∪ L2) × (R1 ∪ R2)
                            result[i] = [list(set(L1) | set(L2)), list(set(R1) | set(R2))]
                            result.pop(j)
                            merged = True
                            break
                    
                    j += 1
                if merged:
                    break
                i += 1
        
        return result
    
    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 50
    
    seed = 0
    while time.time() < deadline:
        bicliques = greedy_cover(seed)
        bicliques = try_merge(bicliques)
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            if best_count <= 4:
                break
        seed += 1
    
    if best_bicliques is None:
        return [[[u],[v for a,v in edges if a==u]] for u in range(n) if any(a==u for a,v in edges)]
    
    result = []
    for l, r in best_bicliques:
        result.append([list(map(int, l)), list(map(int, r))])
        
    return result
