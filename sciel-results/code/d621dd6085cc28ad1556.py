import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = 75.0  # Safety margin (90s limit, leave 15s buffer)
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # remaining_edges as set of (u, v) with u < v
    remaining_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        remaining_edges.add((u, v))
        
    def edge_in_remaining(u, v):
        if u > v:
            u, v = v, u
        return (u, v) in remaining_edges
    
    def expand_biclique(L, R, remaining):
        """Expand L and R to be maximal w.r.t. remaining edges."""
        changed = True
        while changed:
            changed = False
            # Try adding to L
            best_candidate = None
            best_score = -1
            for u_prime in range(n):
                if u_prime in L:
                    continue
                # Check connectivity to all of R
                valid = True
                for r in R:
                    if not edge_in_remaining(u_prime, r):
                        valid = False
                        break
                if valid:
                    # Score: how many additional R candidates this enables
                    score = 0
                    for v_prime in range(n):
                        if v_prime in R:
                            continue
                        # Check if v_prime connects to all of L + u_prime
                        ok = True
                        for l in L:
                            if not edge_in_remaining(v_prime, l):
                                ok = False
                                break
                        if ok and edge_in_remaining(v_prime, u_prime):
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = u_prime
            if best_candidate is not None:
                L.add(best_candidate)
                changed = True
            
            # Try adding to R
            best_candidate = None
            best_score = -1
            for v_prime in range(n):
                if v_prime in R:
                    continue
                valid = True
                for l in L:
                    if not edge_in_remaining(v_prime, l):
                        valid = False
                        break
                if valid:
                    # Score: how many additional L candidates this enables
                    score = 0
                    for u_prime in range(n):
                        if u_prime in L:
                            continue
                        ok = True
                        for r in R:
                            if not edge_in_remaining(u_prime, r):
                                ok = False
                                break
                        if ok and edge_in_remaining(u_prime, v_prime):
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = v_prime
            if best_candidate is not None:
                R.add(best_candidate)
                changed = True
        return L, R
    
    def count_biclique_edges(L, R, remaining):
        """Count how many edges in L x R are in remaining."""
        count = 0
        for l in L:
            for r in R:
                if l != r and edge_in_remaining(l, r):
                    count += 1
        return count
    
    def find_best_biclique(remaining):
        """Find the biclique that covers the most edges from remaining."""
        best_L = None
        best_R = None
        best_edges = 0
        
        # Sample a subset of seed edges for efficiency
        seed_edges = list(remaining)
        # If too many, sample
        if len(seed_edges) > 200:
            import random
            random.seed(42)
            seed_edges = random.sample(seed_edges, 200)
        
        for u, v in seed_edges:
            if time.time() > deadline:
                break
            L = {u}
            R = {v}
            L, R = expand_biclique(L, R, remaining)
            ec = count_biclique_edges(L, R, remaining)
            if ec > best_edges:
                best_edges = ec
                best_L = set(L)
                best_R = set(R)
        
        return best_L, best_R, best_edges
    
    bicliques = []
    
    # Phase 1: Greedy partition with best-biclique selection
    while remaining_edges and time.time() < deadline:
        L, R, ec = find_best_biclique(remaining_edges)
        
        if L is None or R is None or ec == 0:
            # Fallback: pick any edge
            u, v = next(iter(remaining_edges))
            L = {u}
            R = {v}
        
        # Remove covered edges
        for l in L:
            for r in R:
                if l != r:
                    key = (min(l, r), max(l, r))
                    if key in remaining_edges:
                        remaining_edges.remove(key)
        
        bicliques.append([list(L), list(R)])
    
    # Phase 2: Fallback for any remaining edges
    if remaining_edges:
        for u, v in remaining_edges:
            bicliques.append([[u], [v]])
    
    # Phase 3: Try to merge bicliques
    # Two bicliques [L1,R1] and [L2,R2] can be merged into [L1∪L2, R1∪R2]
    # if all cross-edges exist in the original graph
    merged = True
    while merged and time.time() < deadline - 5:
        merged = False
        for i in range(len(bicliques)):
            if time.time() > deadline - 5:
                break
            L1, R1 = bicliques[i][0], bicliques[i][1]
            L1_set, R1_set = set(L1), set(R1)
            for j in range(i+1, len(bicliques)):
                L2, R2 = bicliques[j][0], bicliques[j][1]
                L2_set, R2_set = set(L2), set(R2)
                
                # Check if we can merge: L1∪L2 x R1∪R2 must all be edges
                # i.e., for all l in L1∪L2, r in R1∪R2, edge (l,r) exists
                new_L = L1_set | L2_set
                new_R = R1_set | R2_set
                
                valid = True
                for l in new_L:
                    for r in new_R:
                        if l == r:
                            continue
                        if r not in adj[l]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Also need to ensure no overlap issues: 
                # The merged biclique's cross edges must be a superset of both
                # Actually we just need all cross pairs to be edges, which we checked.
                
                # Merge: replace bicliques[i] with merged, remove j
                bicliques[i] = [list(new_L), list(new_R)]
                bicliques.pop(j)
                merged = True
                break
            if merged:
                break
    
    # Ensure types are correct: vertices are Python int, sides are lists
    result = []
    for bic in bicliques:
        L = [int(v) for v in bic[0]]
        R = [int(v) for v in bic[1]]
        result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
