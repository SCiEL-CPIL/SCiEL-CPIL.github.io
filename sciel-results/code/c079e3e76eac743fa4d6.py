import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def get_neighbors(mask):
        """Extract list of vertex indices from a bitmask."""
        result = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            result.append(v)
            m ^= lsb
        return result

    def expand_biclique(L, R, rem_mask, max_iter=20):
        """Expand biclique (L, R) using iterative common neighbor expansion."""
        L = set(L)
        R = set(R)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return L, R
            
            changed = False
            
            # Expand L: find u' not in L such that u' is connected to all in R in rem_mask
            mask_R = 0
            for rv in R:
                mask_R |= (1 << rv)
            
            if mask_R:
                for u_prime in range(n):
                    if u_prime in L:
                        continue
                    # Check if u_prime has edges to all vertices in R in rem_mask
                    if (rem_mask[u_prime] & mask_R) == mask_R:
                        L.add(u_prime)
                        changed = True
            
            # Expand R: find v' not in R such that v' is connected to all in L in rem_mask
            mask_L = 0
            for lu in L:
                mask_L |= (1 << lu)
            
            if mask_L:
                for v_prime in range(n):
                    if v_prime in R:
                        continue
                    # Check if v_prime has edges to all vertices in L in rem_mask
                    valid = True
                    for lu in L:
                        if not (rem_mask[lu] & (1 << v_prime)):
                            valid = False
                            break
                    if valid:
                        R.add(v_prime)
                        changed = True
            
            if not changed:
                break
        
        return L, R

    def greedy_partition(rem_mask_init, random_start=True):
        """Greedy biclique partition using random edge selection."""
        rem_mask = rem_mask_init[:]
        edges_left = len(edges)
        current_bicliques = []
        
        while edges_left > 0:
            if time.time() > deadline:
                return None
            
            # Collect all remaining edges
            remaining_edges = []
            for u in range(n):
                if rem_mask[u]:
                    vs = get_neighbors(rem_mask[u])
                    for v in vs:
                        if u < v:  # Avoid double counting
                            remaining_edges.append((u, v))
            
            if not remaining_edges:
                break
            
            # Select starting edge
            if random_start:
                u, v = random.choice(remaining_edges)
            else:
                # Pick edge with highest degree sum
                u, v = max(remaining_edges, key=lambda e: bin(rem_mask[e[0]]).count('1') + bin(rem_mask[e[1]]).count('1'))
            
            # Expand biclique from this edge
            L, R = expand_biclique({u}, {v}, rem_mask)
            
            if not L or not R:
                # Fallback: single edge
                L, R = {u}, {v}
            
            size = len(L) * len(R)
            if size <= 0:
                break
            
            current_bicliques.append([sorted(L), sorted(R)])
            
            # Update rem_mask: remove all edges between L and R
            mask_R = 0
            for rv in R:
                mask_R |= (1 << rv)
            mask_L = 0
            for lu in L:
                mask_L |= (1 << lu)
            
            for u in L:
                rem_mask[u] &= ~mask_R
            for v in R:
                rem_mask[v] &= ~mask_L
            
            edges_left -= size
        
        if edges_left != 0:
            return None
        
        return current_bicliques

    # Initial solution: greedy with random restarts
    best_bicliques = None
    best_count = float('inf')
    
    # Try multiple restarts
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        result = greedy_partition(adj[:], random_start=True)
        if result is not None:
            count = len(result)
            if count < best_count:
                best_count = count
                best_bicliques = result
                # Early termination if we found a very good solution
                if best_count <= 5:
                    break
    
    # Local search: try to merge bicliques
    if best_bicliques is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_bicliques)):
                    if time.time() > deadline:
                        break
                    
                    # Try to merge biclique i and j
                    L1, R1 = set(best_bicliques[i][0]), set(best_bicliques[i][1])
                    L2, R2 = set(best_bicliques[j][0]), set(best_bicliques[j][1])
                    
                    # Try different orientations for merging
                    merged_options = [
                        (L1 | L2, R1 | R2),
                        (L1 | R2, R1 | L2),
                        (R1 | L2, L1 | R2),
                        (R1 | R2, L1 | L2),
                    ]
                    
                    for L_m, R_m in merged_options:
                        # Check if merged biclique is valid: all cross edges must be in original graph
                        valid = True
                        for u in L_m:
                            for v in R_m:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        # Check if merged biclique covers exactly the edges from bicliques i and j
                        # (no extra edges that aren't in the original set)
                        covered_edges = set()
                        for u in L_m:
                            for v in R_m:
                                if u < v:
                                    covered_edges.add((u, v))
                                else:
                                    covered_edges.add((v, u))
                        
                        # Get edges from bicliques i and j
                        original_edges = set()
                        for u in L1:
                            for v in R1:
                                if u < v:
                                    original_edges.add((u, v))
                                else:
                                    original_edges.add((v, u))
                        for u in L2:
                            for v in R2:
                                if u < v:
                                    original_edges.add((u, v))
                                else:
                                    original_edges.add((v, u))
                        
                        # Check if covered edges are a subset of original edges
                        # and that we're not introducing new edges
                        if covered_edges == original_edges:
                            # Valid merge!
                            new_bicliques = [b for k, b in enumerate(best_bicliques) if k not in (i, j)]
                            new_bicliques.append([sorted(L_m), sorted(R_m)])
                            best_bicliques = new_bicliques
                            best_count = len(new_bicliques)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Final local search: try to replace individual bicliques with better ones
    if best_bicliques is not None and time.time() < deadline:
        for iteration in range(5):
            if time.time() > deadline:
                break
            for i in range(len(best_bicliques)):
                if time.time() > deadline:
                    break
                
                # Get edges covered by biclique i
                L, R = set(best_bicliques[i][0]), set(best_bicliques[i][1])
                covered_edges = set()
                for u in L:
                    for v in R:
                        if u < v:
                            covered_edges.add((u, v))
                        else:
                            covered_edges.add((v, u))
                
                # Try to find a smaller biclique that covers these edges
                # This is complex, so we'll skip for now
                pass
    
    # Return best found
    if best_bicliques is None:
        # Fallback: one biclique per edge
        result = []
        for u, v in edges:
            result.append([[u], [v]])
        return result
    
    # Convert to required format
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    
    return result
