import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill
    
    n = P['n']
    all_edges = set()
    for e in P['edges']:
        a, b = e
        all_edges.add((min(a,b), max(a,b)))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (a,b) in all_edges:
        adj[a].add(b)
        adj[b].add(a)
    
    def greedy_biclique_partition(remaining_edges, adj, seed=None):
        if seed is not None:
            random.seed(seed)
        
        bicliques = []
        rem = set(remaining_edges)
        
        while rem:
            if time.time() > deadline:
                # Return star partition as fallback
                break
            
            # Pick an edge to anchor
            # Choose edge with max product of remaining degrees
            best_edge = None
            best_score = -1
            edge_list = list(rem)
            # For large sets, sample
            if len(edge_list) > 100:
                sample = random.sample(edge_list, min(100, len(edge_list)))
            else:
                sample = edge_list
            
            for (a,b) in sample:
                deg_a = len(adj[a])
                deg_b = len(adj[b])
                score = deg_a * deg_b
                if score > best_score:
                    best_score = score
                    best_edge = (a,b)
            
            if best_edge is None:
                best_edge = edge_list[0]
            
            a, b = best_edge
            
            # Expand into a biclique starting from edge (a,b)
            L = {a}
            R = {b}
            
            # Iteratively grow
            improved = True
            while improved:
                improved = False
                
                # Find candidates for L: vertices connected to ALL in R
                # and not already in L
                common_R = set.intersection(*[adj[v] for v in R]) if R else set()
                cand_L = common_R - L
                
                # Find candidates for R: vertices connected to ALL in L
                common_L = set.intersection(*[adj[v] for v in L]) if L else set()
                cand_R = common_L - R
                
                # Try adding each candidate and pick the one that maximizes |L|*|R|
                best_gain = 0
                best_add = None  # (side, vertex)
                
                for v in cand_L:
                    new_L = L | {v}
                    new_R = R
                    # Check: all pairs in new_L x new_R must be edges
                    # v must connect to all in R (already ensured by common_R)
                    gain = len(new_L) * len(new_R) - len(L) * len(R)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = ('L', v)
                
                for v in cand_R:
                    new_L = L
                    new_R = R | {v}
                    gain = len(new_L) * len(new_R) - len(L) * len(R)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = ('R', v)
                
                if best_add is not None:
                    side, v = best_add
                    if side == 'L':
                        L.add(v)
                    else:
                        R.add(v)
                    improved = True
            
            # Convert to sorted lists
            L_list = sorted(L)
            R_list = sorted(R)
            
            # Verify: all cross pairs must be edges
            valid = True
            for u in L_list:
                for v in R_list:
                    e = (min(u,v), max(u,v))
                    if e not in rem:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback: just use the single edge
                L_list = [a]
                R_list = [b]
            
            bicliques.append([L_list, R_list])
            
            # Remove edges in this biclique
            for u in L_list:
                for v in R_list:
                    e = (min(u,v), max(u,v))
                    rem.discard(e)
        
        return bicliques, rem
    
    # Try multiple seeds and keep the best
    best_result = None
    best_count = float('inf')
    
    # First, try a deterministic approach (seed=0)
    for seed in [0, 1, 42, 123, 7, 99]:
        if time.time() > deadline:
            break
        bicliques, remaining = greedy_biclique_partition(all_edges, adj, seed=seed)
        
        if remaining:
            # Add remaining edges as individual bicliques
            for (a,b) in remaining:
                bicliques.append([[a],[b]])
        
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_result = bicliques
        
        if best_count <= 5:
            break
    
    if best_result is None:
        # Fallback: star partition
        best_result = [[[u],[v for a,v in P['edges'] if a==u]] for u in range(n) if any(a==u for a,v in P['edges'])]
    
    # Ensure types are correct: integers as Python int, lists
    final = []
    for bic in best_result:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final.append([left, right])
    
    return final
