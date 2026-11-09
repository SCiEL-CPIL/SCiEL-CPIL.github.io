import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix and edge set for O(1) lookup
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
        
    uncovered = set(edge_set)
    bicliques = []
    
    # Helper to check if a set of vertices forms a biclique with another set
    # L x R must be subset of edges
    def is_valid_biclique(L, R):
        for u in L:
            for v in R:
                if not adj[u][v]:
                    return False
        return True
        
    # Greedy expansion to find a biclique covering many uncovered edges
    def find_biclique(start_edge):
        u, v = start_edge
        L = [u]
        R = [v]
        
        # Pre-compute neighbors in original graph for speed
        # We only add vertices that maintain the biclique property
        
        while True:
            improved = False
            
            # Try adding to L
            best_gain_l = 0
            best_v_l = -1
            for w in range(n):
                if w in L:
                    continue
                # Check if w can be added to L
                # w must be connected to all v in R
                can_add = True
                for v in R:
                    if not adj[w][v]:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Calculate gain: number of uncovered edges (w, v) for v in R
                gain = 0
                for v in R:
                    e = (min(w, v), max(w, v))
                    if e in uncovered:
                        gain += 1
                
                if gain > best_gain_l:
                    best_gain_l = gain
                    best_v_l = w
            
            if best_v_l != -1:
                L.append(best_v_l)
                improved = True
            
            # Try adding to R
            best_gain_r = 0
            best_v_r = -1
            for w in range(n):
                if w in R:
                    continue
                # Check if w can be added to R
                # w must be connected to all u in L
                can_add = True
                for u in L:
                    if not adj[u][w]:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Calculate gain: number of uncovered edges (u, w) for u in L
                gain = 0
                for u in L:
                    e = (min(u, w), max(u, w))
                    if e in uncovered:
                        gain += 1
                
                if gain > best_gain_r:
                    best_gain_r = gain
                    best_v_r = w
            
            if best_v_r != -1:
                R.append(best_v_r)
                improved = True
                
            if not improved:
                break
                
        return L, R

    while uncovered:
        if time.time() > deadline:
            break
            
        # Try to find a good biclique
        # Strategy: Try expanding from a few random uncovered edges and pick the best one
        best_L, best_R = None, None
        best_covered = 0
        
        # Sample a few uncovered edges to start from
        sample_size = min(10, len(uncovered))
        sample_edges = random.sample(list(uncovered), sample_size)
        
        for e in sample_edges:
            L, R = find_biclique(e)
            # Count how many uncovered edges this biclique covers
            covered = 0
            for u in L:
                for v in R:
                    e2 = (min(u, v), max(u, v))
                    if e2 in uncovered:
                        covered += 1
                        
            if covered > best_covered:
                best_covered = covered
                best_L, best_R = L, R
                
        if best_L is None or best_covered == 0:
            # Fallback: just take one edge
            e = next(iter(uncovered))
            best_L, best_R = [e[0]], [e[1]]
            
        bicliques.append([best_L, best_R])
        
        # Remove covered edges
        for u in best_L:
            for v in best_R:
                e = (min(u, v), max(u, v))
                if e in uncovered:
                    uncovered.remove(e)
                    
    # Fallback if we didn't finish in time (should not happen for n=30)
    if uncovered:
        for e in uncovered:
            bicliques.append([[e[0]], [e[1]]])
            
    # Convert to required format: list of [left, right]
    # Ensure types are int and lists
    result = []
    for L, R in bicliques:
        result.append([[int(v) for v in L], [int(v) for v in R]])
        
    return result
