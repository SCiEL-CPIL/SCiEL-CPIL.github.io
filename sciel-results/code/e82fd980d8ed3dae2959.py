"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    best_score = -1
    best_subset = None
    start_time = time.time()
    
    # Theoretical max distinct degrees is 9 (0 to 8)
    # We search all subsets of size >= 8
    for k in range(8, n + 1):
        # Check time limit
        if time.time() - start_time > 55:
            break
            
        for combo in itertools.combinations(range(n), k):
            s = set(combo)
            degrees = set()
            for v in combo:
                deg = 0
                for u in adj[v]:
                    if u in s:
                        deg += 1
                degrees.add(deg)
            
            num_distinct = len(degrees)
            if num_distinct > best_score:
                best_score = num_distinct
                best_subset = list(combo)
                
                # If we hit the theoretical max, we can stop
                if best_score == 9:
                    return best_subset
                    
    if best_subset is None:
        return list(range(8))
        
    return best_subset
# EVOLVE-BLOCK-END
