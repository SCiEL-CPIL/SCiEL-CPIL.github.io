import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Any element in the set (other than 255) must share a factor with 255
    # so must be divisible by 3, 5, or 17
    candidates = [i for i in range(1, 256) if i != 255 and (i % 3 == 0 or i % 5 == 0 or i % 17 == 0)]
    
    n = len(candidates)
    
    # Build adjacency: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Bron-Kerbosch with pivot to find maximum clique
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        if not P:
            return
        
        # Pivot selection: choose vertex in P ∪ X with maximum degree in P
        pivot = None
        max_deg = -1
        # Convert to list for stable iteration and combination
        union_px = list(P) + list(X)
        for v in union_px:
            deg = sum(1 for u in P if adj[v][u])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Candidates: vertices in P not adjacent to pivot
        if pivot is not None:
            candidates_list = [v for v in P if not adj[pivot][v]]
        else:
            candidates_list = list(P)
        
        for v in candidates_list:
            # Use sets for P and X to maintain O(1) lookup/add/remove in recursion
            new_R = R + [v]
            new_P = set(u for u in P if adj[v][u])
            new_X = set(u for u in X if adj[v][u])
            bron_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
    
    # Use sets for P and X for efficiency
    P = set(range(n))
    X = set()
    bron_kerbosch([], P, X)
    
    # The clique among candidates, plus 255
    result = [candidates[i] for i in best_clique] + [255]
    
    return result
