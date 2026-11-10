import random
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    N = 128
    # Precompute gcds
    gcds = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcds[i][j] = gcd(i, j)
            gcds[j][i] = gcd(i, j)
    
    # Precompute bad triples
    # A triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcds[a][b]
            for c in range(b+1, N+1):
                if gcds[a][c] == g_ab and gcds[b][c] == g_ab:
                    bad_triples.append((a,b,c))
                    
    # Local Search
    best_set = set()
    best_size = 0
    
    # Greedy initialization
    current_set = set()
    # Order by descending "badness" potential? Or just 1..N
    # Let's try adding in order 1..N
    for i in range(1, N+1):
        # Check if adding i creates a bad triple
        valid = True
        # We need to check if there exist a,b in current_set such that (a,b,i) is bad
        # This is O(|S|^2) per addition. |S| <= 128. 128^2 * 128 = 2M ops. Fast.
        # But we can just check against precomputed bad triples involving i.
        # Better: just check all pairs in current_set.
        for a in current_set:
            for b in current_set:
                if a == b: continue
                # Check if (a,b,i) is bad
                if gcds[a][b] == gcds[a][i] == gcds[b][i]:
                    valid = False
                    break
            if not valid: break
        if valid:
            current_set.add(i)
            
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(current_set)
        
    # Local Search
    # We will try to improve by swapping or adding/removing
    # Since greedy might be suboptimal, we can try to remove elements to allow others?
    # Actually, Max Independent Set is usually solved by removing vertices from edges.
    # Let's restart with a different heuristic:
    # Start with all vertices, remove those in bad triples.
    
    for restart in range(10):
        if time.time() > deadline: break
        
        # Random initialization
        current_set = set(random.sample(range(1, N+1), N//2))
        
        # Remove vertices to make it independent
        # Find bad triples in current_set
        # Iterate until no bad triples
        changed = True
        while changed:
            changed = False
            # Find a bad triple
            # To speed up, we can just pick a random triple from bad_triples and check if it's in current_set
            # But we need to find ANY bad triple.
            # Let's iterate through bad_triples
            found_triple = None
            for t in bad_triples:
                if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                    found_triple = t
                    break
            
            if found_triple:
                # Remove one element randomly
                v_to_remove = random.choice(found_triple)
                current_set.remove(v_to_remove)
                changed = True
        
        # Now current_set is independent.
        # Try to add vertices
        improved = True
        while improved:
            improved = False
            candidates = list(set(range(1, N+1)) - current_set)
            random.shuffle(candidates)
            for v in candidates:
                # Check if v can be added
                valid = True
                for a in current_set:
                    for b in current_set:
                        if a == b: continue
                        if gcds[a][b] == gcds[a][v] == gcds[b][v]:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    current_set.add(v)
                    improved = True
                    break # Restart candidate loop? Or continue?
                    # If we add v, other candidates might become invalid.
                    # So break and restart the while loop.
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    return list(best_set)
