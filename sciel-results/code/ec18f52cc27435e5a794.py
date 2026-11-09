import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 128
    # Precompute GCDs
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
            
    def is_bad_triple(a, b, c):
        return g[a][b] == g[a][c] == g[b][c]
        
    def count_bad_triples(S):
        # S is a set or list
        # For speed, use list of indices
        count = 0
        L = list(S)
        n = len(L)
        for i in range(n):
            a = L[i]
            for j in range(i+1, n):
                b = L[j]
                gab = g[a][b]
                for k in range(j+1, n):
                    c = L[k]
                    if g[a][c] == gab and g[b][c] == gab:
                        count += 1
        return count

    def get_bad_triples(S):
        L = list(S)
        n = len(L)
        triples = []
        for i in range(n):
            a = L[i]
            for j in range(i+1, n):
                b = L[j]
                gab = g[a][b]
                for k in range(j+1, n):
                    c = L[k]
                    if g[a][c] == gab and g[b][c] == gab:
                        triples.append((a,b,c))
        return triples

    # Initial Set: All numbers
    S = set(range(1, N+1))
    
    # Phase 1: Greedy Removal to satisfy constraint
    while time.time() < deadline:
        bads = get_bad_triples(S)
        if not bads:
            break
            
        # Count frequency of each element in bad triples
        freq = {}
        for a,b,c in bads:
            freq[a] = freq.get(a, 0) + 1
            freq[b] = freq.get(b, 0) + 1
            freq[c] = freq.get(c, 0) + 1
            
        # Remove element with max frequency
        # Tie-breaker: remove the one that is "most composite" or just random?
        # Let's pick the one with max freq, then random among ties.
        max_freq = max(freq.values())
        candidates = [x for x, f in freq.items() if f == max_freq]
        to_remove = random.choice(candidates)
        S.remove(to_remove)
        
    # Phase 2: Greedy Addition
    # Try to add back numbers that were removed or not in S
    # We maintain a list of candidates not in S
    candidates = [x for x in range(1, N+1) if x not in S]
    # Sort candidates by "potential" - maybe number of GCDs > 1 with S?
    # Or just random order
    
    random.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        # Check if adding x creates any bad triple
        # A bad triple involving x must be (a, b, x) where a,b in S
        # and g(a,b) == g(a,x) == g(b,x)
        valid = True
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            gax = g[a][x]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if g[a][b] == gax and g[b][x] == gax:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(x)
            
    # Phase 3: Local Search (Swap)
    # Try to swap an element in S with an element not in S to increase size or maintain size with fewer conflicts?
    # We want to maximize size. So we only swap if we can add more?
    # Actually, if S is maximal (no single addition possible), we might be stuck in a local optimum.
    # We can try removing 2 elements and adding 3? Or removing 1 and adding 2?
    # Given time, simple swap (remove 1, add 1) doesn't increase size.
    # We need to increase size. So we need to find a way to add 1 element.
    # If no single element can be added, try to remove a "bad" element and add a "good" one?
    # But that keeps size same.
    # To increase size, we must find a configuration where we can add k elements.
    
    # Let's try a random restart if we have time, keeping the best solution found.
    best_S = set(S)
    best_size = len(S)
    
    restarts = 0
    while time.time() < deadline and restarts < 20:
        # Random perturbation
        S_curr = set(best_S)
        # Remove a random element
        if S_curr:
            S_curr.remove(random.choice(list(S_curr)))
        
        # Try to add back
        candidates = [x for x in range(1, N+1) if x not in S_curr]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            valid = True
            S_list = list(S_curr)
            for i in range(len(S_list)):
                a = S_list[i]
                gax = g[a][x]
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    if g[a][b] == gax and g[b][x] == gax:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S_curr.add(x)
                
        if len(S_curr) > best_size:
            best_S = S_curr
            best_size = len(S_curr)
            
        restarts += 1
        
    return list(best_S)
