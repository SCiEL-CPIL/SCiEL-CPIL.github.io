import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcds[a][b]
            for c in range(b + 1, N + 1):
                gbc = gcds[b][c]
                if gab == gbc:
                    gca = gcds[c][a]
                    if gca == gab:
                        bad_triples.append((a, b, c))
                        
    # Map element to list of bad triples it participates in
    elem_triples = {x: [] for x in range(1, N + 1)}
    for idx, t in enumerate(bad_triples):
        for x in t:
            elem_triples[x].append(idx)
            
    def is_valid_addition(S_set, x):
        for idx in elem_triples[x]:
            t = bad_triples[idx]
            if t[0] in S_set and t[1] in S_set and t[2] in S_set:
                return False
        return True

    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    best_set = set()
    best_size = 0
    
    def greedy_init():
        # Start with multiples of 2 or 3
        S = set(x for x in range(1, N + 1) if x % 2 == 0 or x % 3 == 0)
        return S

    def local_search(S):
        S = set(S)
        
        # Phase 1: Remove elements to make it valid
        # We repeatedly find a bad triple and remove the "most constrained" element
        while True:
            if time.time() > deadline:
                break
            
            # Find a bad triple in S
            bad_idx = -1
            for idx in range(len(bad_triples)):
                t = bad_triples[idx]
                if t[0] in S and t[1] in S and t[2] in S:
                    bad_idx = idx
                    break
            
            if bad_idx == -1:
                break
            
            t = bad_triples[bad_idx]
            candidates = [x for x in t if x in S]
            
            # Heuristic: remove the element that is in the most active bad triples
            best_rem = candidates[0]
            max_viol = -1
            for x in candidates:
                viol = 0
                for idx2 in elem_triples[x]:
                    t2 = bad_triples[idx2]
                    if t2[0] in S and t2[1] in S and t2[2] in S:
                        viol += 1
                if viol > max_viol:
                    max_viol = viol
                    best_rem = x
            S.remove(best_rem)
            
        # Phase 2: Try to add elements
        not_in_S = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(not_in_S)
        for x in not_in_S:
            if time.time() > deadline:
                break
            if is_valid_addition(S, x):
                S.add(x)
                
        return S

    # Run multiple restarts
    for run in range(100):
        if time.time() > deadline - 5:
            break
            
        S = greedy_init()
        S = local_search(S)
        
        if len(S) > len(best_set):
            best_set = set(S)
            
    # Final polish
    not_in_best = [x for x in elements if x not in best_set]
    random.shuffle(not_in_best)
    for x in not_in_best:
        if is_valid_addition(best_set, x):
            best_set.add(x)

    return list(best_set)
