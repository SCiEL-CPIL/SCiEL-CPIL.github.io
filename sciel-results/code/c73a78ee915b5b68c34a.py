import random
import time
import itertools

def construct():
    n = 16
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2) configurations
    k222s = []
    # Matchings of 6 indices
    matchings = [
        [(0,1),(2,3),(4,5)], [(0,2),(1,3),(4,5)], [(0,3),(1,2),(4,5)],
        [(0,1),(2,4),(3,5)], [(0,2),(1,4),(3,5)], [(0,4),(1,2),(3,5)],
        [(0,1),(2,5),(3,4)], [(0,2),(1,5),(3,4)], [(0,5),(1,2),(3,4)],
        [(0,3),(1,4),(2,5)], [(0,4),(1,3),(2,5)], [(0,5),(1,4),(2,3)],
        [(0,3),(1,5),(2,4)], [(0,4),(1,5),(2,3)], [(0,5),(1,3),(2,4)]
    ]
    for verts in itertools.combinations(range(n), 6):
        for m in matchings:
            pairs = [(verts[m[0][0]], verts[m[0][1]]),
                     (verts[m[1][0]], verts[m[1][1]]),
                     (verts[m[2][0]], verts[m[2][1]])]
            t_set = set()
            for a in pairs[0]:
                for b in pairs[1]:
                    for c in pairs[2]:
                        t = (min(a,b,c), min(max(a,b),c), max(max(a,b),c))
                        t_set.add(t)
            k222s.append(list(t_set))
            
    triple_to_k222s = [[] for _ in range(num_triples)]
    for i, k222 in enumerate(k222s):
        for t in k222:
            idx = triple_to_idx[t]
            triple_to_k222s[idx].append(i)
            
    num_k222s = len(k222s)
    
    best_set = set()
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial seed: all triples with 0
    seed_set = set(t for t in triples if 0 in t)
    best_set = seed_set
    
    # Greedy restarts
    while time.time() < deadline:
        random.shuffle(triples)
        counts = [0] * num_k222s
        current_set = set()
        
        for t in triples:
            idx = triple_to_idx[t]
            can_add = True
            for k_idx in triple_to_k222s[idx]:
                if counts[k_idx] == 7:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(t)
                for k_idx in triple_to_k222s[idx]:
                    counts[k_idx] += 1
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    return [list(t) for t in best_set]
