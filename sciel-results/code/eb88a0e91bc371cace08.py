import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe
    
    n = 16
    # Generate all triples
    triples = []
    triple_indices = {}
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
                triple_indices[(a, b, c)] = idx
                idx += 1
    
    total_triples = len(triples)
    
    # Pre-compute all K(2,2,2)
    k222s = []
    triple_to_k222s = [[] for _ in range(total_triples)]
    
    vertices = list(range(n))
    for v6 in itertools.combinations(vertices, 6):
        v = list(v6)
        for i in range(1, 6):
            p1 = (v[0], v[i])
            remaining = [v[j] for j in range(6) if j != 0 and j != i]
            r = remaining
            
            # 3 ways to pair remaining 4
            # 1
            p2 = (r[0], r[1])
            p3 = (r[2], r[3])
            # 2
            p2b = (r[0], r[2])
            p3b = (r[1], r[3])
            # 3
            p2c = (r[0], r[3])
            p3c = (r[1], r[2])
            
            partitions = [
                [p1, p2, p3],
                [p1, p2b, p3b],
                [p1, p2c, p3c]
            ]
            
            for pairs in partitions:
                t_indices = set()
                for mask in range(8):
                    t = (
                        pairs[0][0] if (mask & 1) == 0 else pairs[0][1],
                        pairs[1][0] if (mask & 2) == 0 else pairs[1][1],
                        pairs[2][0] if (mask & 4) == 0 else pairs[2][1]
                    )
                    t_sorted = tuple(sorted(t))
                    if t_sorted in triple_indices:
                        t_indices.add(triple_indices[t_sorted])
                
                if len(t_indices) == 8:
                    k222_idx = len(k222s)
                    k222s.append(list(t_indices))
                    for t_idx in t_indices:
                        triple_to_k222s[t_idx].append(k222_idx)
    
    best_set = set()
    best_size = 0
    
    while time.time() < deadline:
        random.shuffle(triples)
        current_set = set()
        k222_counts = [0] * len(k222s)
        critical_k222s = set()
        
        for t in triples:
            t_idx = triple_indices[t]
            
            safe = True
            for k_idx in triple_to_k222s[t_idx]:
                if k_idx in critical_k222s:
                    safe = False
                    break
            
            if safe:
                current_set.add(t_idx)
                for k_idx in triple_to_k222s[t_idx]:
                    k222_counts[k_idx] += 1
                    if k222_counts[k_idx] == 7:
                        critical_k222s.add(k_idx)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    result = [list(triples[i]) for i in best_set]
    return result
