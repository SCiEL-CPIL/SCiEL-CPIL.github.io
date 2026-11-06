import time
import random
from itertools import combinations

# Precompute popcounts for masks up to 16 bits
POP_COUNT = [0] * (1 << 16)
for i in range(1, 1 << 16):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

def _greedy_vertex_cover(triples, n=16):
    if not triples:
        return 0
        
    freq = [0] * n
    for t in triples:
        for v in t:
            freq[v] += 1
            
    active_edges = set(triples)
    cover_size = 0
    
    while active_edges:
        max_v = -1
        max_f = -1
        for v in range(n):
            if freq[v] > max_f:
                max_f = freq[v]
                max_v = v
                
        if max_f == 0:
            break
            
        cover_size += 1
        
        edges_to_remove = []
        for e in active_edges:
            if max_v in e:
                edges_to_remove.append(e)
                
        for e in edges_to_remove:
            active_edges.remove(e)
            for v in e:
                if v != max_v:
                    freq[v] -= 1
                    
    return cover_size

def _calc_alpha(triples):
    n = 16
    edge_masks = []
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        edge_masks.append(mask)
        
    max_alpha = 0
    for mask in range(1 << n):
        pc = POP_COUNT[mask]
        if pc <= max_alpha:
            continue
        is_independent = True
        for em in edge_masks:
            if (mask & em) == em:
                is_independent = False
                break
        if is_independent:
            max_alpha = pc
    return max_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_alpha = 16
    best_triples = []
    
    all_triples = list(combinations(range(16), 3))
    
    while time.time() < deadline:
        random.shuffle(all_triples)
        
        current_triples = []
        used_pairs = set()
        
        for t in all_triples:
            a, b, c = t
            p1 = (min(a,b), max(a,b))
            p2 = (min(a,c), max(a,c))
            p3 = (min(b,c), max(b,c))
            
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
                
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            current_triples.append(t)
            
        cover_size = _greedy_vertex_cover(current_triples)
        lower_bound_alpha = 16 - cover_size
        
        if lower_bound_alpha > best_alpha:
            continue
            
        alpha = _calc_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in current_triples]
            
            if best_alpha <= 5:
                break

    return best_triples
