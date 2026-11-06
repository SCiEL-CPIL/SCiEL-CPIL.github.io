import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    all_triples.sort()
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    k222_configs = []
    
    for points in combinations(range(n), 6):
        p0,p1,p2,p3,p4,p5 = points
        for j in range(1, 6):
            pair1 = (p0, points[j])
            rem = [points[k] for k in range(6) if k != 0 and k != j]
            pairings = [
                ((rem[0],rem[1]), (rem[2],rem[3])),
                ((rem[0],rem[2]), (rem[1],rem[3])),
                ((rem[0],rem[3]), (rem[1],rem[2])),
            ]
            for pair2, pair3 in pairings:
                edges = []
                for a in pair1:
                    for b in pair2:
                        for c in pair3:
                            t = tuple(sorted((a,b,c)))
                            edges.append(triple_to_idx[t])
                k222_configs.append(edges)
    
    num_k222 = len(k222_configs)
    
    # For each triple, list of K(2,2,2) configs it belongs to
    triple_to_k222 = [[] for _ in range(num_triples)]
    for i, edges in enumerate(k222_configs):
        for e in edges:
            triple_to_k222[e].append(i)
    
    edge_count = [0] * num_k222
    result = []
    
    for t_idx in range(num_triples):
        if time.time() - start_time > deadline - 5:
            break
        
        can_add = True
        for k222_idx in triple_to_k222[t_idx]:
            if edge_count[k222_idx] == 7:
                can_add = False
                break
        
        if can_add:
            result.append(list(all_triples[t_idx]))
            for k222_idx in triple_to_k222[t_idx]:
                edge_count[k222_idx] += 1
    
    return result
