import itertools
import random
import time

N = 16

def construct():
    points = list(itertools.product(range(N), repeat=2))
    n_points = N * N
    
    # Precompute iso_with_pair
    iso_with_pair = [[set() for _ in range(n_points)] for _ in range(n_points)]
    
    for i in range(n_points):
        p_i = points[i]
        for j in range(i + 1, n_points):
            p_j = points[j]
            d_ij = (p_i[0] - p_j[0])**2 + (p_i[1] - p_j[1])**2
            
            for k in range(n_points):
                if k == i or k == j:
                    continue
                p_k = points[k]
                d_ik = (p_i[0] - p_k[0])**2 + (p_i[1] - p_k[1])**2
                d_jk = (p_j[0] - p_k[0])**2 + (p_j[1] - p_k[1])**2
                
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    iso_with_pair[i][j].add(k)
                    iso_with_pair[j][i].add(k)
                    
    def greedy_maximal(order):
        S = set()
        for idx in order:
            can_add = True
            S_list = list(S)
            for a in S_list:
                for b in S_list:
                    if a >= b:
                        continue
                    if idx in iso_with_pair[a][b]:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.add(idx)
        return S

    def shrink_maximal(S):
        S_list = list(S)
        block_count = [0] * n_points
        
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for q in iso_with_pair[a][b]:
                    if q not in S:
                        block_count[q] += 1
        
        changed = True
        while changed:
            changed = False
            for p in list(S):
                count_p = [0] * n_points
                for s in S:
                    if s == p:
                        continue
                    for q in iso_with_pair[p][s]:
                        if q not in S:
                            count_p[q] += 1
                
                can_remove = True
                for q in range(n_points):
                    if q in S:
                        continue
                    if block_count[q] == count_p[q]:
                        can_remove = False
                        break
                
                if can_remove:
                    S.remove(p)
                    changed = True
                    for s in S:
                        for q in iso_with_pair[p][s]:
                            if q not in S:
                                block_count[q] -= 1
                    break
        
        return S

    best_S = None
    best_size = float('inf')
    
    start_time = time.time()
    max_time = 280
    
    # Run trials
    for trial in range(5000):
        if time.time() - start_time > max_time:
            break
        
        order = list(range(n_points))
        random.shuffle(order)
        
        S = greedy_maximal(order)
        S = shrink_maximal(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = S
            
    if best_S is None:
        best_S = greedy_maximal(list(range(n_points)))
        
    return [points[i] for i in best_S]
