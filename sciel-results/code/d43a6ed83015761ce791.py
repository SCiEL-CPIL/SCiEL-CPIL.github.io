import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    vertices = list(range(1, N + 1))
    
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    conflict = [set() for _ in range(N)]
    has_self_loop = [False] * N
    
    for i in range(N):
        v = i + 1
        for s in squares:
            u = s - v
            if 1 <= u <= N:
                conflict[i].add(u - 1)
                if u == v:
                    has_self_loop[i] = True

    def greedy_mis():
        in_set = [False] * N
        degrees = [len(conflict[i]) for i in range(N)]
        sorted_vertices = sorted(range(N), key=lambda x: degrees[x])
        
        for v in sorted_vertices:
            if has_self_loop[v]:
                continue
            compatible = True
            for u in range(N):
                if in_set[u] and u in conflict[v]:
                    compatible = False
                    break
            if compatible:
                in_set[v] = True
                
        return [i+1 for i in range(N) if in_set[i]]

    def local_search(initial_set, seed=42):
        random.seed(seed)
        S = set(v - 1 for v in initial_set)
        best_set = S
        best_size = len(S)
        
        improved = True
        while improved:
            improved = False
            removal_order = list(S)
            random.shuffle(removal_order)
            
            for v in removal_order:
                if time.time() > deadline:
                    break
                
                S_temp = S - {v}
                candidates = []
                for u in range(N):
                    if u in S_temp or has_self_loop[u]:
                        continue
                    ok = True
                    for w in S_temp:
                        if w in conflict[u]:
                            ok = False
                            break
                    if ok:
                        candidates.append(u)
                            
                cand_sorted = sorted(candidates, key=lambda x: len(conflict[x]))
                
                added = []
                added_set = set()
                for u in cand_sorted:
                    ok = True
                    for w in added:
                        if w in conflict[u]:
                            ok = False
                            break
                    if ok:
                        added.append(u)
                        added_set.add(u)
                        
                new_set = S_temp.union(added_set)
                
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set
                    improved = True
                    S = new_set
                    break
                    
        return [i+1 for i in best_set]

    current_set = greedy_mis()
    best_set = local_search(current_set, seed=1)
    
    # Try a few more seeds to escape local optima
    for i in range(2, 20):
        if time.time() > deadline:
            break
        cand = local_search(current_set, seed=i)
        if len(cand) > len(best_set):
            best_set = cand

    return best_set
