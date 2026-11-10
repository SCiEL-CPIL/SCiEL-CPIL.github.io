import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def compute_new_sums(sums, max_k, x):
        new_sums = [0] * (max_k + 2)
        for k in range(max_k + 1, 0, -1):
            new_sums[k] = sums[k] | (sums[k-1] << x)
        new_sums[0] = sums[0]
        return new_sums, max_k + 1

    def check_disjointness(sums, max_k):
        union = 0
        for k in range(max_k + 1):
            s = sums[k]
            if s & union:
                return False
            union |= s
        return True

    best_set = []
    best_len = 0
    
    nogoods = set()
    nogoods_by_elem = {}

    def add_nogood(ng):
        nonlocal nogoods, nogoods_by_elem
        if ng not in nogoods:
            nogoods.add(ng)
            for e in ng:
                if e not in nogoods_by_elem:
                    nogoods_by_elem[e] = set()
                nogoods_by_elem[e].add(ng)

    def is_forbidden(chosen, x):
        # Check if there is a nogood N such that N is subset of (chosen U {x})
        # Since chosen is valid, N must contain x.
        # So we check nogoods containing x.
        if x not in nogoods_by_elem:
            return False
        
        for ng in nogoods_by_elem[x]:
            # Check if ng - {x} is subset of chosen
            # ng is a frozenset
            # We can iterate ng
            valid = True
            for e in ng:
                if e != x and e not in chosen:
                    valid = False
                    break
            if valid:
                return True
        return False

    def dfs(chosen, sums, max_k, candidates, depth):
        nonlocal best_set, best_len
        
        if time.time() > deadline:
            return

        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = list(chosen)
        
        if len(chosen) + len(candidates) <= best_len:
            return

        # Iterate candidates (sorted descending)
        for i in range(len(candidates)):
            if time.time() > deadline:
                return
            
            x = candidates[i]
            
            if is_forbidden(chosen, x):
                continue

            new_sums, new_max_k = compute_new_sums(sums, max_k, x)
            
            if check_disjointness(new_sums, new_max_k):
                new_chosen = chosen | {x}
                remaining = candidates[:i] + candidates[i+1:]
                dfs(new_chosen, new_sums, new_max_k, remaining, depth + 1)
            else:
                add_nogood(chosen | {x})

    initial_candidates = list(range(256, 0, -1))
    initial_sums = [0] * 61
    initial_sums[0] = 1 << 0
    
    sys.setrecursionlimit(10000)
    
    dfs(set(), initial_sums, 0, initial_candidates, 0)
    
    if not best_set:
        best_set = [1]
        
    return list(best_set)
