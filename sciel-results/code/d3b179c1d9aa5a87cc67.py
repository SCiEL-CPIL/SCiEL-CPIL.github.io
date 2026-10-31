import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_score = -1
    best_set = [0, 1, 3]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small n and M
    # We know n=6, M=17 gives ~1.109
    # Let's search n up to 8, M up to 40
    
    for n in range(3, 9):
        # Search M from n-1 to 40
        for M in range(n - 1, 41):
            if time.time() > deadline - 10:
                break
            # Generate combinations of n-1 elements from 1..M
            # This is too slow for large M.
            # Instead, use local search starting from random sets
            
    # Phase 2: Local search for specific n
    def local_search(start_set, max_iters=5000):
        nonlocal best_score, best_set
        curr = sorted(set(start_set))
        if not curr or 0 not in curr or len(curr) < 2:
            return curr
        
        curr_score = compute_score(curr)
        no_improve_count = 0
        max_no_improve = 20
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            n_curr = len(curr)
            max_cur = max(curr)
            
            # Try adding an element
            if n_curr < 10:
                search_limit = max_cur + 10
                for new_el in range(1, search_limit + 1):
                    if new_el in curr:
                        continue
                    cand = sorted(curr + [new_el])
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        improved = True
                        no_improve_count = 0
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        break
            
            # Try replacing an element
            if n_curr >= 2 and not improved:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    search_limit = max_cur + 10
                    for new_el in range(1, search_limit + 1):
                        if new_el in curr:
                            continue
                        cand = list(curr)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < 2:
                            continue
                        s = compute_score(cand)
                        if s > curr_score:
                            curr = cand
                            curr_score = s
                            improved = True
                            no_improve_count = 0
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            break
                    if improved:
                        break
            
            # Try removing an element
            if n_curr > 2 and not improved:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(curr) if j != i]
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        improved = True
                        no_improve_count = 0
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        break
            
            if improved:
                continue
            
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
            
            # Random perturbation
            if random.random() < 0.1:
                n = random.randint(3, min(8, n_curr + 2))
                max_m = max(10, max_cur + random.randint(0, 5))
                if n - 1 > 0:
                    inner = random.sample(range(1, max_m + 1), min(n - 1, max_m))
                    cand = sorted([0] + inner)
                else:
                    cand = [0]
                if len(cand) >= 2:
                    s = compute_score(cand)
                    if s > curr_score:
                        curr = cand
                        curr_score = s
                        no_improve_count = 0
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
        
        return curr

    # Start from known good sets
    known_good = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 9, 11, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 25]
    ]
    
    for start in known_good:
        if time.time() > deadline - 5:
            break
        local_search(start)

    # Random restarts
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        n = random.randint(3, 8)
        max_m = random.randint(10, 30)
        inner = random.sample(range(1, max_m + 1), min(n - 1, max_m))
        start = sorted([0] + inner)
        local_search(start)

    return list(best_set)
