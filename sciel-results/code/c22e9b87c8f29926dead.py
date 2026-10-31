import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        # Vectorized sumset and diffset
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
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
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 1: Try known good Golomb rulers as initial candidates
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 31, 34],
        [0, 1, 4, 10, 18, 23, 31, 42, 45],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 54],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 65],
    ]
    
    for gr in golomb_rulers:
        if time.time() > deadline - 15:
            break
        try_set(gr)
    
    # Phase 2: Exhaustive search for small M
    for M in range(1, 14):
        if time.time() > deadline - 15:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 15:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
    
    # Phase 3: Heuristic Search with multiple strategies
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current) if current else 0
        
        # Strategy 1: Try adding an element - more targeted search
        if not improved:
            candidates_to_add = set()
            # Nearby positions
            for i in range(1, max_cur + 50):
                if i not in current:
                    candidates_to_add.add(i)
            # Larger jumps based on current structure
            if len(current) > 2:
                for _ in range(50):
                    candidates_to_add.add(random.randint(max_cur + 1, max_cur + 300))
            # Try positions that create interesting difference patterns
            for _ in range(30):
                candidates_to_add.add(random.randint(1, max_cur + 100))
                
            for new_el in sorted(candidates_to_add):
                if time.time() > deadline - 10:
                    break
                if new_el in current:
                    continue
                cand = sorted(current + [new_el])
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # Strategy 2: Try replacing an element with better neighborhood search
        if not improved and len(current) >= 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                old_el = current[i]
                # Wider neighborhood for replacement
                for delta in range(-50, 51):
                    if time.time() > deadline - 10:
                        break
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # Strategy 3: Try removing an element
        if not improved and len(current) > 3:
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        # Strategy 4: Try swapping two elements
        if not improved and len(current) > 3:
            for i in range(1, len(current)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline - 10:
                        break
                    cand = list(current)
                    cand[i], cand[j] = cand[j], cand[i]
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation to escape local optima
            if random.random() < 0.7:
                perturbation_type = random.random()
                if perturbation_type < 0.25:
                    # Swap two random elements
                    if len(current) > 2:
                        i = random.randint(1, len(current)-1)
                        j = random.randint(1, len(current)-1)
                        if i != j:
                            cand = list(current)
                            cand[i], cand[j] = cand[j], cand[i]
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                elif perturbation_type < 0.5:
                    # Add a random element
                    new_el = random.randint(1, max_cur + 150)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)
                elif perturbation_type < 0.75:
                    # Replace a random element
                    if len(current) > 1:
                        i = random.randint(1, len(current)-1)
                        new_el = random.randint(1, max_cur + 80)
                        if new_el not in current:
                            cand = list(current)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 in cand:
                                try_set(cand)
                                current = list(best_set)
                else:
                    # Remove a random element
                    if len(current) > 2:
                        i = random.randint(1, len(current)-1)
                        cand = [x for j, x in enumerate(current) if j != i]
                        try_set(cand)
                        current = list(best_set)
    
    return list(best_set)
