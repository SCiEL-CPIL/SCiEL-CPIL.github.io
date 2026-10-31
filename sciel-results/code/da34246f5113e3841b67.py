import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit
    
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
        
        # Vectorized sum and diff computation
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
        # Limit size to prevent O(n^2) blowup in compute_score if n gets large
        if len(cand) > 60: 
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Try structured sets
    # Geometric progressions with various bases and starting points
    for base in [2, 3, 4, 5, 7, 8, 10, 12, 15]:
        for start in [1, 2, 3]:
            g = [0, start]
            while g[-1] * base < 20000:
                g.append(g[-1] * base)
            try_set(g)
    
    # Power-of-prime constructions
    for p in [2, 3, 5, 7, 11]:
        g = [0]
        val = 1
        while val < 20000:
            g.append(val)
            val *= p
        try_set(g)
    
    # Golomb ruler-like sets (greedy construction)
    def generate_golomb_like(n, max_val=10000):
        """Generate a set that tries to have unique differences"""
        U = [0, 1]
        U_set = set(U)
        for _ in range(2, n):
            best_pos = -1
            best_score_local = -1
            # Limit search range for speed
            for pos in range(2, min(max_val, 5000) + 1):
                if pos in U_set:
                    continue
                # Check if this creates new differences
                new_diffs = set()
                for u in U:
                    new_diffs.add(pos - u)
                    new_diffs.add(u - pos)
                # We want to maximize unique differences
                all_diffs = set()
                for i in range(len(U)):
                    for j in range(len(U)):
                        all_diffs.add(U[i] - U[j])
                new_unique = len(new_diffs - all_diffs)
                if new_unique > best_score_local:
                    best_score_local = new_unique
                    best_pos = pos
                if best_score_local >= len(U):  # Perfect: all new
                    break
            if best_pos > 0:
                U.append(best_pos)
                U_set.add(best_pos)
            else:
                break
        return U
    
    for n in [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]:
        g = generate_golomb_like(n, 5000)
        try_set(g)
    
    # Phase 2: Exhaustive search for very small sets
    # Limit M to 10 to keep 2^M manageable (1024)
    for M in range(1, 11):
        if time.time() > deadline - 20:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 20:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
    
    # Phase 2b: Try sets with large gaps (sparse sets)
    # These can have |U-U| >> |U+U|
    for max_el in [100, 200, 500, 1000, 2000, 5000]:
        if time.time() > deadline - 15:
            break
        # Random sparse sets
        for trial in range(20):
            if time.time() > deadline - 15:
                break
            n_el = random.randint(3, min(12, max_el // 10 + 3))
            cand = {0}
            while len(cand) < n_el:
                cand.add(random.randint(1, max_el))
            try_set(list(cand))
    
    # Phase 3: Local search from best found
    current = list(best_set)
    no_improve = 0
    max_no_improve = 500
    
    # Helper: generate smart candidates for addition
    def gen_add_candidates(cur_set, max_cur):
        candidates = set()
        cur_set_s = set(cur_set)
        
        # 1. Points just above max
        for delta in range(1, min(10, max_cur // 3 + 5)):
            candidates.add(max_cur + delta)
        
        # 2. Geometric extensions
        if max_cur > 0:
            for ratio in [2, 3, 4]:
                candidates.add(max_cur * ratio)
                candidates.add(max_cur * ratio + 1)
                candidates.add(max_cur * ratio - 1)
        
        # 3. Random large jumps
        for _ in range(10):
            candidates.add(random.randint(max_cur + 1, max_cur * 2 + 50))
        
        # 4. Small integers not yet in set (limited)
        for i in range(1, min(20, max_cur + 1)):
            if i not in cur_set_s:
                candidates.add(i)
        
        return candidates
    
    # Helper: generate candidates for replacement
    def gen_replace_candidates(cur_set, idx, max_cur):
        candidates = set()
        cur_set_s = set(cur_set)
        old_el = cur_set[idx]
        
        # Nearby values
        for delta in range(-10, 11):
            new_el = old_el + delta
            if new_el >= 1 and new_el != old_el and new_el not in cur_set_s:
                candidates.add(new_el)
        
        # Geometric relations to other elements
        for other in cur_set:
            if other == 0 or other == old_el:
                continue
            for ratio in [2, 3]:
                candidates.add(other * ratio)
                candidates.add(other * ratio + 1)
        
        # Random
        for _ in range(5):
            candidates.add(random.randint(1, max_cur * 2 + 50))
        
        return candidates
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        n_cur = len(current)
        
        # 1. Try adding an element
        candidates_to_add = gen_add_candidates(current, max_cur)
        
        for new_el in candidates_to_add:
            if new_el in current or new_el < 1:
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
        
        # 2. Try replacing an element
        if len(current) >= 3:
            for i in range(1, len(current)):  # Skip 0
                old_el = current[i]
                candidates = gen_replace_candidates(current, i, max_cur)
                
                for new_el in candidates:
                    if new_el < 1 or new_el in current or new_el == old_el:
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
        
        # 3. Try removing an element
        if len(current) > 3:
            for i in range(1, len(current)):  # Skip 0
                cand = [x for j, x in enumerate(current) if j != i]
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation to escape local minima
            if no_improve % 5 == 0:
                perturbation_type = random.random()
                
                if perturbation_type < 0.3 and len(current) > 2:
                    # Swap two elements
                    i = random.randint(1, len(current) - 1)
                    j = random.randint(1, len(current) - 1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand and len(cand) == len(current):
                            try_set(cand)
                            current = list(best_set)
                
                elif perturbation_type < 0.6:
                    # Add a random element
                    new_el = random.randint(1, max_cur + 100)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)
                
                elif perturbation_type < 0.8:
                    # Remove a random element
                    if len(current) > 3:
                        idx = random.randint(1, len(current) - 1)
                        cand = [x for j, x in enumerate(current) if j != idx]
                        try_set(cand)
                        current = list(best_set)
                
                else:
                    # Large random addition
                    new_el = random.randint(max_cur + 1, max_cur * 2 + 100)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)
        
        # Periodic restart from a different starting point
        if no_improve > 100 and no_improve % 50 == 0:
            # Try a fresh random start
            n_el = random.randint(4, 15)
            max_val = random.randint(100, 2000)
            cand = {0}
            while len(cand) < n_el:
                cand.add(random.randint(1, max_val))
            try_set(list(cand))
            current = list(best_set)
            no_improve = 0

    return list(best_set)
