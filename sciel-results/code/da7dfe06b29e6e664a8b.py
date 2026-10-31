import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        """Compute the score for a set U"""
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        n = len(U_arr)
        
        # For small sets, use direct computation
        if n <= 30:
            sums = U_arr[:, None] + U_arr[None, :]
            diffs = U_arr[:, None] - U_arr[None, :]
            n_sum = len(np.unique(sums))
            n_diff = len(np.unique(diffs))
        else:
            # For larger sets, use FFT-based approach
            # Create binary indicator vector
            max_val = max_u
            size = 2 * max_val + 1
            indicator = np.zeros(size, dtype=np.float64)
            indicator[U_arr] = 1.0
            
            # Sumset: |U+U| is the number of positions where convolution of indicator with itself is > 0
            # Difference set: |U-U| is the number of positions where cross-correlation is > 0
            
            # For sumset: convolve indicator with itself
            # The sum u+v ranges from 0 to 2*max_u
            # We need to count distinct sums
            sumset = np.zeros(2 * max_val + 1, dtype=np.int64)
            for u in U_arr:
                sumset[u] += 1
            
            # Actually, for large sets, let's just use numpy unique but be smart
            # Generate all pairs more efficiently
            # Use the fact that we only need unique values
            U_sorted = np.sort(U_arr)
            
            # For sums: generate upper triangle
            n_half = (n * (n + 1)) // 2
            sums = np.empty(n_half, dtype=np.int64)
            idx = 0
            for i in range(n):
                sums[idx:idx+n-i] = U_sorted[i] + U_sorted[i:]
                idx += n - i
            n_sum = len(np.unique(sums))
            
            # For differences: all pairs
            diffs = U_sorted[:, None] - U_sorted[None, :]
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

    # Phase 1: Try structured sets with different strategies
    
    # Geometric progressions with various bases
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        g = [0, 1]
        while g[-1] * base < 20000:
            g.append(g[-1] * base)
        try_set(g)
    
    # Sparse arithmetic progressions
    for start in range(1, 10):
        for step in [3, 4, 5, 7, 8, 10, 13, 16, 20]:
            a = [0]
            val = start
            while val < 5000:
                a.append(val)
                val += step
            try_set(a)
    
    # Golomb ruler-like sets (greedy construction) - improved version
    def generate_golomb_like(n, max_val=10000, seed_val=1):
        """Generate a set that tries to have unique differences"""
        U = [0, seed_val]
        U_set = set(U)
        existing_diffs = set()
        for u in U:
            for v in U:
                existing_diffs.add(u - v)
        
        for _ in range(2, n):
            best_pos = -1
            best_score_local = -1
            best_new_diffs = 0
            # Try positions more sparsely for larger n
            step = 1 if max_val < 500 else 2
            for pos in range(max(U) + 1, max_val + 1, step):
                if pos in U_set:
                    continue
                # Check if this creates new differences
                new_diffs_count = 0
                for u in U:
                    d1 = pos - u
                    d2 = u - pos
                    if d1 not in existing_diffs:
                        new_diffs_count += 1
                    if d2 not in existing_diffs:
                        new_diffs_count += 1
                if new_diffs_count > best_score_local:
                    best_score_local = new_diffs_count
                    best_pos = pos
                    best_new_diffs = new_diffs_count
                if best_score_local >= 2 * len(U) - 1:  # Maximum possible
                    break
            if best_pos > 0:
                U.append(best_pos)
                U_set.add(best_pos)
                for u in U:
                    if u != best_pos:
                        existing_diffs.add(best_pos - u)
                        existing_diffs.add(u - best_pos)
            else:
                break
        return U
    
    # Generate Golomb-like sets with different parameters
    for n in [4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        for seed_val in [1, 2, 3, 5]:
            g = generate_golomb_like(n, 20000, seed_val)
            try_set(g)
    
    # Phase 2: Exhaustive search for small sets
    if time.time() < deadline - 30:
        for M in range(1, 13):
            if time.time() > deadline - 30:
                break
            for mask in range(1, 1 << M):
                if time.time() > deadline - 30:
                    break
                cand = [0]
                for i in range(M):
                    if mask & (1 << i):
                        cand.append(i + 1)
                try_set(cand)
    
    # Phase 3: More aggressive local search with multiple restarts
    initial_sets = []
    
    # Start from best found
    initial_sets.append(list(best_set))
    
    # Start from various structured sets
    for base in [2, 3, 5]:
        g = [0, 1]
        while g[-1] * base < 3000:
            g.append(g[-1] * base)
        initial_sets.append(g)
    
    for step in [3, 5, 7]:
        a = [0]
        val = 1
        while val < 2000:
            a.append(val)
            val += step
        initial_sets.append(a)
    
    # Random sparse sets
    for _ in range(10):
        n = random.randint(5, 20)
        max_val = random.randint(100, 2000)
        cand = {0}
        for _ in range(n - 1):
            cand.add(random.randint(1, max_val))
        initial_sets.append(sorted(cand))
    
    # Local search function
    def local_search(initial, max_iterations=200):
        current = list(initial)
        cur_score = compute_score(current)
        if cur_score <= -1:
            return current, cur_score
        
        no_improve = 0
        max_no_improve = 50
        
        for it in range(max_iterations):
            if time.time() > deadline - 15:
                break
            
            improved = False
            max_cur = max(current)
            n_cur = len(current)
            
            # 1. Try adding an element
            candidates_to_add = set()
            # Small integers
            for i in range(1, min(max_cur + 30, 2000)):
                if i not in current:
                    candidates_to_add.add(i)
            # Gaps
            for i in range(1, n_cur - 1):
                gap_start = current[i] + 1
                gap_end = current[i + 1] - 1
                for j in range(gap_start, min(gap_end + 1, gap_start + 10)):
                    if j not in current:
                        candidates_to_add.add(j)
            # Large jumps
            for _ in range(30):
                candidates_to_add.add(random.randint(max_cur + 1, max_cur + 300))
            
            for new_el in candidates_to_add:
                if new_el in current or new_el < 1:
                    continue
                cand = sorted(current + [new_el])
                s = compute_score(cand)
                if s > cur_score:
                    cur_score = s
                    current = cand
                    improved = True
                    no_improve = 0
                    break
            
            # 2. Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    for delta in range(-20, 21):
                        new_el = old_el + delta
                        if new_el < 1 or new_el in current or new_el == old_el:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > cur_score:
                            cur_score = s
                            current = cand
                            improved = True
                            no_improve = 0
                            break
                    if improved:
                        break
            
            # 3. Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > cur_score:
                        cur_score = s
                        current = cand
                        improved = True
                        no_improve = 0
                        break
            
            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                # Random perturbation
                if len(current) > 2:
                    i = random.randint(1, len(current) - 1)
                    new_val = random.randint(1, max_cur + 100)
                    cand = list(current)
                    cand[i] = new_val
                    cand = sorted(set(cand))
                    if 0 in cand:
                        s = compute_score(cand)
                        if s > cur_score * 0.95:  # Accept worse with some probability
                            cur_score = s
                            current = cand
                        elif random.random() < 0.1:
                            cur_score = s
                            current = cand
                else:
                    new_el = random.randint(1, max_cur + 200)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        s = compute_score(cand)
                        if s > cur_score or random.random() < 0.1:
                            cur_score = s
                            current = cand
        
        return current, cur_score
    
    # Run local search from multiple initial points
    for init in initial_sets:
        if time.time() > deadline - 20:
            break
        result, score = local_search(init)
        if score > best_score:
            best_score = score
            best_set = result
    
    # Final polish: try more aggressive moves on best set
    if time.time() < deadline - 10 and len(best_set) > 2:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            i = random.randint(1, len(best_set) - 1)
            old = best_set[i]
            new = random.randint(1, max(best_set) + 500)
            if new != old:
                cand = list(best_set)
                cand[i] = new
                cand = sorted(set(cand))
                if 0 in cand:
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = cand
    
    return list(best_set)
