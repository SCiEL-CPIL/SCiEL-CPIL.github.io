import numpy as np
import random
import time
import itertools

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Generate all words as index arrays
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices and ball_sets
    ball_arrays = []
    ball_sets = []
    
    # Precompute base powers
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    for i in range(total):
        w = words[i]
        ball = [i]
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball.append(new_idx)
        ball_arrays.append(np.array(ball, dtype=np.int32))
        ball_sets.append(set(ball))

    # Helper to check validity
    def is_valid_cover(code):
        if not code:
            return False
        covered = np.zeros(total, dtype=bool)
        for c in code:
            covered[ball_arrays[c]] = True
        return np.all(covered)

    # Greedy Cover with randomization for diversity
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total, dtype=bool)
        code = set()
        remaining = np.ones(total, dtype=bool)
        
        order = list(range(total))
        random.shuffle(order)
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                ball = ball_arrays[c]
                new_cov = np.count_nonzero(remaining[ball])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            if best_c == -1:
                break
            code.add(best_c)
            remaining[ball_arrays[best_c]] = False
            
        return code

    # Helper to get uncovered points
    def get_uncovered(code, cnt):
        return np.where(cnt == 0)[0]

    # Local Search with 2-for-2 swaps
    def local_search(initial_code, cnt, deadline, max_streak=2000):
        current_code = set(initial_code)
        cnt = cnt.copy()
        best_size = len(current_code)
        best_code_snapshot = set(current_code)
        best_cnt_snapshot = cnt.copy()
        
        no_improve_streak = 0
        
        while time.time() < deadline - 2.0 and no_improve_streak < max_streak:
            improved = False
            
            # 1. Delete Phase - remove redundant codewords
            redundant_found = False
            code_list = list(current_code)
            random.shuffle(code_list)
            
            for c in code_list:
                ball = ball_arrays[c]
                if np.all(cnt[ball] >= 2):
                    current_code.remove(c)
                    cnt[ball] -= 1
                    improved = True
                    redundant_found = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code_snapshot = set(current_code)
                        best_cnt_snapshot = cnt.copy()
                    break
            
            if redundant_found:
                no_improve_streak = 0
                continue
            
            # 2. 1-for-1 Swap Phase (existing)
            min_c = -1
            min_uc = 100
            for c in current_code:
                ball = ball_arrays[c]
                uc = np.sum(cnt[ball] == 1)
                if uc > 0 and uc < min_uc:
                    min_uc = uc
                    min_c = c
                    
            swap_performed = False
            
            if min_c != -1:
                U = [p for p in ball_arrays[min_c] if cnt[p] == 1]
                cand_set = set(ball_sets[U[0]])
                for u in U[1:]:
                    cand_set &= set(ball_sets[u])
                cand_set -= current_code
                
                if cand_set:
                    c_new = random.choice(list(cand_set))
                    current_code.remove(min_c)
                    current_code.add(c_new)
                    cnt[ball_arrays[min_c]] -= 1
                    cnt[ball_arrays[c_new]] += 1
                    swap_performed = True
                    improved = True
            
            # 3. 2-for-2 Swap Phase (NEW) - escape local optima
            if not swap_performed and time.time() < deadline - 5.0:
                # Try to find two codewords to remove and two to add
                code_list = list(current_code)
                if len(code_list) >= 2:
                    # Sample pairs to avoid O(n^2)
                    num_trials = min(50, len(code_list) * 2)
                    for _ in range(num_trials):
                        idx1, idx2 = random.sample(range(len(code_list)), 2)
                        c1, c2 = code_list[idx1], code_list[idx2]
                        
                        # Check if removing both leaves some uncovered
                        ball1 = ball_arrays[c1]
                        ball2 = ball_arrays[c2]
                        
                        # Points that would become uncovered if we remove c1 and c2
                        # A point p is uncovered if cnt[p] == 1 and p is in ball1 XOR ball2 (only covered by one of them)
                        # Actually, p is uncovered if (p in ball1 and cnt[p]==1) OR (p in ball2 and cnt[p]==1) AND not both
                        # More precisely: after removal, cnt'[p] = cnt[p] - (1 if p in ball1) - (1 if p in ball2)
                        # We need cnt'[p] >= 1 for all p, i.e., cnt[p] - (1 if p in ball1) - (1 if p in ball2) >= 1
                        # So for p in ball1 ∩ ball2: need cnt[p] >= 3
                        # For p in ball1 \ ball2: need cnt[p] >= 2
                        # For p in ball2 \ ball1: need cnt[p] >= 2
                        # For p not in ball1 ∪ ball2: need cnt[p] >= 1 (always true if currently covered)
                        
                        valid = True
                        # Check intersection
                        inter = set(ball1) & set(ball2)
                        for p in inter:
                            if cnt[p] < 3:
                                valid = False
                                break
                        
                        if valid:
                            # Check symmetric difference
                            sym_diff = set(ball1) ^ set(ball2)
                            for p in sym_diff:
                                if cnt[p] < 2:
                                    valid = False
                                    break
                        
                        if valid:
                            # Find candidates to add
                            # We need to cover all points that were only covered by c1 or c2
                            # Uncovered after removal: points where cnt'[p] == 0
                            # These are points p where cnt[p] == 1 and p is in (ball1 \ ball2) or (ball2 \ ball1)
                            # Actually, if cnt[p]==1 and p in ball1 only, then after removing c1, cnt'[p]=0
                            # Similarly for ball2 only
                            
                            # Let's find the set of points that MUST be covered by new codewords
                            must_cover = set()
                            for p in set(ball1) - set(ball2):
                                if cnt[p] == 1:
                                    must_cover.add(p)
                            for p in set(ball2) - set(ball1):
                                if cnt[p] == 1:
                                    must_cover.add(p)
                            
                            if not must_cover:
                                # No points become uncovered, so we can just remove both
                                current_code.discard(c1)
                                current_code.discard(c2)
                                cnt[ball1] -= 1
                                cnt[ball2] -= 1
                                improved = True
                                if len(current_code) < best_size:
                                    best_size = len(current_code)
                                    best_code_snapshot = set(current_code)
                                    best_cnt_snapshot = cnt.copy()
                                swap_performed = True
                                break
                            
                            # Find candidate codewords that cover all must_cover points
                            # A candidate c_new covers must_cover if must_cover ⊆ ball(c_new)
                            # We need two such candidates (or one that covers all)
                            
                            # First try: find one codeword that covers all must_cover
                            all_candidates = []
                            for c_new in range(total):
                                if c_new in current_code or c_new == c1 or c_new == c2:
                                    continue
                                if must_cover.issubset(ball_sets[c_new]):
                                    all_candidates.append(c_new)
                            
                            if len(all_candidates) >= 1:
                                # Pick one or two
                                if len(all_candidates) >= 2:
                                    c_new1, c_new2 = random.sample(all_candidates, 2)
                                else:
                                    c_new1 = all_candidates[0]
                                    c_new2 = c_new1  # Will handle below
                                
                                if c_new1 != c_new2:
                                    current_code.discard(c1)
                                    current_code.discard(c2)
                                    current_code.add(c_new1)
                                    current_code.add(c_new2)
                                    cnt[ball_arrays[c1]] -= 1
                                    cnt[ball_arrays[c2]] -= 1
                                    cnt[ball_arrays[c_new1]] += 1
                                    cnt[ball_arrays[c_new2]] += 1
                                    improved = True
                                    if len(current_code) < best_size:
                                        best_size = len(current_code)
                                        best_code_snapshot = set(current_code)
                                        best_cnt_snapshot = cnt.copy()
                                    swap_performed = True
                                    break
                                else:
                                    # Only one candidate, use it twice? No, must be distinct.
                                    # Skip this case
                                    pass
                            elif len(all_candidates) == 1:
                                # Only one candidate, can't do 2-for-2 with distinct codewords
                                # Could do 2-for-1 but that's different
                                pass
            
            # 4. Random Swap Phase (fallback)
            if not swap_performed:
                for _ in range(50):
                    c_old = random.choice(list(current_code))
                    c_new = random.randint(0, total - 1)
                    if c_new in current_code:
                        continue
                    
                    diff = ball_sets[c_old] - ball_sets[c_new]
                    if diff:
                        valid = True
                        for p in diff:
                            if cnt[p] < 2:
                                valid = False
                                break
                        if not valid:
                            continue
                    
                    current_code.remove(c_old)
                    current_code.add(c_new)
                    cnt[ball_arrays[c_old]] -= 1
                    cnt[ball_arrays[c_new]] += 1
                    swap_performed = True
                    improved = True
                    break
            
            if not improved:
                no_improve_streak += 1
            else:
                no_improve_streak = 0
        
        # Final cleanup
        while True:
            deleted = False
            for c in list(current_code):
                ball = ball_arrays[c]
                if np.all(cnt[ball] >= 2):
                    current_code.remove(c)
                    cnt[ball] -= 1
                    deleted = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code_snapshot = set(current_code)
                        best_cnt_snapshot = cnt.copy()
            if not deleted:
                break
        
        return best_code_snapshot, best_cnt_snapshot

    # Main: Run multiple restarts with different seeds
    best_overall = None
    best_overall_size = float('inf')
    
    num_restarts = 5
    restart_time_budget = (deadline - t0 - 10.0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Greedy with different seed
        init_code = greedy_cover(seed=r * 12345 + 67890)
        
        if not init_code:
            continue
        
        # Build counts
        cnt = np.zeros(total, dtype=np.int32)
        for c in init_code:
            cnt[ball_arrays[c]] += 1
        
        # Local search
        search_deadline = min(t0 + restart_time_budget * (r + 1) + 10.0, deadline - 5.0)
        found_code, found_cnt = local_search(init_code, cnt, search_deadline)
        
        if found_code and len(found_code) < best_overall_size:
            # Validate
            if is_valid_cover(found_code):
                best_overall = found_code
                best_overall_size = len(found_code)
    
    # If no valid result from restarts, fallback to single greedy
    if best_overall is None:
        best_overall = greedy_cover()
        if not is_valid_cover(best_overall):
            # Critical failure
            best_overall = set(range(total))
    
    # Convert back to list of lists
    result = []
    for idx in best_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
