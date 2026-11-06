import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = np.arange(target_lo, target_hi + 1)
    
    def get_counts(A):
        """Calculate representation counts for all target sums."""
        counts = np.zeros(target_hi + 1, dtype=np.int32)
        A_arr = np.array(A, dtype=np.int32)
        for i in range(len(A_arr)):
            for j in range(i, len(A_arr)):
                s = A_arr[i] + A_arr[j]
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts
    
    def get_score(counts):
        """Return (uncovered_count, peak_count)."""
        target_counts = counts[target_lo:target_hi+1]
        uncovered = int(np.sum(target_counts == 0))
        peak = int(np.max(target_counts)) if len(target_counts) > 0 else 0
        return (uncovered, peak)
    
    def get_variance(counts):
        """Return variance of target counts (for balance polish)."""
        target_counts = counts[target_lo:target_hi+1].astype(np.float64)
        if len(target_counts) == 0:
            return 0.0
        return float(np.var(target_counts))
    
    def apply_swap(counts, A_set, x, y):
        """
        Apply swap: remove x, add y. Modifies counts in place.
        Assumes x in A_set, y not in A_set.
        """
        # Remove x
        for z in A_set:
            if z == x:
                continue
            s = x + z
            if target_lo <= s <= target_hi:
                counts[s] -= 1
        s2x = 2 * x
        if target_lo <= s2x <= target_hi:
            counts[s2x] -= 1
        
        # Add y (with the set already having x removed, so z != y is automatic if y not in set)
        for z in A_set:
            if z == y:
                continue
            s = y + z
            if target_lo <= s <= target_hi:
                counts[s] += 1
        s2y = 2 * y
        if target_lo <= s2y <= target_hi:
            counts[s2y] += 1
    
    def rollback_swap(counts, A_set, x, y):
        """
        Rollback swap: remove y, add x. Modifies counts in place.
        Assumes y in A_set, x not in A_set.
        """
        # Remove y
        for z in A_set:
            if z == y:
                continue
            s = y + z
            if target_lo <= s <= target_hi:
                counts[s] -= 1
        s2y = 2 * y
        if target_lo <= s2y <= target_hi:
            counts[s2y] -= 1
        
        # Add x
        for z in A_set:
            if z == x:
                continue
            s = x + z
            if target_lo <= s <= target_hi:
                counts[s] += 1
        s2x = 2 * x
        if target_lo <= s2x <= target_hi:
            counts[s2x] += 1
    
    best_A = None
    best_score = (float('inf'), float('inf'))
    best_var = float('inf')
    
    # Phase 1: Generate structured initial candidates
    candidates = []
    
    # 1. Even numbers in range
    candidates.append(list(range(64, 129, 2)))
    
    # 2. Multiples of small integers
    candidates.append(list(range(0, 129, 2)))
    candidates.append(list(range(0, 129, 3)))
    
    # 3. Random sets of various sizes
    for k in [30, 35, 40, 45, 50]:
        for _ in range(5):
            candidates.append(random.sample(range(n + 1), k))
    
    # 4. Greedy balanced construction
    def greedy_balanced_build(target_size=45):
        """Build a set greedily, preferring elements that reduce peak/variance."""
        A = set()
        base_elements = random.sample(range(0, n + 1), min(10, target_size))
        A.update(base_elements)
        
        counts = get_counts(list(A))
        
        while len(A) < target_size:
            if time.time() > deadline - 5:
                break
            best_y = None
            best_metric = None
            
            candidates_y = [y for y in range(n + 1) if y not in A]
            random.shuffle(candidates_y)
            candidates_y = candidates_y[:30]
            
            for y in candidates_y:
                temp_counts = counts.copy()
                for z in A:
                    s = y + z
                    if target_lo <= s <= target_hi:
                        temp_counts[s] += 1
                s2y = 2 * y
                if target_lo <= s2y <= target_hi:
                    temp_counts[s2y] += 1
                
                score = get_score(temp_counts)
                var = get_variance(temp_counts)
                
                metric = (score[0], score[1], var)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_y = y
            
            if best_y is not None:
                A.add(best_y)
                counts = get_counts(list(A))
        
        return list(A)
    
    for _ in range(3):
        candidates.append(greedy_balanced_build(45))
    
    # Evaluate all candidates and pick best few for local search
    scored_candidates = []
    for A in candidates:
        if time.time() > deadline - 5:
            break
        score = get_score(get_counts(A))
        var = get_variance(get_counts(A))
        scored_candidates.append((score, var, A))
    
    scored_candidates.sort(key=lambda x: (x[0][0], x[0][1], x[1]))
    
    # Phase 2: Enhanced local search with balance polish and SA escape
    def local_search_enhanced(initial_A, max_steps=5000):
        """Local search with swap moves, balance polish, and SA escape."""
        A = list(initial_A)
        A_set = set(A)
        counts = get_counts(A)
        curr_score = get_score(counts)
        curr_var = get_variance(counts)
        
        best_A_local = A[:]
        best_score_local = curr_score
        best_var_local = curr_var
        
        no_improve = 0
        max_no_improve = 300
        sa_temperature = 0.5
        sa_cooling = 0.995
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            # Decide move type
            use_sa = no_improve > max_no_improve and random.random() < sa_temperature
            use_polish = (not use_sa) and (no_improve > max_no_improve // 2)
            
            x = random.choice(A)
            y = random.randint(0, n)
            attempts = 0
            while y in A_set and attempts < 15:
                y = random.randint(0, n)
                attempts += 1
            if y in A_set:
                continue
            
            # Apply swap
            old_counts = counts.copy()
            apply_swap(counts, A_set, x, y)
            A_set.remove(x)
            A_set.add(y)
            A[A.index(x)] = y  # Keep list consistent (though we use set for logic)
            # Actually, let's keep A as a list and sync carefully
            # Let's just rebuild A from A_set for simplicity in tracking, or just use A_set
            # Wait, A.index(x) might fail if x is not in A? No, x was chosen from A.
            # But if we have duplicates? The problem says A is a subset, so no duplicates.
            
            new_score = get_score(counts)
            new_var = get_variance(counts)
            
            accept = False
            if new_score < curr_score:
                accept = True
                no_improve = 0
            elif new_score == curr_score:
                if use_polish:
                    if new_var < curr_var:
                        accept = True
                        no_improve = 0
                elif use_sa:
                    # Accept worse moves with probability
                    # Simple heuristic: accept if peak didn't increase, or random
                    if new_score[1] <= curr_score[1]:
                        accept = True
                    elif random.random() < sa_temperature:
                        accept = True
            else:
                if use_sa:
                    # Accept significantly worse moves with low prob to escape
                    if random.random() < sa_temperature * 0.1:
                        accept = True
            
            if accept:
                curr_score = new_score
                curr_var = new_var
                if curr_score < best_score_local or (curr_score == best_score_local and curr_var < best_var_local):
                    best_score_local = curr_score
                    best_var_local = curr_var
                    best_A_local = list(A_set)
            else:
                # Rollback
                counts = old_counts
                A_set.remove(y)
                A_set.add(x)
                A[A.index(y)] = x
                no_improve += 1
                sa_temperature *= sa_cooling
        
        return best_A_local, best_score_local, best_var_local
    
    # Run local search on top candidates
    for score, var, A in scored_candidates[:5]:
        if time.time() > deadline - 5:
            break
        
        A_res, score_res, var_res = local_search_enhanced(A, max_steps=3000)
        
        if score_res < best_score or (score_res == best_score and var_res < best_var):
            best_score = score_res
            best_var = var_res
            best_A = A_res
        
        # Early exit if we found a good solution
        if best_score[0] == 0 and best_score[1] <= 4:
            break
    
    # If still not good enough, try more aggressive search
    if best_score[0] > 0 or best_score[1] > 5:
        if time.time() < deadline - 10:
            # Try larger random sets with local search
            for k in [45, 50, 55]:
                if time.time() > deadline - 5:
                    break
                for _ in range(3):
                    if time.time() > deadline - 3:
                        break
                    init_A = random.sample(range(n + 1), k)
                    A_res, score_res, var_res = local_search_enhanced(init_A, max_steps=2000)
                    if score_res < best_score or (score_res == best_score and var_res < best_var):
                        best_score = score_res
                        best_var = var_res
                        best_A = A_res
                    if best_score[0] == 0 and best_score[1] <= 4:
                        break
                if best_score[0] == 0 and best_score[1] <= 4:
                    break
    
    if best_A is None:
        # Fallback
        return list(range(64, 129))
    
    return best_A
