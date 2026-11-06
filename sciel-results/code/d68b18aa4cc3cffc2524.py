import random
import time
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
    n = 128
    target_lo = 128
    target_hi = 256
    max_sum = 2 * n
    target_size = target_hi - target_lo + 1  # 129
    
    # Precompute pair contributions
    # For each element y, what sums does it contribute to when paired with each z?
    # We'll use a matrix approach for vectorization
    
    def compute_counts_vectorized(A_arr):
        """Compute sum counts using vectorized numpy operations.
        A_arr: numpy array of elements in A (sorted)
        Returns: counts array of size max_sum+1
        """
        m = len(A_arr)
        if m == 0:
            return np.zeros(max_sum + 1, dtype=np.int32)
        
        # All pairs (i,j) with i <= j
        # Use broadcasting
        a = A_arr[:, np.newaxis]  # (m, 1)
        b = A_arr[np.newaxis, :]  # (1, m)
        sums = a + b  # (m, m)
        
        # Only upper triangle (including diagonal) for unordered pairs
        mask = np.triu(np.ones((m, m), dtype=bool))
        valid_sums = sums[mask]
        
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        # Count occurrences
        np.add.at(counts, valid_sums, 1)
        
        return counts
    
    def get_score(counts):
        """Returns (uncovered, peak, sum_sq) - lower is better"""
        target_counts = counts[target_lo:target_hi+1]
        uncovered = np.sum(target_counts == 0)
        peak = np.max(target_counts) if len(target_counts) > 0 else 0
        sum_sq = np.sum(target_counts.astype(np.float64) ** 2)
        return (int(uncovered), int(peak), float(sum_sq))
    
    def score_cost(score):
        """Convert score tuple to single cost for comparison"""
        uncovered, peak, sum_sq = score
        return uncovered * 10000 + peak * 100 + sum_sq / 1000.0
    
    def try_add(A_set, counts, y):
        """Try adding y to A, return new counts and score improvement"""
        if y in A_set:
            return None
        # New counts: add y+z for all z in A_set, plus 2*y
        new_counts = counts.copy()
        A_list = list(A_set)
        for z in A_list:
            s = y + z
            if s <= max_sum:
                new_counts[s] += 1
        s2y = 2 * y
        if s2y <= max_sum:
            new_counts[s2y] += 1
        return new_counts
    
    def try_remove(A_set, counts, x):
        """Try removing x from A, return new counts"""
        if x not in A_set:
            return None
        new_counts = counts.copy()
        A_list = list(A_set)
        A_list.remove(x)
        for z in A_list:
            s = x + z
            if s <= max_sum:
                new_counts[s] -= 1
        s2x = 2 * x
        if s2x <= max_sum:
            new_counts[s2x] -= 1
        return new_counts
    
    def try_swap(A_set, counts, x, y):
        """Try swapping x with y in A"""
        if x not in A_set or y in A_set:
            return None
        new_counts = counts.copy()
        # Remove x
        A_list = list(A_set)
        A_list.remove(x)
        for z in A_list:
            s = x + z
            if s <= max_sum:
                new_counts[s] -= 1
        s2x = 2 * x
        if s2x <= max_sum:
            new_counts[s2x] -= 1
        # Add y
        A_list.append(y)
        for z in A_list:
            if z != y:
                s = y + z
                if s <= max_sum:
                    new_counts[s] += 1
        s2y = 2 * y
        if s2y <= max_sum:
            new_counts[s2y] += 1
        return new_counts
    
    # Fast score computation using numpy
    def fast_score(counts):
        tc = counts[target_lo:target_hi+1].astype(np.int32)
        uncovered = int(np.sum(tc == 0))
        peak = int(np.max(tc)) if len(tc) > 0 else 0
        return (uncovered, peak)
    
    # Load-balanced greedy construction
    def greedy_load_balanced(max_A_size=65, time_budget=8.0):
        """Build A greedily by always adding the element that best balances loads"""
        A = set()
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        # Start with a few random elements for diversity
        seed_size = random.randint(5, 15)
        seed = random.sample(range(n+1), seed_size)
        for y in seed:
            A.add(y)
        
        # Recompute counts
        A_list = sorted(A)
        A_arr = np.array(A_list, dtype=np.int32)
        counts = compute_counts_vectorized(A_arr)
        
        best_score = get_score(counts)
        
        start_greedy = time.time()
        
        for _ in range(max_A_size - len(A)):
            if time.time() - start_greedy > time_budget:
                break
            
            # Find the element to add that minimizes the resulting peak
            # Consider candidates that help with low-count sums
            current_counts = counts[target_lo:target_hi+1].astype(np.int32)
            min_count = np.min(current_counts)
            
            # Focus on elements that can help cover the lowest-count sums
            candidates = [y for y in range(n+1) if y not in A]
            random.shuffle(candidates)
            
            best_y = None
            best_new_counts = None
            best_new_score = None
            
            # Evaluate a subset of candidates
            eval_limit = 40
            for y in candidates[:eval_limit]:
                # Quick check: what sums would y help?
                helpful = 0
                for s in range(target_lo, target_hi+1):
                    if counts[s] <= min_count + 1:
                        z = s - y
                        if 0 <= z <= n and (z in A or z == y):
                            helpful += 1
                
                if helpful == 0 and min_count > 0:
                    continue
                
                # Full evaluation
                new_counts = counts.copy()
                for z in A:
                    s = y + z
                    if s <= max_sum:
                        new_counts[s] += 1
                s2y = 2 * y
                if s2y <= max_sum:
                    new_counts[s2y] += 1
                
                ns = get_score(new_counts)
                if best_new_score is None or ns < best_new_score:
                    best_new_score = ns
                    best_new_counts = new_counts
                    best_y = y
            
            if best_y is not None:
                A.add(best_y)
                counts = best_new_counts
                best_score = best_new_score
            else:
                break
        
        return A, counts, best_score
    
    # Local search with swap moves
    def local_search(A_init, counts_init, max_steps=5000, time_budget=15.0):
        """Local search using swap and add/remove moves"""
        A = set(A_init)
        counts = counts_init.copy()
        curr_score = get_score(counts)
        best_A = set(A)
        best_score = curr_score
        best_cost = score_cost(curr_score)
        
        start_ls = time.time()
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() - start_ls > time_budget:
                break
            
            uncovered, peak, sum_sq = curr_score
            
            if uncovered > 0:
                # Must add elements to cover gaps
                uncovered_sums = [s for s in range(target_lo, target_hi+1) if counts[s] == 0]
                s_target = random.choice(uncovered_sums)
                
                # Find candidates that can cover s_target
                candidates = []
                for y in range(max(0, s_target - n), min(n, s_target) + 1):
                    z = s_target - y
                    if y not in A and (z in A or z == y):
                        candidates.append(y)
                
                if candidates:
                    y = random.choice(candidates)
                    new_counts = counts.copy()
                    for z in A:
                        s = y + z
                        if s <= max_sum:
                            new_counts[s] += 1
                    s2y = 2 * y
                    if s2y <= max_sum:
                        new_counts[s2y] += 1
                    A.add(y)
                    new_score = get_score(new_counts)
                    
                    if new_score[0] < curr_score[0] or (new_score[0] == curr_score[0] and new_score < curr_score):
                        counts = new_counts
                        curr_score = new_score
                        if score_cost(curr_score) < best_cost:
                            best_score = curr_score
                            best_A = set(A)
                            best_cost = score_cost(curr_score)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        A.remove(y)
                        no_improve += 1
                else:
                    no_improve += 1
                    
            else:
                # Full coverage: minimize peak via swaps
                # Target the hottest sums
                hot_sums = [s for s in range(target_lo, target_hi+1) if counts[s] == peak]
                s_hot = random.choice(hot_sums)
                
                # Find pairs in A that sum to s_hot
                pairs_at_hot = []
                A_list = sorted(A)
                for a in A_list:
                    b = s_hot - a
                    if b >= a and b in A:
                        pairs_at_hot.append((a, b))
                
                if not pairs_at_hot:
                    no_improve += 1
                    continue
                
                a, b = random.choice(pairs_at_hot)
                
                # Try swapping a with a random y
                candidates = [y for y in range(n+1) if y not in A]
                random.shuffle(candidates)
                candidates = candidates[:25]
                
                best_swap_y = None
                best_swap_counts = None
                best_swap_score = None
                
                for y in candidates:
                    new_counts = try_swap(A, counts, a, y)
                    if new_counts is None:
                        continue
                    ns = get_score(new_counts)
                    # Accept if peak decreases or same peak with lower variance
                    if ns[0] == 0 and (ns[1] < peak or (ns[1] == peak and ns[2] < sum_sq)):
                        if best_swap_score is None or ns < best_swap_score:
                            best_swap_score = ns
                            best_swap_counts = new_counts
                            best_swap_y = y
                
                if best_swap_y is not None:
                    A.remove(a)
                    A.add(best_swap_y)
                    counts = best_swap_counts
                    curr_score = best_swap_score
                    if score_cost(curr_score) < best_cost:
                        best_score = curr_score
                        best_A = set(A)
                        best_cost = score_cost(curr_score)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 500:
                break
        
        return best_A, best_score
    
    # Simulated annealing with fast evaluation
    def run_sa(A_init, counts_init, max_steps=20000, T_start=50.0, T_end=0.5, alpha=0.9995, time_budget=20.0):
        A = set(A_init)
        counts = counts_init.copy()
        curr_score = get_score(counts)
        best_A = set(A)
        best_score = curr_score
        best_cost = score_cost(curr_score)
        
        T = T_start
        start_sa = time.time()
        
        for step in range(max_steps):
            if time.time() - start_sa > time_budget:
                break
            
            uncovered, peak, sum_sq = curr_score
            
            if uncovered > 0:
                p_add = 0.35
                p_swap = 0.55
                p_remove = 0.10
            else:
                p_add = 0.05
                p_swap = 0.65
                p_remove = 0.30
            
            r = random.random()
            
            if r < p_add:
                if len(A) < n + 1:
                    candidates = [i for i in range(n+1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        new_counts = counts.copy()
                        for z in A:
                            s = y + z
                            if s <= max_sum:
                                new_counts[s] += 1
                        s2y = 2 * y
                        if s2y <= max_sum:
                            new_counts[s2y] += 1
                        A.add(y)
                        new_score = get_score(new_counts)
                        
                        old_cost = score_cost(curr_score)
                        new_cost = score_cost(new_score)
                        d = new_cost - old_cost
                        
                        if d < 0 or random.random() < np.exp(-d / max(T, 0.01)):
                            curr_score = new_score
                            if score_cost(curr_score) < best_cost:
                                best_score = curr_score
                                best_A = set(A)
                                best_cost = score_cost(curr_score)
                        else:
                            A.remove(y)
                            
            elif r < p_add + p_swap:
                if len(A) > 0:
                    x = random.choice(list(A))
                    candidates = [i for i in range(n+1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        new_counts = try_swap(A, counts, x, y)
                        if new_counts is not None:
                            A.remove(x)
                            A.add(y)
                            counts = new_counts
                            new_score = get_score(counts)
                            
                            old_cost = score_cost(curr_score)
                            new_cost = score_cost(new_score)
                            d = new_cost - old_cost
                            
                            if d < 0 or random.random() < np.exp(-d / max(T, 0.01)):
                                curr_score = new_score
                                if score_cost(curr_score) < best_cost:
                                    best_score = curr_score
                                    best_A = set(A)
                                    best_cost = score_cost(curr_score)
                            else:
                                A.remove(y)
                                A.add(x)
                                counts = try_swap(A, counts, y, x)
                                if counts is None:
                                    counts = compute_counts_vectorized(np.array(sorted(A), dtype=np.int32))
                                curr_score = get_score(counts)
                            
            else:
                if len(A) > 5:
                    x = random.choice(list(A))
                    new_counts = try_remove(A, counts, x)
                    if new_counts is not None:
                        A.remove(x)
                        counts = new_counts
                        new_score = get_score(counts)
                        
                        old_cost = score_cost(curr_score)
                        new_cost = score_cost(new_score)
                        d = new_cost - old_cost
                        
                        if d < 0 or random.random() < np.exp(-d / max(T, 0.01)):
                            curr_score = new_score
                            if score_cost(curr_score) < best_cost:
                                best_score = curr_score
                                best_A = set(A)
                                best_cost = score_cost(curr_score)
                        else:
                            A.add(x)
                            counts = try_add(A, counts, x)
                            if counts is None:
                                counts = compute_counts_vectorized(np.array(sorted(A), dtype=np.int32))
                            curr_score = get_score(counts)
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_A, best_score
    
    # Main search
    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))
    best_cost = float('inf')
    
    # Phase 1: Generate diverse initial candidates
    inits = []
    
    # Load-balanced greedy constructions
    for trial in range(6):
        if time.time() > deadline - 30:
            break
        A_g, c_g, s_g = greedy_load_balanced(max_A_size=60, time_budget=5.0)
        inits.append((s_g, A_g, c_g))
    
    # Structured: arithmetic progressions
    for step in [2, 3, 4]:
        for start in range(step):
            A_ap = set(range(start, n+1, step))
            A_arr = np.array(sorted(A_ap), dtype=np.int32)
            c_ap = compute_counts_vectorized(A_arr)
            s_ap = get_score(c_ap)
            inits.append((s_ap, A_ap, c_ap))
    
    # Random sets of various sizes
    for k in [40, 50, 55, 60]:
        A_rand = set(random.sample(range(n+1), k))
        A_arr = np.array(sorted(A_rand), dtype=np.int32)
        c_rand = compute_counts_vectorized(A_arr)
        s_rand = get_score(c_rand)
        inits.append((s_rand, A_rand, c_rand))
    
    # Middle-weighted random
    for _ in range(3):
        A_mw = set()
        for i in range(55):
            if i < 15:
                A_mw.add(random.randint(0, 32))
            elif i < 35:
                A_mw.add(random.randint(32, 96))
            else:
                A_mw.add(random.randint(96, 128))
        A_arr = np.array(sorted(A_mw), dtype=np.int32)
        c_mw = compute_counts_vectorized(A_arr)
        s_mw = get_score(c_mw)
        inits.append((s_mw, A_mw, c_mw))
    
    # Sort by score
    inits.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on top inits
    for i, (s_init, A_init, c_init) in enumerate(inits[:6]):
        if time.time() > deadline - 25:
            break
        A_ls, s_ls = local_search(A_init, c_init, max_steps=4000, time_budget=8.0)
        cost_ls = score_cost(s_ls)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_score = s_ls
            best_A = A_ls
    
    # Phase 3: SA on the best result
    if best_A is not None and time.time() < deadline - 12:
        A_arr = np.array(sorted(best_A), dtype=np.int32)
        c_best = compute_counts_vectorized(A_arr)
        A_sa, s_sa = run_sa(best_A, c_best, max_steps=15000, T_start=40.0, T_end=0.3, alpha=0.9997, time_budget=18.0)
        cost_sa = score_cost(s_sa)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_score = s_sa
            best_A = A_sa
    
    # Phase 4: Final local search polish
    if best_A is not None and time.time() < deadline - 5:
        A_arr = np.array(sorted(best_A), dtype=np.int32)
        c_best = compute_counts_vectorized(A_arr)
        A_final, s_final = local_search(best_A, c_best, max_steps=3000, time_budget=10.0)
        cost_final = score_cost(s_final)
        if cost_final < best_cost:
            best_cost = cost_final
            best_score = s_final
            best_A = A_final
    
    # Fallback
    if best_A is None:
        # Return a valid covering set
        A_fb = set(range(0, n+1, 2)) | set(range(1, n+1, 4))
        # Ensure coverage
        counts_fb = np.zeros(max_sum + 1, dtype=np.int32)
        A_list = sorted(A_fb)
        A_arr = np.array(A_list, dtype=np.int32)
        counts_fb = compute_counts_vectorized(A_arr)
        while any(counts_fb[s] == 0 for s in range(target_lo, target_hi+1)):
            for y in range(n+1):
                if y not in A_fb:
                    A_fb.add(y)
                    A_list = sorted(A_fb)
                    A_arr = np.array(A_list, dtype=np.int32)
                    counts_fb = compute_counts_vectorized(A_arr)
                    break
        best_A = A_fb
    
    return list(best_A)
