import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 128
    target_lo = 128
    target_hi = 256
    max_sum = 2 * n

    def get_score(counts, A_size):
        """
        Returns (uncovered, peak, variance_sum)
        Lower is better.
        """
        target_counts = counts[target_lo:target_hi+1]
        uncovered = 0
        peak = 0
        sum_sq = 0
        total = 0
        for c in target_counts:
            if c == 0:
                uncovered += 1
            if c > peak:
                peak = c
            sum_sq += c * c
            total += c
        
        return (uncovered, peak, sum_sq)

    def init_counts(A):
        counts = [0] * (max_sum + 1)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                counts[s] += 1
        return counts

    def add_element(A, counts, y):
        """Add y to A and update counts. Returns new A set."""
        A.add(y)
        for z in A:
            s = y + z
            if s <= max_sum:
                counts[s] += 1
        return A

    def remove_element(A, counts, x):
        """Remove x from A and update counts. Returns new A set."""
        A.remove(x)
        for z in A:
            s = x + z
            if s <= max_sum:
                counts[s] -= 1
        s_xx = 2 * x
        if s_xx <= max_sum:
            counts[s_xx] -= 1
        return A

    def swap_element(A, counts, x, y):
        """Swap x with y in A. x in A, y not in A."""
        # Remove x
        A.remove(x)
        for z in A:
            s = x + z
            if s <= max_sum:
                counts[s] -= 1
        s_xx = 2 * x
        if s_xx <= max_sum:
            counts[s_xx] -= 1
            
        # Add y
        A.add(y)
        for z in A:
            s = y + z
            if s <= max_sum:
                counts[s] += 1
        return A

    def compute_target_demand(A):
        """
        For each target sum s in [target_lo, target_hi], compute the number of
        pairs (a, b) with a <= b, a, b in A, a + b = s.
        Returns dict {s: count}.
        """
        demand = {}
        A_list = sorted(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    demand[s] = demand.get(s, 0) + 1
        return demand

    def greedy_edge_picking(max_iters=200, max_A_size=80):
        """
        Greedy load-balancing: start with a random seed set, then iteratively
        add the element that best reduces the peak target-sum multiplicity
        while maintaining full coverage.
        """
        A = set(random.sample(range(0, n + 1), min(15, max_A_size)))
        counts = init_counts(A)
        score = get_score(counts, len(A))
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if len(A) >= max_A_size:
                break
            
            uncovered, peak, sum_sq = score
            
            # If still uncovered, prioritize coverage
            if uncovered > 0:
                # Find uncovered sums
                uncovered_sums = [s for s in range(target_lo, target_hi + 1) if counts[s] == 0]
                # For each candidate y, check if it helps cover uncovered sums
                best_y = None
                best_key = None
                candidates = [y for y in range(n + 1) if y not in A]
                random.shuffle(candidates)
                candidates = candidates[:40]
                
                for y in candidates:
                    # Count how many uncovered sums y can help cover
                    help_count = 0
                    for s in uncovered_sums:
                        # y + z = s => z = s - y, need z in A or z == y
                        z = s - y
                        if z in A or (z == y and y in A):
                            help_count += 1
                        elif z == y:
                            # 2y = s, need y in A (it is being added)
                            help_count += 1
                    # Also penalize if y increases peak too much
                    temp_peak = peak
                    for z in A:
                        s = y + z
                        if target_lo <= s <= target_hi:
                            if counts[s] + 1 > temp_peak:
                                temp_peak = counts[s] + 1
                    s2y = 2 * y
                    if target_lo <= s2y <= target_hi:
                        if counts[s2y] + 1 > temp_peak:
                            temp_peak = counts[s2y] + 1
                    
                    key = (-help_count, temp_peak)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_y = y
            else:
                # Full coverage: minimize peak
                best_y = None
                best_key = None
                candidates = [y for y in range(n + 1) if y not in A]
                random.shuffle(candidates)
                candidates = candidates[:50]
                
                for y in candidates:
                    # Simulate adding y
                    temp_counts = counts[:]
                    for z in A:
                        s = y + z
                        if s <= max_sum:
                            temp_counts[s] += 1
                    s2y = 2 * y
                    if s2y <= max_sum:
                        temp_counts[s2y] += 1
                    
                    temp_score = get_score(temp_counts, len(A) + 1)
                    key = temp_score
                    if best_key is None or key < best_key:
                        best_key = key
                        best_y = y
            
            if best_y is not None:
                add_element(A, counts, best_y)
                score = get_score(counts, len(A))
        
        return A, counts, score

    def hot_sum_local_search(A, counts, max_steps=10000):
        """
        Local search focused on reducing peak by targeting the hottest sums.
        For each hot sum s (high count), try to find a pair (a, b) in A with a+b=s
        and replace one of them with an element that reduces the load on s.
        """
        A = set(A)
        counts = counts[:]
        curr_score = get_score(counts, len(A))
        best_A_local = set(A)
        best_score_local = curr_score
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            uncovered, peak, sum_sq = curr_score
            
            if uncovered > 0:
                # Try to add an element that covers uncovered sums
                uncovered_sums = [s for s in range(target_lo, target_hi + 1) if counts[s] == 0]
                if not uncovered_sums:
                    break
                s_hot = random.choice(uncovered_sums)
                # Find candidates y such that y + (s_hot - y) covers s_hot
                candidates = []
                for y in range(max(0, s_hot - n), min(n, s_hot) + 1):
                    z = s_hot - y
                    if y not in A and z in A:
                        candidates.append(y)
                    elif y not in A and y == z:
                        candidates.append(y)
                if candidates:
                    y = random.choice(candidates)
                    add_element(A, counts, y)
                    new_score = get_score(counts, len(A))
                    if new_score[0] < curr_score[0] or (new_score[0] == curr_score[0] and new_score < curr_score):
                        curr_score = new_score
                        if curr_score < best_score_local:
                            best_score_local = curr_score
                            best_A_local = set(A)
                    else:
                        remove_element(A, counts, y)
            else:
                # Full coverage: target the hottest sum
                # Find sums with count == peak
                hot_sums = [s for s in range(target_lo, target_hi + 1) if counts[s] == peak]
                if not hot_sums:
                    break
                s_hot = random.choice(hot_sums)
                
                # Find all pairs in A that sum to s_hot
                pairs_at_hot = []
                A_list = sorted(A)
                for i in range(len(A_list)):
                    a = A_list[i]
                    b = s_hot - a
                    if b >= a and b in A:
                        pairs_at_hot.append((a, b))
                
                if not pairs_at_hot:
                    # Shouldn't happen if counts[s_hot] == peak > 0
                    break
                
                # Pick a random pair from the hot sum
                a, b = random.choice(pairs_at_hot)
                
                # Try to swap a with some y not in A
                # We want y such that the new pair (y, b) or (y, a) doesn't create
                # as much load on s_hot, and ideally reduces peak elsewhere
                candidates = [y for y in range(n + 1) if y not in A]
                random.shuffle(candidates)
                candidates = candidates[:30]
                
                best_swap = None
                best_swap_score = None
                for y in candidates:
                    # Try swapping a with y
                    swap_element(A, counts, a, y)
                    new_score = get_score(counts, len(A))
                    # Check if this helps: peak reduced or same peak but lower variance
                    if new_score[0] == curr_score[0]:
                        if new_score[1] < curr_score[1]:
                            accept = True
                        elif new_score[1] == curr_score[1]:
                            if new_score[2] < curr_score[2]:
                                accept = True
                            else:
                                accept = False
                        else:
                            accept = False
                    else:
                        accept = False
                    
                    if accept:
                        if best_swap_score is None or new_score < best_swap_score:
                            best_swap_score = new_score
                            best_swap = y
                    # Revert
                    swap_element(A, counts, y, a)
                
                if best_swap is not None:
                    swap_element(A, counts, a, best_swap)
                    curr_score = best_swap_score
                    if curr_score < best_score_local:
                        best_score_local = curr_score
                        best_A_local = set(A)
        
        return best_A_local, best_score_local

    def run_sa(initial_A, initial_counts, max_steps, T_start, T_end, alpha):
        A = set(initial_A)
        counts = initial_counts[:]
        curr_score = get_score(counts, len(A))
        best_A_local = set(A)
        best_score_local = curr_score
        
        T = T_start
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            uncovered, peak, _ = curr_score
            
            if uncovered > 0:
                p_add = 0.4
                p_swap = 0.5
                p_remove = 0.1
            else:
                p_add = 0.1
                p_swap = 0.6
                p_remove = 0.3
            
            r = random.random()
            
            if r < p_add:
                if len(A) < n + 1:
                    candidates = [i for i in range(n+1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        add_element(A, counts, y)
                        new_score = get_score(counts, len(A))
                        
                        old_cost = curr_score[0]*1000 + curr_score[1]*10 + curr_score[2]/100.0
                        new_cost = new_score[0]*1000 + new_score[1]*10 + new_score[2]/100.0
                        d_cost = new_cost - old_cost
                        
                        if d_cost < 0:
                            accept = True
                        else:
                            accept = (random.random() < pow(2.71828, -d_cost / T))
                        
                        if accept:
                            curr_score = new_score
                            if curr_score < best_score_local:
                                best_score_local = curr_score
                                best_A_local = set(A)
                        else:
                            remove_element(A, counts, y)
                            
            elif r < p_add + p_swap:
                if len(A) > 0:
                    x = random.choice(list(A))
                    candidates = [i for i in range(n+1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        swap_element(A, counts, x, y)
                        new_score = get_score(counts, len(A))
                        
                        old_cost = curr_score[0]*1000 + curr_score[1]*10 + curr_score[2]/100.0
                        new_cost = new_score[0]*1000 + new_score[1]*10 + new_score[2]/100.0
                        d_cost = new_cost - old_cost
                        
                        if d_cost < 0:
                            accept = True
                        else:
                            accept = (random.random() < pow(2.71828, -d_cost / T))
                        
                        if accept:
                            curr_score = new_score
                            if curr_score < best_score_local:
                                best_score_local = curr_score
                                best_A_local = set(A)
                        else:
                            swap_element(A, counts, y, x)
                            
            else:
                if len(A) > 1:
                    x = random.choice(list(A))
                    remove_element(A, counts, x)
                    new_score = get_score(counts, len(A))
                    
                    old_cost = curr_score[0]*1000 + curr_score[1]*10 + curr_score[2]/100.0
                    new_cost = new_score[0]*1000 + new_score[1]*10 + new_score[2]/100.0
                    d_cost = new_cost - old_cost
                    
                    if d_cost < 0:
                        accept = True
                    else:
                        accept = (random.random() < pow(2.71828, -d_cost / T))
                    
                    if accept:
                        curr_score = new_score
                        if curr_score < best_score_local:
                            best_score_local = curr_score
                            best_A_local = set(A)
                    else:
                        add_element(A, counts, x)
            
            T *= alpha
            if T < T_end:
                T = T_end
                
        return best_A_local, best_score_local

    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))
    
    # Phase 1: Generate initial candidates using greedy edge-picking
    inits = []
    
    # Greedy edge-picking with various parameters
    for trial in range(8):
        A_greedy, counts_greedy, score_greedy = greedy_edge_picking(max_iters=150, max_A_size=70)
        inits.append((score_greedy, A_greedy, counts_greedy))
    
    # Some random inits for diversity
    for k in [40, 50, 60]:
        A_rand = set(random.sample(range(n+1), k))
        counts_rand = init_counts(A_rand)
        score_rand = get_score(counts_rand, len(A_rand))
        inits.append((score_rand, A_rand, counts_rand))
    
    # Structured: Arithmetic Progressions
    for step in [2, 3]:
        for start in range(step):
            A_ap = set(range(start, n+1, step))
            counts_ap = init_counts(A_ap)
            score_ap = get_score(counts_ap, len(A_ap))
            inits.append((score_ap, A_ap, counts_ap))
    
    inits.sort(key=lambda x: x[0])
    
    # Phase 2: Run hot-sum local search on top inits
    for score_init, A_init, counts_init in inits[:5]:
        if time.time() > deadline - 10.0:
            break
        A_polished, score_polished = hot_sum_local_search(A_init, counts_init, max_steps=8000)
        if score_polished < best_score:
            best_score = score_polished
            best_A = A_polished
    
    # Phase 3: SA on the best result so far
    if best_A is not None and time.time() < deadline - 15.0:
        counts_best = init_counts(best_A)
        A_res, score_res = run_sa(best_A, counts_best, 30000, 40.0, 0.1, 0.999)
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
    
    # Phase 4: Final polish
    if best_A is not None and time.time() < deadline - 3.0:
        counts_best = init_counts(best_A)
        A_final, score_final = hot_sum_local_search(best_A, counts_best, max_steps=15000)
        if score_final < best_score:
            best_score = score_final
            best_A = A_final

    if best_A is None:
        return list(range(0, n+1))
        
    return list(best_A)
