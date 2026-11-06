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

    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))
    
    # Generate initial candidates
    inits = []
    
    # 1. Random sets of various sizes
    for k in [30, 40, 50, 60, 70]:
        for _ in range(5):
            inits.append(set(random.sample(range(n+1), k)))
            
    # 2. Structured: Arithmetic Progressions
    for step in [2, 3, 4]:
        for start in range(step):
            A = set(range(start, n+1, step))
            inits.append(A)
            
    # 3. "Middle-heavy" random
    for _ in range(5):
        A = set()
        for i in range(50):
            if i < 20:
                A.add(random.randint(0, n//4))
            elif i < 40:
                A.add(random.randint(n//4, 3*n//4))
            else:
                A.add(random.randint(3*n//4, n))
        inits.append(A)

    # 4. Greedy Balanced Build (from inspiration)
    def greedy_balanced_build(target_size=45):
        A = set()
        base_elements = random.sample(range(0, n + 1), min(10, target_size))
        A.update(base_elements)
        counts = init_counts(A)
        
        while len(A) < target_size:
            if time.time() > deadline - 5:
                break
            best_y = None
            best_metric = None
            
            candidates_y = [y for y in range(n + 1) if y not in A]
            random.shuffle(candidates_y)
            candidates_y = candidates_y[:30]
            
            for y in candidates_y:
                temp_counts = counts[:]
                for z in A:
                    s = y + z
                    if s <= max_sum:
                        temp_counts[s] += 1
                s2y = 2 * y
                if s2y <= max_sum:
                    temp_counts[s2y] += 1
                
                score = get_score(temp_counts, len(A) + 1)
                metric = score
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_y = y
            
            if best_y is not None:
                add_element(A, counts, best_y)
        
        return A

    for _ in range(5):
        inits.append(greedy_balanced_build(45))

    # Evaluate inits
    scored_inits = []
    for A_init in inits:
        counts = init_counts(A_init)
        score = get_score(counts, len(A_init))
        scored_inits.append((score, A_init, counts))
        
    scored_inits.sort(key=lambda x: x[0])
    
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

    # Run SA from top inits
    # We have ~30s left maybe? 
    # Let's allocate time.
    
    if time.time() < deadline - 5.0:
        best_init_score, best_init_A, best_init_counts = scored_inits[0]
        
        A_res, score_res = run_sa(best_init_A, best_init_counts, 50000, 50.0, 0.1, 0.999)
        
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
            
    if time.time() < deadline - 5.0 and len(scored_inits) > 1:
        second_init_score, second_init_A, second_init_counts = scored_inits[1]
        if second_init_A != best_A:
             A_res, score_res = run_sa(second_init_A, second_init_counts, 30000, 50.0, 0.1, 0.999)
             if score_res < best_score:
                 best_score = score_res
                 best_A = A_res

    # Polish phase: Local search focused on variance reduction at fixed peak
    def polish(A, counts, max_steps=5000):
        A = set(A)
        counts = counts[:]
        curr_score = get_score(counts, len(A))
        best_A_local = set(A)
        best_score_local = curr_score
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            # Try a swap
            if len(A) > 0:
                x = random.choice(list(A))
                candidates = [i for i in range(n+1) if i not in A]
                if candidates:
                    y = random.choice(candidates)
                    swap_element(A, counts, x, y)
                    new_score = get_score(counts, len(A))
                    
                    # Only accept if uncovered is same and (peak is lower OR (peak is same and sum_sq is lower))
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
                        curr_score = new_score
                        if curr_score < best_score_local:
                            best_score_local = curr_score
                            best_A_local = set(A)
                    else:
                        swap_element(A, counts, y, x)
                        
        return best_A_local, best_score_local

    if best_A is not None and time.time() < deadline - 2.0:
        counts_best = init_counts(best_A)
        A_polished, score_polished = polish(best_A, counts_best, 10000)
        if score_polished < best_score:
            best_score = score_polished
            best_A = A_polished

    if best_A is None:
        return list(range(0, n+1))
        
    return list(best_A)
