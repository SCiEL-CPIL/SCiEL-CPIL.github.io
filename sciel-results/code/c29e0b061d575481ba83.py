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
    
    # counts[s] is the number of pairs {a,b} in A such that a+b=s
    # We only care about s in [target_lo, target_hi] for scoring, 
    # but we need to maintain counts for all s in [0, max_sum] 
    # to correctly update when adding/removing elements?
    # Actually, when adding y, we update counts[y+z]. 
    # If y+z is outside [target_lo, target_hi], we don't care for the score, 
    # but we might care if we later remove y? No, if it's outside, it stays outside.
    # However, to keep the logic simple and correct, we maintain counts for 0..max_sum.
    # max_sum = 256. Array size 257. Very small.
    
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
        
        # Variance is proportional to sum_sq - (total^2 / len)
        # We minimize peak primarily. Variance is secondary.
        # To make variance comparable, we can just use sum_sq.
        # But if peak is 0, we want to minimize sum_sq? 
        # If uncovered > 0, we prioritize covering.
        
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
        y_list = list(A)
        # We need to add pairs (y, z) for all z in A.
        # Since we just added y, A now contains y.
        # We need to add 1 to counts[y+z] for all z in A.
        # This includes z=y (pair {y,y}).
        for z in A:
            s = y + z
            if s <= max_sum:
                counts[s] += 1
        return A

    def remove_element(A, counts, x):
        """Remove x from A and update counts. Returns new A set."""
        A.remove(x)
        # We need to subtract 1 from counts[x+z] for all z in A (after removal).
        # The pairs involving x were {x, z} for z in A_old.
        # A_old = A_new U {x}.
        # So pairs are {x, z} for z in A_new, plus {x, x}.
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
        # Pick more from the middle
        for i in range(50):
            if i < 20:
                A.add(random.randint(0, n//4))
            elif i < 40:
                A.add(random.randint(n//4, 3*n//4))
            else:
                A.add(random.randint(3*n//4, n))
        inits.append(A)

    # Evaluate inits
    for A_init in inits:
        counts = init_counts(A_init)
        score = get_score(counts, len(A_init))
        if score < best_score:
            best_score = score
            best_A = set(A_init)
            
    # Simulated Annealing
    # We run SA from the best initial state, and maybe restart a few times.
    
    # To maximize exploration, we can run multiple SA chains.
    # But time is limited. Let's run one long SA from the best init, 
    # and if time permits, restart from other good inits.
    
    # Let's collect top 5 inits
    scored_inits = []
    for A_init in inits:
        counts = init_counts(A_init)
        score = get_score(counts, len(A_init))
        scored_inits.append((score, A_init, counts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Run SA on the best init
    # And potentially others if time allows.
    
    def run_sa(initial_A, initial_counts, max_steps, T_start, T_end, alpha):
        A = set(initial_A)
        counts = initial_counts[:]
        curr_score = get_score(counts, len(A))
        best_A_local = set(A)
        best_score_local = curr_score
        
        T = T_start
        # Cooling schedule
        # We want to finish by max_steps or time.
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            # Generate Move
            # Probabilities depend on current state
            uncovered, peak, _ = curr_score
            
            if uncovered > 0:
                # Need to add elements or swap to cover gaps
                p_add = 0.4
                p_swap = 0.5
                p_remove = 0.1
            else:
                # Covered, minimize peak
                # If peak is high, remove or swap
                # If peak is low, maybe add to reduce variance? 
                # Actually, adding elements increases counts, likely increasing peak.
                # So if covered, we should try to remove or swap.
                p_add = 0.1
                p_swap = 0.6
                p_remove = 0.3
            
            r = random.random()
            
            old_A_size = len(A)
            
            if r < p_add:
                # Add random element
                if len(A) < n + 1:
                    candidates = [i for i in range(n+1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        # Apply move
                        add_element(A, counts, y)
                        new_score = get_score(counts, len(A))
                        
                        # Acceptance
                        delta = new_score[1] - curr_score[1] # Peak diff
                        # If uncovered changed, it's a huge delta
                        if new_score[0] < curr_score[0]:
                            accept = True
                        elif new_score[0] == curr_score[0]:
                            if new_score[1] < curr_score[1]:
                                accept = True
                            elif new_score[1] == curr_score[1]:
                                # Check variance (sum_sq)
                                if new_score[2] < curr_score[2]:
                                    accept = True
                                else:
                                    # SA accept worse variance?
                                    # Delta in objective. 
                                    # Objective: (uncovered, peak, sum_sq)
                                    # We want to minimize.
                                    # If peak same, we accept worse sum_sq with prob exp(-delta/T)
                                    # But sum_sq is large. Let's normalize?
                                    # Or just accept if peak improves or same and sum_sq improves.
                                    # For SA, we need a scalar cost.
                                    # Cost = uncovered * 1000 + peak * 10 + sum_sq / 100
                                    old_cost = curr_score[0]*1000 + curr_score[1]*10 + curr_score[2]/100.0
                                    new_cost = new_score[0]*1000 + new_score[1]*10 + new_score[2]/100.0
                                    d_cost = new_cost - old_cost
                                    if d_cost < 0:
                                        accept = True
                                    else:
                                        accept = (random.random() < pow(2.71828, -d_cost / T))
                                # end if
                            else:
                                # Peak increased
                                old_cost = curr_score[0]*1000 + curr_score[1]*10 + curr_score[2]/100.0
                                new_cost = new_score[0]*1000 + new_score[1]*10 + new_score[2]/100.0
                                d_cost = new_cost - old_cost
                                accept = (random.random() < pow(2.71828, -d_cost / T))
                        else:
                            # Uncovered increased
                            old_cost = curr_score[0]*1000 + curr_score[1]*10 + curr_score[2]/100.0
                            new_cost = new_score[0]*1000 + new_score[1]*10 + new_score[2]/100.0
                            d_cost = new_cost - old_cost
                            accept = (random.random() < pow(2.71828, -d_cost / T))
                        
                        if accept:
                            curr_score = new_score
                            if curr_score < best_score_local:
                                best_score_local = curr_score
                                best_A_local = set(A)
                        else:
                            # Revert
                            remove_element(A, counts, y)
                            # curr_score unchanged
                            
            elif r < p_add + p_swap:
                # Swap
                if len(A) > 0:
                    x = random.choice(list(A))
                    candidates = [i for i in range(n+1) if i not in A]
                    if candidates:
                        y = random.choice(candidates)
                        # Apply swap
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
                            # Revert: swap y back to x
                            swap_element(A, counts, y, x)
                            
            else:
                # Remove
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
                        # Revert: add x back
                        add_element(A, counts, x)
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
                
        return best_A_local, best_score_local

    # Run SA from top inits
    # We have ~30s left maybe? 
    # Let's allocate time.
    
    # Sort inits by score
    # We already have best_score from simple eval.
    
    # Run SA on the best init
    if time.time() < deadline - 5.0:
        # Get best init
        best_init_score, best_init_A, best_init_counts = scored_inits[0]
        
        # Run SA
        # Parameters:
        # Steps: 50000 (fast steps)
        # T_start: 50, T_end: 0.1, alpha: 0.999
        A_res, score_res = run_sa(best_init_A, best_init_counts, 50000, 50.0, 0.1, 0.999)
        
        if score_res < best_score:
            best_score = score_res
            best_A = A_res
            
    # If time remains, run SA on 2nd best init
    if time.time() < deadline - 5.0 and len(scored_inits) > 1:
        second_init_score, second_init_A, second_init_counts = scored_inits[1]
        # Only if it's not identical to best
        if second_init_A != best_A:
             A_res, score_res = run_sa(second_init_A, second_init_counts, 30000, 50.0, 0.1, 0.999)
             if score_res < best_score:
                 best_score = score_res
                 best_A = A_res

    # If still uncovered or high peak, try a "Repair" phase?
    # The SA should handle it.
    
    if best_A is None:
        # Fallback
        return list(range(0, n+1))
        
    return list(best_A)
