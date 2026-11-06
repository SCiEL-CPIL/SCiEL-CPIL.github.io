import random
import time

def construct():
    n = 128
    target_start = 128
    target_end = 256
    
    def get_counts(A_set):
        counts = [0] * (2 * n + 1)
        A_list = list(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                counts[s] += 1
        return counts

    def is_valid(counts):
        for s in range(target_start, target_end + 1):
            if counts[s] == 0:
                return False
        return True

    def get_peak(counts):
        return max(counts[target_start:target_end + 1])

    def get_variance(counts):
        vals = counts[target_start:target_end + 1]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        return sum((x - mean) ** 2 for x in vals)

    start_time = time.time()
    timeout = 55.0
    
    best_peak = float('inf')
    best_A = None
    
    # Limit number of full restarts to ensure we finish
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        initial_size = random.randint(30, 45)
        A = set(random.sample(range(n + 1), initial_size))
        
        current_counts = get_counts(A)
        current_peak = get_peak(current_counts) if is_valid(current_counts) else float('inf')
        current_var = get_variance(current_counts) if is_valid(current_counts) else float('inf')
        
        local_improvements = 0
        max_local_steps = 100 # Limit local search depth per restart
        
        for _ in range(max_local_steps):
            if time.time() - start_time > timeout:
                break
                
            improved = False
            
            # Generate candidates
            candidates_to_add = []
            if not is_valid(current_counts):
                missing_sums = [s for s in range(target_start, target_end + 1) if current_counts[s] == 0]
                for s in missing_sums[:5]:
                    for a in A:
                        x = s - a
                        if 0 <= x <= n and x not in A:
                            candidates_to_add.append(x)
            
            all_candidates = [x for x in range(n + 1) if x not in A]
            random.shuffle(all_candidates)
            candidates_to_add.extend(all_candidates[:10])
            candidates_to_add = list(set(candidates_to_add))
            
            if not candidates_to_add:
                break
                
            # Try swaps
            # To speed up, we only try a subset of 'a's if the set is large?
            # No, iterate all 'a's but break early if found.
            
            for a in list(A):
                if time.time() - start_time > timeout:
                    break
                    
                best_b = None
                best_new_peak = float('inf')
                best_new_var = float('inf')
                best_new_counts = None
                best_new_valid = False
                
                for b in candidates_to_add:
                    if time.time() - start_time > timeout:
                        break
                        
                    new_A = A.copy()
                    new_A.remove(a)
                    new_A.add(b)
                    
                    new_counts = get_counts(new_A)
                    
                    if is_valid(new_counts):
                        new_peak = get_peak(new_counts)
                        new_var = get_variance(new_counts)
                        
                        if new_peak < best_new_peak:
                            best_new_peak = new_peak
                            best_new_var = new_var
                            best_new_counts = new_counts
                            best_new_valid = True
                            best_b = b
                        elif new_peak == best_new_peak and new_var < best_new_var:
                            best_new_var = new_var
                            best_new_counts = new_counts
                            best_new_valid = True
                            best_b = b
                        elif new_peak == best_new_peak + 1 and new_var < best_new_var * 0.8:
                            if random.random() < 0.1:
                                best_new_peak = new_peak
                                best_new_var = new_var
                                best_new_counts = new_counts
                                best_new_valid = True
                                best_b = b

                if best_b is not None and best_new_valid:
                    if best_new_peak < current_peak or (best_new_peak == current_peak and best_new_var < current_var) or (best_new_peak == current_peak + 1 and best_new_var < current_var * 0.8 and random.random() < 0.2):
                        A = A.copy()
                        A.remove(a)
                        A.add(best_b)
                        current_counts = best_new_counts
                        current_peak = best_new_peak
                        current_var = best_new_var
                        improved = True
                        local_improvements += 1
                        break # Restart a-loop

            if not improved:
                # Perturbation
                if len(A) > 5:
                    for _ in range(2):
                        a = random.choice(list(A))
                        candidates = [x for x in range(n + 1) if x not in A]
                        if not candidates: continue
                        b = random.choice(candidates)
                        
                        A = A.copy()
                        A.remove(a)
                        A.add(b)
                        
                        current_counts = get_counts(A)
                        if is_valid(current_counts):
                            current_peak = get_peak(current_counts)
                            current_var = get_variance(current_counts)
                        else:
                            current_peak = float('inf')
                            current_var = float('inf')
                else:
                    break

        # Update global best
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = set(A)
            
        # If we found a very good peak, we might want to spend more time polishing?
        # But for now, we just restart.

    if best_A is None:
        return list(range(0, n + 1, 4))
        
    return list(best_A)

# EVOLVE-BLOCK-END
