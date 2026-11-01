import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin for 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute base for index <-> word conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls for all indices
    # ball_indices[i] is a numpy array of indices within Hamming distance 'radius'
    ball_indices = [None] * total
    for i in range(total):
        temp = i
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        
        ball = [i]
        # Radius 1 logic: change one coordinate to any other value in 0..q-1
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                ball.append(i + (v - orig) * base[coord])
        ball_indices[i] = np.array(ball, dtype=np.int32)

    # Helper to convert index to list of digits
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        return w

    # Greedy initialization
    def greedy_init():
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        
        # Precompute a list of all indices for random access
        all_indices = list(range(total))
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            
            # Optimization: If uncovered is small, we might find a codeword quickly.
            # But standard greedy checks all.
            # To speed up, we can sample if time is tight, but 2187 is small enough to scan.
            # However, scanning 2187 * 15 ops is ~30k ops. Doing this ~150 times is ~4.5M ops.
            # This is fast in Python if vectorized, but loops are slow.
            
            # Let's use numpy for the coverage check in greedy to be fast.
            # We want max( count of 0s in covered[ball_indices[c]] )
            
            # Since we do this repeatedly, let's just loop.
            # We can optimize by only checking candidates that cover at least one uncovered point?
            # Hard to know without checking.
            
            # Randomized greedy to break symmetry
            random.shuffle(all_indices)
            
            for c in all_indices:
                ball = ball_indices[c]
                # Count how many in this ball are currently uncovered
                new_cov = np.count_nonzero(covered[ball] == 0)
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            
            code.append(best_c)
            covered[ball_indices[best_c]] = 1
            uncovered_count = np.count_nonzero(covered == 0)
            
        return code

    # Simulated Annealing / Local Search
    def local_search(initial_code, time_budget):
        # State: Set of codewords (indices)
        code_set = set(initial_code)
        
        # Coverage count: count[i] = number of codewords covering index i
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1
        
        uncovered = np.count_nonzero(cnt == 0)
        
        # We want to minimize size, subject to uncovered == 0.
        # Penalty for uncovered is high.
        # Cost = len(code_set) + 1000 * uncovered
        
        best_code = list(code_set)
        best_size = len(code_set)
        
        # Temperature
        T = 2.0
        cooling = 0.99995
        min_T = 0.05
        
        t_start = time.time()
        
        # Precompute list of all indices for random sampling
        all_indices = list(range(total))
        
        # Precompute ball membership as a set for O(1) checks
        ball_set = {c: set(b) for c, b in enumerate(ball_indices)}
        
        # Track consecutive non-improving steps for temperature reset
        steps_since_improve = 0
        
        while time.time() - t_start < time_budget:
            # Random move type with adjusted probabilities
            r = random.random()
            
            if r < 0.35:
                # Try to remove a codeword
                if len(code_set) == 0:
                    continue
                c_out = random.choice(list(code_set))
                
                # Calculate delta
                # Remove c_out
                ball = ball_indices[c_out]
                
                # Check if any point in ball becomes uncovered
                # new_cnt = cnt - 1 for points in ball
                # If any cnt[x] == 1, it becomes 0 -> uncovered increases
                delta_uncovered = np.count_nonzero(cnt[ball] == 1)
                
                delta_cost = -1 + 1000 * delta_uncovered
                
                if delta_cost < 0 or (T > min_T and random.random() < np.exp(-delta_cost / T)):
                    # Accept
                    code_set.remove(c_out)
                    cnt[ball] -= 1
                    uncovered += delta_uncovered
                    
                    if uncovered == 0:
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
                    else:
                        steps_since_improve += 1
                        
            elif r < 0.55:
                # Try to add a codeword (useful if we are in a bad state with uncovered points)
                if uncovered == 0:
                    # Skip pure adds if covered, to keep size down
                    continue
                
                c_in = random.choice(all_indices)
                if c_in in code_set:
                    continue
                
                ball = ball_indices[c_in]
                # Points in ball that are currently uncovered (cnt == 0)
                # Adding c_in will cover them.
                # delta_uncovered is negative (reduction)
                num_covered_new = np.count_nonzero(cnt[ball] == 0)
                delta_uncovered = -num_covered_new
                
                delta_cost = 1 + 1000 * delta_uncovered
                
                if delta_cost < 0 or (T > min_T and random.random() < np.exp(-delta_cost / T)):
                    # Accept
                    code_set.add(c_in)
                    cnt[ball] += 1
                    uncovered += delta_uncovered
                    steps_since_improve += 1
                    
            else:
                # Swap: remove one, add one
                if len(code_set) == 0:
                    continue
                c_out = random.choice(list(code_set))
                c_in = random.choice(all_indices)
                
                if c_in in code_set or c_in == c_out:
                    continue
                
                ball_out = ball_indices[c_out]
                ball_in = ball_indices[c_in]
                
                # Calculate change in uncovered count
                affected = np.union1d(ball_out, ball_in)
                
                # Current cnt for affected
                old_cnt = cnt[affected]
                
                # New cnt for affected
                in_out = np.isin(affected, ball_out)
                in_in = np.isin(affected, ball_in)
                
                new_cnt = old_cnt - in_out.astype(np.int32) + in_in.astype(np.int32)
                
                # Old uncovered in affected: count of old_cnt == 0
                old_uncov = np.count_nonzero(old_cnt == 0)
                # New uncovered in affected: count of new_cnt == 0
                new_uncov = np.count_nonzero(new_cnt == 0)
                
                delta_uncovered = new_uncov - old_uncov
                
                delta_cost = 0 + 1000 * delta_uncovered # Size doesn't change
                
                if delta_cost < 0 or (T > min_T and random.random() < np.exp(-delta_cost / T)):
                    # Accept
                    code_set.remove(c_out)
                    code_set.add(c_in)
                    
                    # Update cnt
                    cnt[ball_out] -= 1
                    cnt[ball_in] += 1
                    uncovered += delta_uncovered
                    steps_since_improve += 1

            # Cooling
            T *= cooling
            if T < min_T:
                # If stuck for a while, reset temperature to escape local minima
                if steps_since_improve > 500:
                    T = 1.0
                    steps_since_improve = 0
                else:
                    T = min_T

        # Final polish: greedily remove redundant codewords
        code_set = set(best_code)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1
        
        improved = True
        while improved:
            improved = False
            to_remove = []
            for c in list(code_set):
                ball = ball_indices[c]
                if np.count_nonzero(cnt[ball] == 1) == 0:
                    to_remove.append(c)
            for c in to_remove:
                code_set.remove(c)
                cnt[ball_indices[c]] -= 1
                improved = True
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)

        return best_code

    # Run multiple restarts
    best_overall = []
    best_size_overall = total

    # 1. Greedy + LS
    greedy_code = greedy_init()
    # Allocate most time to the first run
    ls_code = local_search(greedy_code, time_budget=55.0)
    best_size_overall = len(ls_code)
    best_overall = ls_code

    # 2. More restarts with less time each
    seed = 1
    while time.time() < deadline - 10.0:
        greedy_code = greedy_init() # Re-randomize greedy? 
        # The greedy_init uses random.shuffle, so it's different each time.
        
        # If we have time, run LS on this new greedy solution
        time_left = deadline - 10.0 - time.time()
        if time_left < 8.0:
            break
            
        # Shorter budget for subsequent runs
        tb = min(8.0, time_left / 2.0)
        
        ls_code = local_search(greedy_code, time_budget=tb)
        
        if len(ls_code) < best_size_overall:
            best_size_overall = len(ls_code)
            best_overall = ls_code
            
        seed += 1
    
    # Final polish pass: aggressively remove redundant codewords
    if best_overall:
        code_set = set(best_overall)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1
        
        improved = True
        while improved:
            improved = False
            to_remove = []
            for c in list(code_set):
                ball = ball_indices[c]
                if np.count_nonzero(cnt[ball] == 1) == 0:
                    to_remove.append(c)
            for c in to_remove:
                code_set.remove(c)
                cnt[ball_indices[c]] -= 1
                improved = True
        
        if len(code_set) < best_size_overall:
            best_size_overall = len(code_set)
            best_overall = list(code_set)

    # Convert indices to words
    result = []
    for idx in best_overall:
        result.append(idx_to_word(idx))
        
    # Verify coverage (safety check)
    if result:
        covered = np.zeros(total, dtype=np.uint8)
        for word in result:
            idx = 0
            for i, d in enumerate(word):
                idx += d * base[i]
            covered[ball_indices[idx]] = 1
        if np.count_nonzero(covered == 0) > 0:
            # Coverage failed, return the best we can (shouldn't happen)
            pass
        
    return result
