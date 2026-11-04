import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    # neighbors[idx] is a list of 17 indices
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        # Use Luby weight^2 for selection probability
        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            # Luby weight^2: pick with probability proportional to count^2
            weights = uncovered_counts.astype(np.float64)
            weights[weights == 0] = 0
            weights_sq = weights * weights
            total_w = weights_sq.sum()
            if total_w == 0:
                break
            probs = weights_sq / total_w
            best_idx = rng.choice(total_points, p=probs)
            
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            # Vectorized check for redundancy
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    coverage_count[ball] -= 1
                    selected_list.pop(i)
                    changed = True
                    break # Restart scan to handle cascading removals
            
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        """
        Local search with simulated annealing.
        Objective: minimize code size.
        Moves: 
          1. Remove a redundant codeword (always accepted if valid).
          2. Swap a codeword c for c' (accepted if it doesn't break coverage, or with prob exp(-delta/T) if it increases size temporarily).
          3. Add a codeword (only if it allows subsequent pruning, or via annealing).
        We focus on swaps and removals.
        """
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(42)
        
        # Simulated Annealing parameters
        # Temperature schedule: exponential cooling
        T_start = 10.0
        T_end = 0.1
        # We don't know the number of steps in advance, so we use time-based or step-based.
        # Let's use a fixed number of iterations per "phase" or just cool down over time.
        # Simpler: T = T_start * (T_end/T_start)^(elapsed_time / total_time)
        
        start_time = time.time()
        total_time = deadline - start_time
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            # Calculate current temperature
            elapsed = time.time() - start_time
            if elapsed >= total_time:
                break
            progress = elapsed / total_time
            T = T_start * (T_end / T_start) ** progress
            
            # Choose a random codeword to modify
            c = selected_list[rng.integers(len(selected_list))]
            
            # Decide move type: 80% swap, 20% removal attempt (if possible)
            # Removal is only possible if redundant, which prune handles. 
            # So we focus on swaps.
            
            # Pick a candidate c_prime from neighbors of c
            # Exclude c itself (index 0 in neighbors[c])
            # Also exclude any c' already in selected_set to avoid duplicates
            c_prime_candidates = neighbors[c][1:]
            
            # Filter out those already in the set
            # This is O(16) check against a set
            valid_candidates = [x for x in c_prime_candidates if x not in selected_set]
            
            if not valid_candidates:
                continue
            
            c_prime = valid_candidates[rng.integers(len(valid_candidates))]
            
            # Tentative swap
            coverage_count[neighbors[c]] -= 1
            coverage_count[neighbors[c_prime]] += 1
            
            # Check validity
            # If any point in neighbors[c] has count 0, the swap breaks coverage.
            # Note: points in neighbors[c_prime] that were 0 become 1, so they are fine.
            # We only care about points that lost coverage (neighbors[c] \ neighbors[c_prime]).
            # But checking np.any(coverage_count[neighbors[c]] == 0) is sufficient and fast.
            if np.any(coverage_count[neighbors[c]] == 0):
                # Invalid swap (breaks coverage)
                # Revert
                coverage_count[neighbors[c_prime]] -= 1
                coverage_count[neighbors[c]] += 1
                continue
            
            # Valid swap. 
            # Does it allow pruning?
            # Try to prune. If pruning reduces size, accept.
            # If not, accept with probability based on T? 
            # Actually, the objective is size. A swap that doesn't change size is neutral.
            # A swap that allows pruning is good.
            # A swap that *prevents* pruning (i.e., makes code "tighter" but same size) might be needed to escape local minima?
            # Usually, we want to reduce size. 
            # Let's accept if it leads to a smaller code after pruning.
            # If it leads to the same size, accept with probability 0.5 (random walk) or based on T?
            # Since we are minimizing, neutral moves help exploration.
            
            # Perform swap in lists
            pos = selected_list.index(c)
            selected_list[pos] = c_prime
            selected_set.discard(c)
            selected_set.add(c_prime)
            
            # Try to prune
            pruned_list = prune(selected_list)
            
            if len(pruned_list) < len(selected_list):
                # Improvement
                selected_list = pruned_list
                selected_set = set(selected_list)
                # Recalculate coverage
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[neighbors[idx]] += 1
                
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
            else:
                # No improvement in size. 
                # Accept with some probability to allow exploration?
                # If we always revert, we get stuck in local minima of "no immediate improvement".
                # Let's accept neutral swaps with probability 0.5 * exp(-1/T) or similar.
                # Or just accept with 0.5.
                if rng.random() < 0.5:
                    # Keep the swap (neutral move)
                    # Note: selected_list is already updated to include c_prime and exclude c, but not pruned.
                    # Since prune didn't remove anything, selected_list is the post-swap list.
                    # We need to ensure coverage_count is consistent. It is, because we updated it for the swap.
                    pass
                else:
                    # Revert
                    selected_list[pos] = c
                    selected_set.discard(c_prime)
                    selected_set.add(c)
                    coverage_count[neighbors[c_prime]] -= 1
                    coverage_count[neighbors[c]] += 1
        
        return best_code

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Run greedy instances
    best_code = None
    best_len = float('inf')
    
    # Run a few greedy instances
    # The current program runs 20, but breaks if time > deadline - 60.
    # Let's run as many as possible in the first 40 seconds.
    greedy_deadline = start_time + 40.0
    
    for seed in range(50):
        if time.time() > greedy_deadline:
            break
        selected = luby_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = luby_greedy(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search with Simulated Annealing
    # Use the remaining time
    best_code = local_search_optimize(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
