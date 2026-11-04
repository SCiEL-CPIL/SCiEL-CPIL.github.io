import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute balls
    # idx_to_word helper
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    balls = np.empty((total_points, ball_size), dtype=np.int32)
    neighbors = [None] * total_points
    
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        balls[idx, 0] = idx
        neighbor_idx = 1
        nbrs = []
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
            balls[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            nbrs.append(n_idx)
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            balls[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            nbrs.append(n_idx)
            
        neighbors[idx] = nbrs

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            # Uncovered count for each ball
            # ~is_covered[balls] is boolean matrix
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            # Luby: probability proportional to score^2
            # Only consider sets with max score? Or all?
            # Standard Luby uses all, but weighting heavily favors max.
            # Let's pick from max candidates to be safe and fast, 
            # or use proper Luby.
            # Proper Luby:
            scores = uncovered_counts.copy()
            # Set scores of covered points to 0? No, scores are counts of uncovered points.
            # If a set covers 0 uncovered points, score is 0.
            
            # We want to pick a set with high score.
            # Let's just pick randomly from the set of sets with max score.
            # This is "Greedy with Random Tie-Breaking".
            # True Luby might be better.
            # Let's stick to Random Tie-Breaking for speed, it's robust.
            
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return selected_indices
            
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i, idx in enumerate(selected_list):
                ball = balls[idx]
                # If all points in this ball have count > 1, it's redundant
                if np.all(coverage_count[ball] > 1):
                    # Remove
                    rem_idx = selected_list.pop(i)
                    coverage_count[balls[rem_idx]] -= 1
                    changed = True
                    break # Restart scan after change
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        
        # Initial prune
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        while time.time() < deadline:
            # Try to prune
            pruned = prune(selected_list)
            if len(pruned) < len(selected_list):
                selected_list = pruned
                # Recalculate counts
                coverage_count = np.zeros(total_points, dtype=np.int32)
                for idx in selected_list:
                    coverage_count[balls[idx]] += 1
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                continue
            
            # Try to find a beneficial swap
            # A swap c -> c' is beneficial if it allows pruning of some d
            # We iterate through codewords
            swapped = False
            n_selected = len(selected_list)
            # Shuffle order to avoid bias?
            # Just iterate
            for i in range(n_selected):
                if time.time() > deadline:
                    break
                c = selected_list[i]
                for c_prime in neighbors[c]:
                    if c_prime in selected_list:
                        continue
                    
                    # Tentatively swap
                    # Update counts
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1
                    
                    # Check if any codeword is now redundant
                    # Only codewords affected by the change in counts
                    # Affected points: balls[c] U balls[c_prime]
                    # Affected codewords: those whose balls intersect affected points
                    # This is hard to precompute quickly for each swap.
                    # Approximation: Check all codewords? O(N * 17) = 500 * 17 = 8500 ops.
                    # 8000 swaps * 8500 ops = 68M ops. Too slow in Python.
                    
                    # Optimization: 
                    # Precompute for each point p, which codewords cover it?
                    # coverers[p] = list of indices in selected_list that cover p.
                    # This is dynamic.
                    
                    # Let's use a faster heuristic:
                    # Just check if the swap itself creates a redundancy in c_prime's ball?
                    # No, we want to remove *other* codewords.
                    
                    # Let's try a simpler local search: 
                    # Random Valid Move (Random Walk)
                    # Pick random c, random neighbor c'.
                    # If swap is valid (all points still covered), accept.
                    # This explores the space.
                    
                    # Revert tentatively
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    break # Break inner loop to do one random move?
                
                # Do one random valid move
                c = selected_list[i]
                c_prime = neighbors[c][random.randint(0, len(neighbors[c])-1)]
                if c_prime in selected_list:
                    continue
                
                # Check validity
                # Points in balls[c] that are NOT in balls[c_prime] must be covered by others.
                # Points in balls[c_prime] must be covered (by c_prime or others).
                
                # Simpler: 
                # Decrement c, Increment c_prime.
                # Check if min count in (balls[c] \ balls[c_prime]) >= 1?
                # And min count in balls[c_prime] >= 1?
                
                coverage_count[balls[c]] -= 1
                coverage_count[balls[c_prime]] += 1
                
                valid = True
                # Check points in c's ball that are lost
                # We can just check if any point in balls[c] has count 0
                # And any point in balls[c_prime] has count 0?
                # Actually, if we swap, points in intersection are fine.
                # Points in c \ c_prime: need count >= 1.
                # Points in c_prime \ c: need count >= 1.
                
                # Efficient check:
                # We know coverage_count before swap.
                # Let's just check the union of balls.
                union_mask = np.zeros(total_points, dtype=bool)
                union_mask[balls[c]] = True
                union_mask[balls[c_prime]] = True
                union_points = np.where(union_mask)[0]
                
                if np.any(coverage_count[union_points] < 1):
                    # Invalid, revert
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                else:
                    # Valid, accept
                    selected_list[i] = c_prime
                    swapped = True
                    break # Restart outer loop after a swap
            
            if swapped:
                # Try to prune after swap
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[balls[idx]] += 1
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]

        return best_code

    start_time = time.time()
    deadline = start_time + 110.0
    
    # Run a few greedy instances and pick the best after pruning
    best_code = None
    best_len = float('inf')
    
    # Run 5 greedy instances
    for seed in range(5):
        if time.time() > deadline - 60: # Leave 60s for local search
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
    
    # Local Search on the best code
    best_code = local_search_optimize(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
