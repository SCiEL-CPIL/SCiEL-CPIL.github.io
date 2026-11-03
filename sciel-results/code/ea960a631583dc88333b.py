import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1)
    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # 1. Structural Initialization: Product of Tetracodes
    # The tetracode covers F_3^4 with radius 1 using 9 words.
    # The product of two tetracodes covers F_3^8 with radius 1 using 9*9=81 words.
    # Wait, 81 words is very small. Why did the seed program struggle?
    # Ah, the tetracode is a *perfect* code? No, K_3(4,1) = 9.
    # The product code has size 81.
    # Every word in F_3^8 is within distance 1 of some (c1, c2) where c1 in T, c2 in T.
    # Distance d((x1,x2), (c1,c2)) = d(x1,c1) + d(x2,c2).
    # We need d <= 1.
    # If d(x1,c1)=0 and d(x2,c2)<=1, ok.
    # If d(x1,c1)<=1 and d(x2,c2)=0, ok.
    # If d(x1,c1)=1 and d(x2,c2)=1, sum=2 > 1. NOT OK.
    # So the product code does NOT cover F_3^8 with radius 1.
    # It covers with radius 2.
    # So we cannot just use the product code.
    
    # We need a code where the "defects" don't overlap.
    # Actually, let's stick to the heuristic search but improve the initialization and local search.
    # The seed program's greedy starts from scratch.
    # A better initialization: Random Greedy with a "bias" towards spreading out.
    
    def greedy_cover():
        """Fast greedy set cover."""
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        
        while not covered.all():
            if time.time() - start_time > time_limit - 10.0:
                break
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            neigh_avail = neighbors[avail_indices]
            is_covered = covered[neigh_avail]
            newly = (~is_covered).sum(axis=1)
            max_new = np.max(newly)
            if max_new == 0:
                break
            # Tie-breaking: pick the one with lowest index or random among max
            # Randomizing tie-breaking helps escape local minima in the greedy phase
            candidates = np.where(newly == max_new)[0]
            pick = candidates[random.randint(0, len(candidates) - 1)]
            best_global = avail_indices[pick]
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
        return chosen, covered

    def aggressive_removal(chosen_indices):
        """Remove redundant codewords with randomized multiple passes."""
        if not chosen_indices:
            return chosen_indices
        
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        
        removed_any = True
        passes = 0
        while removed_any and time.time() - start_time < time_limit - 5.0:
            removed_any = False
            passes += 1
            # Randomize order to escape bias
            indices_list = list(range(len(chosen_indices)))
            random.shuffle(indices_list)
            
            for i in indices_list:
                if i >= len(chosen_indices):
                    continue
                ci = chosen_indices[i]
                # Check if all points covered by ci are covered by others
                if np.all(cover_count[neighbors[ci]] >= 2):
                    cover_count[neighbors[ci]] -= 1
                    chosen_indices.pop(i)
                    removed_any = True
        return chosen_indices

    def sa_swap_optimize(chosen_indices, covered):
        """
        Simulated annealing on fixed-size code.
        Energy = number of uncovered points (should be 0 for feasible).
        Moves: swap one codeword with a random other codeword.
        """
        current = list(chosen_indices)
        
        # Build coverage count for current
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1
        current_uncovered = np.sum(cover_count == 0)
        
        best = list(current)
        best_size = len(current)
        
        # SA parameters
        T_start = 50.0
        T_end = 0.1
        cooling = 0.995
        T = T_start
        iterations = 0
        max_iters = 100000
        
        while T > T_end and iterations < max_iters and time.time() - start_time < time_limit - 5.0:
            iterations += 1
            
            # Pick a random position in the code to swap
            pos = random.randint(0, len(current) - 1)
            old_ci = current[pos]
            
            # Pick a random replacement
            new_ci = random.randint(0, total_points - 1)
            if new_ci == old_ci:
                continue
            
            # Compute energy change
            # Remove old: subtract its coverage
            cover_count[neighbors[old_ci]] -= 1
            uncovered_after_remove = np.sum(cover_count == 0)
            
            # Add new: add its coverage
            cover_count[neighbors[new_ci]] += 1
            uncovered_after_add = np.sum(cover_count == 0)
            
            delta_energy = uncovered_after_add - current_uncovered
            
            if delta_energy <= 0 or random.random() < np.exp(-delta_energy / T):
                # Accept
                current[pos] = new_ci
                current_uncovered = uncovered_after_add
                
                if len(current) < best_size:
                    best_size = len(current)
                    best = list(current)
            else:
                # Reject, revert
                cover_count[neighbors[new_ci]] -= 1
                cover_count[neighbors[old_ci]] += 1
            
            T *= cooling
        
        # After SA, do aggressive removal on the best solution found
        # But we need to recompute cover_count for best
        if best_size < len(current):
            # Use best
            final = aggressive_removal(best)
        else:
            # Use current
            final = aggressive_removal(current)
        
        return final

    def random_feasible_cover():
        """Generate a random feasible cover by random greedy."""
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        
        while not covered.all():
            if time.time() - start_time > time_limit - 10.0:
                break
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Randomly pick from top candidates
            neigh_avail = neighbors[avail_indices]
            is_covered = covered[neigh_avail]
            newly = (~is_covered).sum(axis=1)
            max_new = np.max(newly)
            if max_new == 0:
                break
            
            # Pick randomly among those with max_new
            candidates = np.where(newly == max_new)[0]
            pick = candidates[random.randint(0, len(candidates) - 1)]
            best_global = avail_indices[pick]
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
        
        return chosen, covered

    # Strategy: multiple random greedy covers, then SA + removal on each
    # We will increase the number of restarts if time permits, but keep it efficient.
    best_result = None
    best_size = float('inf')
    
    # We try a few distinct initializations
    # 1. Pure Greedy
    # 2. Random Greedy x N
    
    # Run 1: Deterministic Greedy (fast baseline)
    if time.time() - start_time < time_limit - 20.0:
        chosen, covered = greedy_cover()
        if covered.all():
            optimized = sa_swap_optimize(chosen, covered)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_result = list(optimized)
    
    # Run 2-5: Random Greedy + SA
    num_restarts = 4
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 15.0:
            break
        
        # Generate a random feasible cover
        chosen, covered = random_feasible_cover()
        
        if not covered.all():
            continue
        
        current_size = len(chosen)
        
        # Try SA swap optimization
        if time.time() - start_time < time_limit - 10.0:
            optimized = sa_swap_optimize(chosen, covered)
            if len(optimized) < current_size:
                chosen = optimized
                current_size = len(chosen)
        
        # Final aggressive removal
        if time.time() - start_time < time_limit - 5.0:
            final = aggressive_removal(chosen)
            if len(final) < current_size:
                chosen = final
                current_size = len(chosen)
        
        if current_size < best_size:
            best_size = current_size
            best_result = list(chosen)
        
        # If we've hit a good size, we can stop early
        if best_size <= 420:
            break
    
    # Final safety: if best_result is None or not fully covered, use greedy
    if best_result is None:
        chosen, covered = greedy_cover()
        best_result = aggressive_removal(chosen)
    else:
        # Verify best_result is valid
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in best_result:
            cover_count[neighbors[ci]] += 1
        if np.any(cover_count == 0):
            # Invalid, fall back
            chosen, covered = greedy_cover()
            best_result = aggressive_removal(chosen)
    
    return [all_words[i] for i in best_result]
