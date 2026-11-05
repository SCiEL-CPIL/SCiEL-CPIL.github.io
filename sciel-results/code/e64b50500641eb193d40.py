import numpy as np
import time
import itertools
import random

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

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx

    # Generate the van Laarhoven 486-word code
    def generate_van_laarhoven_code():
        """
        Generate the 486-word ternary covering code for n=8, radius=1.
        This is based on the construction by van Laarhoven, Aarts, van Lint and Wille (1989).
        
        The code is constructed using a specific pattern. We'll use a known good initial
        set and then optimize it.
        """
        # Use a greedy construction as a starting point, then optimize
        # We'll generate multiple random greedy codes and pick the best
        return None  # Placeholder, we'll use greedy instead

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        # Build reverse mapping
        rev_neighbors = [[] for _ in range(total_points)]
        for b in range(total_points):
            for p in neighbors[b]:
                rev_neighbors[p].append(b)
        
        while np.any(uncovered_counts > 0):
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    for b_prime in rev_neighbors[p]:
                        uncovered_counts[b_prime] -= 1
            
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
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = neighbors[idx]
                if np.all(coverage_count[ball] > 1):
                    rem_idx = selected_list.pop(i)
                    coverage_count[neighbors[rem_idx]] -= 1
                    changed = True
                    break
        return selected_list

    def defect_local_search(selected_indices, deadline):
        """
        Defect minimization local search:
        - Start with a valid code
        - Iteratively try to remove codewords or swap them to reduce total code size
        - Use precomputed neighbor intersections to evaluate swaps efficiently
        """
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
        
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(123)
        selected_set = set(selected_list)
        
        # Precompute: for each pair of codewords, the number of points covered by both
        # This helps identify which swaps are most beneficial
        # But this is O(m^2 * ball_size) which might be too slow for large m
        # Instead, we'll use a simpler approach: for each codeword, compute how many
        # points it uniquely covers
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            improved = False
            
            # Phase 1: Try to remove redundant codewords
            for i in range(len(selected_list)):
                c = selected_list[i]
                ball = neighbors[c]
                if np.all(coverage_count[ball] > 1):
                    coverage_count[ball] -= 1
                    selected_list.pop(i)
                    selected_set.remove(c)
                    improved = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swaps - move a codeword to a better position
            # For each codeword, check if moving it to a neighbor position reduces total size
            c = selected_list[rng.integers(len(selected_list))]
            ball_c = neighbors[c]
            
            # Points that would become uncovered if we remove c
            risky_mask = coverage_count[ball_c] == 1
            num_risky = np.sum(risky_mask)
            
            if num_risky == 0:
                # c is redundant, but we already checked above. This shouldn't happen.
                continue
            
            # Try moving c to each of its neighbors
            best_swap = None
            best_defect = num_risky  # Current "defect" if we remove c
            
            for pos in range(1, ball_size):
                c_prime = neighbors[c][pos]
                
                if c_prime in selected_set:
                    continue
                
                ball_prime = neighbors[c_prime]
                
                # If we swap c -> c_prime:
                # Points that lose coverage: ball_c \ ball_prime (where count was 1)
                # Points that gain coverage: ball_prime \ ball_c
                # Net change in covered points: |ball_prime \ ball_c| - |ball_c \ ball_prime|
                # But we only care about points that were uniquely covered by c
                
                # Points in ball_c that are NOT in ball_prime and had count 1
                # These would become uncovered
                # We need to check: for each point in ball_c with count 1, is it in ball_prime?
                
                # Vectorized approach:
                # risky_points are points in ball_c with coverage_count == 1
                # A swap is valid if all risky_points are in ball_prime
                
                if num_risky > 0:
                    # Check if all risky points are covered by c_prime
                    risky_points = ball_c[risky_mask]
                    # Check which risky points are in ball_prime
                    # Use set intersection for efficiency
                    if len(risky_points) <= 5:  # Small number, use direct check
                        all_covered = True
                        for rp in risky_points:
                            if not np.any(ball_prime == rp):
                                all_covered = False
                                break
                    else:
                        # Use np.isin for larger sets
                        all_covered = np.all(np.isin(risky_points, ball_prime))
                    
                    if not all_covered:
                        continue
                
                # The swap is valid. Now check if it enables further pruning
                # Simulate the swap
                coverage_count[ball_c] -= 1
                coverage_count[ball_prime] += 1
                
                # Check if any codeword becomes redundant
                can_prune = False
                for j in range(len(selected_list)):
                    if j == selected_list.index(c):
                        continue
                    cj = selected_list[j]
                    ball_j = neighbors[cj]
                    if np.all(coverage_count[ball_j] > 1):
                        can_prune = True
                        break
                
                if can_prune:
                    # This swap enables a prune, which is good
                    best_swap = c_prime
                    best_defect = -1  # Prefer swaps that enable pruning
                    break
                else:
                    # Revert
                    coverage_count[ball_prime] -= 1
                    coverage_count[ball_c] += 1
            
            if best_swap is not None:
                # Perform the swap
                idx_in_list = selected_list.index(c)
                selected_list[idx_in_list] = best_swap
                selected_set.remove(c)
                selected_set.add(best_swap)
                
                # Update coverage
                coverage_count[ball_c] -= 1
                coverage_count[neighbors[best_swap]] += 1
                
                # Try to prune
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    selected_list = pruned
                    selected_set = set(selected_list)
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in selected_list:
                        coverage_count[neighbors[idx]] += 1
                    
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                else:
                    # Keep the swap even if no immediate prune, as it may help later
                    pass
            
            # Phase 3: Random perturbation to escape local minima
            if rng.random() < 0.1:  # 10% chance of perturbation
                if len(selected_list) > 10:
                    # Remove a random codeword and re-add a better one
                    c = selected_list[rng.integers(len(selected_list))]
                    ball_c = neighbors[c]
                    coverage_count[ball_c] -= 1
                    
                    idx_in_list = selected_list.index(c)
                    selected_list.pop(idx_in_list)
                    selected_set.remove(c)
                    
                    # Find a codeword that covers the most newly uncovered points
                    newly_uncovered = np.where(coverage_count == 0)[0]
                    if len(newly_uncovered) > 0:
                        best_new = None
                        best_coverage = 0
                        for candidate in neighbors[newly_uncovered[0]]:
                            if candidate not in selected_set:
                                cov = np.sum(coverage_count[neighbors[candidate]] == 0)
                                if cov > best_coverage:
                                    best_coverage = cov
                                    best_new = candidate
                        if best_new is not None:
                            selected_list.append(best_new)
                            selected_set.add(best_new)
                            coverage_count[neighbors[best_new]] += 1
        
        return best_code

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Run greedy instances to get a good starting point
    best_code = None
    best_len = float('inf')
    
    for seed in range(3):
        if time.time() > deadline - 60:
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
    
    # Defect minimization local search
    best_code = defect_local_search(best_code, deadline)
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
