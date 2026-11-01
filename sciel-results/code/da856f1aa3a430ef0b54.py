import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit
    
    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: all words (worst case) or greedy
        total = q ** n
        # Precompute balls
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = q ** i
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)
        
        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
            max_idx = np.argmax(counts)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(len(balls)):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break
        
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result

    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17
    
    # Precompute balls as lists of integers
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(ball)
    
    # Precompute adjacency: for each word y, which codewords x have y in their ball
    # Due to symmetry of Hamming distance, y in balls[x] iff x in balls[y]
    # So we can use balls as the inverse mapping
    # But for faster updates, precompute as numpy arrays
    ball_arrays = np.array(balls, dtype=np.int32)  # shape (6561, 17)
    
    # Precompute: for each position i and value v, the mask of words with that value
    # This helps with fast coverage updates
    # Actually, let's use a different approach: precompute the "co-occurrence" structure
    
    # For vectorized greedy: when we pick codeword x, we need to decrement counts
    # for all codewords y that share any word in their ball with x's ball.
    # Two codewords x and y have overlapping balls iff d(x,y) <= 2
    # So we need to know, for each x, which y are within distance 2 of x.
    
    # Precompute distance-2 neighborhoods for each codeword
    # dist2_neighbors[x] = list of y such that d(x,y) <= 2
    dist2_neighbors = []
    for x in range(total):
        digits_x = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits_x.append(d)
            tmp //= 3
        
        neighbors = [x]  # distance 0
        # Distance 1
        for i in range(8):
            for new_d in range(3):
                if new_d != digits_x[i]:
                    y = x + (new_d - digits_x[i]) * (3 ** i)
                    neighbors.append(y)
        # Distance 2
        for i in range(8):
            for j in range(i+1, 8):
                for new_di in range(3):
                    if new_di == digits_x[i]:
                        continue
                    for new_dj in range(3):
                        if new_dj == digits_x[j]:
                            continue
                        y = x + (new_di - digits_x[i]) * (3 ** i) + (new_dj - digits_x[j]) * (3 ** j)
                        neighbors.append(y)
        dist2_neighbors.append(neighbors)
    
    dist2_arrays = np.array(dist2_neighbors, dtype=np.int32)
    
    # Current best solution
    best_size = total
    best_solution = None
    
    # Helper function for local search
    def local_search(initial_set, time_limit):
        current_set = set(initial_set)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[ball_arrays[c]] += 1
        
        def is_redundant(c):
            return np.all(cover_count[ball_arrays[c]] > 1)
        
        best_local = len(current_set)
        best_local_set = set(current_set)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_set:
                    continue
                if is_redundant(c):
                    cover_count[ball_arrays[c]] -= 1
                    current_set.remove(c)
                    improved = True
                    if len(current_set) < best_local:
                        best_local = len(current_set)
                        best_local_set = set(current_set)
        
        return list(best_local_set)
    
    # Helper: greedy cover with vectorized count updates
    def greedy_cover(noise_scale=0.0, time_limit=None):
        covered = np.zeros(total, dtype=bool)
        counts = np.full(total, ball_size, dtype=np.int32)
        if noise_scale > 0:
            noise = np.random.uniform(0, 1, size=total)
            counts_float = counts.astype(float) + noise * noise_scale
        else:
            counts_float = counts.astype(float)
        
        selected = []
        
        while True:
            if time_limit is not None and time.time() > time_limit:
                break
            max_idx = int(np.argmax(counts_float))
            if counts[max_idx] == 0:
                break
            
            selected.append(max_idx)
            
            # Get newly covered words
            ball = ball_arrays[max_idx]
            new_covered = ball[~covered[ball]]
            
            if len(new_covered) == 0:
                # No new coverage, skip (shouldn't happen if counts are correct)
                counts_float[max_idx] = -1
                continue
            
            # For each newly covered word y, decrement counts of all codewords
            # that are within distance 2 of max_idx (since they share y in their ball)
            # Actually, we need to decrement counts of codewords z such that
            # z's ball contains y. This means d(z, y) <= 1.
            # So for each new y, we need to find all z with d(z,y) <= 1.
            # That's balls[y] = ball_arrays[y]
            
            # Vectorized approach: for each new y, get ball_arrays[y] and decrement
            # But we need to be careful not to double-decrement if multiple new y's
            # point to the same z.
            
            # Simpler: just iterate over new_covered and use ball_arrays
            for y in new_covered:
                covered[y] = True
                # Decrement counts for all z in ball_arrays[y]
                z_list = ball_arrays[y]
                counts[z_list] -= 1
                counts_float[z_list] -= 1
            
            if np.all(covered):
                break
        
        return selected
    
    # Strategy 1: Standard greedy
    if time.time() < deadline - 20:
        greedy_result = greedy_cover(time_limit=deadline - 20)
        if len(greedy_result) < best_size:
            best_size = len(greedy_result)
            best_solution = greedy_result[:]
        
        # Local search on greedy result
        if time.time() < deadline - 10:
            ls_result = local_search(greedy_result, deadline - 10)
            if len(ls_result) < best_size:
                best_size = len(ls_result)
                best_solution = ls_result[:]
    
    # Strategy 2: Latin hypercube initialization
    # Generate diverse codewords by ensuring each position has balanced digit distribution
    # and maximizing minimum pairwise distance
    
    def latin_hypercube_init(target_size, time_limit):
        """Generate diverse codewords using Latin hypercube sampling."""
        # Create a pool of candidate codewords with good diversity
        # Use a greedy packing approach: pick codewords far from existing ones
        
        selected = []
        # Start with a random codeword
        selected.append(random.randint(0, total - 1))
        
        # Compute distances from selected to all others
        # For efficiency, maintain a "min_dist" array
        min_dist = np.full(total, 100, dtype=np.int32)
        
        # Precompute digits for all words for fast distance computation
        # Actually, let's just compute on the fly for the first few
        
        while len(selected) < target_size and time.time() < time_limit:
            # Find the word with maximum min_dist
            max_idx = int(np.argmax(min_dist))
            if min_dist[max_idx] <= 1:
                # All remaining words are within distance 1 of some selected word
                # Pick randomly from uncovered
                break
            selected.append(max_idx)
            
            # Update min_dist for all words
            # Compute distance from max_idx to all others
            digits_new = []
            tmp = max_idx
            for i in range(8):
                d = tmp % 3
                digits_new.append(d)
                tmp //= 3
            
            for x in range(total):
                if x in selected:
                    continue
                # Compute Hamming distance
                tmp = x
                dist = 0
                for i in range(8):
                    d = tmp % 3
                    if d != digits_new[i]:
                        dist += 1
                    tmp //= 3
                if dist < min_dist[x]:
                    min_dist[x] = dist
            
            # Mark selected as having distance 0
            min_dist[max_idx] = 0
        
        return selected
    
    if time.time() < deadline - 15:
        # Start with a diverse set of ~50 codewords
        diverse_init = latin_hypercube_init(50, deadline - 25)
        
        # Use these as starting points for greedy
        # Modify greedy to start with initial set
        covered = np.zeros(total, dtype=bool)
        counts = np.full(total, ball_size, dtype=np.int32)
        selected = list(diverse_init)
        
        # Apply initial coverage
        for c in selected:
            ball = ball_arrays[c]
            new_covered = ball[~covered[ball]]
            for y in new_covered:
                covered[y] = True
                z_list = ball_arrays[y]
                counts[z_list] -= 1
        
        # Continue greedy
        while True:
            if time.time() > deadline - 15:
                break
            max_idx = int(np.argmax(counts))
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            
            ball = ball_arrays[max_idx]
            new_covered = ball[~covered[ball]]
            if len(new_covered) == 0:
                counts[max_idx] = -1
                continue
            
            for y in new_covered:
                covered[y] = True
                z_list = ball_arrays[y]
                counts[z_list] -= 1
            
            if np.all(covered):
                break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected[:]
        
        # Local search
        if time.time() < deadline - 8:
            ls_result = local_search(selected, deadline - 8)
            if len(ls_result) < best_size:
                best_size = len(ls_result)
                best_solution = ls_result[:]
    
    # Strategy 3: Multiple random restarts with noisy greedy
    for restart in range(8):
        if time.time() > deadline - 8:
            break
        noise_scales = [0.05, 0.1, 0.2, 0.03, 0.15, 0.08, 0.25, 0.06]
        ns = noise_scales[restart % len(noise_scales)]
        
        greedy_result = greedy_cover(noise_scale=ns, time_limit=deadline - 8)
        if len(greedy_result) < best_size:
            best_size = len(greedy_result)
            best_solution = greedy_result[:]
        
        # Local search
        if time.time() < deadline - 5:
            ls_result = local_search(greedy_result, deadline - 5)
            if len(ls_result) < best_size:
                best_size = len(ls_result)
                best_solution = ls_result[:]
    
    # Strategy 4: Final polish - try add-remove swaps
    if best_solution is not None and time.time() < deadline - 3:
        current_set = set(best_solution)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[ball_arrays[c]] += 1
        
        # Try removing redundant codewords
        for c in list(current_set):
            if np.all(cover_count[ball_arrays[c]] > 1):
                cover_count[ball_arrays[c]] -= 1
                current_set.remove(c)
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
        
        # Try 2-for-1 swaps: remove 2, add 1 if it improves
        if time.time() < deadline - 2:
            # Find pairs of redundant codewords
            redundant = [c for c in current_set if np.all(cover_count[ball_arrays[c]] > 1)]
            random.shuffle(redundant)
            
            for i in range(min(len(redundant), 20)):
                if time.time() > deadline - 2:
                    break
                if i >= len(redundant):
                    break
                c1 = redundant[i]
                if c1 not in current_set:
                    continue
                
                # Temporarily remove c1
                cover_count[ball_arrays[c1]] -= 1
                uncovered_by_c1 = ball_arrays[c1][cover_count[ball_arrays[c1]] == 0]
                
                if len(uncovered_by_c1) == 0:
                    # c1 is truly redundant, already handled above
                    cover_count[ball_arrays[c1]] += 1
                    continue
                
                # Find a codeword that covers all of uncovered_by_c1
                # Check if any codeword covers all these words
                # For each candidate z, check if all words in uncovered_by_c1 are in ball_arrays[z]
                # This is expensive, so limit candidates
                candidates_to_try = list(current_set - {c1})
                random.shuffle(candidates_to_try)
                
                for c2 in candidates_to_try[:50]:
                    if c2 not in current_set:
                        continue
                    if np.all(cover_count[ball_arrays[c2]] > 1):
                        # Temporarily remove c2
                        cover_count[ball_arrays[c2]] -= 1
                        uncovered_by_c2 = ball_arrays[c2][cover_count[ball_arrays[c2]] == 0]
                        
                        if len(uncovered_by_c2) == 0:
                            # Both are redundant, remove both
                            current_set.remove(c1)
                            current_set.remove(c2)
                            if len(current_set) < best_size:
                                best_size = len(current_set)
                                best_solution = list(current_set)
                            # Restore cover_count
                            cover_count[ball_arrays[c1]] += 1
                            cover_count[ball_arrays[c2]] += 1
                            break
                        else:
                            # Check if one codeword can cover both sets
                            # Find z that covers uncovered_by_c1 and uncovered_by_c2
                            all_uncovered = np.unique(np.concatenate([uncovered_by_c1, uncovered_by_c2]))
                            # Check if there's a codeword whose ball contains all of all_uncovered
                            # For each w in all_uncovered, find candidates in ball_arrays[w]
                            # Intersect these sets
                            if len(all_uncovered) <= 4:
                                candidate_set = set(range(total))
                                for w in all_uncovered:
                                    candidate_set &= set(ball_arrays[w].tolist())
                                for z in candidate_set:
                                    if z not in current_set:
                                        # Try adding z
                                        current_set.add(z)
                                        current_set.remove(c1)
                                        current_set.remove(c2)
                                        if len(current_set) < best_size:
                                            best_size = len(current_set)
                                            best_solution = list(current_set)
                                        # Restore
                                        current_set.remove(z)
                                        current_set.add(c1)
                                        current_set.add(c2)
                                        break
                            cover_count[ball_arrays[c2]] += 1
                
                # Restore c1
                cover_count[ball_arrays[c1]] += 1
    
    # Convert best_solution to list of digit lists
    if best_solution is None:
        # Fallback: use standard greedy
        best_solution = greedy_cover(time_limit=deadline - 2)
    
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)
    
    return result
