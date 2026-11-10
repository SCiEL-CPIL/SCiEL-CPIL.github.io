import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy set cover
        total_words = q ** n
        def int_to_tuple(w):
            digits = []
            for _ in range(n):
                digits.append(w % q)
                w //= q
            return tuple(reversed(digits))
        
        def tuple_to_int(t):
            w = 0
            mult = 1
            for d in reversed(t):
                w += d * mult
                mult *= q
            return w

        neighbors = [None] * total_words
        for w in range(total_words):
            t = int_to_tuple(w)
            ball = {w}
            for i in range(n):
                orig = t[i]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_t = list(t)
                    new_t[i] = new_val
                    ball.add(tuple_to_int(tuple(new_t)))
            neighbors[w] = list(ball)
        
        covered = [False] * total_words
        uncovered_count = total_words
        codewords_list = []
        available_candidates = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            for c in available_candidates:
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available_candidates.remove(best_candidate)
            for w in neighbors[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        
        result = [list(int_to_tuple(c)) for c in codewords_list]
        return result

    # Specialized solver for q=3, n=7, r=1
    total_words = 2187
    
    # Precompute neighbors
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    # neighbors_list[w] is a list of integers
    # neighbors_set[w] is a set of integers (for O(1) lookup if needed, but we mostly iterate)
    neighbors_list = [None] * total_words
    for w in range(total_words):
        ball = [w]
        for i in range(n):
            p = powers[i]
            digit = (w // p) % 3
            if digit != 0:
                ball.append(w - digit * p)
            if digit != 1:
                ball.append(w + (1 - digit) * p)
            if digit != 2:
                ball.append(w + (2 - digit) * p)
        neighbors_list[w] = ball

    # Precompute a numpy array for fast vectorized operations if needed
    # covers[c, w] is True if c covers w
    # This is a 2187x2187 boolean matrix. ~4.7 MB. Fast to create.
    covers = np.zeros((total_words, total_words), dtype=bool)
    for c in range(total_words):
        covers[c, neighbors_list[c]] = True

    def greedy_set_cover():
        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        codewords_list = []
        available = set(range(total_words))
        
        # Precompute neighbor arrays for vectorized scoring
        neighbor_arrays = [np.array(neighbors_list[w], dtype=np.int32) for w in range(total_words)]
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            # Randomize tie-breaking to explore different greedy paths
            candidates = list(available)
            random.shuffle(candidates)
            
            for c in candidates:
                # Count uncovered neighbors using numpy
                score = np.sum(~covered[neighbor_arrays[c]])
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == 15:
                        break
            
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available.remove(best_candidate)
            for w in neighbors_list[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        return codewords_list

    def local_search(code_list, time_limit):
        code_set = set(code_list)
        # cover_count[w] = number of codewords covering w
        cover_count = np.zeros(total_words, dtype=np.int16)
        for c in code_set:
            cover_count[neighbors_list[c]] += 1
            
        start_time = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        no_improvement_count = 0
        
        # Precompute neighbor arrays for vectorized operations
        neighbor_arrays = [np.array(neighbors_list[w], dtype=np.int32) for w in range(total_words)]
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            indices = list(range(len(code_list)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > time_limit:
                    break
                c = code_list[idx]
                # Check if any neighbor has cover_count 1
                if np.min(cover_count[neighbor_arrays[c]]) > 1:
                    # Remove
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbor_arrays[c]] -= 1
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    no_improvement_count = 0
                    break # Restart scan
            
            if improved:
                continue
                
            # 2. Try Swaps (vectorized scoring)
            if code_list:
                # Pick a random codeword to potentially swap out
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                # Find unique words (cover_count == 1) for c
                unique_mask = (cover_count[neighbor_arrays[c]] == 1)
                unique_words = neighbor_arrays[c][unique_mask]
                
                if len(unique_words) > 0:
                    # Find candidate c' not in code_set that covers all unique_words
                    # Vectorize: for each c_prime, check if it covers all unique_words
                    # covers[c_prime, unique_words] should be all True
                    
                    best_c_prime = -1
                    best_score = -1
                    
                    # Precompute penalty for this c (words with count 2 that will become 1)
                    non_unique_mask = ~unique_mask
                    at_risk_words = neighbor_arrays[c][non_unique_mask]
                    penalty = np.sum(cover_count[at_risk_words] == 2) * 2
                    
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        
                        # Check if c_prime covers all unique_words
                        if not np.all(covers[c_prime, unique_words]):
                            continue
                        
                        # Score the swap using vectorized operations
                        # Gain: words in neighbors[c_prime] with low cover_count
                        c_prime_neighbors = neighbor_arrays[c_prime]
                        gain = np.sum(cover_count[c_prime_neighbors] == 1) * 2 + np.sum(cover_count[c_prime_neighbors] == 2)
                        
                        net_score = gain - penalty
                        
                        if net_score > best_score:
                            best_score = net_score
                            best_c_prime = c_prime
                            if best_score > 10: # Good enough
                                break
                    
                    if best_c_prime != -1:
                        # Perform Swap
                        del code_list[idx]
                        code_set.remove(c)
                        cover_count[neighbor_arrays[c]] -= 1
                        
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        cover_count[neighbor_arrays[best_c_prime]] += 1
                        
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                break
            
            # 3. Try 2-for-1 replacement: remove 2 codewords, add 1 that covers their union
            if not improved and len(code_list) >= 2 and time.time() < time_limit - 1:
                # Pick 2 random codewords
                i1, i2 = random.sample(range(len(code_list)), 2)
                c1, c2 = code_list[i1], code_list[i2]
                
                # Words that would become uncovered if we remove both c1 and c2
                # A word w is "protected" if it's covered by at least one of c1, c2
                # And if we remove both, w becomes uncovered only if cover_count[w] == 1 and w is in ball(c1) or ball(c2)
                # Actually: w becomes uncovered iff cover_count[w] == 1 and w in (ball(c1) union ball(c2))
                # But we need cover_count[w] >= 1 for w in ball(c1) or ball(c2)
                # After removing c1 and c2: cover_count[w] decreases by (1 if w in ball(c1) else 0) + (1 if w in ball(c2) else 0)
                # w becomes uncovered iff new_count == 0, i.e., old_count == number of removals covering w
                
                ball1 = set(neighbors_list[c1])
                ball2 = set(neighbors_list[c2])
                union_ball = ball1 | ball2
                
                # Find words that would become uncovered
                critical_words = []
                for w in union_ball:
                    removals = (1 if w in ball1 else 0) + (1 if w in ball2 else 0)
                    if cover_count[w] == removals:
                        critical_words.append(w)
                
                if len(critical_words) <= 15:
                    # Try to find a single c' that covers all critical_words
                    best_c_prime = -1
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        if np.all(covers[c_prime, critical_words]):
                            best_c_prime = c_prime
                            break
                    
                    if best_c_prime != -1:
                        # Check if this is an improvement (it should be, since we remove 2 add 1)
                        # Remove c1 and c2, add c_prime
                        # Need to handle indices carefully
                        if i1 > i2:
                            i1, i2 = i2, i1
                        c1, c2 = code_list[i2], code_list[i1]
                        
                        del code_list[i1]
                        code_set.remove(c1)
                        cover_count[neighbor_arrays[c1]] -= 1
                        
                        del code_list[i2]
                        code_set.remove(c2)
                        cover_count[neighbor_arrays[c2]] -= 1
                        
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        cover_count[neighbor_arrays[best_c_prime]] += 1
                        
                        improved = True
                        no_improvement_count = 0
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
            
            # 4. Perturbation if stuck
            if no_improvement_count > 15 and code_list:
                # Remove a random few codewords
                num_to_remove = max(1, len(code_list) // 15)
                remove_indices = random.sample(range(len(code_list)), num_to_remove)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    c = code_list[idx]
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbor_arrays[c]] -= 1
                
                # Re-fill greedily
                covered = (cover_count > 0)
                uncovered_count = int(np.sum(~covered))
                
                # Greedy fill
                available = set(range(total_words)) - code_set
                while uncovered_count > 0 and time.time() < time_limit:
                    best_candidate = -1
                    best_score = -1
                    candidates = list(available)
                    random.shuffle(candidates)
                    
                    for c in candidates:
                        score = int(np.sum(~covered[neighbor_arrays[c]]))
                        if score > best_score:
                            best_score = score
                            best_candidate = c
                            if best_score == 15:
                                break
                    
                    if best_candidate == -1:
                        break
                    code_list.append(best_candidate)
                    code_set.add(best_candidate)
                    for w in neighbors_list[best_candidate]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                    cover_count[neighbor_arrays[best_candidate]] += 1
                    available.remove(best_candidate)
                
                no_improvement_count = 0

        return best_code

    # Execution
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Run multiple greedy passes and keep the best
    best_initial = None
    best_initial_size = float('inf')
    greedy_count = 0
    
    while time.time() < deadline - 15.0: # Leave 15s for local search on best
        initial_code = greedy_set_cover()
        greedy_count += 1
        if len(initial_code) < best_initial_size:
            best_initial_size = len(initial_code)
            best_initial = initial_code
            # If we find a very good one, we can break early?
            # 186 is the target. If we find < 190, it's good.
            if best_initial_size <= 190:
                break
        # After a few good runs, stop if we're not improving
        if greedy_count > 5 and best_initial_size <= 192:
            break
        
    # 2. Local Search on the best initial code
    if best_initial is None:
        best_initial = greedy_set_cover()
        
    final_code = local_search(best_initial, deadline - 2.0)
    
    # 3. Final Cleanup
    code_set = set(final_code)
    cover_count = np.zeros(total_words, dtype=np.int16)
    for c in code_set:
        cover_count[neighbors_list[c]] += 1
        
    final_list = list(code_set)
    neighbor_arrays = [np.array(neighbors_list[w], dtype=np.int32) for w in range(total_words)]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(final_list)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            c = final_list[idx]
            if np.min(cover_count[neighbor_arrays[c]]) > 1:
                del final_list[idx]
                code_set.remove(c)
                cover_count[neighbor_arrays[c]] -= 1
                improved = True

    # Convert to list of lists
    result = []
    for c in final_list:
        digits = []
        w = c
        for _ in range(n):
            digits.append(w % q)
            w //= q
        result.append(digits[::-1])
        
    return result
