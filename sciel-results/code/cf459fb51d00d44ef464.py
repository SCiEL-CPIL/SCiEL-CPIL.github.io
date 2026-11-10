import time
import random
import itertools
import sys

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for safety, though task is fixed
        # Use seed logic for other params
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return []

    total_words = q ** n  # 2187
    
    # Helper to convert int to tuple and vice versa
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

    # 1. Precompute neighbors (balls of radius 1)
    # Ball size for q=3, n=7, r=1 is 1 + 7*2 = 15
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
                new_w = tuple_to_int(new_t)
                ball.add(new_w)
        neighbors[w] = list(ball)
        
    # For faster lookup in local search, create a set for each neighbor list?
    # Or just use list and 'in' check. List of 15 elements is fine.
    # But for "covers all unique words", we check if w in neighbors[c_prime].
    # Let's precompute neighbor sets for O(1) lookup.
    neighbor_sets = [set(lst) for lst in neighbors]

    def greedy_set_cover(randomize=False):
        covered = [False] * total_words
        uncovered_count = total_words
        codewords_list = []
        available_candidates = list(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            # Optimization: Randomize candidates to break symmetry?
            # Or just pick the first one with max score.
            if randomize:
                random.shuffle(available_candidates)
            # We iterate through all available candidates.
            # To speed up, we could maintain a priority queue, but 2187 is small.
            for c in available_candidates:
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == len(neighbors[c]): # Perfect cover for this step
                        break
            
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available_candidates.remove(best_candidate)
            for w in neighbors[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        return codewords_list

    def local_search(code_list, time_limit):
        # Convert to set for fast lookup
        code_set = set(code_list)
        cover_count = [0] * total_words
        for c in code_set:
            for w in neighbors[c]:
                cover_count[w] += 1
        
        start_time = time.time()
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            # Iterate in random order to avoid bias
            indices = list(range(len(code_list)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c = code_list[i]
                can_remove = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove
                    del code_list[i]
                    code_set.remove(c)
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    improved = True
                    break # Restart scan after removal
            
            if improved:
                continue
                
            # 2. Try Swaps
            # Pick a random codeword to try swapping
            if code_list:
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                # Find unique words for c
                unique_words = []
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if unique_words:
                    # Find candidate c' not in code_set that covers all unique_words
                    best_c_prime = -1
                    best_score = -1
                    
                    # Iterate through all words not in code
                    # To speed up, we can sample, but 2187 is small enough to scan all
                    # But scanning all 2187 for every swap attempt might be slow if many attempts.
                    # Let's limit the scan or sample.
                    # Actually, 2187 * 15 checks is ~30k ops. If we do 1000 swaps/sec, that's 30M ops/sec.
                    # Python is slower. Let's try to optimize.
                    
                    # Optimization: Precompute for each word w, the list of codewords that cover it?
                    # No, cover_count is dynamic.
                    
                    # Let's just scan, but maybe break early if we find a good one.
                    # Heuristic: Prefer c_prime that covers many words with cover_count == 2.
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        
                        # Check if c_prime covers all unique_words
                        covers_all = True
                        for w in unique_words:
                            if w not in neighbor_sets[c_prime]:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score: How many words does c_prime cover that have low cover count?
                        score = 0
                        for w in neighbors[c_prime]:
                            if cover_count[w] == 2:
                                score += 1
                            elif cover_count[w] == 1:
                                # If c_prime covers a word that is unique to another codeword,
                                # that other codeword might become removable?
                                score += 2 
                        
                        if score > best_score:
                            best_score = score
                            best_c_prime = c_prime
                            if best_score > 10: # Good enough?
                                break
                    
                    if best_c_prime != -1:
                        # Perform Swap
                        # Remove c
                        del code_list[idx]
                        code_set.remove(c)
                        for w in neighbors[c]:
                            cover_count[w] -= 1
                        
                        # Add c_prime
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        for w in neighbors[best_c_prime]:
                            cover_count[w] += 1
                        
                        improved = True
                    else:
                        # No swap found for this c
                        pass
                else:
                    # Should have been removed in step 1
                    pass
            else:
                break

        return code_list

    # Execution
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s margin for safety

    # 1. Greedy
    # We will run multiple greedy instances with different random seeds to get diverse starting points
    # Then apply local search to each.
    
    best_code = None
    best_len = float('inf')
    
    # Run 1: Deterministic Greedy
    initial_code = greedy_set_cover(randomize=False)
    # Run local search
    # Split time among restarts?
    # Let's say we do 5 restarts.
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        
        if r == 0:
            current_code = initial_code
        else:
            # Randomized greedy
            current_code = greedy_set_cover(randomize=True)
            
        # Run local search with time limit
        current_code = local_search(current_code, time.time() + time_per_restart)
        
        # Check if this is better
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code
            
        # If we reached the theoretical lower bound (146), we can stop
        if best_len <= 146:
            break
            
    if best_code is None:
        best_code = initial_code # Fallback to last run
        
    # 3. Final Cleanup (Removals)
    code_set = set(best_code)
    cover_count = [0] * total_words
    for c in code_set:
        for w in neighbors[c]:
            cover_count[w] += 1
            
    final_list = list(code_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order for final cleanup
        indices = list(range(len(final_list)))
        random.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            c = final_list[i]
            can_remove = True
            for w in neighbors[c]:
                if cover_count[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                del final_list[i]
                code_set.remove(c)
                for w in neighbors[c]:
                    cover_count[w] -= 1
                improved = True

    # Convert to list of lists
    result = []
    for c in final_list:
        t = int_to_tuple(c)
        result.append(list(t))
        
    return result
