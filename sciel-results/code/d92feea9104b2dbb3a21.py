import time
import random
import itertools
import sys
import heapq

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for safety, though task is fixed
        # Use seed logic for other params
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return []

    total_words = q ** n
    
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

    def greedy_set_cover(randomize=True):
        covered = [False] * total_words
        uncovered_count = total_words
        codewords_list = []
        available_candidates = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            best_tiebreaker = -1
            
            # If randomizing, shuffle candidates to break symmetry
            candidates = list(available_candidates)
            if randomize:
                random.shuffle(candidates)
            
            for c in candidates:
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                # Tie-breaking: random or based on word value
                tie = random.randint(0, 1000) if randomize else c
                if score > best_score or (score == best_score and tie > best_tiebreaker):
                    best_score = score
                    best_tiebreaker = tie
                    best_candidate = c
                    if best_score == len(neighbors[c]):
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
        code_set = set(code_list)
        cover_count = [0] * total_words
        for c in code_set:
            for w in neighbors[c]:
                cover_count[w] += 1
        
        start_time = time.time()
        no_improvement_count = 0
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            for i in range(len(code_list) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                c = code_list[i]
                can_remove = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    del code_list[i]
                    code_set.remove(c)
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    improved = True
                    no_improvement_count = 0
                    break
            
            if improved:
                continue
                
            # 2. Try Swaps
            if code_list:
                # Pick a random codeword
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                unique_words = []
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if unique_words:
                    best_c_prime = -1
                    best_score = -1
                    
                    # Scan for best replacement
                    # Optimization: If unique_words is large, few candidates will work.
                    # If unique_words is small (e.g., 1), many candidates work.
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        
                        # Check if c_prime covers all unique_words
                        covers_all = True
                        for w in unique_words:
                            if w not in neighbors[c_prime]:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score: Prefer c_prime that "helps" other codewords become removable.
                        # This means c_prime should cover words that are currently covered by exactly 2 codewords.
                        # By adding c_prime, those words become covered by 3, but wait...
                        # Actually, we are REMOVING c. So words in neighbors[c] \ unique_words lose a cover.
                        # Words in neighbors[c_prime] gain a cover.
                        
                        # Let's analyze the change in "removability" of other codewords.
                        # A codeword c_other is removable if all its neighbors have count >= 2.
                        # If we swap c -> c_prime:
                        # - For w in neighbors[c] \ unique_words: count decreases by 1.
                        #   If count was 2, it becomes 1. This might make c_other (if it covers w) NON-removable?
                        #   No, if count becomes 1, w is unique to c_other. c_other is NOT removable.
                        #   So we want to AVOID covering words with count 2 in neighbors[c] \ unique_words?
                        #   Actually, c_prime might cover some of these.
                        
                        # Simpler heuristic: Maximize the number of words in neighbors[c_prime] that have count == 2.
                        # Why? If c_prime covers a word w with count 2, w becomes count 3.
                        # The other codeword covering w (say c_x) now has w covered by c_prime and c_x (and maybe others).
                        # If c_x had w as its ONLY unique word, c_x is still not removable if count(w) becomes 1?
                        # Wait. If count(w) was 2 (covered by c and c_x), and we remove c, count(w) becomes 1 (unique to c_x).
                        # If c_prime ALSO covers w, count(w) becomes 2 (covered by c_x and c_prime).
                        # This preserves the "removability" of c_x (it doesn't become stuck with a unique word).
                        # If c_prime did NOT cover w, w becomes unique to c_x, making c_x potentially non-removable (if w was the only reason it wasn't removable before? No, if w was count 2, c_x was removable if all its other neighbors were count >= 2. If w becomes 1, c_x is NOT removable).
                        
                        # So, we WANT c_prime to cover words in neighbors[c] that have count 2 (and are not in unique_words).
                        # Let's call this set "critical_words".
                        
                        score = 0
                        for w in neighbors[c_prime]:
                            # We care about words that were covered by c (and thus will lose a cover)
                            # If w is in neighbors[c] and not in unique_words, its count drops by 1.
                            # If its count was 2, it becomes 1. This is bad for removability of the other coverer.
                            # If c_prime covers w, it adds 1 back. Net change 0.
                            # So we want c_prime to cover as many "count 2" words from neighbors[c] as possible.
                            
                            if w in neighbors[c] and cover_count[w] == 2:
                                score += 10
                            elif cover_count[w] == 1:
                                # Covering a unique word of another codeword makes that word count 2.
                                # This helps the other codeword become removable (if it was stuck due to this word).
                                score += 5
                            
                        if score > best_score:
                            best_score = score
                            best_c_prime = c_prime
                            if best_score > 20: # Early break if very good
                                break
                    
                    if best_c_prime != -1:
                        # Perform Swap
                        del code_list[idx]
                        code_set.remove(c)
                        for w in neighbors[c]:
                            cover_count[w] -= 1
                        
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        for w in neighbors[best_c_prime]:
                            cover_count[w] += 1
                        
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                break
                
            if no_improvement_count > 500:
                # Restart or break? 
                # If we are stuck, breaking this instance and restarting might be better.
                # But we handle restarts in the main loop.
                break

        return code_list

    # Execution
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s margin for output conversion
    
    best_code = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Greedy is fast (~0.1s). Local search takes the rest.
    # Let's try 5 restarts with ~20s each.
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Randomized Greedy
        initial_code = greedy_set_cover(randomize=True)
        
        # Local Search
        current_code = local_search(initial_code, time.time() + time_per_restart)
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code
            
        if best_len <= 156: # Lower bound
            break
            
    if best_code is None:
        best_code = initial_code
        
    # Final Cleanup
    code_set = set(best_code)
    cover_count = [0] * total_words
    for c in code_set:
        for w in neighbors[c]:
            cover_count[w] += 1
            
    final_list = list(code_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(final_list) - 1, -1, -1):
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
    
    result = []
    for c in final_list:
        t = int_to_tuple(c)
        result.append(list(t))
        
    return result
