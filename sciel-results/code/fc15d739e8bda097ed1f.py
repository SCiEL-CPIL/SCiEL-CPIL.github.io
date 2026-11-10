import time
import random
import itertools
import sys
import numpy as np

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for safety, though task is fixed
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
        
    # Precompute neighbor sets for O(1) lookup in local search
    neighbor_sets = [set(lst) for lst in neighbors]

    def generate_random_linear_code(dim):
        """Generates a random linear code of given dimension over GF(3)."""
        # Generate random basis vectors
        basis = []
        while len(basis) < dim:
            v = np.random.randint(0, 3, n)
            # Check linear independence
            # We can do this by checking if v is in the span of current basis
            # For small dim (4) and small field (3), we can just generate all combinations
            # and check if v is unique among them.
            # Or simpler: Gaussian elimination on the matrix of basis + v.
            if not basis:
                if np.any(v != 0):
                    basis.append(v)
                continue
            
            # Check if v is linearly dependent on basis
            # We want to see if v can be written as sum a_i * b_i
            # This is a system of linear equations.
            # Since dim is small, we can just iterate all possible coefficients
            # but that's 3^k. For k=4, that's 81 checks.
            # A faster way for small n:
            # Let M be the matrix with basis vectors as rows.
            # Augment with v. Perform Gaussian elimination mod 3.
            # If the last row becomes [0..0 | non-zero], it's independent.
            
            # Actually, simpler:
            # Just generate a random matrix of size dim x n, check rank.
            # If rank < dim, retry.
            # For dim=4, n=7, probability of full rank is high.
            
            # Let's just use a simple check:
            # Create all linear combinations of current basis.
            # If v is not in the set, it's independent.
            span = {tuple(np.zeros(n, dtype=int))}
            for coeff in itertools.product(range(3), repeat=len(basis)):
                vec = np.zeros(n, dtype=int)
                for i, c in enumerate(coeff):
                    if c != 0:
                        vec += c * basis[i]
                vec = vec % 3
                span.add(tuple(vec))
            
            if tuple(v) not in span:
                basis.append(v)

        # Generate all codewords
        codewords = []
        for coeff in itertools.product(range(3), repeat=dim):
            vec = np.zeros(n, dtype=int)
            for i, c in enumerate(coeff):
                if c != 0:
                    vec += c * basis[i]
            vec = vec % 3
            # Convert to int
            w = 0
            mult = 1
            for d in reversed(vec):
                w += d * mult
                mult *= q
            codewords.append(w)
        
        return codewords

    def greedy_extend(initial_codewords, time_limit):
        """Extends an initial set of codewords to a full cover using greedy selection."""
        covered = [False] * total_words
        uncovered_count = total_words
        code_set = set(initial_codewords)
        
        # Mark words covered by initial codewords
        for c in code_set:
            for w in neighbors[c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        
        if uncovered_count == 0:
            return list(code_set)

        available_candidates = [w for w in range(total_words) if w not in code_set]
        
        while uncovered_count > 0:
            if time.time() > time_limit:
                break
                
            best_candidate = -1
            best_score = -1
            
            # To speed up, we can sample candidates if the list is large, 
            # but 2187 is small.
            # However, checking all 2100 candidates every step might be slow if many steps.
            # Greedy usually converges in ~100-150 steps. 100 * 2100 * 15 ops ~ 3M ops. Fast enough.
            
            for c in available_candidates:
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == 15: # Max possible
                        break
            
            if best_candidate == -1:
                break
                
            code_set.add(best_candidate)
            available_candidates.remove(best_candidate)
            
            for w in neighbors[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    
        return list(code_set)

    def local_search(code_list, time_limit):
        """Refines the code using removals and swaps."""
        code_set = set(code_list)
        cover_count = [0] * total_words
        for c in code_set:
            for w in neighbors[c]:
                cover_count[w] += 1
        
        start_time = time.time()
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
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
                        # We want to increase redundancy of other codewords to enable removal.
                        score = 0
                        for w in neighbors[c_prime]:
                            if cover_count[w] == 2:
                                score += 1
                            elif cover_count[w] == 1:
                                # This word is unique to some other codeword c''.
                                # If we swap c with c', c'' might become removable?
                                # No, c'' still covers w.
                                # But c' covers w too. So cover_count[w] becomes 2.
                                # This makes c'' potentially removable if it has no other unique words.
                                score += 2 
                        
                        if score > best_score:
                            best_score = score
                            best_c_prime = c_prime
                            if best_score > 10: # Good enough
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
                else:
                    # Should have been removed in step 1
                    pass
            else:
                break

        return code_list

    # Execution
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s margin for safety

    best_code = None
    best_len = float('inf')
    
    # Strategy:
    # 1. Run a few iterations of "Random Linear Code Init + Greedy Extend + Local Search"
    # 2. Also run standard Greedy + Local Search as a baseline/fallback
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Determine time budget for this restart
        remaining_time = deadline - time.time()
        time_per_restart = remaining_time / (num_restarts - r)
        
        # Allocate time: 20% for init/greedy, 80% for local search
        time_init = time_per_restart * 0.2
        time_ls = time_per_restart * 0.8
        
        # 1. Generate Initial Code
        if r < 5:
            # First 5 restarts: Random Linear Code (Dimension 4)
            # This is the "structural" idea
            initial_code = generate_random_linear_code(4)
            # Greedy Extend
            current_code = greedy_extend(initial_code, time.time() + time_init)
        else:
            # Next 5 restarts: Standard Greedy (for diversity)
            # We can just use a randomized greedy or standard greedy
            # Let's use standard greedy for r=5, randomized for r>5
            # But greedy_extend handles the extension. 
            # For standard greedy, we start with empty set.
            initial_code = []
            current_code = greedy_extend(initial_code, time.time() + time_init)
            
        if not current_code:
            continue
            
        # 2. Local Search
        current_code = local_search(current_code, time.time() + time_ls)
        
        # 3. Check if better
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code
            
        # If we reached the theoretical lower bound (146), we can stop
        if best_len <= 146:
            break
            
    if best_code is None:
        # Fallback: Just run a quick greedy if everything failed
        best_code = greedy_extend([], time.time() + 5.0)
        if not best_code:
            return []

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

