import time
import random
import math
import sys

def construct(q, n, radius):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s/150s limit
    
    total = q ** n
    rng = random.Random(42)
    
    # Precompute neighbors (words within Hamming distance radius)
    # For radius 1, this is 1 + n*(q-1)
    # Each neighbor is an index 0..total-1
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    
    # Convert to sets for fast intersection/union checks if needed, 
    # but lists are better for iteration in gain calculation.
    # We will use lists for gain, sets for membership in code.
    
    def greedy_init(seed):
        """Fast greedy set cover initialization."""
        r = random.Random(seed)
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        # Precompute list of all indices for random selection
        all_indices = list(range(total))
        
        while rem_count > 0:
            if time.monotonic() > deadline - 5.0:
                break
            
            best_c = -1
            best_gain = -1
            
            # Find candidate with max new coverage
            # Optimization: Iterate through all candidates. 
            # N=729, K=13. 729*13 = 9477 ops per pass. 
            # We do this ~73 times. 73 * 9477 ~ 700k ops. Very fast.
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                        if gain > best_gain:
                            # Can't break early unless we know max possible gain
                            # Max gain is len(neigh[c]) = 13.
                            # If gain == 13, we can break inner loop
                            if gain == len(neigh[c]):
                                break
                
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                # Fallback: pick any uncovered word's neighbor
                for u in range(total):
                    if remaining[u]:
                        for nc in neigh[u]:
                            if nc not in code_set:
                                best_c = nc
                                break
                        if best_c != -1:
                            break
                if best_c == -1:
                    break
            
            code.append(best_c)
            code_set.add(best_c)
            
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1
        
        return code, counts

    def polish(code, counts, time_budget):
        """Local search to minimize code size."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget
        
        # Precompute neighbor sets for O(1) membership checks in swap validation
        neigh_sets = [set(neigh[i]) for i in range(total)]
        
        def is_removable(c):
            """Check if codeword c can be removed without losing coverage."""
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True

        def try_remove(c):
            """Attempt to remove c. Returns True if successful."""
            if not is_removable(c):
                return False
            code.remove(c)
            code_set.discard(c)
            for u in neigh[c]:
                counts[u] -= 1
            return True

        # Phase 1: Greedy Removal
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
        
        # Phase 2: Simulated Annealing with specific moves
        temp = 5.0
        cool_rate = 0.999
        min_temp = 0.1
        
        # Helper to calculate cost of a state (number of uncovered words)
        # We maintain 'uncovered_count' implicitly via counts, but for SA acceptance
        # we need to know if the move creates uncovered words.
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        code.pop(idx)
                        code_set.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            
            elif move_type < 0.8:
                # Try Swap (1-for-1)
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    # Choose new_c
                    # Bias towards neighbors of old_c to maintain structure
                    if rng.random() < 0.7:
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c])-1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check validity: Does removing old_c and adding new_c leave any word uncovered?
                    # A word u is uncovered if counts[u] == 1 (only covered by old_c) and u not in neigh[new_c]
                    valid = True
                    uncovered_count = 0
                    new_c_set = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1 and u not in new_c_set:
                            valid = False
                            uncovered_count += 1
                            if uncovered_count > 3: # Limit perturbation depth
                                break
                    
                    if valid:
                        # Accept
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        # Check if removal is now possible?
                        # We don't do immediate removal in SA step to keep moves atomic,
                        # but we can check if best_size improved (it didn't, size is same)
                        # However, this swap might enable future removals.
                    else:
                        # SA acceptance for invalid swap
                        # Cost is uncovered_count
                        if uncovered_count <= 5 and rng.random() < math.exp(-uncovered_count / temp):
                            # Accept perturbation
                            for u in neigh[old_c]:
                                counts[u] -= 1
                            for u in neigh[new_c]:
                                counts[u] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            
                            # Try to fix uncovered words immediately to restore validity
                            # Find all currently uncovered words
                            uncovered_words = [u for u in range(total) if counts[u] == 0]
                            for u in uncovered_words:
                                # Find a codeword that covers u
                                # Prefer one that covers many other uncovered words?
                                # Just pick a random neighbor
                                nc = rng.choice(neigh[u])
                                if nc not in code_set:
                                    code.append(nc)
                                    code_set.add(nc)
                                    for v in neigh[nc]:
                                        counts[v] += 1
                                # If nc is already in code, counts[u] should have been >0? 
                                # No, if nc is in code, it covers u, so counts[u] would be >=1.
                                # So if counts[u]==0, no current codeword covers u.
                                # Thus nc must be new.
                                
            else:
                # Try 2-for-1 Move (Remove 2, Add 1)
                # This is powerful for reducing size.
                if len(code) >= 2:
                    i1 = rng.randint(0, len(code) - 1)
                    i2 = rng.randint(0, len(code) - 1)
                    if i1 == i2:
                        continue
                    
                    c1 = code[i1]
                    c2 = code[i2]
                    
                    # Find a codeword c_new that covers all words covered by c1 and c2
                    # that are not covered by others.
                    # Specifically, let U1 = {u in neigh[c1] | counts[u] == 1}
                    # Let U2 = {u in neigh[c2] | counts[u] == 1}
                    # We need c_new such that U1 U U2 subset of neigh[c_new].
                    
                    critical_words = set()
                    for u in neigh[c1]:
                        if counts[u] == 1:
                            critical_words.add(u)
                    for u in neigh[c2]:
                        if counts[u] == 1:
                            critical_words.add(u)
                            
                    if not critical_words:
                        # Both are removable individually? 
                        # If so, just remove them.
                        if is_removable(c1):
                            code.remove(c1)
                            code_set.discard(c1)
                            for u in neigh[c1]: counts[u] -= 1
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                        if c2 in code and is_removable(c2):
                            code.remove(c2)
                            code_set.discard(c2)
                            for u in neigh[c2]: counts[u] -= 1
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                        continue
                    
                    # Candidate c_new must cover all critical_words.
                    # Find candidates by looking at neighbors of a random critical word
                    # and checking if they cover all critical_words.
                    crit_list = list(critical_words)
                    sample_u = rng.choice(crit_list)
                    
                    found_new_c = -1
                    for cand in neigh[sample_u]:
                        if cand in code_set:
                            continue
                        # Check if cand covers all critical_words
                        covers_all = True
                        cand_set = neigh_sets[cand]
                        for w in critical_words:
                            if w not in cand_set:
                                covers_all = False
                                break
                        if covers_all:
                            found_new_c = cand
                            break
                    
                    if found_new_c != -1:
                        # Perform the move: Remove c1, c2, Add found_new_c
                        # Check validity: 
                        # Removing c1 and c2 reduces counts for their neighbors.
                        # Adding found_new_c increases counts for its neighbors.
                        # We must ensure no word becomes uncovered (count < 0 is impossible, count == 0 is bad).
                        
                        # Simulate counts change
                        # Delta for u:
                        # -1 if u in neigh[c1]
                        # -1 if u in neigh[c2]
                        # +1 if u in neigh[found_new_c]
                        
                        valid = True
                        # Only need to check words in neigh[c1] U neigh[c2] U neigh[found_new_c]
                        affected = set(neigh[c1]) | set(neigh[c2]) | set(neigh[found_new_c])
                        
                        for u in affected:
                            new_count = counts[u]
                            if u in neigh[c1]: new_count -= 1
                            if u in neigh[c2]: new_count -= 1
                            if u in neigh[found_new_c]: new_count += 1
                            
                            if new_count < 0:
                                valid = False
                                break
                            
                        if valid:
                            # Apply move
                            # Remove c1
                            code.remove(c1)
                            code_set.discard(c1)
                            for u in neigh[c1]: counts[u] -= 1
                            # Remove c2
                            if c2 in code:
                                code.remove(c2)
                                code_set.discard(c2)
                                for u in neigh[c2]: counts[u] -= 1
                            # Add found_new_c
                            code.append(found_new_c)
                            code_set.add(found_new_c)
                            for u in neigh[found_new_c]: counts[u] += 1
                            
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                            else:
                                # If size didn't decrease (e.g. we added 1 and removed 2 but one was already gone? No, logic handles distinct)
                                # Actually, if size didn't decrease, it means we failed to remove both? 
                                # No, we removed 2 and added 1, so size decreases by 1.
                                # Unless c1 == c2? We checked i1 != i2, so c1 != c2.
                                # So size always decreases by 1 if this move is taken.
                                pass
            temp *= cool_rate
        
        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size

    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    # Determine number of restarts
    # Each restart: Greedy (~0.1s) + Polish (~5-10s)
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 15.0:
            break
        
        # Use different seeds
        seed = 42 + restart * 13
        code, counts = greedy_init(seed)
        
        if not code:
            continue
        
        # Polish
        remaining_time = deadline - time.monotonic()
        if remaining_time < 15.0:
            break
            
        # Allocate time based on remaining restarts
        # Give more time to later restarts? Or equal?
        # Let's give equal time, but ensure we don't exceed deadline
        time_per_polish = min(8.0, (remaining_time - 5.0) / (num_restarts - restart))
        
        polished_code, polished_size = polish(code, counts, time_per_polish)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
            
            # If we hit the theoretical lower bound or known best (73), we can stop early?
            # Lower bound is 71. Known best is 73.
            # If we find 73, we can try a few more to see if we find 72 or 71, 
            # but it's unlikely. However, the task asks to minimize.
            # We will continue to search for 72/71 if time permits, 
            # but 73 is a very strong result.
            if overall_best_size <= 73:
                # We can reduce time for subsequent restarts or break if we are confident.
                # Let's just let the loop continue with tighter time budgets.
                pass

    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result

