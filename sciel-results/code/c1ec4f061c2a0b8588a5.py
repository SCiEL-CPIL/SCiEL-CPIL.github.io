import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    total_words = q ** n
    
    # 1. Precompute neighbors (balls of radius 1)
    # For q=3, n=8, radius=1, ball size is 1 + 8*2 = 17
    balls = []
    for i in range(total_words):
        # Convert i to digits
        d = []
        temp = i
        for _ in range(n):
            d.append(temp % q)
            temp //= q
        d.reverse() # Most significant digit first? Doesn't matter as long as consistent.
        
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                # Convert back to int
                val = 0
                for digit in d:
                    val = val * q + digit
                neighbors.append(val)
            d[pos] = orig
        balls.append(neighbors)
        
    # Precompute inverse index: for each word, which centers cover it?
    # inverse[w] = list of c such that w in balls[c]
    inverse = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls[c]:
            inverse[w].append(c)

    # 2. Greedy Initialization
    uncovered = [True] * total_words
    code = []
    cover_count = [0] * total_words
    
    # Greedy: pick center that covers most uncovered points
    while any(uncovered) and time.time() < deadline:
        best_center = -1
        max_gain = 0
        # We can optimize this, but for 6561 words, a simple loop might be slow if done 500 times.
        # 500 * 6561 * 17 ~ 55M ops. Might be tight in Python.
        # Let's use a faster heuristic or just random init + repair.
        # Actually, let's use a "random greedy" or just start with a random set and repair.
        # Better: Start with a random set of size ~500, then repair uncovered points.
        
        # Let's try a hybrid: Greedy for first 100, then stop and let local search do the work.
        # Or just use a structured initialization.
        
        # Let's stick to a fast greedy using numpy if possible, or just a simple loop.
        # Given the small size, let's just do a simple greedy but limit iterations or use randomization.
        
        # Actually, a better init:
        # Randomly select ~450 codewords.
        # Then repair: while there are uncovered points, pick a random uncovered point, 
        # and add a random center that covers it.
        
        random.seed(42)
        initial_size = 450
        code = random.sample(range(total_words), initial_size)
        
        # Calculate cover_count
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        # Repair uncovered points
        uncovered = [w for w in range(total_words) if cover_count[w] == 0]
        while uncovered and time.time() < deadline:
            w = uncovered.pop()
            # Pick a random center from inverse[w]
            candidates = inverse[w]
            c = random.choice(candidates)
            if c not in code:
                code.append(c)
                for w2 in balls[c]:
                    cover_count[w2] += 1
            # Re-check if any other points are still uncovered? 
            # This is inefficient. 
            # Let's just do a full greedy pass on the remaining uncovered.
            
        # Actually, let's just use a proper greedy set cover. It's the most reliable init.
        # Optimize greedy:
        # Precompute ball as numpy array?
        # balls_np = np.array(balls)
        
        # Let's restart with a cleaner greedy implementation.
        code = []
        cover_count = [0] * total_words
        uncovered_mask = [1] * total_words # 1 if uncovered
        
        # To speed up greedy, we can just pick random centers until coverage is high, then trim.
        # But let's try to be smart.
        
        # We'll use a "Greedy with Randomization" (Lazy Greedy)
        # Or just:
        # 1. Random init with 400 codewords.
        # 2. Repair loop: find uncovered, add best center.
        
        code = random.sample(range(total_words), 400)
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        while True:
            if time.time() > deadline - 20: break # Leave time for search
            uncovered = [w for w in range(total_words) if cover_count[w] == 0]
            if not uncovered:
                break
            # Pick one uncovered point
            w = uncovered[0]
            # Find candidate centers that cover w
            candidates = inverse[w]
            # Pick the one that covers the most other uncovered points
            best_c = candidates[0]
            max_cov = 0
            for c in candidates:
                cov = 0
                for w2 in balls[c]:
                    if cover_count[w2] == 0:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_c = c
            code.append(best_c)
            for w2 in balls[best_c]:
                cover_count[w2] += 1

    # 3. Local Search: Trim and Swap
    code_set = set(code)
    
    def is_redundant(c):
        # Check if all points in balls[c] are covered by at least 2 codewords
        # i.e. min(cover_count[w] for w in balls[c]) >= 2
        # If we remove c, cover_count[w] becomes cover_count[w]-1.
        # We need cover_count[w]-1 >= 1 => cover_count[w] >= 2.
        for w in balls[c]:
            if cover_count[w] == 1:
                return False
        return True

    def remove_code(c):
        code_set.remove(c)
        for w in balls[c]:
            cover_count[w] -= 1
        code.remove(c)

    def add_code(c):
        code_set.add(c)
        for w in balls[c]:
            cover_count[w] += 1
        code.append(c)

    def swap_code(old_c, new_c):
        code_set.remove(old_c)
        code_set.add(new_c)
        code[code.index(old_c)] = new_c
        
        for w in balls[old_c]:
            cover_count[w] -= 1
        for w in balls[new_c]:
            cover_count[w] += 1

    # Phase 1: Aggressive Trimming
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            if is_redundant(c):
                remove_code(c)
                changed = True

    # Phase 2: Simulated Annealing for Swaps
    # Objective: Minimize len(code). 
    # Move: Swap c with c'. 
    # Constraint: Coverage must be maintained.
    # If we swap c with c', we must ensure that all points previously covered ONLY by c 
    # are now covered by c' (or others).
    # Unique points of c: U_c = {w in balls[c] | cover_count[w] == 1}
    # If U_c is empty, we can just remove c (handled in trimming).
    # If U_c is not empty, c' MUST cover all points in U_c.
    
    temp = 10.0
    no_improve_count = 0
    
    while time.time() < deadline:
        if temp < 0.01:
            break
            
        improved = False
        
        # Try to remove redundant codes first
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline: break
            c = code[i]
            if is_redundant(c):
                remove_code(c)
                improved = True
                # Restart loop after removal as indices changed
                break
        
        if improved:
            continue
            
        # Try swaps
        # Shuffle code to avoid bias
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline: break
            
            c = code[idx]
            # Find unique points
            unique = [w for w in balls[c] if cover_count[w] == 1]
            
            if not unique:
                # Should have been removed, but maybe race condition or just redundant
                remove_code(c)
                improved = True
                break
                
            # Find candidates c' that cover all points in 'unique'
            # Start with candidates for the first unique point
            candidates = set(inverse[unique[0]])
            # Intersect with inverse of other unique points
            for w in unique[1:]:
                candidates.intersection_update(inverse[w])
                
            # Remove current code and any code that is already in the set
            candidates.difference_update(code_set)
            
            if not candidates:
                continue
                
            # Evaluate candidates
            # We want to pick a candidate that minimizes the "cost"
            # Cost could be the number of new unique points created by c'
            # Or simply pick a random one to allow exploration.
            
            best_candidate = None
            best_cost = float('inf')
            
            # Limit candidates to avoid slow loops
            candidate_list = list(candidates)
            if len(candidate_list) > 50:
                random.shuffle(candidate_list)
                candidate_list = candidate_list[:50]
                
            for c_prime in candidate_list:
                # Simulate swap
                # Calculate new unique points for c_prime
                # Note: cover_count is for the CURRENT state.
                # If we swap c -> c_prime:
                # cover_count_new[w] = cover_count[w] - 1 (if w in balls[c]) + 1 (if w in balls[c_prime])
                
                new_unique_count = 0
                for w in balls[c_prime]:
                    # Calculate effective cover count if we do the swap
                    count = cover_count[w]
                    if w in balls[c]:
                        count -= 1
                    # If w is in balls[c_prime], count += 1 (already added in logic above? 
                    # No, we are checking balls[c_prime], so the +1 is implicit for w in balls[c_prime].
                    # So effective count is cover_count[w] - (1 if w in balls[c] else 0) + 1
                    # Wait, w is in balls[c_prime]. So it gets +1.
                    # Does it get -1? Only if w is also in balls[c].
                    eff_count = cover_count[w] + 1
                    if w in balls[c]:
                        eff_count -= 1
                    
                    if eff_count == 1:
                        new_unique_count += 1
                        
                # We want to minimize new_unique_count
                if new_unique_count < best_cost:
                    best_cost = new_unique_count
                    best_candidate = c_prime
                    
                    if best_cost == 0:
                        break # Perfect swap, no new unique points
                
            if best_candidate is not None:
                # Accept swap?
                # If best_cost is low, it's good.
                # If best_cost is high, it might be bad, but SA allows it.
                # Delta E = best_cost (roughly)
                if best_cost <= 1 or random.random() < np.exp(-best_cost / temp):
                    swap_code(c, best_candidate)
                    improved = True
                    break # Restart after swap
        
        if not improved:
            no_improve_count += 1
            temp *= 0.95
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0
            temp *= 0.99 # Cool down slowly if improving

    # Final Trim
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            if is_redundant(c):
                remove_code(c)
                changed = True

    # Convert to required format
    result = []
    for c in code:
        d = []
        temp_c = c
        for _ in range(n):
            d.append(temp_c % q)
            temp_c //= q
        d.reverse()
        result.append(d)

    return result
