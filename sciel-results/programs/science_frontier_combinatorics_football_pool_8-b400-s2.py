import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n  # 6561
    
    # Precompute digits for all words for final output
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists
    # For q=3, n=8, radius=1: 1 + 8*(3-1) = 17 neighbors
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)
        
    def generate_greedy_code():
        cov_count = np.zeros(total_words, dtype=np.int32)
        is_code = np.zeros(total_words, dtype=bool)
        code_list = []
        
        # Potential: number of uncovered neighbors
        # Initially all 17 are uncovered
        potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        uncovered_remaining = total_words
        
        while uncovered_remaining > 0 and time.time() < deadline - 30.0:
            # Find word covering most uncovered points
            # Using argmax on potential is O(N). With N=6561, it's fast.
            # We could use a heap, but argmax on numpy is very fast (C loop)
            c = np.argmax(potential)
            if potential[c] == 0:
                break
                
            code_list.append(c)
            is_code[c] = True
            potential[c] = 0 # Prevent re-picking
            
            neigh_c = neighbors[c]
            for x in neigh_c:
                if cov_count[x] == 0:
                    cov_count[x] = 1
                    uncovered_remaining -= 1
                    # Update potentials of neighbors of x
                    potential[neighbors[x]] -= 1
                else:
                    cov_count[x] += 1
        return code_list, cov_count, is_code

    def perform_local_search(code, cov_count, is_code, max_time):
        # code: list of ints
        # cov_count: numpy array
        # is_code: numpy array bool
        end_time = time.time() + max_time
        random.seed(42)
        stuck_counter = 0
        best_code = code[:]
        best_size = len(code)
        best_cov_count = cov_count.copy()
        best_is_code = is_code.copy()
        
        while time.time() < end_time:
            if len(code) == 0:
                break
                
            # 1. Try Deletion
            idx = random.randint(0, len(code) - 1)
            c = code[idx]
            
            # Check if all neighbors are covered > 1 time
            # If yes, remove c
            if np.all(cov_count[neighbors[c]] > 1):
                cov_count[neighbors[c]] -= 1
                is_code[c] = False
                code.pop(idx)
                stuck_counter = 0
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cov_count = cov_count.copy()
                    best_is_code = is_code.copy()
                continue

            # 2. Deletion failed. Check if c is removable by swapping?
            # Find risky points: neighbors of c with coverage exactly 1
            neigh_c = neighbors[c]
            risky_mask = cov_count[neigh_c] == 1
            risky_points = neigh_c[risky_mask]
            
            # If no risky points, it means we *could* delete, but np.all check above 
            # would have caught it if ALL were >1. 
            # Wait, if np.all(cov_count[neighbors[c]] > 1) is False, then at least one is 1.
            # So risky_points is non-empty if deletion failed.
            
            if len(risky_points) == 0:
                # Should not happen if deletion check failed, but safety
                cov_count[neighbors[c]] -= 1
                is_code[c] = False
                code.pop(idx)
                stuck_counter = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cov_count = cov_count.copy()
                    best_is_code = is_code.copy()
                continue

            # Find a candidate c2 such that c2 covers all risky_points
            # c2 must be in intersection of N(r) for all r in risky_points
            # And c2 should not be in current code (to maintain size or reduce, 
            # but if we swap, size stays same. To reduce size we need deletion.
            # However, sometimes a swap allows a subsequent deletion.
            # Also, c2 must not be c itself.
            
            # Start with neighbors of first risky point
            # Note: N(r) includes r itself and its neighbors.
            # c2 must be in N(r) means dist(c2, r) <= 1.
            
            candidates = set(neighbors[risky_points[0]].tolist())
            candidates.discard(c) # c is already removed? No, we are considering swapping c -> c2.
            # c is in code. c2 must be NOT in code.
            
            for r in risky_points[1:]:
                candidates &= set(neighbors[r].tolist())
                if not candidates:
                    break
            
            # Filter out current code words
            code_set = set(code)
            valid_candidates = [c2 for c2 in candidates if c2 not in code_set]
            
            if valid_candidates:
                # Pick a random candidate
                c2 = random.choice(valid_candidates)
                
                # Perform Swap: remove c, add c2
                # 1. Remove c
                cov_count[neighbors[c]] -= 1
                is_code[c] = False
                code.pop(idx)
                code.append(c2) # Keep list, index doesn't matter for logic, just distinctness
                is_code[c2] = True
                
                # 2. Add c2
                cov_count[neighbors[c2]] += 1
                
                # Since c2 covers all risky points, and c was the only cover for them,
                # coverage of risky points is restored.
                # Other points in N(c) \ N(c2) lose coverage from c.
                # But we checked that they had coverage > 1 (or were not risky).
                # If a point was in N(c) but NOT in N(c2), and had cov_count 1, it would have been in risky_points.
                # So all points in N(c) that had count 1 are in N(c2).
                # Points with count > 1 remain covered.
                # So the swap is always valid in terms of coverage!
                
                stuck_counter = 0
            else:
                stuck_counter += 1
                
                # If stuck a lot, maybe try a random perturbation?
                # For now, just continue.
                
            # Check for improvement (size didn't change, so no improvement in size)
            # But we might escape local minima for future deletions.
            # We don't update best_code here because size is same.
            
            # Optional: Occasionally try to delete multiple? No, single step.
            
        return best_code, best_cov_count, best_is_code

    # 1. Initialize
    code, cov_count, is_code = generate_greedy_code()
    
    # 2. Local Search
    # Use remaining time for search
    time_left = deadline - time.time()
    code, cov_count, is_code = perform_local_search(code, cov_count, is_code, time_left)
    
    # 3. Final Verification and Cleanup
    # Recompute counts to be sure
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        final_cov_count[neighbors[c]] += 1
        
    if not np.all(final_cov_count > 0):
        # Fallback to greedy if search broke something (shouldn't happen)
        code, _, _ = generate_greedy_code()
        
    return [list(digits_of[i]) for i in code]
