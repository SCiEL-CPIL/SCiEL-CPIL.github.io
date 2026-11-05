import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    # Precompute digits for all words
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

    # --- Phase 1: Fast Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []

    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words

    # Precompute the union of neighbors for each potential center
    # This allows O(1) lookup of which points a center can cover
    # But we still need to update potential efficiently
    
    # Use a different greedy strategy: prioritize centers that cover the most uncovered points
    # We'll maintain a priority queue-like structure using numpy
    
    while uncovered_remaining > 0 and time.time() < deadline - 30.0:
        # Find the center that covers the most currently uncovered points
        # potential[x] = number of uncovered points that x can cover
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                # For each neighbor y of x, x was uncovered, so y loses one potential
                # But we need to be careful: only if y could cover x
                # Actually, potential[y] counts uncovered points in neighbors[y]
                # If x was uncovered and x is in neighbors[y], then potential[y] should decrease
                # So we need to check which centers y have x in their neighbor list
                # This is expensive to do naively
                
                # Alternative: just recompute potential for affected centers
                # For now, let's use a simpler update:
                # When x becomes covered, any center y that covers x should have its potential decreased
                # We can precompute reverse neighbors: for each point x, which centers cover it
                # But that's the same as neighbors[x]
                # So: for each y in neighbors[x], if y is still a potential center, potential[y] -= 1
                # But we need to make sure we don't double-count
                
                # Actually, the current code has a bug: it does potential[neighbors[x]] -= 1
                # which is correct: for each center y that can cover x, if x was uncovered,
                # then y's potential decreases by 1
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # --- Phase 2: Iterated Local Search with improved moves ---
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)

    random.seed(42)
    stuck_counter = 0
    no_improve_counter = 0

    def try_remove(c, idx):
        """Try to remove codeword c at index idx. Returns True if successful."""
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            return True
        return False

    def try_swap(c1, c1_idx):
        """Try to swap c1 with a better codeword. Returns True if successful."""
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size
        # Find risky points: x in neighbors[c1] where cov_count[x] == 1
        risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]

        if not risky_points:
            return try_remove(c1, c1_idx)

        # Find c2 not in code such that c2 covers all risky points.
        candidates = set(neighbors[risky_points[0]].tolist())
        candidates.discard(c1)

        for x in risky_points[1:]:
            candidates &= set(neighbors[x].tolist())
            if not candidates:
                break

        valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]

        if not valid_candidates:
            return False

        # Pick the candidate that covers the most additional points (greedy heuristic)
        best_c2 = None
        best_gain = -1
        for c2 in valid_candidates[:20]:  # Limit to first 20 for speed
            # Count how many new points c2 covers
            gain = 0
            for x in neighbors[c2]:
                if current_cov_count[x] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c2 = c2

        if best_c2 is None:
            return False

        c2 = best_c2
        current_cov_count[neighbors[c1]] -= 1
        current_is_code[c1] = False
        current_code_set.remove(c1)
        current_code[c1_idx] = c2

        current_cov_count[neighbors[c2]] += 1
        current_is_code[c2] = True
        current_code_set.add(c2)
        return True

    def try_two_for_one(c1, c1_idx, c2, c2_idx):
        """Try to remove two codewords c1, c2 and add one that covers all their uniquely covered points."""
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size
        # Find points uniquely covered by c1 or c2
        risky = set()
        for x in neighbors[c1]:
            if current_cov_count[x] == 1:
                risky.add(x)
        for x in neighbors[c2]:
            if current_cov_count[x] == 1:
                risky.add(x)
        
        if not risky:
            # Both can be removed
            if try_remove(c1, c1_idx) and try_remove(c2, c2_idx - 1 if c2_idx > c1_idx else c2_idx):
                return True
            return False
        
        risky_list = list(risky)
        # Find a codeword that covers all risky points
        candidates = set(neighbors[risky_list[0]].tolist())
        candidates.discard(c1)
        candidates.discard(c2)
        for x in risky_list[1:]:
            candidates &= set(neighbors[x].tolist())
            if not candidates:
                break
        
        valid_candidates = [c for c in candidates if c not in current_code_set]
        if not valid_candidates:
            return False
        
        c_new = random.choice(valid_candidates)
        
        # Remove c1 and c2
        if c1_idx < c2_idx:
            current_code.pop(c2_idx)
            current_code.pop(c1_idx)
        else:
            current_code.pop(c1_idx)
            current_code.pop(c2_idx)
        current_size -= 2
        
        current_cov_count[neighbors[c1]] -= 1
        current_is_code[c1] = False
        current_code_set.remove(c1)
        current_cov_count[neighbors[c2]] -= 1
        current_is_code[c2] = False
        current_code_set.remove(c2)
        
        # Add c_new
        current_code.append(c_new)
        current_size += 1
        current_cov_count[neighbors[c_new]] += 1
        current_is_code[c_new] = True
        current_code_set.add(c_new)
        
        return True

    while time.time() < deadline - 5.0:
        if current_size == 0:
            break
        
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        made_progress = False
        
        # 1. Try to remove a word
        if try_remove(c, idx):
            made_progress = True
            stuck_counter = 0
            no_improve_counter = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            no_improve_counter += 1
        
        # 2. If stuck, try a swap
        if not made_progress and stuck_counter > 3 and time.time() < deadline - 10.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            if try_swap(c1, c1_idx):
                made_progress = True
                stuck_counter = 0
                no_improve_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
        
        # 3. If still stuck, try a 2-for-1 move
        if not made_progress and stuck_counter > 10 and time.time() < deadline - 15.0 and current_size > 1:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            c2_idx = random.randint(0, current_size - 1)
            c2 = current_code[c2_idx]
            
            if c1_idx != c2_idx:
                if try_two_for_one(c1, c1_idx, c2, c2_idx):
                    made_progress = True
                    stuck_counter = 0
                    no_improve_counter = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
        
        if made_progress:
            stuck_counter = 0
        else:
            stuck_counter += 1

    # --- Phase 3: Simulated Annealing / Random Restart Polish ---
    # Use the best code found so far as a starting point for a more aggressive polish.
    # We perform a series of random restarts or aggressive local searches on the best code.
    
    # First, a deterministic removal pass
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    # If we still have time, try a few random restarts from the best solution
    # with a higher probability of accepting non-improving moves (simulated annealing style)
    # or simply more aggressive swaps.
    
    # Let's try a few random restarts of the local search starting from the best_code
    # but with a different random seed and maybe a different order of operations.
    
    if time.time() < deadline - 10.0:
        # Try 3 random restarts
        for restart in range(3):
            if time.time() >= deadline - 5.0:
                break
                
            # Start from a copy of the best code
            rc_code = best_code[:]
            rc_size = len(rc_code)
            rc_cov = np.zeros(total_words, dtype=np.int32)
            for idx in rc_code:
                rc_cov[neighbors[idx]] += 1
            rc_set = set(rc_code)
            
            # Local search with random restart logic
            # We can try to swap out codewords more aggressively.
            # For simplicity, we just run a few more iterations of the same loop but with a new seed.
            random.seed(42 + restart * 1000)
            
            stuck = 0
            while time.time() < deadline - 5.0 - restart * 2.0:
                if rc_size == 0:
                    break
                idx = random.randint(0, rc_size - 1)
                c = rc_code[idx]
                
                if np.all(rc_cov[neighbors[c]] > 1):
                    rc_cov[neighbors[c]] -= 1
                    rc_set.remove(c)
                    rc_code.pop(idx)
                    rc_size -= 1
                    stuck = 0
                    
                    if rc_size < best_size:
                        best_size = rc_size
                        best_code = rc_code[:]
                else:
                    stuck += 1
                    
                if stuck > 5:
                    c1_idx = random.randint(0, rc_size - 1)
                    c1 = rc_code[c1_idx]
                    risky_points = [x for x in neighbors[c1] if rc_cov[x] == 1]
                    
                    if not risky_points:
                        rc_cov[neighbors[c1]] -= 1
                        rc_set.remove(c1)
                        rc_code.pop(c1_idx)
                        rc_size -= 1
                        stuck = 0
                        if rc_size < best_size:
                            best_size = rc_size
                            best_code = rc_code[:]
                    else:
                        candidates = set(neighbors[risky_points[0]].tolist())
                        candidates.discard(c1)
                        for x in risky_points[1:]:
                            candidates &= set(neighbors[x].tolist())
                            if not candidates:
                                break
                        valid_candidates = [c2 for c2 in candidates if c2 not in rc_set]
                        if valid_candidates:
                            c2 = random.choice(valid_candidates)
                            rc_cov[neighbors[c1]] -= 1
                            rc_set.remove(c1)
                            rc_code[c1_idx] = c2
                            rc_cov[neighbors[c2]] += 1
                            rc_set.add(c2)
                            stuck = 0
                    stuck = 0

    # Final verification and cleanup
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    # One last removal pass
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]

    return [list(digits_of[i]) for i in final_code]
