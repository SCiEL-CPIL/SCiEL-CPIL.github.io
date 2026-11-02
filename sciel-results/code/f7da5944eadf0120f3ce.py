import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    # The task is specific to q=3, n=8, radius=1.
    # If parameters differ, fall back to a simple heuristic.
    if q != 3 or n != 8 or radius != 1:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: random sampling or product (inefficient but valid if small enough)
        # For large spaces this would fail, but the task is fixed.
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors (balls) for all words using numpy for speed
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    balls = np.zeros((total_words, num_neighbors), dtype=np.int32)
    
    for w in range(total_words):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute ball sets for fast intersection checks if needed, 
    # but we will primarily use vectorized counting.
    # To speed up candidate search, we can precompute a lookup: 
    # for each word, which codewords cover it? No, that's dynamic.
    
    # Instead, we use the "critical set" approach:
    # For a codeword c, its critical words are those with count==1.
    # A valid replacement c' must cover all critical words.
    
    best_size = total_words
    best_set = set(range(total_words)) # Start with all words? No, start with greedy.
    
    # 1. Greedy Set Cover Initialization
    # We want a good starting point.
    # Standard greedy: pick word covering most uncovered.
    covered = np.zeros(total_words, dtype=np.bool_)
    code_indices = []
    
    # Optimization: Precompute the list of neighbors for each word as a list for quick access
    # balls is already (total, 17).
    
    while not np.all(covered):
        if time.time() > deadline - 5.0:
            break
            
        # Find word covering most uncovered
        # Vectorized approach:
        # For each word i, count how many of its neighbors are NOT covered.
        # uncovered_mask = ~covered
        # gain[i] = sum(balls[i] & uncovered_mask)
        
        # This is a matrix multiplication or sum over axis.
        # balls is (N, K). uncovered is (N,).
        # We want sum over K of balls[i, k] in uncovered.
        
        # Efficient way:
        # Create a binary matrix M where M[i, j] = 1 if j is in ball(i).
        # Then gain = M @ uncovered_vector.
        # But building M is expensive.
        
        # Alternative:
        # gain[i] = np.sum(covered[balls[i]] == 0)
        # This is a loop over N (6561). Each step is 17 lookups.
        # 6561 * 17 = ~110k ops. Very fast in Python? Maybe.
        # Let's try to vectorize the selection.
        
        # We can update gains incrementally?
        # Initially gain[i] = 17 for all.
        # When we cover a word w, we need to decrement gain[i] for all i such that w in ball(i).
        # This is the inverse mapping.
        
        # Let's precompute the inverse: for each word w, which codewords cover it?
        # coverers[w] = list of i such that w in ball(i)
        # Since the graph is symmetric (distance <= 1), coverers[w] = balls[w].
        # Yes! The covering relation is symmetric.
        # So if we pick codeword c, it covers balls[c].
        # If a word w becomes covered, it affects the gain of all c' such that w in ball(c').
        # Which is c' in balls[w].
        
        gains = np.full(total_words, num_neighbors, dtype=np.int32)
        
        while not np.all(covered):
            if time.time() > deadline - 5.0:
                break
                
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if not all covered and graph connected, but safe.
                
            # Pick one with max gain (random among ties to avoid bias? Or first?)
            # Let's pick random among those with max gain to diversify.
            candidates = np.where(gains == max_gain)[0]
            c = random.choice(candidates)
            
            code_indices.append(c)
            
            # Update covered and gains
            new_covered = balls[c]
            
            # For each word w in new_covered that was NOT covered before:
            # It is now covered.
            # We need to decrement gains for all codewords c' that cover w.
            # i.e., c' in balls[w].
            
            # Identify which words in new_covered are newly covered.
            newly = new_covered[~covered[new_covered]]
            
            if len(newly) > 0:
                # For each w in newly, find all c' in balls[w] and decrement gains[c']
                # We can do this in a loop over newly.
                # len(newly) is at most 17.
                for w in newly:
                    # coverers of w are balls[w]
                    gains[balls[w]] -= 1
                    
            covered[new_covered] = True
            
    initial_code = set(code_indices)
    
    # 2. Redundancy Removal (Pruning)
    # Iterate until no more removals possible.
    
    def prune(code_set):
        if not code_set:
            return set()
        
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            cover_count[balls[c]] += 1
            
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Find codewords that are redundant.
            # A codeword c is redundant if all w in ball(c) have cover_count[w] > 1.
            
            # Vectorized check:
            # For each c in code_set, check if min(cover_count[balls[c]]) > 1.
            # We can compute this for all c in code_set at once?
            # Let's just loop through the set, it's small (< 500).
            
            for c in list(code_set):
                # Check if any word in ball(c) has count 1
                # If min(cover_count[balls[c]]) == 1, not redundant.
                # If min > 1, redundant.
                if np.min(cover_count[balls[c]]) > 1:
                    code_set.remove(c)
                    cover_count[balls[c]] -= 1
                    changed = True
                    # Restart loop to catch cascading redundancies?
                    # Or continue? Restarting is safer for finding all.
                    # But breaking and restarting is slower.
                    # Let's just break and restart the while loop.
                    break
                    
        return code_set

    current_set = prune(initial_code)
    
    # 3. Local Search (1-Swaps)
    # Try to swap codewords to reduce size further or enable future removals.
    # Actually, 1-swaps keep size constant. We need to find a swap that allows a subsequent removal.
    # Strategy:
    # 1. Pick a codeword c.
    # 2. Identify its "critical" words U_c = {w in ball(c) | cover_count[w] == 1}.
    # 3. If U_c is empty, c is redundant (handled by prune).
    # 4. Find c' not in current_set such that U_c subset of ball(c').
    # 5. If such c' exists, perform swap: remove c, add c'.
    # 6. Re-prune. If size decreased, keep. If not, maybe keep if it leads to future decrease?
    #    Simpler: Only keep swap if it leads to immediate removal (size decrease).
    #    Or: Perform swap, prune, if size < old_size, keep. Else revert.
    
    def local_search(code_set):
        if not code_set:
            return set()
            
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            cover_count[balls[c]] += 1
            
        improved = True
        iterations = 0
        max_iterations = 5000 # Safety
        
        while improved and time.time() < deadline - 2.0 and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Shuffle candidates to avoid bias
            candidates = list(code_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if c not in code_set:
                    continue
                
                # Find critical words for c
                # U_c = words in ball(c) with count 1
                c_ball = balls[c]
                U_c_mask = cover_count[c_ball] == 1
                U_c = c_ball[U_c_mask]
                
                if len(U_c) == 0:
                    # Should have been pruned, but just in case
                    code_set.remove(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate c'
                # c' must cover all words in U_c.
                # Intersection of balls[U_c]
                # Since U_c is small (usually 1-3), this is fast.
                
                # Start with ball of first word in U_c
                # We need a set of candidates.
                # Using sets for intersection is slow in Python if done repeatedly.
                # Let's use a boolean mask or just list intersection.
                
                # Since balls are small (17 elements), we can do:
                # candidates = set(balls[U_c[0]])
                # for w in U_c[1:]:
                #    candidates &= set(balls[w])
                
                # But we want to exclude current_set.
                
                # Optimization:
                # If len(U_c) == 1:
                #    c' can be any word in balls[U_c[0]] \ {c} (and not in code_set? No, c' must be new)
                #    Actually, c' must be in balls[U_c[0]].
                #    And c' not in code_set.
                
                # Let's collect candidates.
                if len(U_c) == 1:
                    w0 = U_c[0]
                    # Potential c' are in balls[w0]
                    potential = balls[w0]
                else:
                    # Intersect
                    # Use a temporary array or set
                    # Since size is small, set intersection is fine.
                    pot_set = set(balls[U_c[0]].tolist())
                    for w in U_c[1:]:
                        pot_set.intersection_update(balls[w].tolist())
                    potential = np.array(list(pot_set), dtype=np.int32)
                
                # Filter out existing codewords
                # Create a mask of words in code_set
                # To make this fast, we can maintain a boolean array `in_code`.
                
                # Let's maintain `in_code` array outside the loop for efficiency?
                # It changes on swap.
                
                # For now, check membership.
                # potential is small.
                valid_candidates = []
                for pc in potential:
                    if pc not in code_set:
                        valid_candidates.append(pc)
                
                if not valid_candidates:
                    continue
                    
                # Score candidates
                # We want a c' that, after swap, allows a removal.
                # Hard to predict.
                # Heuristic: Pick c' that maximizes overlap with "low count" words?
                # Or just random.
                # Let's try random first.
                c_new = random.choice(valid_candidates)
                
                # Perform swap
                old_size = len(code_set)
                
                code_set.remove(c)
                cover_count[c_ball] -= 1
                
                code_set.add(c_new)
                cover_count[balls[c_new]] += 1
                
                # Re-prune
                # We need to update cover_count properly? 
                # The prune function recalculates cover_count from scratch.
                # That's O(N * |code|). N=6561, |code|~500. 3M ops. Fast enough.
                # But we already have cover_count.
                # Let's write a prune that uses existing count.
                
                # Inline prune:
                changed = True
                while changed and time.time() < deadline - 2.0:
                    changed = False
                    for cc in list(code_set):
                        if np.min(cover_count[balls[cc]]) > 1:
                            code_set.remove(cc)
                            cover_count[balls[cc]] -= 1
                            changed = True
                            break
                            
                new_size = len(code_set)
                
                if new_size < old_size:
                    improved = True
                    # Keep the change
                    break # Restart main loop with new set
                else:
                    # Revert
                    code_set.remove(c_new)
                    cover_count[balls[c_new]] -= 1
                    
                    code_set.add(c)
                    cover_count[c_ball] += 1
                    # Re-prune after revert? 
                    # The revert restores the exact previous state, so no prune needed.
                    # But wait, the prune inside might have removed other things?
                    # No, we only pruned if we kept the swap.
                    # If we revert, we are back to pre-swap state.
                    # So we just continue to next candidate.
                    
        return code_set

    # Run local search
    final_set = local_search(current_set)
    
    # Convert to list of lists
    result = []
    for idx in final_set:
        w = [0] * n
        temp = idx
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        result.append(w)
        
    return result
