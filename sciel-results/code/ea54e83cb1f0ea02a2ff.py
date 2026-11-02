import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    total = q ** n
    # For q=3, n=7, total=2187. Ball size = 1 + 7*2 = 15.
    
    # 1. Precompute word coordinates and ball indices
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball indices (Hamming distance <= 1)
    ball_indices = []
    for i in range(total):
        w = words[i]
        ball = [i]
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                # Calculate new index
                # Index is base-q representation. 
                # i = sum(w[j] * q^(n-1-j))
                # Change w[coord] to v:
                # diff = (v - orig) * q^(n-1-coord)
                power = q ** (n - 1 - coord)
                new_idx = i + (v - orig) * power
                ball.append(new_idx)
        ball_indices.append(np.array(ball, dtype=np.int32))
        
    # Precompute coverers: coverers[p] is a boolean mask of length total, 
    # where coverers[p][c] is True if c covers p (i.e., p is in ball_indices[c])
    # This is the transpose of the ball_mask.
    # Memory: 2187 * 2187 bytes ~ 4.7 MB. Acceptable.
    coverers = np.zeros((total, total), dtype=bool)
    for c in range(total):
        coverers[ball_indices[c], c] = True
        
    # Helper to check time
    def time_ok():
        return time.time() < deadline

    def greedy_init(order):
        remaining = np.ones(total, dtype=bool)
        code = []
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                # Count how many remaining points c covers
                new_cov = np.count_nonzero(remaining & coverers[:, c])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == 15: # Max ball size
                        break
            code.append(best_c)
            remaining[coverers[:, best_c]] = False
        return code

    def local_search(code_set, cnt):
        """
        Aggressive local search to minimize code size.
        Returns (code_set, cnt, improved_bool)
        """
        improved = True
        while improved and time_ok():
            improved = False
            
            # 1. Deletion Pass
            for c in list(code_set):
                # Check if all points in ball(c) have cnt >= 2
                if np.all(cnt[ball_indices[c]] >= 2):
                    code_set.remove(c)
                    cnt[ball_indices[c]] -= 1
                    improved = True
            
            if not time_ok():
                break

            # 2. 1-for-1 Swap Pass
            # We look for c in code_set and w not in code_set
            # such that unique_points(c) subset of ball(w)
            
            # To speed up, we can iterate through code_set
            code_list = list(code_set)
            random.shuffle(code_list)
            
            for c in code_list:
                if not time_ok():
                    break
                if c not in code_set: # Might have been removed
                    continue
                
                # Get unique points of c
                # unique if cnt[p] == 1 and c covers p
                ball_c = ball_indices[c]
                unique_mask = (cnt[ball_c] == 1)
                if not np.any(unique_mask):
                    # Should have been deleted in step 1
                    continue
                    
                unique_points = ball_c[unique_mask]
                
                # Find candidates w that cover all unique_points
                # Start with all True, intersect with coverers[p] for p in unique_points
                candidates = np.ones(total, dtype=bool)
                for p in unique_points:
                    candidates &= coverers[p]
                
                # Exclude current code
                for c2 in code_set:
                    candidates[c2] = False
                    
                valid_candidates = np.where(candidates)[0]
                if len(valid_candidates) > 0:
                    # Pick one randomly
                    w = random.choice(valid_candidates)
                    
                    # Apply swap
                    code_set.remove(c)
                    code_set.add(w)
                    cnt[ball_indices[c]] -= 1
                    cnt[ball_indices[w]] += 1
                    improved = True
                    break # Restart loop as state changed
            
            if not time_ok():
                break

            # 3. 2-for-1 Swap Pass
            # Remove c1, c2, add w.
            # Valid if unique(c1) U unique(c2) subset of ball(w)
            
            code_list2 = list(code_set)
            random.shuffle(code_list2)
            
            found_2for1 = False
            # Limit iterations to avoid O(N^2) explosion if N is large, but N~170 is small.
            # 170 * 170 = 28900 pairs. Each check is fast.
            for i in range(len(code_list2)):
                if not time_ok():
                    break
                c1 = code_list2[i]
                if c1 not in code_set:
                    continue
                
                ball_c1 = ball_indices[c1]
                unique_mask1 = (cnt[ball_c1] == 1)
                unique_points1 = ball_c1[unique_mask1]
                
                for j in range(i + 1, len(code_list2)):
                    if not time_ok():
                        break
                    c2 = code_list2[j]
                    if c2 not in code_set:
                        continue
                    
                    ball_c2 = ball_indices[c2]
                    unique_mask2 = (cnt[ball_c2] == 1)
                    unique_points2 = ball_c2[unique_mask2]
                    
                    # Combine unique points
                    # If both are empty, they should have been deleted.
                    # We need a word w that covers unique1 U unique2.
                    
                    # Optimization: If unique1 is empty, we only need to cover unique2.
                    # But wait, if unique1 is empty, c1 is redundant?
                    # No, c1 might be needed to cover points that are *not* unique to c1 but are unique to c2?
                    # No, "unique to c1" means covered ONLY by c1.
                    # If c1 has no unique points, it can be deleted.
                    # So if unique1 is empty, we should have deleted c1.
                    # Thus, we can assume unique1 is non-empty.
                    
                    all_unique = np.concatenate([unique_points1, unique_points2])
                    all_unique = np.unique(all_unique)
                    
                    if len(all_unique) == 0:
                         # Both redundant? Shouldn't happen.
                         code_set.remove(c1)
                         code_set.remove(c2)
                         cnt[ball_c1] -= 1
                         cnt[ball_c2] -= 1
                         improved = True
                         found_2for1 = True
                         break
                         
                    candidates = np.ones(total, dtype=bool)
                    for p in all_unique:
                        candidates &= coverers[p]
                        
                    for c3 in code_set:
                        candidates[c3] = False
                        
                    valid_candidates = np.where(candidates)[0]
                    if len(valid_candidates) > 0:
                        w = random.choice(valid_candidates)
                        
                        # Apply 2-for-1
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(w)
                        cnt[ball_c1] -= 1
                        cnt[ball_c2] -= 1
                        cnt[ball_indices[w]] += 1
                        improved = True
                        found_2for1 = True
                        break
                        
                if found_2for1:
                    break
            
            if found_2for1:
                continue # Restart local search loop

        return code_set, cnt

    # Main Loop
    best_code = None
    best_size = total + 1
    best_cnt = None

    # Strategy: Multiple random greedy inits, then local search.
    num_inits = 20
    inits_done = 0
    
    while inits_done < num_inits and time_ok():
        # Generate a random order for greedy
        order = list(range(total))
        random.shuffle(order)
        
        code = greedy_init(order)
        code_set = set(code)
        
        # Build counts
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1
            
        # Local Search
        code_set, cnt = local_search(code_set, cnt)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_cnt = cnt.copy()
            
        inits_done += 1
        
        # If we found a very good solution (e.g., < 170), we can spend more time on it?
        # For now, just keep going with new inits.

    if best_code is None:
        # Fallback
        best_code = list(range(0, total, 10))
        
    # Final cleanup/deletion pass on the best code found
    final_code_set = set(best_code)
    final_cnt = best_cnt.copy()
    
    # One more aggressive deletion/swap pass on the best solution
    # Re-run local search on the best solution to squeeze out more
    final_code_set, final_cnt = local_search(final_code_set, final_cnt)
    
    if len(final_code_set) < best_size:
        best_code = list(final_code_set)
        best_size = len(final_code_set)

    # Convert indices to words
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
