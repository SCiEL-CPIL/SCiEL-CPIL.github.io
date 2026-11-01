import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Base values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Generate all words as index arrays
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    # Precompute ball arrays as numpy
    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

    # Precompute ball_mask: ball_mask[i, j] is True if j is in ball of i
    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, ball_arrays[i]] = True

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        remaining = np.ones(total, dtype=bool)
        code = []
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                new_cov = np.count_nonzero(remaining & ball_mask[c])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            remaining[ball_mask[best_c]] = False
        return code

    def delete_redundant(code_set, cnt):
        """Remove redundant codewords where all points in ball have count >= 2."""
        changed = False
        for c in list(code_set):
            if np.all(cnt[ball_arrays[c]] >= 2):
                code_set.remove(c)
                cnt[ball_arrays[c]] -= 1
                changed = True
        return changed

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1
    best_cnt = None

    strategies = ["natural", "random1", "random2", "random3", "random4", "random5", "random6", "random7"]

    for s in strategies:
        if time.time() > deadline - 40:
            break
        if s == "natural":
            order = list(range(total))
        else:
            order = list(range(total))
            random.shuffle(order)
        
        code = greedy_cover(order)
        code_set = set(code)
        
        # Build counts
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arrays[c]] += 1
        
        # Delete redundant
        while delete_redundant(code_set, cnt):
            pass
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_cnt = cnt.copy()

    if best_code is None:
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # === Local Search with Efficient Swap Checks ===
    
    def compute_unique_points_fast(code_set, cnt):
        """
        Fast version: only compute for codewords that might have unique points.
        """
        unique = {}
        for c in code_set:
            ball = ball_arrays[c]
            if np.any(cnt[ball] == 1):
                mask = (cnt[ball] == 1)
                unique[c] = ball[mask]
            else:
                unique[c] = np.array([], dtype=np.int32)
        return unique

    def is_valid_swap(c_remove, c_add, unique_points):
        """
        Check if swapping c_remove for c_add maintains coverage.
        Valid if unique_points[c_remove] is subset of ball[c_add].
        """
        u = unique_points[c_remove]
        if len(u) == 0:
            return True
        return np.all(ball_mask[c_add, u])

    def apply_swap(c_remove, c_add, code_set, cnt):
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_arrays[c_remove]] -= 1
        cnt[ball_arrays[c_add]] += 1

    def is_valid_2for1(c1, c2, c_add, unique_points):
        """
        Check if removing c1, c2 and adding c_add maintains coverage.
        """
        u1 = unique_points[c1]
        u2 = unique_points[c2]
        if len(u1) == 0 and len(u2) == 0:
            return True
        if len(u1) > 0:
            if not np.all(ball_mask[c_add, u1]):
                return False
        if len(u2) > 0:
            if not np.all(ball_mask[c_add, u2]):
                return False
        return True

    def apply_2for1(c1, c2, c_add, code_set, cnt):
        code_set.remove(c1)
        code_set.remove(c2)
        code_set.add(c_add)
        cnt[ball_arrays[c1]] -= 1
        cnt[ball_arrays[c2]] -= 1
        cnt[ball_arrays[c_add]] += 1

    def is_valid_3for2(c1, c2, c3, c_add1, c_add2, unique_points):
        """
        Check if removing c1,c2,c3 and adding c_add1, c_add2 maintains coverage.
        """
        u1 = unique_points[c1]
        u2 = unique_points[c2]
        u3 = unique_points[c3]
        
        if len(u1) > 0:
            if not np.all(ball_mask[c_add1, u1] | ball_mask[c_add2, u1]):
                return False
        if len(u2) > 0:
            if not np.all(ball_mask[c_add1, u2] | ball_mask[c_add2, u2]):
                return False
        if len(u3) > 0:
            if not np.all(ball_mask[c_add1, u3] | ball_mask[c_add2, u3]):
                return False
        return True

    def apply_3for2(c1, c2, c3, c_add1, c_add2, code_set, cnt):
        code_set.remove(c1)
        code_set.remove(c2)
        code_set.remove(c3)
        code_set.add(c_add1)
        code_set.add(c_add2)
        cnt[ball_arrays[c1]] -= 1
        cnt[ball_arrays[c2]] -= 1
        cnt[ball_arrays[c3]] -= 1
        cnt[ball_arrays[c_add1]] += 1
        cnt[ball_arrays[c_add2]] += 1

    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    no_improve_streak = 0
    max_streak = 500
    
    # Precompute candidate pool for swaps: all words not in current code
    # We'll maintain this dynamically
    
    while time.time() < deadline - 10.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Compute unique points
        unique_points = compute_unique_points_fast(current_code, cnt)
        
        # 2. Try Deletions
        for c in list(current_code):
            if len(unique_points[c]) == 0:
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                break
                
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 3. Try 1-for-1 swaps
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            if time.time() > deadline - 10.0:
                break
            if c not in current_code:
                continue
            
            u = unique_points[c]
            if len(u) == 0:
                continue
            
            # Find candidate replacements: words w not in current_code
            # such that all unique_points are in ball(w)
            candidates = np.ones(total, dtype=bool)
            for pt in u:
                candidates &= ball_mask[pt]
            
            # Exclude current codewords
            for c2 in current_code:
                candidates[c2] = False
            
            valid_candidates = np.where(candidates)[0]
            if len(valid_candidates) > 0:
                # Choose the one that maximizes overlap with current code's coverage
                # (to minimize disruption) - or just pick randomly
                w = random.choice(valid_candidates)
                
                # Apply swap
                apply_swap(c, w, current_code, cnt)
                improved = True
                break
        
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 4. Try 2-for-1 swaps (remove 2, add 1)
        if len(current_code) >= 2:
            found_2for1 = False
            code_list2 = list(current_code)
            random.shuffle(code_list2)
            
            for i in range(min(30, len(code_list2))):
                if time.time() > deadline - 10.0:
                    break
                c1 = code_list2[i]
                if c1 not in current_code:
                    continue
                
                u1 = unique_points[c1]
                
                for j in range(min(15, len(code_list2))):
                    if j == i:
                        continue
                    c2 = code_list2[j]
                    if c2 not in current_code or c2 == c1:
                        continue
                    
                    u2 = unique_points[c2]
                    
                    # Combine unique points from both
                    if len(u1) == 0 and len(u2) == 0:
                        # Both can be deleted
                        current_code.remove(c1)
                        current_code.remove(c2)
                        cnt[ball_arrays[c1]] -= 1
                        cnt[ball_arrays[c2]] -= 1
                        improved = True
                        found_2for1 = True
                        break
                    
                    all_unique = np.concatenate([u1, u2])
                    all_unique = np.unique(all_unique)
                    
                    # Find w such that all all_unique ⊆ ball(w)
                    candidates = np.ones(total, dtype=bool)
                    for pt in all_unique:
                        candidates &= ball_mask[pt]
                    
                    # Exclude current codewords
                    for c3 in current_code:
                        candidates[c3] = False
                    
                    valid_candidates = np.where(candidates)[0]
                    if len(valid_candidates) > 0:
                        w = random.choice(valid_candidates)
                        
                        # Apply 2-for-1
                        apply_2for1(c1, c2, w, current_code, cnt)
                        improved = True
                        found_2for1 = True
                        break
                
                if found_2for1:
                    break
            
            if improved:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                no_improve_streak = 0
                continue

        # 5. Try 3-for-2 swaps (remove 3, add 2)
        if len(current_code) >= 3:
            found_3for2 = False
            code_list3 = list(current_code)
            random.shuffle(code_list3)
            
            for i in range(min(20, len(code_list3))):
                if time.time() > deadline - 10.0:
                    break
                c1 = code_list3[i]
                if c1 not in current_code:
                    continue
                
                u1 = unique_points[c1]
                
                for j in range(min(10, len(code_list3))):
                    if j == i:
                        continue
                    c2 = code_list3[j]
                    if c2 not in current_code or c2 == c1:
                        continue
                    
                    u2 = unique_points[c2]
                    
                    for k in range(min(5, len(code_list3))):
                        if k == i or k == j:
                            continue
                        c3 = code_list3[k]
                        if c3 not in current_code or c3 == c1 or c3 == c2:
                            continue
                        
                        u3 = unique_points[c3]
                        
                        # Find w1, w2 such that unique points are covered
                        # This is more complex, so we use a heuristic approach
                        # Try to find w1 that covers most of the unique points
                        all_unique = np.concatenate([u1, u2, u3])
                        all_unique = np.unique(all_unique)
                        
                        if len(all_unique) == 0:
                            # All can be deleted
                            current_code.remove(c1)
                            current_code.remove(c2)
                            current_code.remove(c3)
                            cnt[ball_arrays[c1]] -= 1
                            cnt[ball_arrays[c2]] -= 1
                            cnt[ball_arrays[c3]] -= 1
                            improved = True
                            found_3for2 = True
                            break
                        
                        # Try to find two words that together cover all unique points
                        # For simplicity, we'll try a greedy approach
                        remaining_unique = set(all_unique.tolist())
                        
                        # Find first word w1
                        candidates1 = np.ones(total, dtype=bool)
                        for pt in all_unique:
                            candidates1[pt] = True  # Simplified: just find any word in ball of first point
                        if len(all_unique) > 0:
                            candidates1 = ball_mask[all_unique[0]].copy()
                        
                        for c3b in current_code:
                            candidates1[c3b] = False
                        
                        valid_candidates1 = np.where(candidates1)[0]
                        if len(valid_candidates1) == 0:
                            continue
                        
                        w1 = random.choice(valid_candidates1)
                        
                        # Update remaining unique points
                        covered_by_w1 = ball_mask[w1]
                        remaining_unique_after_w1 = [pt for pt in all_unique if not covered_by_w1[pt]]
                        
                        if len(remaining_unique_after_w1) == 0:
                            # w1 covers everything, we just need one more word (or none)
                            # But we need to add 2 words, so add a dummy
                            # Actually, if w1 covers everything, we can add any other word
                            candidates2 = np.ones(total, dtype=bool)
                            for c3c in current_code:
                                candidates2[c3c] = False
                            candidates2[w1] = False
                            valid_candidates2 = np.where(candidates2)[0]
                            if len(valid_candidates2) > 0:
                                w2 = random.choice(valid_candidates2)
                                apply_3for2(c1, c2, c3, w1, w2, current_code, cnt)
                                improved = True
                                found_3for2 = True
                                break
                            continue
                        
                        # Find second word w2 that covers remaining_unique_after_w1
                        candidates2 = np.ones(total, dtype=bool)
                        for pt in remaining_unique_after_w1:
                            candidates2 &= ball_mask[pt]
                        
                        for c3d in current_code:
                            candidates2[c3d] = False
                        candidates2[w1] = False
                        
                        valid_candidates2 = np.where(candidates2)[0]
                        if len(valid_candidates2) > 0:
                            w2 = random.choice(valid_candidates2)
                            apply_3for2(c1, c2, c3, w1, w2, current_code, cnt)
                            improved = True
                            found_3for2 = True
                            break
                        
                    if found_3for2:
                        break
                
                if found_3for2:
                    break
            
            if improved:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                no_improve_streak = 0
                continue

        if not improved:
            no_improve_streak += 1

    # Final: one more delete pass
    while delete_redundant(current_code, cnt):
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code_list = list(current_code)

    # Convert back to list of lists
    result = []
    for idx in best_code_list:
        w = []
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)

    return result
