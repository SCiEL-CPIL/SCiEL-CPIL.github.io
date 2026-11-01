"""Combined program: vectorized greedy + efficient local search with 2-for-1 moves."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

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

    rng = random.Random(42)
    strategies = ["natural"] + [f"random{i}" for i in range(8)]

    for s in strategies:
        if time.time() > deadline - 50:
            break
        if s == "natural":
            order = list(range(total))
        else:
            order = list(range(total))
            rng.shuffle(order)
        
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
        # Fallback: simple greedy
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # === Local Search with Efficient Swap Checks ===
    
    def compute_unique_points(code_set, cnt):
        """Compute unique points for each codeword."""
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
        """Check if swapping c_remove for c_add maintains coverage."""
        u = unique_points.get(c_remove, np.array([], dtype=np.int32))
        if len(u) == 0:
            return True
        return np.all(ball_mask[c_add, u])

    def apply_swap(c_remove, c_add, code_set, cnt):
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_arrays[c_remove]] -= 1
        cnt[ball_arrays[c_add]] += 1

    def is_valid_2for1(c1, c2, c_add, unique_points):
        """Check if removing c1, c2 and adding c_add maintains coverage."""
        u1 = unique_points.get(c1, np.array([], dtype=np.int32))
        u2 = unique_points.get(c2, np.array([], dtype=np.int32))
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

    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    no_improve_streak = 0
    max_streak = 300
    
    while time.time() < deadline - 5.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Compute unique points
        unique_points = compute_unique_points(current_code, cnt)
        
        # 2. Try Deletions
        deletion_found = False
        for c in list(current_code):
            if len(unique_points.get(c, np.array([], dtype=np.int32))) == 0:
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                deletion_found = True
                break
        
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            if deletion_found:
                continue

        # 3. Try 1-for-1 swaps
        code_list = list(current_code)
        rng.shuffle(code_list)
        
        swap_found = False
        for c in code_list:
            if time.time() > deadline - 5.0:
                break
            if c not in current_code:
                continue
            
            u = unique_points.get(c, np.array([], dtype=np.int32))
            if len(u) == 0:
                # Already handled in deletion
                continue
            
            # Find candidates that cover all unique points
            # Sample some candidates
            for _ in range(30):
                new_c = rng.randint(0, total - 1)
                if new_c in current_code:
                    continue
                if is_valid_swap(c, new_c, unique_points):
                    apply_swap(c, new_c, current_code, cnt)
                    improved = True
                    swap_found = True
                    break
            
            if swap_found:
                break
        
        if improved:
            no_improve_streak = 0
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            continue

        # 4. Try 2-for-1 swaps (remove 2, add 1)
        code_list = list(current_code)
        rng.shuffle(code_list)
        
        twofor1_found = False
        for i in range(min(50, len(code_list))):
            if time.time() > deadline - 5.0:
                break
            c1 = code_list[i]
            if c1 not in current_code:
                continue
            u1 = unique_points.get(c1, np.array([], dtype=np.int32))
            if len(u1) == 0:
                continue
            
            for j in range(i + 1, min(i + 20, len(code_list))):
                c2 = code_list[j]
                if c2 not in current_code:
                    continue
                u2 = unique_points.get(c2, np.array([], dtype=np.int32))
                
                # Find a candidate that covers both u1 and u2
                for _ in range(20):
                    new_c = rng.randint(0, total - 1)
                    if new_c in current_code:
                        continue
                    if is_valid_2for1(c1, c2, new_c, unique_points):
                        apply_2for1(c1, c2, new_c, current_code, cnt)
                        improved = True
                        twofor1_found = True
                        break
                
                if twofor1_found:
                    break
            
            if twofor1_found:
                break
        
        if improved:
            no_improve_streak = 0
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            continue

        no_improve_streak += 1

    # Convert back to list of words
    result = [words[c].tolist() for c in best_code_list]
    
    # Verify coverage (quick check)
    final_cnt = np.zeros(total, dtype=np.int32)
    for c in best_code_list:
        final_cnt[ball_arrays[c]] += 1
    if np.any(final_cnt == 0):
        # Coverage broken; fall back to greedy
        code = greedy_cover(list(range(total)))
        result = [words[c].tolist() for c in code]
    
    return result
# EVOLVE-BLOCK-END
