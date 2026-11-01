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

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = np.ones(total, dtype=bool)
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                ball = ball_arrays[c]
                new_cov = np.count_nonzero(remaining[ball])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_arrays[best_c]] = True
            remaining[ball_arrays[best_c]] = False
        return code

    def delete_redundant(code_set, cnt):
        """Remove redundant codewords where all points in ball have count >= 2."""
        changed = False
        for c in list(code_set):
            ball = ball_arrays[c]
            if np.all(cnt[ball] >= 2):
                code_set.remove(c)
                cnt[ball] -= 1
                changed = True
        return changed

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1

    strategies = [
        "natural",
        "random1",
        "random2",
        "random3",
        "random4",
        "random5",
    ]

    for s in strategies:
        if time.time() > deadline - 30:
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
        # Fallback
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # === Simulated Annealing Phase ===
    current_code = set(best_code)
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_arrays[c]] += 1

    best_size = len(current_code)
    best_code = list(current_code)
    best_cnt = cnt.copy()

    # Simulated annealing parameters
    initial_temp = 0.5
    cooling_rate = 0.9995
    min_temp = 0.01
    temp = initial_temp

    # Precompute overlap for faster swap validation
    overlap = np.zeros((total, total), dtype=np.int8)
    balls_containing = [[] for _ in range(total)]
    for i in range(total):
        for p in ball_indices[i]:
            balls_containing[p].append(i)
    
    for p in range(total):
        bc = balls_containing[p]
        for a in range(len(bc)):
            for b in range(a+1, len(bc)):
                i = bc[a]
                j = bc[b]
                overlap[i, j] += 1
                overlap[j, i] += 1

    def is_valid_swap(c_remove, c_add, code_set, cnt):
        """Check if swapping c_remove for c_add maintains coverage."""
        if c_add in code_set or c_add == c_remove:
            return False
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        in_r_not_a = ball_r[np.isin(ball_r, ball_a, assume_unique=False, invert=True)]
        if len(in_r_not_a) > 0 and np.any(cnt[in_r_not_a] < 2):
            return False
        return True

    def apply_swap(c_remove, c_add, code_set, cnt):
        """Apply a swap."""
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_r] -= 1
        cnt[ball_a] += 1

    def undo_swap(c_remove, c_add, code_set, cnt):
        """Undo a swap."""
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        code_set.remove(c_add)
        code_set.add(c_remove)
        cnt[ball_r] += 1
        cnt[ball_a] -= 1

    # Precompute point -> set of codewords covering it (for faster swap validation)
    point_covered_by = [set() for _ in range(total)]
    for c in range(total):
        for p in ball_indices[c]:
            point_covered_by[p].add(c)

    # Simulated annealing loop
    no_improve_streak = 0
    max_streak = 2000

    while time.time() < deadline - 10.0 and no_improve_streak < max_streak:
        improved = False

        # 1. Delete phase
        while delete_redundant(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                best_cnt = cnt.copy()
            improved = True

        if not improved:
            no_improve_streak += 1
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
            continue
        else:
            no_improve_streak = 0

        # 2. Informed 1-for-1 swap phase
        swap_attempts = 0
        max_swap_attempts = 100
        while swap_attempts < max_swap_attempts and time.time() < deadline - 10.0:
            swap_attempts += 1
            if len(current_code) == 0:
                break

            c_remove = random.choice(list(current_code))
            if c_remove not in current_code:
                continue
            c_add = random.randint(0, total - 1)

            if c_add == c_remove or c_add in current_code:
                continue

            ov = overlap[c_remove, c_add]
            if ov >= ball_size:
                # ball_r fully contained in ball_a: always valid
                apply_swap(c_remove, c_add, current_code, cnt)
                improved = True
            else:
                # Need to check non-overlapping points
                ball_r = ball_arrays[c_remove]
                ball_a = ball_arrays[c_add]
                mask_in_a = np.isin(ball_r, ball_a, assume_unique=False)
                points_to_check = ball_r[~mask_in_a]
                if len(points_to_check) > 0 and np.any(cnt[points_to_check] < 2):
                    # Invalid swap - skip
                    continue
                else:
                    apply_swap(c_remove, c_add, current_code, cnt)
                    improved = True

            # After swap, try to delete
            while delete_redundant(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    best_cnt = cnt.copy()
                improved = True

        # 3. 2-for-1 swap: try replacing two codewords with one
        two_for_one_attempts = 0
        max_two_for_one = 50
        while two_for_one_attempts < max_two_for_one and time.time() < deadline - 10.0:
            two_for_one_attempts += 1
            if len(current_code) < 2:
                break

            # Pick two codewords to remove
            c1, c2 = random.sample(list(current_code), 2)
            if c1 not in current_code or c2 not in current_code:
                continue

            ball1 = ball_arrays[c1]
            ball2 = ball_arrays[c2]
            
            # Try a few random c_add candidates
            for _ in range(3):
                c_add = random.randint(0, total - 1)
                if c_add in current_code:
                    continue
                ball_a = ball_arrays[c_add]
                
                # Check: after removing c1 and c2, adding c_add, all points still covered
                mask1_in_a = np.isin(ball1, ball_a, assume_unique=False)
                mask2_in_a = np.isin(ball2, ball_a, assume_unique=False)
                
                b1_not_a = ball1[~mask1_in_a]
                b2_not_a = ball2[~mask2_in_a]
                
                valid = True
                if len(b1_not_a) > 0:
                    in_b2 = np.isin(b1_not_a, ball2, assume_unique=False)
                    if np.any(in_b2):
                        if np.any(cnt[b1_not_a[in_b2]] < 3):
                            valid = False
                    if valid and np.any(~in_b2):
                        if np.any(cnt[b1_not_a[~in_b2]] < 2):
                            valid = False
                
                if valid and len(b2_not_a) > 0:
                    in_b1 = np.isin(b2_not_a, ball1, assume_unique=False)
                    if np.any(in_b1):
                        if np.any(cnt[b2_not_a[in_b1]] < 3):
                            valid = False
                    if valid and np.any(~in_b1):
                        if np.any(cnt[b2_not_a[~in_b1]] < 2):
                            valid = False
                
                if not valid:
                    continue
                
                # Valid 2-for-1 swap
                current_code.remove(c1)
                current_code.remove(c2)
                current_code.add(c_add)
                cnt[ball1] -= 1
                cnt[ball2] -= 1
                cnt[ball_a] += 1
                improved = True
                break  # Break inner loop, try another pair

            # After 2-for-1, try to delete more
            while delete_redundant(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    best_cnt = cnt.copy()
                improved = True

        # 4. 3-for-2 swap: replace three codewords with two (escape local minima)
        three_for_two_attempts = 0
        max_three_for_two = 30
        while three_for_two_attempts < max_three_for_two and time.time() < deadline - 10.0:
            three_for_two_attempts += 1
            if len(current_code) < 3:
                break

            c1, c2, c3 = random.sample(list(current_code), 3)
            if c1 not in current_code or c2 not in current_code or c3 not in current_code:
                continue

            ball1 = ball_arrays[c1]
            ball2 = ball_arrays[c2]
            ball3 = ball_arrays[c3]

            # Union of the three balls
            union_mask = np.zeros(total, dtype=bool)
            union_mask[ball1] = True
            union_mask[ball2] = True
            union_mask[ball3] = True
            union_points = np.where(union_mask)[0]

            # Try a few random pairs of replacement codewords
            for _ in range(3):
                d1, d2 = random.sample(range(total), 2)
                if d1 in current_code or d2 in current_code:
                    continue
                if d1 == d2:
                    continue

                ball_d1 = ball_arrays[d1]
                ball_d2 = ball_arrays[d2]

                # Points in union not covered by d1 or d2
                not_in_d1 = np.isin(union_points, ball_d1, assume_unique=False)
                not_in_d2 = np.isin(union_points, ball_d2, assume_unique=False)
                uncovered = union_points[~(not_in_d1 | not_in_d2)]

                if len(uncovered) > 0:
                    # Check if these points are still covered by other codewords
                    if np.any(cnt[uncovered] < 2):
                        continue

                # Also need: for points only covered by one of c1,c2,c3 (and not d1,d2),
                # they must have cnt >= 2 before removal. The above check handles this
                # since cnt includes c1,c2,c3 contributions.
                # Actually we need: after removing c1,c2,c3 and adding d1,d2,
                # every point in union is covered.
                # A point p in union:
                #   - if p is in ball_d1 or ball_d2: covered by new codewords
                #   - else: must be covered by some other codeword in current_code
                #     (i.e., cnt[p] - [1 if p in ball1] - [1 if p in ball2] - [1 if p in ball3] >= 1)
                #     which means cnt[p] >= 1 + (1 if p in ball1) + (1 if p in ball2) + (1 if p in ball3)
                #     But since p is in union, at least one of ball1,ball2,ball3 contains p.
                #     So we need cnt[p] >= 2 + (number of other balls among c1,c2,c3 containing p)
                #     Simpler: just check cnt[p] >= 2 for all uncovered points (conservative)
                #     and cnt[p] >= 3 for points in exactly one of the three balls and not in d1/d2.
                # Let's do a precise check:
                valid = True
                if len(uncovered) > 0:
                    # For each uncovered point, compute how many of c1,c2,c3 cover it
                    in_b1 = np.isin(uncovered, ball1, assume_unique=False)
                    in_b2 = np.isin(uncovered, ball2, assume_unique=False)
                    in_b3 = np.isin(uncovered, ball3, assume_unique=False)
                    num_cover = in_b1.astype(np.int32) + in_b2.astype(np.int32) + in_b3.astype(np.int32)
                    # Need cnt[p] - num_cover[p] >= 1  =>  cnt[p] >= num_cover[p] + 1
                    if np.any(cnt[uncovered] < num_cover + 1):
                        valid = False

                if not valid:
                    continue

                # Apply 3-for-2 swap
                current_code.remove(c1)
                current_code.remove(c2)
                current_code.remove(c3)
                current_code.add(d1)
                current_code.add(d2)
                cnt[ball1] -= 1
                cnt[ball2] -= 1
                cnt[ball3] -= 1
                cnt[ball_d1] += 1
                cnt[ball_d2] += 1
                improved = True
                break

            # After 3-for-2, try to delete more
            while delete_redundant(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    best_cnt = cnt.copy()
                improved = True

        if not improved:
            no_improve_streak += 1
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
        else:
            no_improve_streak = 0

    # Final: high-precision deletion polish pass
    # Repeat delete + single-swap attempts until no improvement
    polish_rounds = 0
    max_polish_rounds = 50
    while polish_rounds < max_polish_rounds and time.time() < deadline - 2.0:
        polish_rounds += 1
        polish_improved = False

        # Delete redundant
        while delete_redundant(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                best_cnt = cnt.copy()
            polish_improved = True

        if not polish_improved:
            # Try a few single swaps
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                if len(current_code) == 0:
                    break
                c_remove = random.choice(list(current_code))
                if c_remove not in current_code:
                    continue
                c_add = random.randint(0, total - 1)
                if c_add == c_remove or c_add in current_code:
                    continue

                ov = overlap[c_remove, c_add]
                if ov >= ball_size:
                    apply_swap(c_remove, c_add, current_code, cnt)
                    polish_improved = True
                else:
                    ball_r = ball_arrays[c_remove]
                    ball_a = ball_arrays[c_add]
                    mask_in_a = np.isin(ball_r, ball_a, assume_unique=False)
                    points_to_check = ball_r[~mask_in_a]
                    if len(points_to_check) > 0 and np.any(cnt[points_to_check] < 2):
                        continue
                    else:
                        apply_swap(c_remove, c_add, current_code, cnt)
                        polish_improved = True

                while delete_redundant(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        best_cnt = cnt.copy()
                    polish_improved = True

        if not polish_improved:
            break

    # Final: one more delete pass
    while delete_redundant(current_code, cnt):
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)

    # Convert back to list of lists
    result = []
    for idx in best_code:
        w = []
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)

    return result
