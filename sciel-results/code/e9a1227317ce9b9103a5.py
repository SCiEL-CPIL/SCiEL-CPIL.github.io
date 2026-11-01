import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

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

    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, ball_arrays[i]] = True

    # Precompute: for each pair (i,j), whether ball[i] subset of ball[j]
    # This is expensive (2187^2 * 15) but we can do it smartly
    # Actually, let's skip full precomputation and do it on-the-fly in pruning

    def greedy_cover(order=None):
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

    def greedy_cover_degree_based():
        """Greedy with dynamic degree-based ordering: at each step, pick the word
        that covers the most uncovered points, but break ties by preferring words
        with higher degree in the uncovered subgraph."""
        remaining = np.ones(total, dtype=bool)
        code = []
        # Precompute degree of each vertex in the "ball graph"
        # degree[i] = number of j such that i is in ball[j] = number of j at dist <= 1 from i
        # For a regular graph this is constant, but in the uncovered subgraph it varies
        # We'll just use the standard greedy but with a twist: process in an order
        # that favors "central" words first
        # Central words: those whose ball overlaps most with other balls
        # Approximate by: words with index in middle range tend to be more central
        # Actually, let's just do standard greedy but try multiple random orders
        # and a "reverse natural" order
        order = list(range(total))
        # Sort by a heuristic: words with more "neighbors" in the ball graph
        # For this regular graph, all have same degree, so use random
        random.shuffle(order)
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
        changed = False
        for c in list(code_set):
            if np.all(cnt[ball_arrays[c]] >= 2):
                code_set.remove(c)
                cnt[ball_arrays[c]] -= 1
                changed = True
        return changed

    def dominance_prune(code_set, cnt):
        """
        Stronger pruning: remove codeword c if there exists another codeword d
        such that every point in ball[c] is covered by at least one codeword
        other than c. This is equivalent to: for all p in ball[c], cnt[p] >= 2
        after removing c. But we can be smarter: check if ball[c] is subset of
        union of balls of other codewords.
        
        Actually the simple check is: if for all p in ball[c], cnt[p] >= 2, then
        removing c doesn't uncover any point. This is the same as delete_redundant.
        
        A stronger check: even if some points in ball[c] have cnt==1, they might
        still be covered by another codeword's ball. But that's exactly what cnt==1
        means - they're covered by exactly one codeword (c). So if any point in
        ball[c] has cnt==1, c is essential.
        
        So the only improvement is to check in a smarter order or do batch removals.
        """
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                if np.all(cnt[ball_arrays[c]] >= 2):
                    code_set.remove(c)
                    cnt[ball_arrays[c]] -= 1
                    changed = True
        return changed

    def compute_unique_points_fast(code_set, cnt):
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

    def local_search(code_set, cnt, time_limit=1.0):
        start_time = time.time()
        no_improve_streak = 0
        max_streak = 300
        while time.time() - start_time < time_limit and no_improve_streak < max_streak:
            improved = False
            unique_points = compute_unique_points_fast(code_set, cnt)
            
            # Deletions
            for c in list(code_set):
                if len(unique_points[c]) == 0:
                    code_set.remove(c)
                    cnt[ball_arrays[c]] -= 1
                    improved = True
                    break
            if improved:
                no_improve_streak = 0
                continue

            # 2-for-1
            code_list = list(code_set)
            random.shuffle(code_list)
            found = False
            for i in range(min(30, len(code_list))):
                c1 = code_list[i]
                u1 = unique_points[c1]
                if len(u1) == 0:
                    continue
                for j in range(i + 1, min(i + 4, len(code_list))):
                    c2 = code_list[j]
                    u2 = unique_points[c2]
                    if len(u1) == 0 and len(u2) == 0:
                        continue
                    for _ in range(8):
                        c_add = random.randint(0, total - 1)
                        if c_add in code_set:
                            continue
                        if is_valid_2for1(c1, c2, c_add, unique_points):
                            apply_2for1(c1, c2, c_add, code_set, cnt)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if improved:
                no_improve_streak = 0
                continue

            # 1-for-1 swaps
            swap_attempts = 0
            while swap_attempts < 15:
                swap_attempts += 1
                if len(code_set) == 0:
                    break
                c_remove = random.choice(list(code_set))
                u_rem = unique_points[c_remove]
                if len(u_rem) == 0:
                    code_set.remove(c_remove)
                    cnt[ball_arrays[c_remove]] -= 1
                    improved = True
                    break
                for _ in range(8):
                    c_add = random.randint(0, total - 1)
                    if c_add in code_set or c_add == c_remove:
                        continue
                    if is_valid_swap(c_remove, c_add, unique_points):
                        apply_swap(c_remove, c_add, code_set, cnt)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                no_improve_streak = 0
                continue

            # 3-for-2
            if len(code_set) >= 3:
                found = False
                for _ in range(8):
                    c1, c2, c3 = random.sample(list(code_set), 3)
                    c_add1 = random.randint(0, total - 1)
                    c_add2 = random.randint(0, total - 1)
                    if c_add1 in code_set or c_add2 in code_set or c_add1 == c_add2:
                        continue
                    if is_valid_3for2(c1, c2, c3, c_add1, c_add2, unique_points):
                        apply_3for2(c1, c2, c3, c_add1, c_add2, code_set, cnt)
                        improved = True
                        found = True
                        break
                if improved:
                    no_improve_streak = 0
                    continue

            if not improved:
                no_improve_streak += 1
        return code_set, cnt

    # === Initial solution ===
    best_code = None
    best_size = total + 1
    best_cnt = None

    strategies = ["natural", "reverse", "rand1", "rand2", "rand3", "rand4", "rand5", "rand6", "rand7", "rand8"]

    for s in strategies:
        if time.time() > deadline - 35:
            break
        if s == "natural":
            order = list(range(total))
        elif s == "reverse":
            order = list(range(total - 1, -1, -1))
        else:
            order = list(range(total))
            random.shuffle(order)
        
        code = greedy_cover(order)
        code_set = set(code)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arrays[c]] += 1
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

    # Polish initial best
    current_code = set(best_code)
    cnt = best_cnt.copy()
    current_code, cnt = local_search(current_code, cnt, time_limit=4.0)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        best_cnt = cnt.copy()

    # === Simulated Annealing ===
    T_start = 1.5
    T_end = 0.005
    cooling_rate = 0.993
    T = T_start

    sa_best_code = set(best_code)
    sa_best_cnt = best_cnt.copy()
    sa_best_size = best_size

    current_code = set(best_code)
    cnt = best_cnt.copy()
    current_size = len(current_code)

    all_indices = np.arange(total)
    restart_count = 0
    max_restarts = 10

    while time.time() < deadline - 8.0:
        if T < T_end:
            restart_count += 1
            if restart_count > max_restarts:
                break
            T = T_start
            current_code = set(sa_best_code)
            cnt = sa_best_cnt.copy()
            current_size = len(current_code)
            # Perturb: remove a few random codewords and re-greedy
            if len(current_code) > 10:
                to_remove = random.sample(list(current_code), min(3, len(current_code) // 10))
                for c in to_remove:
                    current_code.remove(c)
                    cnt[ball_arrays[c]] -= 1
                current_size = len(current_code)
                # Re-cover uncovered points
                uncovered = np.zeros(total, dtype=bool)
                for c in current_code:
                    uncovered[ball_arrays[c]] = True
                # Greedy fill
                remaining = ~uncovered
                while np.any(remaining):
                    best_c = -1
                    best_new = -1
                    for c in range(total):
                        if c in current_code:
                            continue
                        new_cov = np.count_nonzero(remaining & ball_mask[c])
                        if new_cov > best_new:
                            best_new = new_cov
                            best_c = c
                            if best_new == ball_size:
                                break
                    if best_c == -1:
                        break
                    current_code.add(best_c)
                    cnt[ball_arrays[best_c]] += 1
                    current_size += 1
                    remaining[ball_mask[best_c]] = False

        improved = False
        unique_points = compute_unique_points_fast(current_code, cnt)

        # Deletions
        for c in list(current_code):
            if len(unique_points[c]) == 0:
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                current_size -= 1
                improved = True
                break
        if improved:
            if current_size < sa_best_size:
                sa_best_size = current_size
                sa_best_code = set(current_code)
                sa_best_cnt = cnt.copy()
            T *= cooling_rate
            continue

        # 2-for-1
        code_list = list(current_code)
        random.shuffle(code_list)
        found = False
        for i in range(min(20, len(code_list))):
            c1 = code_list[i]
            u1 = unique_points[c1]
            if len(u1) == 0:
                continue
            for j in range(i + 1, min(i + 3, len(code_list))):
                c2 = code_list[j]
                u2 = unique_points[c2]
                if len(u1) == 0 and len(u2) == 0:
                    continue
                for _ in range(6):
                    c_add = random.randint(0, total - 1)
                    if c_add in current_code:
                        continue
                    if is_valid_2for1(c1, c2, c_add, unique_points):
                        apply_2for1(c1, c2, c_add, current_code, cnt)
                        current_size -= 1
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if improved:
            if current_size < sa_best_size:
                sa_best_size = current_size
                sa_best_code = set(current_code)
                sa_best_cnt = cnt.copy()
            T *= cooling_rate
            continue

        # 1-for-1 swaps
        swap_attempts = 0
        while swap_attempts < 12:
            swap_attempts += 1
            if len(current_code) == 0:
                break
            c_remove = random.choice(list(current_code))
            u_rem = unique_points[c_remove]
            if len(u_rem) == 0:
                current_code.remove(c_remove)
                cnt[ball_arrays[c_remove]] -= 1
                current_size -= 1
                improved = True
                break
            for _ in range(6):
                c_add = random.randint(0, total - 1)
                if c_add in current_code or c_add == c_remove:
                    continue
                if is_valid_swap(c_remove, c_add, unique_points):
                    apply_swap(c_remove, c_add, current_code, cnt)
                    improved = True
                    break
            if improved:
                break
        if improved:
            T *= cooling_rate
            continue

        # 3-for-2
        if len(current_code) >= 3:
            found = False
            for _ in range(6):
                c1, c2, c3 = random.sample(list(current_code), 3)
                c_add1 = random.randint(0, total - 1)
                c_add2 = random.randint(0, total - 1)
                if c_add1 in current_code or c_add2 in current_code or c_add1 == c_add2:
                    continue
                if is_valid_3for2(c1, c2, c3, c_add1, c_add2, unique_points):
                    apply_3for2(c1, c2, c3, c_add1, c_add2, current_code, cnt)
                    current_size -= 1
                    improved = True
                    found = True
                    break
            if improved:
                if current_size < sa_best_size:
                    sa_best_size = current_size
                    sa_best_code = set(current_code)
                    sa_best_cnt = cnt.copy()
                T *= cooling_rate
                continue

        if not improved:
            T *= cooling_rate

    # Final polish
    final_code = set(sa_best_code)
    final_cnt = sa_best_cnt.copy()
    final_code, final_cnt = local_search(final_code, final_cnt, time_limit=4.0)
    while delete_redundant(final_code, final_cnt):
        pass

    if len(final_code) < sa_best_size:
        sa_best_size = len(final_code)
        sa_best_code = final_code

    result = []
    for idx in sa_best_code:
        w = []
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
    
    return result
