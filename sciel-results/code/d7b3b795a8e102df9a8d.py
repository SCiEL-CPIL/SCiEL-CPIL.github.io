import math
import time
import numpy as np
import random

def construct():
    START = time.time()
    DEADLINE = START + 140  # 140s safety margin before 150s limit

    def elapsed():
        return time.time() - START

    def time_ok(limit_frac=1.0):
        return elapsed() < DEADLINE - 5 * limit_frac

    def compute_score_fast(u):
        """Compute score using numpy for speed on larger sets."""
        u_arr = np.array(sorted(set(u)), dtype=np.int64)
        n = len(u_arr)
        if n == 0:
            return -1.0
        if 0 not in u_arr:
            return -1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0

        # Sumset: all a+b for a,b in u
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))

        # Diffset: all a-b for a,b in u
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))

        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def compute_score_slow(u):
        """Fallback for very small sets or when numpy is slow."""
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
        r = len(diffset) / len(sumset)
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def compute_score(u):
        if len(u) <= 10:
            return compute_score_slow(u)
        else:
            return compute_score_fast(u)

    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known optimal/near-optimal Golomb rulers
    golomb_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],       # n=7 optimal
        [0, 1, 4, 10, 18, 23, 29, 34],   # n=8 optimal
        [0, 1, 4, 10, 18, 23, 29, 37, 45], # n=9 (near-optimal)
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52], # n=10 (near-optimal)
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 60], # n=11 (near-optimal)
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 60, 70], # n=12
    ]

    best_set = [0, 1, 3]
    best_score = compute_score(best_set)

    # Evaluate all known Golomb rulers
    for u in golomb_candidates:
        if not time_ok():
            break
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)

    # Diff-guided local search: prioritize moves that add new differences over new sums
    def local_search_diff_guided(start_u, time_budget=30):
        """Hill-climbing local search with diff-guided scoring."""
        start_t = time.time()
        current_u = sorted(set(start_u))
        if 0 not in current_u:
            current_u = sorted([0] + current_u)
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score

        # Precompute sum and diff sets
        def get_sum_diff_sets(u):
            u_set = set(u)
            sum_set = set()
            diff_set = set()
            for a in u_set:
                for b in u_set:
                    sum_set.add(a + b)
                    diff_set.add(a - b)
            return sum_set, diff_set

        while time.time() - start_t < time_budget and time_ok():
            improved = False
            cur_sum_set, cur_diff_set = get_sum_diff_sets(current_u)
            n = len(current_u)

            # Neighborhood 1: move a non-zero element with diff-guided scoring
            for i in range(len(current_u)):
                if current_u[i] == 0:
                    continue

                best_new_val = None
                best_new_score = current_score
                elem = current_u[i]

                # Search a range of new values
                search_lo = max(1, elem - 20)
                search_hi = elem + 25
                for new_val in range(search_lo, search_hi + 1):
                    if new_val == elem or new_val in current_u:
                        continue

                    # Estimate new diff and sum counts
                    new_diff_count = 0
                    new_sum_count = 0

                    for a in current_u:
                        if a == elem:
                            continue
                        d1 = new_val - a
                        d2 = a - new_val
                        if d1 not in cur_diff_set:
                            new_diff_count += 1
                        if d2 not in cur_diff_set:
                            new_diff_count += 1
                        s_val = new_val + a
                        if s_val not in cur_sum_set:
                            new_sum_count += 1

                    # Prefer moves that add more diffs than sums
                    net_gain = new_diff_count - new_sum_count * 0.8
                    if net_gain > 0:
                        candidate = list(current_u)
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate or len(candidate) != len(current_u):
                            continue
                        s = compute_score(candidate)
                        if s > best_new_score + 1e-12:
                            best_new_score = s
                            best_new_val = new_val

                if best_new_val is not None:
                    current_u[i] = best_new_val
                    current_u = sorted(set(current_u))
                    current_score = best_new_score
                    if current_score > best_s:
                        best_s = current_score
                        best_u = list(current_u)
                    improved = True
                    break

            # Neighborhood 2: add a new element with diff-guided selection
            if not improved:
                cur_sum_set, cur_diff_set = get_sum_diff_sets(current_u)
                best_candidate = None
                best_candidate_score = current_score

                max_u = max(current_u)
                search_hi = max_u + len(current_u) * len(current_u) + 15

                # Try candidates in order of increasing value
                for new_val in range(1, search_hi + 1):
                    if new_val in current_u:
                        continue

                    new_diff_count = 0
                    new_sum_count = 0
                    for a in current_u:
                        d1 = new_val - a
                        d2 = a - new_val
                        if d1 not in cur_diff_set:
                            new_diff_count += 1
                        if d2 not in cur_diff_set:
                            new_diff_count += 1
                        s_val = new_val + a
                        if s_val not in cur_sum_set:
                            new_sum_count += 1

                    net_gain = new_diff_count - new_sum_count * 0.7
                    if net_gain > 0:
                        candidate = sorted(current_u + [new_val])
                        s = compute_score(candidate)
                        if s > best_candidate_score + 1e-12:
                            best_candidate_score = s
                            best_candidate = new_val

                if best_candidate is not None:
                    current_u = sorted(current_u + [best_candidate])
                    current_score = best_candidate_score
                    if current_score > best_s:
                        best_s = current_score
                        best_u = list(current_u)
                    improved = True

            # Neighborhood 3: remove an element (if len > 2)
            if not improved and n > 2:
                for i in range(len(current_u)):
                    if current_u[i] == 0:
                        continue
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    if len(candidate) >= 2:
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(current_u)
                            improved = True
                            break

            # Neighborhood 4: 2-element swap
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        if current_u[i] == 0 or current_u[j] == 0:
                            continue
                        candidate = list(current_u)
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        candidate = sorted(set(candidate))
                        if 0 not in candidate or len(candidate) != n:
                            continue
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(current_u)
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        return best_u, best_s

    # Try local search from the best Golomb ruler found
    if time_ok(2.0):
        ls_u, ls_s = local_search_diff_guided(best_set, time_budget=25)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u

    # Try local search from other starting points
    if time_ok(3.0):
        for u in golomb_candidates[:6]:
            if not time_ok(3.0):
                break
            ls_u, ls_s = local_search_diff_guided(u, time_budget=10)
            if ls_s > best_score:
                best_score = ls_s
                best_set = ls_u

    # Try a greedy construction approach with diff-guided selection
    if time_ok(4.0):
        def greedy_build_diff_guided(n_target, time_budget=12):
            """Build a Golomb-like ruler greedily, optimizing score with diff-guided selection."""
            start_t = time.time()
            u = [0, 1]
            while len(u) < n_target and time.time() - start_t < time_budget and time_ok():
                max_u = max(u)
                best_new = None
                best_new_score = compute_score(u)
                search_hi = max_u + len(u) * len(u) + 10

                # Get current sum and diff sets
                u_set = set(u)
                cur_sum_set = set()
                cur_diff_set = set()
                for a in u_set:
                    for b in u_set:
                        cur_sum_set.add(a + b)
                        cur_diff_set.add(a - b)

                for new_val in range(max_u + 1, search_hi + 1):
                    candidate = sorted(u + [new_val])
                    if not is_golomb(candidate):
                        continue

                    # Estimate new diffs and sums
                    new_diff_count = 0
                    new_sum_count = 0
                    for a in u_set:
                        d1 = new_val - a
                        d2 = a - new_val
                        if d1 not in cur_diff_set:
                            new_diff_count += 1
                        if d2 not in cur_diff_set:
                            new_diff_count += 1
                        s_val = new_val + a
                        if s_val not in cur_sum_set:
                            new_sum_count += 1

                    net_gain = new_diff_count - new_sum_count * 0.7
                    if net_gain <= 0:
                        continue

                    s = compute_score(candidate)
                    if s > best_new_score:
                        best_new_score = s
                        best_new = candidate
                    if time.time() - start_t > time_budget:
                        break

                if best_new is not None:
                    u = best_new
                else:
                    # Try non-Golomb additions too
                    for new_val in range(max_u + 1, search_hi + 1):
                        candidate = sorted(u + [new_val])
                        s = compute_score(candidate)
                        if s > best_new_score:
                            best_new_score = s
                            best_new = candidate
                        if time.time() - start_t > time_budget:
                            break
                    if best_new is not None:
                        u = best_new
                    else:
                        break
            return u, compute_score(u)

        for n_target in [6, 8, 10, 12]:
            if not time_ok(4.0):
                break
            g_u, g_s = greedy_build_diff_guided(n_target, time_budget=8)
            if g_s > best_score:
                best_score = g_s
                best_set = g_u

    # Final polish: tight local search around the best set
    if time_ok(5.0):
        final_u, final_s = local_search_diff_guided(best_set, time_budget=15)
        if final_s > best_score:
            best_score = final_s
            best_set = final_u

    # Final: ensure we return a valid set containing 0
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set

    return best_set
