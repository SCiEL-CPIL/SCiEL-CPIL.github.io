import time
import numpy as np
import random
import math

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin under 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base for index <-> word
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as index lists and boolean coverage matrix
    ball_indices = [None] * total
    cov = np.zeros((total, total), dtype=np.bool_)

    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q

        ball = [c]
        cov[c, c] = True

        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                ball.append(w)
                cov[c, w] = True

        ball_indices[c] = ball

    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            # Vectorized gain computation
            gains = cov @ (1 - covered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = random.choice(candidates.tolist())
            else:
                best_c = int(candidates[0])

            code.append(best_c)
            covered[cov[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    if np.min(cnt[cov[c]]) >= 2:
                        code_set.remove(c)
                        cnt[cov[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        no_success = 0
        max_streak = 2000

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # 1. Try deletion
            deleted = False
            for c in list(code_set):
                if np.min(cnt[cov[c]]) >= 2:
                    code_set.remove(c)
                    cnt[cov[c]] -= 1
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break

            if deleted:
                continue

            # 2. Try 1-for-1 swap using critical words
            c_remove = random.choice(list(code_set))
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                # Can delete directly
                code_set.remove(c_remove)
                cnt[cov[c_remove]] -= 1
                continue

            critical_words = np.where(critical_mask)[0]

            # Find candidates that cover all critical words
            candidates_mask = np.all(cov[:, critical_words], axis=1)
            in_set_mask = np.zeros(total, dtype=bool)
            for c in code_set:
                in_set_mask[c] = True
            candidates_mask &= ~in_set_mask
            candidates_mask[c_remove] = False  # Exclude the one we're removing

            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                c_add = int(random.choice(candidate_indices))
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt[cov[c_remove]] -= 1
                cnt[cov[c_add]] += 1
                no_success = 0
                try_delete()
            else:
                no_success += 1

            # 3. Try 2-for-1 improvement (remove 2, add 1)
            if no_success > 300 and time.time() - t_start < time_budget - 0.5 and len(code_set) >= 2:
                c_rem1, c_rem2 = random.sample(list(code_set), 2)
                
                # Find union of critical words from both
                crit1 = (cnt == 1) & cov[c_rem1]
                crit2 = (cnt == 1) & cov[c_rem2]
                # Words that are critical for rem1 but not covered by rem2, and vice versa
                # Actually, we need words that would become uncovered if we remove both
                # A word x is at risk if cnt[x] == 1 and (x in ball(rem1) or x in ball(rem2))
                # But if x is in both balls and cnt[x]==2, removing both makes it 0.
                # Let's be conservative: find words with cnt <= 2 that are in the union of balls
                union_ball = cov[c_rem1] | cov[c_rem2]
                at_risk = union_ball & (cnt <= 2)
                
                if not np.any(at_risk):
                    # Safe to remove both, just add one random
                    c_add = random.randint(0, total - 1)
                    if c_add not in code_set and c_add != c_rem1 and c_add != c_rem2:
                        code_set.remove(c_rem1)
                        code_set.remove(c_rem2)
                        code_set.add(c_add)
                        cnt[cov[c_rem1]] -= 1
                        cnt[cov[c_rem2]] -= 1
                        cnt[cov[c_add]] += 1
                        no_success = 0
                        try_delete()
                        continue

                # Try to find a single codeword that covers all at-risk words
                at_risk_words = np.where(at_risk)[0]
                if len(at_risk_words) <= ball_size:  # Reasonable check
                    candidates_mask = np.all(cov[:, at_risk_words], axis=1)
                    in_set_mask = np.zeros(total, dtype=bool)
                    for c in code_set:
                        in_set_mask[c] = True
                    candidates_mask &= ~in_set_mask
                    candidates_mask[c_rem1] = False
                    candidates_mask[c_rem2] = False

                    candidate_indices = np.where(candidates_mask)[0]
                    if len(candidate_indices) > 0:
                        c_add = int(random.choice(candidate_indices))
                        code_set.remove(c_rem1)
                        code_set.remove(c_rem2)
                        code_set.add(c_add)
                        cnt[cov[c_rem1]] -= 1
                        cnt[cov[c_rem2]] -= 1
                        cnt[cov[c_add]] += 1
                        no_success = 0
                        try_delete()
                    else:
                        no_success += 1
                else:
                    no_success += 1

            # 4. Try 1-for-2 swap (remove 1, add 2) to escape local minima
            if no_success > 800 and time.time() - t_start < time_budget - 1.0:
                c_rem = random.choice(list(code_set))
                critical_mask = (cnt == 1) & cov[c_rem]
                critical_words = np.where(critical_mask)[0]

                if len(critical_words) > 0 and len(critical_words) <= 2 * ball_size:
                    # Sample random pairs
                    found = False
                    for _ in range(30):
                        c1 = random.randint(0, total - 1)
                        c2 = random.randint(0, total - 1)
                        if c1 in code_set or c2 in code_set or c1 == c2:
                            continue
                        if c1 == c_rem or c2 == c_rem:
                            continue

                        # Check if c1 and c2 together cover all critical words
                        if np.all(cov[c1, critical_words] | cov[c2, critical_words]):
                            code_set.remove(c_rem)
                            code_set.add(c1)
                            code_set.add(c2)
                            cnt[cov[c_rem]] -= 1
                            cnt[cov[c1]] += 1
                            cnt[cov[c2]] += 1
                            no_success = 0
                            try_delete()
                            found = True
                            break

                    if not found:
                        no_success += 1
                else:
                    no_success += 1

            # 5. Perturb if stuck
            if no_success > max_streak:
                rand_c = random.randint(0, total - 1)
                if rand_c not in code_set:
                    code_set.add(rand_c)
                    cnt[cov[rand_c]] += 1
                    try_delete()
                no_success = 0

        return best_code, best_size

    def final_polish(code_list, time_budget=3.0):
        """Aggressive final pass with multiple deletion and swap rounds."""
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        # Multiple rounds of deletion
        for _ in range(10):
            if time.time() - t_start > time_budget:
                break
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    if np.min(cnt[cov[c]]) >= 2:
                        code_set.remove(c)
                        cnt[cov[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        # Targeted swaps
        for _ in range(50):
            if time.time() - t_start > time_budget:
                break
            if len(code_set) == 0:
                break

            c_remove = random.choice(list(code_set))
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                code_set.remove(c_remove)
                cnt[cov[c_remove]] -= 1
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                continue

            critical_words = np.where(critical_mask)[0]
            candidates_mask = np.all(cov[:, critical_words], axis=1)
            in_set_mask = np.zeros(total, dtype=bool)
            for c in code_set:
                in_set_mask[c] = True
            candidates_mask &= ~in_set_mask
            candidates_mask[c_remove] = False

            candidate_indices = np.where(candidates_mask)[0]
            if len(candidate_indices) > 0:
                c_add = int(random.choice(candidate_indices))
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt[cov[c_remove]] -= 1
                cnt[cov[c_add]] += 1
                # Try delete after swap
                for c2 in list(code_set):
                    if np.min(cnt[cov[c2]]) >= 2:
                        code_set.remove(c2)
                        cnt[cov[c2]] -= 1
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=15.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Final polish on best result
    if time.time() < deadline - 5.0:
        polished_code, polished_size = final_polish(best_code_overall, time_budget=3.0)
        if polished_size < best_size_overall:
            best_size_overall = polished_size
            best_code_overall = polished_code

    # Multi-restart
    seed = 0
    while time.time() < deadline - 8.0:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 8.0 - time.time()
        if time_left < 3.0:
            break
        tb = min(8.0, max(2.0, time_left / 5.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)

        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

            # Polish the new best
            if time.time() < deadline - 3.0:
                polished_code, polished_size = final_polish(best_code_overall, time_budget=2.0)
                if polished_size < best_size_overall:
                    best_size_overall = polished_size
                    best_code_overall = polished_code

        # Early exit if we're close to optimal
        if best_size_overall <= 180:
            break

    # Final conversion
    result = []
    for idx in best_code_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
