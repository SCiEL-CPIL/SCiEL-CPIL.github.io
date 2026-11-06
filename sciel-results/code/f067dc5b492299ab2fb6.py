import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 87.0

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    best_A = None
    best_max = float('inf')

    def compute_counts(ind):
        """Compute unordered pair-sum representation counts for [T_min, T_max]."""
        conv = np.convolve(ind, ind, mode='full')
        counts = np.empty(R, dtype=np.int32)
        for i, s in enumerate(range(T_min, T_max + 1)):
            c_ord = conv[s]
            if s % 2 == 0:
                half = s // 2
                n_eq = 1 if (half < N_max and ind[half] == 1) else 0
            else:
                n_eq = 0
            counts[i] = (c_ord + n_eq) // 2
        return counts

    def max_count(counts):
        return int(np.max(counts))

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def soft_max_score(counts, temperature=1.0):
        """Smooth surrogate: log-sum-exp of counts, penalizes peaks smoothly."""
        # Add small epsilon to avoid log(0) issues, but counts are >= 0
        c = counts.astype(np.float64) + 1e-6
        m = np.max(c)
        # log-sum-exp for numerical stability
        lse = m + np.log(np.sum(np.exp(c - m)))
        return lse

    def repair(A_set, ind, counts):
        """Repair gaps by adding elements with best gain."""
        for _ in range(200):
            if is_valid(counts):
                break
            gaps = np.where(counts == 0)[0]
            best_y = None
            best_gain = -1
            for y in range(N_max):
                if ind[y] == 1:
                    continue
                gain = 0
                for gi in gaps:
                    s = T_min + gi
                    a = s - y
                    if 0 <= a < N_max and (ind[a] == 1 or a == y):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
            if best_y is not None:
                A_set.add(best_y)
                ind[best_y] = 1
                counts = compute_counts(ind)
            else:
                break
        return is_valid(counts)

    def try_improve(A_initial, time_budget):
        nonlocal best_A, best_max
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        counts = compute_counts(ind)

        if not is_valid(counts):
            if not repair(A_set, ind, counts):
                return

        current_max = max_count(counts)
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)

        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = current_max

        # Phase 0: Greedy removal — try removing elements to reduce |A| and peak
        removal_deadline = start_time + time_budget * 0.20
        while time.time() < removal_deadline:
            improved = False
            for x in list(A_cur):
                if len(A_cur) <= 20:
                    break
                ind_test = ind_cur.copy()
                ind_test[x] = 0
                c_test = compute_counts(ind_test)
                if is_valid(c_test):
                    m_test = max_count(c_test)
                    if m_test <= max_cur:
                        # Removing x keeps validity and doesn't increase peak
                        A_cur.discard(x)
                        ind_cur = ind_test
                        counts_cur = c_test
                        max_cur = m_test
                        improved = True
                        if max_cur < best_max:
                            best_max = max_cur
                            best_A = set(A_cur)
                        break
            if not improved:
                break

        # Phase 1: Fast random swaps with incremental count update
        swap_deadline = start_time + time_budget * 0.60
        no_improve = 0

        while time.time() < swap_deadline:
            if len(A_cur) == 0:
                break
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)
            y = random.randint(0, N_max - 1)
            if y == x or ind_cur[y] == 1:
                continue

            # Compute new counts incrementally
            c_new = counts_cur.copy()

            # Subtract contributions of x
            for a in A_cur:
                if a == x:
                    continue
                s = a + x
                if T_min <= s <= T_max:
                    c_new[s - T_min] -= 1
            s_xx = 2 * x
            if T_min <= s_xx <= T_max:
                c_new[s_xx - T_min] -= 1

            # Add contributions of y
            for a in A_cur:
                if a == x:
                    continue
                s = a + y
                if T_min <= s <= T_max:
                    c_new[s - T_min] += 1
            s_yy = 2 * y
            if T_min <= s_yy <= T_max:
                c_new[s_yy - T_min] += 1

            if not is_valid(c_new):
                no_improve += 1
                if no_improve > 2000:
                    break
                continue

            m_new = max_count(c_new)

            if m_new < max_cur:
                A_cur.discard(x)
                A_cur.add(y)
                ind_cur[x] = 0
                ind_cur[y] = 1
                counts_cur = c_new
                max_cur = m_new
                no_improve = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                # Use soft-max as tiebreaker: prefer swaps that reduce overall peak pressure
                old_soft = soft_max_score(counts_cur)
                new_soft = soft_max_score(c_new)
                if new_soft < old_soft or random.random() < 0.10:
                    A_cur.discard(x)
                    A_cur.add(y)
                    ind_cur[x] = 0
                    ind_cur[y] = 1
                    counts_cur = c_new
            else:
                no_improve += 1
                if no_improve > 2000:
                    break

        # Phase 2: Targeted swaps focusing on reducing peak using soft-max guidance
        phase2_deadline = min(start_time + time_budget * 0.85, deadline - 5.0)
        no_improve2 = 0

        while time.time() < phase2_deadline:
            if len(A_cur) == 0:
                break

            max_c = max_count(counts_cur)
            if max_c <= 1:
                break

            # Find sums with maximum count
            high_sums = np.where(counts_cur >= max_c)[0]

            # Try to find a swap that reduces peak by targeting elements
            # that contribute to many high-peak sums
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)

            best_y = None
            best_score = None

            for _ in range(12):
                y = random.randint(0, N_max - 1)
                if y == x or ind_cur[y] == 1:
                    continue

                # Incremental update
                c_new = counts_cur.copy()
                for a in A_cur:
                    if a == x:
                        continue
                    s = a + x
                    if T_min <= s <= T_max:
                        c_new[s - T_min] -= 1
                s_xx = 2 * x
                if T_min <= s_xx <= T_max:
                    c_new[s_xx - T_min] -= 1

                for a in A_cur:
                    if a == x:
                        continue
                    s = a + y
                    if T_min <= s <= T_max:
                        c_new[s - T_min] += 1
                s_yy = 2 * y
                if T_min <= s_yy <= T_max:
                    c_new[s_yy - T_min] += 1

                if not is_valid(c_new):
                    continue

                m_new = max_count(c_new)
                n_high = int(np.sum(c_new >= m_new))
                soft = soft_max_score(c_new)
                score = (m_new, n_high, soft)

                if best_score is None or score < best_score:
                    best_score = score
                    best_y = y

            if best_y is None:
                no_improve2 += 1
                if no_improve2 > 300:
                    break
                continue

            # Apply best swap
            ind_new = ind_cur.copy()
            ind_new[x] = 0
            ind_new[best_y] = 1
            c_new = compute_counts(ind_new)

            if not is_valid(c_new):
                no_improve2 += 1
                if no_improve2 > 300:
                    break
                continue

            m_new = max_count(c_new)

            if m_new < max_cur:
                A_cur.discard(x)
                A_cur.add(best_y)
                ind_cur = ind_new
                counts_cur = c_new
                max_cur = m_new
                no_improve2 = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                old_soft = soft_max_score(counts_cur)
                new_soft = soft_max_score(c_new)
                if new_soft < old_soft or random.random() < 0.05:
                    A_cur.discard(x)
                    A_cur.add(best_y)
                    ind_cur = ind_new
                    counts_cur = c_new
            else:
                no_improve2 += 1
                if no_improve2 > 300:
                    break

        # Phase 3: Final polish — exhaustive local search over single-element moves
        polish_deadline = min(start_time + time_budget * 0.98, deadline - 2.0)
        while time.time() < polish_deadline:
            improved = False
            # Try adding an element
            for y in range(N_max):
                if ind_cur[y] == 1:
                    continue
                ind_test = ind_cur.copy()
                ind_test[y] = 1
                c_test = compute_counts(ind_test)
                if is_valid(c_test):
                    m_test = max_count(c_test)
                    if m_test < max_cur:
                        A_cur.add(y)
                        ind_cur = ind_test
                        counts_cur = c_test
                        max_cur = m_test
                        improved = True
                        if max_cur < best_max:
                            best_max = max_cur
                            best_A = set(A_cur)
                        break
                    elif m_test == max_cur:
                        old_soft = soft_max_score(counts_cur)
                        new_soft = soft_max_score(c_test)
                        if new_soft < old_soft * 0.999:  # Significant soft improvement
                            A_cur.add(y)
                            ind_cur = ind_test
                            counts_cur = c_test
                            improved = True
                        # Don't break on equal max; try other y's
            if improved:
                continue
            # Try removing an element
            for x in list(A_cur):
                if len(A_cur) <= 15:
                    break
                ind_test = ind_cur.copy()
                ind_test[x] = 0
                c_test = compute_counts(ind_test)
                if is_valid(c_test):
                    m_test = max_count(c_test)
                    if m_test < max_cur:
                        A_cur.discard(x)
                        ind_cur = ind_test
                        counts_cur = c_test
                        max_cur = m_test
                        improved = True
                        if max_cur < best_max:
                            best_max = max_cur
                            best_A = set(A_cur)
                        break
                    elif m_test == max_cur:
                        old_soft = soft_max_score(counts_cur)
                        new_soft = soft_max_score(c_test)
                        if new_soft < old_soft * 0.999:
                            A_cur.discard(x)
                            ind_cur = ind_test
                            counts_cur = c_test
                            improved = True
            if not improved:
                break

    # Try multiple diverse starting points
    starts = []
    # Seed: upper half
    starts.append(list(range(64, 129)))
    # Random sizes
    for sz in [30, 35, 40, 45, 50, 55, 60]:
        for trial in range(2):
            random.seed(42 + trial * 1000 + sz)
            starts.append(random.sample(range(0, N_max), sz))
    # Structured starts
    starts.append(list(range(0, 129, 2)))  # evens
    starts.append(list(range(1, 129, 2)))  # odds
    starts.append(list(range(32, 129, 2)))  # evens upper
    starts.append(list(range(33, 129, 2)))  # odds upper
    # Additional structured starts for diversity
    starts.append(list(range(0, 129, 3)))  # multiples of 3
    starts.append(list(range(1, 129, 3)))  # 1 mod 3
    starts.append(list(range(2, 129, 3)))  # 2 mod 3

    for i, s in enumerate(starts):
        remaining = deadline - time.time() - 3.0
        if remaining < 2.0:
            break
        time_budget = min(remaining * 0.65, 10.0)
        if i > 5:
            time_budget = min(time_budget, 5.0)
        try_improve(s, time_budget)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
