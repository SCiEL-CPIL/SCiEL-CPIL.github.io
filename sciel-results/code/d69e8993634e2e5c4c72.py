import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 148.0

    total_space = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Build neighbor masks for each word
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            dv = (i // powers[pos]) % q
            for nv in range(q):
                if nv == dv:
                    continue
                m |= (1 << (i + (nv - dv) * powers[pos]))
        masks[i] = m
    full_mask = (1 << total_space) - 1

    # Precompute digit values for all words
    digit_values = []
    for i in range(total_space):
        dv = []
        for pos in range(n):
            dv.append((i // powers[pos]) % q)
        digit_values.append(tuple(dv))

    def int_to_tuple(v):
        return [0] * n  # placeholder
    def int_to_tuple(v):
        res = [0] * n
        for i in range(n):
            res[i] = (v // powers[i]) % q
        return res

    def compute_counts(selected):
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        return counts

    def polish(selected, time_limit):
        selected = selected[:]
        if not selected:
            return selected
        counts = compute_counts(selected)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit
                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True
        return selected

    def local_search(selected, time_limit):
        selected = selected[:]
        if not selected:
            return selected
        counts = compute_counts(selected)
        selected_set = set(selected)

        for _ in range(60):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit

                if unique_mask == 0:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                best_gain = -1
                best_word = -1
                best_gain2 = -1
                best_word2 = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    if (m2 & unique_mask) == unique_mask:
                        gain = (m2 & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2
                        # Also track second-best for diversity
                        if gain > best_gain2 and w2 != best_word:
                            best_gain2 = gain
                            best_word2 = w2

                if best_word >= 0 and best_gain > 0:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    mm = masks[best_word]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-1 swaps: find best replacement word
        if time.time() < time_limit - 10:
            for _ in range(15):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                for i_idx in range(min(40, len(indices))):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected):
                        continue
                    w1 = selected[idx1]
                    idx2 = random.randint(0, len(selected) - 1)
                    if idx2 == idx1:
                        continue
                    w2 = selected[idx2]
                    m1 = masks[w1]
                    m2 = masks[w2]

                    u_ab = 0
                    inter = m1 & m2
                    mm = inter
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 2:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff1 = m1 & ~m2
                    mm = diff1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff2 = m2 & ~m1
                    mm = diff2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    if u_ab == 0:
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break

                    if u_ab.bit_count() > 12:
                        continue

                    # Find best replacement word (maximize new coverage)
                    best_word = -1
                    best_new_cover = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        m3 = masks[w3]
                        if (m3 & u_ab) == u_ab:
                            new_cover = (m3 & ~(m1 | m2)).bit_count()
                            if new_cover > best_new_cover:
                                best_new_cover = new_cover
                                best_word = w3

                    if best_word >= 0:
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = masks[best_word]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected.append(best_word)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        selected_set.add(best_word)
                        improved = True
                        break

                if improved:
                    selected = polish(selected, time_limit)
                    counts = compute_counts(selected)
                    selected_set = set(selected)

        # 3-for-2 swaps: remove 3 words, add 2 words
        if time.time() < time_limit - 15:
            for _ in range(8):
                if time.time() > time_limit:
                    break
                improved = False
                if len(selected) < 6:
                    break
                indices = list(range(len(selected)))
                random.shuffle(indices)
                for i_idx in range(min(20, len(indices))):
                    i1 = indices[i_idx]
                    if i1 >= len(selected):
                        continue
                    i2 = random.randint(0, len(selected) - 1)
                    if i2 == i1:
                        continue
                    i3 = random.randint(0, len(selected) - 1)
                    if i3 == i1 or i3 == i2:
                        continue
                    w1, w2, w3 = selected[i1], selected[i2], selected[i3]
                    m1, m2, m3 = masks[w1], masks[w2], masks[w3]
                    union123 = m1 | m2 | m3

                    # Find points uniquely covered by these 3 words
                    u_123 = 0
                    mm = union123
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_123 |= lowbit
                        mm ^= lowbit

                    if u_123 == 0:
                        # All 3 can be removed
                        for ww in [w1, w2, w3]:
                            mm = masks[ww]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                        for ii in sorted([i1, i2, i3], reverse=True):
                            selected.pop(ii)
                        for ww in [w1, w2, w3]:
                            selected_set.remove(ww)
                        improved = True
                        break

                    if u_123.bit_count() > 15:
                        continue

                    # Find best 2-word replacement
                    best_pair = None
                    best_new_cover = -1
                    for a in range(total_space):
                        if a in selected_set:
                            continue
                        ma = masks[a]
                        if (ma & u_123) != u_123:
                            continue
                        for b in range(a + 1, total_space):
                            if b in selected_set:
                                continue
                            mb = masks[b]
                            if (mb & u_123) != u_123:
                                continue
                            new_cover = ((ma | mb) & ~union123).bit_count()
                            if new_cover > best_new_cover:
                                best_new_cover = new_cover
                                best_pair = (a, b)

                    if best_pair is not None:
                        a, b = best_pair
                        for ww in [w1, w2, w3]:
                            mm = masks[ww]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                        for ww in [a, b]:
                            mm = masks[ww]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] += 1
                                mm ^= lowbit
                        for ii in sorted([i1, i2, i3], reverse=True):
                            selected.pop(ii)
                        for ww in [w1, w2, w3]:
                            selected_set.remove(ww)
                        selected.append(a)
                        selected.append(b)
                        selected_set.add(a)
                        selected_set.add(b)
                        improved = True
                        break

                if improved:
                    selected = polish(selected, time_limit)
                    counts = compute_counts(selected)
                    selected_set = set(selected)

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()
        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
        return selected

    def solve_ilp(selected, time_limit):
        if not selected or time.time() > time_limit:
            return selected[:]
        m = len(selected)
        if m < 10:
            return selected[:]
        rows = []
        cols = []
        data = []
        for j, w in enumerate(selected):
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                i = lowbit.bit_length() - 1
                rows.append(i)
                cols.append(j)
                data.append(1)
                mm ^= lowbit
        A = csr_matrix((data, (rows, cols)), shape=(total_space, m))
        c = np.ones(m)
        lb = np.ones(total_space)
        ub = np.full(total_space, np.inf)
        lb_vars = np.zeros(m)
        ub_vars = np.ones(m)
        integrality = np.ones(m)
        try:
            res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
                       integrality=integrality, bounds=Bounds(lb_vars, ub_vars),
                       options={"time_limit": max(1.0, time_limit - time.time() - 2.0)})
            if res.status in (0, 1):
                x = res.x
                new_sel = [selected[j] for j in range(m) if x[j] > 0.5]
                if new_sel:
                    covered = 0
                    for w in new_sel:
                        covered |= masks[w]
                    if covered == full_mask:
                        return new_sel
        except Exception:
            pass
        return selected[:]

    # Strategy 1: Greedy + polish + local search
    best_size = total_space
    best_selected = list(range(total_space))

    if time.time() < deadline - 20:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized restarts with varied parameters
    for r in range(60):
        if time.time() > deadline - 40:
            break
        alpha = random.uniform(0.1, 0.8)
        sol = run_greedy(randomized=True, alpha=alpha)
        if len(sol) < best_size + 15:
            polished = polish(sol[:], deadline - 40)
            ls = local_search(polished[:], deadline - 40)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 3: Weighted greedy with different scoring
    if time.time() < deadline - 50:
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()
        # Use a scoring function that balances new coverage vs. redundancy
        while remaining and time.time() < deadline - 50:
            best_score = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                new_cover = (masks[w] & remaining).bit_count()
                overlap = (masks[w] & covered).bit_count()
                # Penalize overlap more heavily to encourage diversity
                score = new_cover * 20 - overlap * 3
                if score > best_score:
                    best_score = score
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        polished = polish(selected[:], deadline - 50)
        ls = local_search(polished[:], deadline - 50)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: MILP on current best
    if time.time() < deadline - 10:
        ilp_res = solve_ilp(best_selected[:], deadline - 8)
        if len(ilp_res) < best_size:
            best_size = len(ilp_res)
            best_selected = ilp_res[:]

    # Final aggressive polish with multiple passes
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            polished = polish(best_selected[:], deadline - 3)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 1)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    # One more polish after final local search
    if time.time() < deadline - 1:
        final_polish = polish(best_selected[:], deadline - 0.5)
        if len(final_polish) < best_size:
            best_size = len(final_polish)
            best_selected = final_polish[:]

    return [int_to_tuple(v) for v in best_selected]
