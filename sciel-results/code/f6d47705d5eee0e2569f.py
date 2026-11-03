import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks for each word
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
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
        """Remove redundant codewords iteratively."""
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
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = compute_counts(selected)
        selected_set = set(selected)

        # 1-for-1 swap and removal
        for _ in range(80):
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
                    # Can remove directly
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

                # Try to swap
                best_gain = -1
                best_word = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    if (m2 & unique_mask) == unique_mask:
                        gain = (m2 & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Remove w
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    # Add best_word
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

        # 2-for-1 swap
        if time.time() < time_limit - 8:
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
                        # Remove both
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

                    if u_ab.bit_count() > 8:
                        continue

                    best_word = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        m3 = masks[w3]
                        if (m3 & u_ab) == u_ab:
                            best_word = w3
                            break

                    if best_word >= 0:
                        # Remove w1, w2
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
                        # Add best_word
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

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()

        while remaining and time.time() < deadline - 8:
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
        """
        Use MILP to find a smaller subset of 'selected' that still covers everything.
        """
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
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=Bounds(lb_vars, ub_vars),
                options={"time_limit": max(1.0, time_limit - time.time() - 2.0)}
            )
            
            if res.status in (0, 1):
                x = res.x
                new_selected = [selected[j] for j in range(m) if x[j] > 0.5]
                if new_selected:
                    covered = 0
                    for w in new_selected:
                        covered |= masks[w]
                    if covered == full_mask:
                        return new_selected
                        
        except Exception:
            pass
        
        return selected[:]

    def solve_ilp_expanded(selected, time_limit):
        """
        MILP with expanded candidate set: current solution + high-gain words from space.
        """
        if not selected or time.time() > time_limit:
            return selected[:]

        m_cur = len(selected)
        if m_cur < 10:
            return selected[:]

        # Build coverage of current solution
        cur_covered = 0
        for w in selected:
            cur_covered |= masks[w]

        # Find candidate words not in selected that have high gain over current coverage
        candidates = set(selected)
        
        # Add top 200 words by new coverage
        gains = []
        for w in range(total_space):
            if w in candidates:
                continue
            gain = (masks[w] & ~cur_covered).bit_count()
            if gain > 0:
                gains.append((gain, w))
        
        gains.sort(reverse=True)
        for _, w in gains[:200]:
            candidates.add(w)
        
        # Also add words that are "close" to current codewords (Hamming distance 1-2)
        for w in selected:
            for pos in range(n):
                digit_val = (w // powers[pos]) % q
                for new_val in range(q):
                    if new_val == digit_val:
                        continue
                    diff = new_val - digit_val
                    w2 = w + diff * powers[pos]
                    if w2 not in candidates:
                        candidates.add(w2)
        
        candidate_list = list(candidates)
        m = len(candidate_list)
        
        if m < 10 or time.time() > time_limit - 5:
            return selected[:]

        # Build constraint matrix
        rows = []
        cols = []
        data = []
        
        for j, w in enumerate(candidate_list):
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
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=Bounds(lb_vars, ub_vars),
                options={"time_limit": max(1.0, time_limit - time.time() - 2.0)}
            )
            
            if res.status in (0, 1):
                x = res.x
                new_selected = [candidate_list[j] for j in range(m) if x[j] > 0.5]
                if new_selected:
                    covered = 0
                    for w in new_selected:
                        covered |= masks[w]
                    if covered == full_mask:
                        return new_selected
                        
        except Exception:
            pass
        
        return selected[:]

    def hungarian_improve(selected, time_limit):
        """
        Use Hungarian algorithm on a bipartite graph to try to reduce codeword count.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = compute_counts(selected)
        selected_set = set(selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                
                # Find words covered ONLY by w
                unique_words = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_words.append(pos)
                    mm ^= lowbit
                
                if not unique_words:
                    # Can remove directly
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
                
                # Try to find a replacement or reassignment
                if len(unique_words) > 10:
                    continue
                    
                best_w2 = -1
                best_gain = -1
                
                u_mask = 0
                for u in unique_words:
                    u_mask |= (1 << u)
                    
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    if (masks[w2] & u_mask) == u_mask:
                        gain = (masks[w2] & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_w2 = w2
                            
                if best_w2 != -1:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                        
                    mm = masks[best_w2]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                        
                    selected[idx] = best_w2
                    selected_set.remove(w)
                    selected_set.add(best_w2)
                    improved = True
                    break
            
            if improved:
                counts = compute_counts(selected)
                selected_set = set(selected)

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 15:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 15)
        ls = local_search(polished[:], deadline - 15)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    for r in range(100):
        if time.time() > deadline - 30:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.6))
        if len(sol) < best_size + 8:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 3: Two-phase greedy
    if time.time() < deadline - 40:
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()

        phase1_limit = int(total_space * 0.5)
        while remaining and len(selected) < phase1_limit and time.time() < deadline - 40:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        while remaining and time.time() < deadline - 40:
            best_overlap = total_space + 1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                new_cover = (masks[w] & remaining).bit_count()
                overlap = (masks[w] & covered).bit_count()
                score = new_cover * 10 - overlap
                if score > best_overlap:
                    best_overlap = score
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        polished = polish(selected[:], deadline - 40)
        ls = local_search(polished[:], deadline - 40)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: MILP on current best
    if time.time() < deadline - 5:
        ilp_result = solve_ilp(best_selected[:], deadline - 3)
        if len(ilp_result) < best_size:
            best_size = len(ilp_result)
            best_selected = ilp_result[:]

    # Strategy 5: MILP with expanded candidate set
    if time.time() < deadline - 10:
        ilp_exp = solve_ilp_expanded(best_selected[:], deadline - 5)
        if len(ilp_exp) < best_size:
            best_size = len(ilp_exp)
            best_selected = ilp_exp[:]

    # Strategy 6: Hungarian/Matching-based improvement
    if time.time() < deadline - 5:
        h_res = hungarian_improve(best_selected[:], deadline - 3)
        if len(h_res) < best_size:
            best_size = len(h_res)
            best_selected = h_res[:]

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Final local search pass
    if time.time() < deadline - 1:
        final_ls = local_search(best_selected[:], deadline - 1)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
