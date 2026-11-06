import time
import random

N = 256
MAX_SUM = 2 * N  # 512


def construct():
    deadline = time.time() + 55.0

    def can_add(x, sel, sum_mask):
        """Check if x can be added to sel maintaining Sidon property."""
        if (sum_mask >> (2 * x)) & 1:
            return False
        for a in sel:
            if (sum_mask >> (x + a)) & 1:
                return False
        return True

    def add_element(x, sel, sel_set, sum_mask):
        """Add x to sel, update sel_set and sum_mask. Returns new sum_mask."""
        sel.append(x)
        sel_set.add(x)
        for a in sel[:-1]:
            sum_mask |= 1 << (x + a)
        sum_mask |= 1 << (2 * x)
        return sum_mask

    def compute_mask(sel):
        """Compute bitmask of all sums a+b for a,b in sel, a<=b."""
        m = 0
        for i in range(len(sel)):
            a = sel[i]
            m |= 1 << (2 * a)
            for j in range(i + 1, len(sel)):
                m |= 1 << (a + sel[j])
        return m

    def is_maximal(sel, sum_mask, sel_set):
        """Check if sel is inclusion-maximal in [1,N]."""
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            if (sum_mask >> (2 * y)) & 1:
                continue
            conflict = False
            for a in sel:
                if (sum_mask >> (y + a)) & 1:
                    conflict = True
                    break
            if not conflict:
                return False
        return True

    def saturate(sel, sel_set, sum_mask):
        """Add all possible elements greedily to make the set maximal."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if x in sel_set:
                    continue
                if can_add(x, sel, sum_mask):
                    sum_mask = add_element(x, sel, sel_set, sum_mask)
                    changed = True
        return sel, sel_set, sum_mask

    def greedy_build(seed):
        """Greedy construction with blocking heuristic, then saturate."""
        rng = random.Random(seed)
        sel = []
        sel_set = set()
        sum_mask = 0
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)

        while remaining and time.time() < deadline:
            best_x = -1
            best_score = -1.0

            # Use a subset of candidates for speed when remaining is large
            if len(remaining) > 50:
                candidates = rng.sample(remaining, 50)
            else:
                candidates = list(remaining)

            for x in candidates:
                if x in sel_set:
                    continue
                if not can_add(x, sel, sum_mask):
                    continue

                # Score: how many remaining elements would be blocked
                block_count = 0.0
                for y in remaining:
                    if y == x:
                        continue
                    # y is blocked if 2y is in new sums or x+y is in new sums
                    if (sum_mask >> (2 * y)) & 1:
                        # Already blocked by existing sums, partial credit
                        block_count += 0.3
                    else:
                        if (sum_mask >> (x + y)) & 1:
                            # x+y already in sums, so adding x blocks y
                            block_count += 1.0
                        elif 2 * y == 2 * x:
                            block_count += 0.5
                # Also count how many existing sel elements' sums conflict
                # (already handled by can_add)

                if block_count > best_score:
                    best_score = block_count
                    best_x = x

            if best_x == -1:
                # Fallback: pick any addable element
                for x in remaining:
                    if x in sel_set:
                        continue
                    if can_add(x, sel, sum_mask):
                        best_x = x
                        break

            if best_x == -1:
                break

            sum_mask = add_element(best_x, sel, sel_set, sum_mask)
            remaining.remove(best_x)

        # Saturate to make maximal
        sel, sel_set, sum_mask = saturate(sel, sel_set, sum_mask)
        return sel

    def local_search_reduce(sel, time_budget=8.0):
        """Try to reduce size via removal + saturation moves."""
        start = time.time()
        current = list(sel)
        current_size = len(current)
        rng = random.Random(12345)

        while time.time() < start + time_budget:
            improved = False
            current_set = set(current)
            current_mask = compute_mask(current)

            # Move type 1: remove one element, saturate, check if smaller
            for i in range(len(current)):
                if time.time() > start + time_budget:
                    break
                x1 = current[i]
                rem_sel = [e for e in current if e != x1]
                rem_set = set(rem_sel)
                rem_mask = compute_mask(rem_sel)

                # Saturate the reduced set
                sat_sel = list(rem_sel)
                sat_set = set(rem_sel)
                sat_mask = rem_mask
                sat_sel, sat_set, sat_mask = saturate(sat_sel, sat_set, sat_mask)

                if len(sat_sel) < current_size:
                    current = sat_sel
                    current_size = len(current)
                    improved = True
                    break

            if improved:
                continue

            # Move type 2: remove two elements, add one, saturate
            for i in range(len(current)):
                if time.time() > start + time_budget:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > start + time_budget:
                        break
                    x1 = current[i]
                    x2 = current[j]
                    rem_sel = [e for e in current if e != x1 and e != x2]
                    rem_set = set(rem_sel)
                    rem_mask = compute_mask(rem_sel)

                    # Try adding each candidate
                    for y in range(1, N + 1):
                        if y in rem_set:
                            continue
                        if not can_add(y, rem_sel, rem_mask):
                            continue

                        new_sel = list(rem_sel)
                        new_set = set(rem_sel)
                        new_mask = rem_mask
                        new_mask = add_element(y, new_sel, new_set, new_mask)

                        # Saturate
                        new_sel, new_set, new_mask = saturate(new_sel, new_set, new_mask)

                        if len(new_sel) < current_size:
                            current = new_sel
                            current_size = len(current)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Move type 3: random perturbation - remove a random element, saturate
            if len(current) < 2:
                break
            x_idx = rng.randint(0, len(current) - 1)
            x = current[x_idx]
            rem_sel = current[:x_idx] + current[x_idx + 1:]
            rem_set = set(rem_sel)
            rem_mask = compute_mask(rem_sel)

            sat_sel = list(rem_sel)
            sat_set = set(rem_sel)
            sat_mask = rem_mask
            sat_sel, sat_set, sat_mask = saturate(sat_sel, sat_set, sat_mask)

            if len(sat_sel) < current_size:
                current = sat_sel
                current_size = len(current)
                improved = True

            if not improved:
                # Try a few more random perturbations
                found = False
                for _ in range(5):
                    if time.time() > start + time_budget:
                        break
                    if len(current) < 2:
                        break
                    x_idx = rng.randint(0, len(current) - 1)
                    x = current[x_idx]
                    rem_sel = current[:x_idx] + current[x_idx + 1:]
                    rem_set = set(rem_sel)
                    rem_mask = compute_mask(rem_sel)

                    sat_sel = list(rem_sel)
                    sat_set = set(rem_sel)
                    sat_mask = rem_mask
                    sat_sel, sat_set, sat_mask = saturate(sat_sel, sat_set, sat_mask)

                    if len(sat_sel) < current_size:
                        current = sat_sel
                        current_size = len(current)
                        found = True
                        break

                if not found:
                    break

        return current

    def backtrack_polish(sel, time_budget=10.0):
        """Try to find a smaller maximal set by backtracking removals."""
        start = time.time()
        best = list(sel)
        best_size = len(best)
        best_set = set(best)
        best_mask = compute_mask(best)

        def try_remove_and_rebuild(current, current_set, current_mask):
            nonlocal best, best_size, best_set, best_mask
            if time.time() > start + time_budget:
                return
            if len(current) >= best_size:
                return

            # Try removing each element
            for i in range(len(current)):
                if time.time() > start + time_budget:
                    return
                x = current[i]
                rem_sel = current[:i] + current[i + 1:]
                rem_set = set(rem_sel)
                rem_mask = compute_mask(rem_sel)

                # Saturate
                sat_sel = list(rem_sel)
                sat_set = set(rem_sel)
                sat_mask = rem_mask
                sat_sel, sat_set, sat_mask = saturate(sat_sel, sat_set, sat_mask)

                if len(sat_sel) < best_size:
                    best = sat_sel
                    best_size = len(sat_sel)
                    best_set = sat_set
                    best_mask = sat_mask
                    # Recurse on the improved set
                    try_remove_and_rebuild(sat_sel, sat_set, sat_mask)

                # Also try: remove x, add a random new element, saturate
                for y in range(1, N + 1):
                    if y in rem_set:
                        continue
                    if not can_add(y, rem_sel, rem_mask):
                        continue
                    new_sel = list(rem_sel)
                    new_set = set(rem_sel)
                    new_mask = rem_mask
                    new_mask = add_element(y, new_sel, new_set, new_mask)
                    sat_sel2 = list(new_sel)
                    sat_set2 = set(new_sel)
                    sat_mask2 = new_mask
                    sat_sel2, sat_set2, sat_mask2 = saturate(sat_sel2, sat_set2, sat_mask2)
                    if len(sat_sel2) < best_size:
                        best = sat_sel2
                        best_size = len(sat_sel2)
                        best_set = sat_set2
                        best_mask = sat_mask2
                        try_remove_and_rebuild(sat_sel2, sat_set2, sat_mask2)
                        break

        try_remove_and_rebuild(list(sel), set(sel), compute_mask(sel))
        return best

    best_set = None
    best_size = float('inf')

    # Phase 1: Generate candidates with greedy + local search
    num_trials = 20
    for i in range(num_trials):
        if time.time() > deadline - 15:
            break
        sel = greedy_build(i)
        if len(sel) < best_size:
            # Apply local search reduction
            reduced = local_search_reduce(sel, time_budget=3.0)
            if len(reduced) < len(sel):
                sel = reduced
            if len(sel) < best_size:
                best_size = len(sel)
                best_set = list(sel)

    if best_set is None:
        best_set = greedy_build(0)
        best_size = len(best_set)

    # Phase 2: Backtracking polish on the best set
    if time.time() < deadline - 5:
        polished = backtrack_polish(best_set, time_budget=10.0)
        if len(polished) < best_size:
            best_size = len(polished)
            best_set = polished

    # Phase 3: Final local search pass if time remains
    if time.time() < deadline - 3:
        final = local_search_reduce(best_set, time_budget=5.0)
        if len(final) < best_size:
            best_size = len(final)
            best_set = final

    return best_set
