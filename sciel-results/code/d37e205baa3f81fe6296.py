import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 135.0  # safety margin before 150s kill

    row_len = k + 1
    d = k * (k + 1) // 2  # differences per row
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(12345)

    def rebuild_row_set(used_set, max_val, jitter=0.0, local_rng=None):
        """Rebuild a single row using set for used differences."""
        if local_rng is None:
            local_rng = rng
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            for trial_x in range(x + 1, max_val + 1):
                new = {trial_x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def rebuild_row_gaps(used_set, max_val, gap_jitter=0.0, local_rng=None):
        """Rebuild a row by choosing gaps (differences between consecutive elements).
        The row is [0, g1, g1+g2, ..., sum(gaps)].
        Each gap must be positive and the cumulative sums must not create duplicate differences."""
        if local_rng is None:
            local_rng = rng
        row = [0]
        row_diffs = set()
        x = 0
        for _ in range(k):
            found = False
            # Determine valid range for next gap
            min_gap = 1
            max_gap = max_val - x
            if max_gap < min_gap:
                return None
            
            # Collect valid gaps in a window first
            window = min(30, max_gap - min_gap + 1)
            candidates = []
            for g in range(min_gap, min_gap + window):
                trial_x = x + g
                if trial_x > max_val:
                    break
                new = {trial_x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    candidates.append(g)
            
            if not candidates:
                # Scan the rest
                for g in range(min_gap + window, max_gap + 1):
                    trial_x = x + g
                    if trial_x > max_val:
                        break
                    new = {trial_x - a for a in row}
                    if not (new & used_set) and not (new & row_diffs):
                        candidates.append(g)
            
            if not candidates:
                return None
            
            # Pick a gap: biased towards smaller unless jitter
            if gap_jitter > 0 and local_rng.random() < gap_jitter:
                g = local_rng.choice(candidates)
            else:
                g = candidates[0]
            
            x += g
            row.append(x)
            new = {x - a for a in row[:-1]}
            row_diffs |= new
        
        return row

    def local_polish(rows, max_passes=8):
        """Reduce scope by locally re-optimizing each row."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 15 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 15:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        for a in range(len(rows[j])):
                            for b in range(a + 1, len(rows[j])):
                                used_other.add(rows[j][b] - rows[j][a])
                best_row = list(rows[i])
                best_max = max(rows[i])
                for attempt in range(30):
                    jitter = 0.0 if attempt < 10 else (0.05 + 0.04 * (attempt - 10))
                    # Alternate between standard rebuild and gap-based rebuild
                    if attempt % 2 == 0:
                        new_row = rebuild_row_set(used_other, best_max - 1, jitter=jitter)
                    else:
                        new_row = rebuild_row_gaps(used_other, best_max - 1, gap_jitter=jitter)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    def kick(rows, num_kicks=1):
        """Perturb by rebuilding random rows with high jitter."""
        rows = [list(r) for r in rows]
        indices = rng.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    for a in range(len(rows[j])):
                        for b in range(a + 1, len(rows[j])):
                            used_other.add(rows[j][b] - rows[j][a])
            success = False
            for attempt in range(15):
                jitter = 0.3 + 0.1 * attempt
                if attempt % 2 == 0:
                    new_row = rebuild_row_set(used_other, best_scope + 30, jitter=jitter)
                else:
                    new_row = rebuild_row_gaps(used_other, best_scope + 30, gap_jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row_set(used_other, best_scope + 40, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    def gap_swap_move(rows, budget=3.0):
        """Try to reduce scope by swapping gaps between rows.
        A gap is the difference between consecutive elements in a row.
        We identify rows with large spans and try to redistribute their gaps."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            # Find the row with the maximum last element
            max_idx = max(range(n_rows), key=lambda i: max(cur[i]))
            row = cur[max_idx]
            
            if max(row) <= 112:
                break
            
            # Get gaps of this row
            gaps = [row[i+1] - row[i] for i in range(len(row)-1)]
            
            # Try to reduce a large gap by moving it to another row
            # Find the largest gap
            max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
            large_gap = gaps[max_gap_idx]
            
            if large_gap <= 2:
                break
            
            # Try to redistribute: reduce this gap and increase a gap in another row
            # We'll try to find a row that can absorb a larger gap
            for other_idx in range(n_rows):
                if other_idx == max_idx:
                    continue
                other_row = cur[other_idx]
                other_gaps = [other_row[i+1] - other_row[i] for i in range(len(other_row)-1)]
                
                # Try to increase the smallest gap in other_row by some amount
                # and decrease the large gap in max_row by the same amount
                min_gap_idx = min(range(len(other_gaps)), key=lambda i: other_gaps[i])
                min_gap = other_gaps[min_gap_idx]
                
                # Try delta values
                for delta in range(1, min(large_gap - 1, 10)):
                    new_large_gap = large_gap - delta
                    new_min_gap = min_gap + delta
                    
                    if new_large_gap <= 0 or new_min_gap <= 0:
                        continue
                    
                    # Check if the modified rows are still valid
                    # Reconstruct max_row with modified gap
                    new_max_row = [0]
                    x = 0
                    for gi, gap in enumerate(gaps):
                        if gi == max_gap_idx:
                            gap = new_large_gap
                        x += gap
                        new_max_row.append(x)
                    
                    # Reconstruct other_row with modified gap
                    new_other_row = [0]
                    x = 0
                    for gi, gap in enumerate(other_gaps):
                        if gi == min_gap_idx:
                            gap = new_min_gap
                        x += gap
                        new_other_row.append(x)
                    
                    # Check validity
                    test_rows = [r[:] for r in cur]
                    test_rows[max_idx] = new_max_row
                    test_rows[other_idx] = new_other_row
                    scope = full_validate(test_rows)
                    if scope is not None and scope < cur_scope:
                        cur = test_rows
                        cur_scope = scope
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break

        if improved:
            return cur, cur_scope
        return None, None

    def targeted_descent(rows, budget=3.0):
        """Try to reduce scope by shrinking the largest entry in max rows."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                # Try a middle position
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(10):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break
                continue

            candidates = set()
            for v in range(lower, min(lower + 25, old_val)):
                candidates.add(v)
            for _ in range(15):
                candidates.add(rng.randint(lower, old_val - 1))

            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                row[pos] = old_val

            if not improved:
                break

        if improved:
            return cur, cur_scope
        return None, None

    def mark_anneal(rows, cur_scope, budget=5.0):
        """Fine-grained simulated annealing on individual marks."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_sc = cur_scope
        best_local = [r[:] for r in rows]
        best_local_sc = cur_scope
        T = 0.8
        while time.time() - start < budget:
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 25
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            cur[ri][pos] = new_val
            new_scope = full_validate(cur)
            if new_scope is not None:
                if new_scope < best_local_sc:
                    best_local_sc = new_scope
                    best_local = [r[:] for r in cur]
                delta = new_scope - cur_sc
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_sc = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                cur[ri][pos] = old_val
            T *= 0.9998
        return best_local, best_local_sc

    def row_swap_search(rows, budget=4.0):
        """Try swapping row order to enable smaller scopes."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        best = [r[:] for r in cur]
        best_sc = cur_scope
        while time.time() - start < budget:
            i = rng.randrange(n_rows)
            j = rng.randrange(n_rows)
            if i == j:
                continue
            cur[i], cur[j] = cur[j], cur[i]
            # Try to polish after swap
            polished = local_polish(cur, max_passes=2)
            ps = full_validate(polished)
            if ps is not None and ps < best_sc:
                best_sc = ps
                best = [r[:] for r in polished]
            else:
                cur[i], cur[j] = cur[j], cur[i]
        return best, best_sc

    # Phase 1: ILS with row reconstruction + local polish
    max_il_iter = 300
    for it in range(max_il_iter):
        if time.time() > deadline - 20:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            if rng.random() < 0.7:
                cur = kick(cur, num_kicks=rng.choice([1, 1, 2]))
            else:
                # Try a row swap perturbation
                i = rng.randrange(n_rows)
                j = rng.randrange(n_rows)
                if i != j:
                    cur[i], cur[j] = cur[j], cur[i]

        cur_scope = full_validate(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Local polish
        if time.time() < deadline - 18:
            polished = local_polish(cur)
            ps = full_validate(polished)
            if ps is not None and ps < best_scope:
                best_scope = ps
                best_rows = [r[:] for r in polished]
                cur = [r[:] for r in polished]

        # Targeted descent
        if time.time() < deadline - 12:
            desc_rows, desc_scope = targeted_descent(cur, budget=1.5)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
                cur = [r[:] for r in desc_rows]

        # Gap swap move (new)
        if time.time() < deadline - 10:
            gap_rows, gap_scope = gap_swap_move(cur, budget=1.0)
            if gap_rows is not None and gap_scope < best_scope:
                best_scope = gap_scope
                best_rows = gap_rows
                cur = [r[:] for r in gap_rows]

    # Phase 2: Row swap search on the best solution
    if time.time() < deadline - 12:
        swap_rows_best, swap_scope = row_swap_search(best_rows, budget=5.0)
        if swap_scope < best_scope:
            best_scope = swap_scope
            best_rows = swap_rows_best

    # Phase 3: Fine-grained mark annealing
    if time.time() < deadline - 8:
        anneal_rows, anneal_scope = mark_anneal(best_rows, best_scope, budget=6.0)
        if anneal_scope < best_scope:
            best_scope = anneal_scope
            best_rows = anneal_rows

    # Phase 4: Final targeted descent
    if time.time() < deadline - 5:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=3.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Phase 5: One more polish pass
    if time.time() < deadline - 3:
        polished = local_polish(best_rows, max_passes=4)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    # Phase 6: Final gap swap polish
    if time.time() < deadline - 2:
        gap_rows, gap_scope = gap_swap_move(best_rows, budget=2.0)
        if gap_rows is not None and gap_scope < best_scope:
            best_scope = gap_scope
            best_rows = gap_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
