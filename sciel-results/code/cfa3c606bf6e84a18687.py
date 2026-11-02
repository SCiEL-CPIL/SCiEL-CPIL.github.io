import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Global best
    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        seen = set()
        conflicts = 0
        for r in rows:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    d = r[j] - ri
                    if d in seen:
                        conflicts += 1
                    else:
                        seen.add(d)
        return conflicts

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def all_diffs_set(rows):
        s = set()
        for r in rows:
            s |= row_diffs_set(r)
        return s

    def check_valid(rows):
        seen = set()
        for r in rows:
            n = len(r)
            for i in range(n):
                ri = r[i]
                for j in range(i + 1, n):
                    d = r[j] - ri
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=15, max_scope=None):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = max_scope if max_scope else 400
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    def exact_polish_row(row, fixed_diffs):
        """Try to reduce each element of row (from right to left) using fixed_diffs from other rows."""
        n = len(row)
        for c_idx in range(n - 1, 0, -1):
            curr_val = row[c_idx]
            prev_val = row[c_idx - 1]
            # Collect diffs that would be freed by changing position c_idx
            remove_set = set()
            for i in range(c_idx):
                remove_set.add(curr_val - row[i])
            for j in range(c_idx + 1, n):
                remove_set.add(row[j] - curr_val)
            # Diffs from this row that don't involve c_idx
            same_row_fixed = set()
            for i in range(n):
                if i == c_idx:
                    continue
                for j in range(i + 1, n):
                    if j == c_idx:
                        continue
                    same_row_fixed.add(row[j] - row[i])
            effective_fixed = set(fixed_diffs) | same_row_fixed
            # Try values from curr_val-1 down to prev_val+1
            found = False
            for new_val in range(curr_val - 1, prev_val, -1):
                if new_val <= prev_val:
                    break
                valid = True
                for i in range(c_idx):
                    d = new_val - row[i]
                    if d in effective_fixed:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(c_idx + 1, n):
                    d = row[j] - new_val
                    if d in effective_fixed:
                        valid = False
                        break
                if not valid:
                    continue
                new_diffs = []
                for i in range(c_idx):
                    new_diffs.append(new_val - row[i])
                for j in range(c_idx + 1, n):
                    new_diffs.append(row[j] - new_val)
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                row[c_idx] = new_val
                found = True
                break
        return row

    def high_precision_polish(rows, time_limit):
        """
        Exhaustive single-element optimization pass.
        For each element (in order of descending value), try every possible
        value in range and pick the one that minimizes the resulting scope
        while maintaining validity. Multiple passes until no improvement.
        """
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        passes = 0
        while improved and (time.time() - start) < time_limit and passes < 20:
            improved = False
            passes += 1
            # Process elements in order of descending value
            all_elements = []
            for r_idx in range(n_rows):
                for c_idx in range(1, len(rows[r_idx])):
                    all_elements.append((rows[r_idx][c_idx], r_idx, c_idx))
            all_elements.sort(reverse=True)
            
            for val, r_idx, c_idx in all_elements:
                if (time.time() - start) > time_limit:
                    break
                row = rows[r_idx]
                curr_val = row[c_idx]
                prev_val = row[c_idx - 1]
                next_val = row[c_idx + 1] if c_idx < len(row) - 1 else None
                
                # Compute fixed diffs (all diffs not involving this element)
                all_diffs = all_diffs_set(rows)
                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                if next_val is not None:
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - curr_val)
                else:
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - curr_val)
                
                fixed_diffs = all_diffs - remove_set
                same_row_fixed = set()
                for i in range(len(row)):
                    if i == c_idx:
                        continue
                    for j in range(i + 1, len(row)):
                        if j == c_idx:
                            continue
                        same_row_fixed.add(row[j] - row[i])
                effective_fixed = fixed_diffs | same_row_fixed
                
                # Try to find a lower value
                best_new_val = curr_val
                for new_val in range(curr_val - 1, prev_val, -1):
                    if new_val <= prev_val:
                        break
                    valid = True
                    for i in range(c_idx):
                        d = new_val - row[i]
                        if d in effective_fixed:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - new_val
                        if d in effective_fixed:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_diffs = [new_val - row[i] for i in range(c_idx)]
                    new_diffs += [row[j] - new_val for j in range(c_idx + 1, len(row))]
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    best_new_val = new_val
                    break  # First valid lower value is best
                
                if best_new_val < curr_val:
                    row[c_idx] = best_new_val
                    improved = True
        return rows

    def local_search_targeted(rows, time_limit):
        """Targeted local search: reduce the largest element first."""
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)

        while time.time() < start + time_limit:
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            if max_c_idx == 0:
                break
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            row = best_rows[max_r_idx]
            all_diffs = all_diffs_set(best_rows)
            remove_set = set()
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, len(row)):
                remove_set.add(row[j] - current_val)
            fixed_diffs = all_diffs - remove_set
            same_row_fixed = set()
            for i in range(len(row)):
                if i == max_c_idx:
                    continue
                for j in range(i + 1, len(row)):
                    if j == max_c_idx:
                        continue
                    same_row_fixed.add(row[j] - row[i])
            effective_fixed = fixed_diffs | same_row_fixed

            found = False
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c_idx):
                    if (new_val - row[i]) in effective_fixed:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(max_c_idx + 1, len(row)):
                    if (row[j] - new_val) in effective_fixed:
                        valid = False
                        break
                if not valid:
                    continue
                new_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_diffs += [row[j] - new_val for j in range(max_c_idx + 1, len(row))]
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                row[max_c_idx] = new_val
                found = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                break
            if not found:
                # Try a random perturbation
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, len(best_rows[r_idx]) - 1)
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx - 1]
                next_v = best_rows[r_idx][c_idx + 1] if c_idx < len(best_rows[r_idx]) - 1 else curr_val + 100
                row = best_rows[r_idx]
                all_diffs = all_diffs_set(best_rows)
                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - curr_val)
                fixed_diffs = all_diffs - remove_set
                same_row_fixed = set()
                for i in range(len(row)):
                    if i == c_idx:
                        continue
                    for j in range(i + 1, len(row)):
                        if j == c_idx:
                            continue
                        same_row_fixed.add(row[j] - row[i])
                effective_fixed = fixed_diffs | same_row_fixed
                low = prev_v + 1
                high = next_v - 1 if c_idx < len(row) - 1 else curr_val + 50
                if low > high:
                    continue
                for _ in range(30):
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    valid = True
                    for i in range(c_idx):
                        if (cand - row[i]) in effective_fixed:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in range(c_idx + 1, len(row)):
                        if (row[j] - cand) in effective_fixed:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_left = [cand - row[i] for i in range(c_idx)]
                    new_right = [row[j] - cand for j in range(c_idx + 1, len(row))]
                    all_new = new_left + new_right
                    if len(set(all_new)) != len(all_new):
                        continue
                    row[c_idx] = cand
                    break
        return best_rows, best_scope

    def sa_search(rows, time_limit):
        """Simulated annealing for conflict-free solutions."""
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_conflicts = current_conflicts
        T = 15.0
        T_min = 0.01
        cooling = 0.997
        start = time.time()
        while time.time() < start + time_limit:
            if T < T_min:
                break
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = current_rows[r_idx]
            old_val = row[pos]
            lower_bound = row[pos - 1] + 1
            if pos < k:
                upper_bound = row[pos + 1] - 1
            else:
                upper_bound = current_scope + 15
            if lower_bound > upper_bound:
                continue
            v = random.randint(lower_bound, upper_bound)
            if v == old_val:
                continue
            row[pos] = v
            new_conflicts = count_conflicts(current_rows)
            new_scope = get_scope(current_rows)
            delta_conflict = new_conflicts - current_conflicts
            delta_scope = new_scope - current_scope
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                elif random.random() < math.exp(-delta_scope / T):
                    accept = True
            else:
                if random.random() < math.exp(-delta_conflict / (T + 0.01)):
                    accept = True
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                    best_conflicts = current_conflicts
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                row[pos] = old_val
            T *= cooling
        return best_rows, best_scope

    def rebuild_row(used_other, jitter=0.0, max_val=None):
        """Rebuild a single row against fixed other rows."""
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else (best_scope + 40)
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and random.random() < jitter and trial_x < 400:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=5):
        """Iteratively rebuild rows to reduce scope."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 8 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 8:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                best_row = list(rows[i])
                best_max = max(rows[i])
                new_row = rebuild_row(used_other, jitter=0.0, max_val=best_max - 1)
                if new_row is not None and max(new_row) < best_max:
                    best_row = new_row
                    best_max = max(new_row)
                for attempt in range(8):
                    jitter = 0.15 + 0.05 * attempt
                    new_row = rebuild_row(used_other, jitter=jitter, max_val=best_max - 1)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    def kick(rows, num_kicks=1):
        """Perturb by rebuilding some rows."""
        rows = [list(r) for r in rows]
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            success = False
            for attempt in range(12):
                jitter = 0.25 + 0.08 * attempt
                new_row = rebuild_row(used_other, jitter=jitter, max_val=best_scope + 25)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, jitter=0.0, max_val=best_scope + 35)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    def beam_search_init(time_limit):
        """
        Beam search with pruning: build rows one at a time, keeping only
        the top beam_width partial configurations by scope.
        """
        start = time.time()
        beam_width = 50
        # Start with first row
        beam = []
        for _ in range(beam_width):
            rand_prob = random.random() * 0.3
            rows = greedy_init(rand_prob, window=10 + int(random.random() * 15), max_scope=best_scope if best_scope < float('inf') else 200)
            beam.append((get_scope(rows), rows))
        
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]
        
        for row_idx in range(1, n_rows):
            if time.time() - start > time_limit:
                break
            new_beam = []
            for scope, rows in beam:
                if time.time() - start > time_limit:
                    break
                # Try to add a new row
                for attempt in range(10):
                    jitter = attempt * 0.1
                    used_other = set()
                    for r in rows:
                        used_other |= row_diffs_set(r)
                    new_row = rebuild_row(used_other, jitter=jitter, max_val=best_scope if best_scope < float('inf') else 200)
                    if new_row is not None:
                        new_rows = rows + [new_row]
                        new_scope = get_scope(new_rows)
                        new_beam.append((new_scope, new_rows))
                        break
            if not new_beam:
                break
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]
        
        if beam:
            return beam[0][1]
        return None

    # === MAIN SEARCH LOOP ===

    # Phase 1: Beam search for initial solution
    beam_time = min(10.0, max(3.0, (deadline - time.time() - 30) * 0.15))
    beam_result = beam_search_init(beam_time)
    if beam_result is not None:
        b_scope = get_scope(beam_result)
        b_conflicts = count_conflicts(beam_result)
        if b_conflicts < best_conflicts or (b_conflicts == best_conflicts and b_scope < best_scope):
            best_conflicts = b_conflicts
            best_scope = b_scope
            best_rows = [r[:] for r in beam_result]

    # Phase 2: Multiple restarts with greedy init + targeted local search
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break
        rand_prob = 0.05 + 0.04 * i
        window = 10 + 4 * i
        max_scope_limit = best_scope - 1 if best_scope < float('inf') else None
        rows = greedy_init(rand_prob, window, max_scope=max_scope_limit)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)

        # Targeted local search
        t1_time = min(4.0, max(2.0, (deadline - time.time() - 25) / max(1, num_restarts - i)))
        t1_rows, t1_scope = local_search_targeted(rows, t1_time)
        t1_conflicts = count_conflicts(t1_rows)
        if t1_conflicts < best_conflicts or (t1_conflicts == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conflicts
            best_scope = t1_scope
            best_rows = [r[:] for r in t1_rows]

        # SA if we have time
        if time.time() < deadline - 25:
            sa_time = min(3.0, (deadline - time.time() - 25) / max(1, num_restarts - i))
            t2_rows, t2_scope = sa_search(t1_rows, sa_time)
            t2_conflicts = count_conflicts(t2_rows)
            if t2_conflicts < best_conflicts or (t2_conflicts == best_conflicts and t2_scope < best_scope):
                best_conflicts = t2_conflicts
                best_scope = t2_scope
                best_rows = [r[:] for r in t2_rows]

        # Early exit if we hit a good solution
        if best_conflicts == 0 and best_scope < 115:
            break

    # Phase 3: ILS (Iterated Local Search) with kicks
    if best_rows is not None:
        remaining = deadline - time.time() - 15
        if remaining > 15:
            il_time = remaining * 0.5
            start_il = time.time()
            while time.time() - start_il < il_time:
                if time.time() > deadline - 15:
                    break
                cur = kick(best_rows, num_kicks=random.choice([1, 1, 2]))
                cur_conflicts = count_conflicts(cur)
                if cur_conflicts > 0:
                    sa_t = min(2.0, (deadline - time.time() - 15) / 3)
                    cur, cur_scope = sa_search(cur, sa_t)
                    cur_conflicts = count_conflicts(cur)
                # Polish
                pol = local_polish(cur, max_passes=3)
                pol_conflicts = count_conflicts(pol)
                pol_scope = get_scope(pol)
                if pol_conflicts < best_conflicts or (pol_conflicts == best_conflicts and pol_scope < best_scope):
                    best_conflicts = pol_conflicts
                    best_scope = pol_scope
                    best_rows = [r[:] for r in pol]

    # Phase 4: Final targeted local search
    if best_rows is not None and time.time() < deadline - 10:
        final_t = min(8.0, deadline - time.time() - 8)
        f1_rows, f1_scope = local_search_targeted(best_rows, final_t)
        f1_conflicts = count_conflicts(f1_rows)
        if f1_conflicts < best_conflicts or (f1_conflicts == best_conflicts and f1_scope < best_scope):
            best_conflicts = f1_conflicts
            best_scope = f1_scope
            best_rows = [r[:] for r in f1_rows]

    # Phase 5: High-precision polish (exhaustive single-element optimization)
    if best_rows is not None and time.time() < deadline - 5:
        polish_t = min(20.0, deadline - time.time() - 3)
        polished = high_precision_polish(best_rows, polish_t)
        pol_conflicts = count_conflicts(polished)
        if pol_conflicts == 0:
            pol_scope = get_scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = [r[:] for r in polished]

    # Fallback if nothing found
    if best_rows is None:
        best_rows = greedy_init(0.1, 20)

    # Final safety: ensure we return valid rows
    if best_conflicts > 0:
        if time.time() < deadline - 2:
            final_rows, final_scope = sa_search(best_rows, deadline - time.time() - 1)
            final_conflicts = count_conflicts(final_rows)
            if final_conflicts < best_conflicts or (final_conflicts == best_conflicts and final_scope < best_scope):
                best_conflicts = final_conflicts
                best_scope = final_scope
                best_rows = [r[:] for r in final_rows]

    return best_rows
