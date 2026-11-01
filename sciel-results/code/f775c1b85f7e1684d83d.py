import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 138.0  # safety margin before 150s kill

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

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def incremental_validate(rows, changed_row_idx, changed_pos, old_val, new_val):
        """Fast incremental check: only recompute diffs for the changed row.
        Returns (is_valid, scope) if we can determine it, else (None, None) to force full validate."""
        row = rows[changed_row_idx]
        # Collect all diffs from other rows
        other_diffs = set()
        for j in range(n_rows):
            if j == changed_row_idx:
                continue
            rj = rows[j]
            for i in range(len(rj)):
                ri = rj[i]
                for jj in range(i + 1, len(rj)):
                    other_diffs.add(rj[jj] - ri)
        
        # Compute new diffs for the changed row
        new_row_diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                new_row_diffs.add(row[j] - ri)
        
        # Check for conflicts
        if new_row_diffs & other_diffs:
            return False, None
        
        # Also check within-row uniqueness (should already be strict increasing, but verify)
        if len(new_row_diffs) != d:
            return False, None
        
        scope = max(max(r) for r in rows)
        return True, scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
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

    rng = random.Random(42)

    def rebuild_row(used_other, max_val, jitter=0.0, local_rng=None):
        """Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed."""
        if local_rng is None:
            local_rng = rng
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < k + 1:
            found = False
            search_end = max_val
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 400:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=6):
        """Reduce scope by locally re-optimizing each row (rebuild against others)."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 12 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                best_row = list(rows[i])
                best_max = max(rows[i])
                # Try to shrink this row's max against the other rows
                for attempt in range(25):
                    jitter = 0.0 if attempt < 8 else (0.05 + 0.03 * (attempt - 8))
                    new_row = rebuild_row(used_other, best_max - 1, jitter=jitter)
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
                    used_other |= row_diffs_set(rows[j])
            success = False
            for attempt in range(12):
                jitter = 0.25 + 0.08 * attempt
                new_row = rebuild_row(used_other, best_scope + 25, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, best_scope + 35, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

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
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(8):
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
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            for _ in range(12):
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

    def mark_anneal(rows, cur_scope, budget=4.0):
        """Fine-grained simulated annealing on individual marks with incremental validation."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_sc = cur_scope
        best_local = [r[:] for r in rows]
        best_local_sc = cur_scope
        T = 0.6
        # Precompute other-row diffs for incremental checking
        other_diffs_cache = [None] * n_rows
        for i in range(n_rows):
            s = set()
            for j in range(n_rows):
                if j == i:
                    continue
                rj = rows[j]
                for ii in range(len(rj)):
                    ri = rj[ii]
                    for jj in range(ii + 1, len(rj)):
                        s.add(rj[jj] - ri)
            other_diffs_cache[i] = s
        
        while time.time() - start < budget:
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 20
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            cur[ri][pos] = new_val
            
            # Incremental check
            row = cur[ri]
            new_row_diffs = set()
            n = len(row)
            for i in range(n):
                ri_val = row[i]
                for j in range(i + 1, n):
                    new_row_diffs.add(row[j] - ri_val)
            
            if new_row_diffs & other_diffs_cache[ri]:
                cur[ri][pos] = old_val
                continue
            if len(new_row_diffs) != d:
                cur[ri][pos] = old_val
                continue
            
            new_scope = max(max(r) for r in cur)
            if new_scope < best_local_sc:
                best_local_sc = new_scope
                best_local = [r[:] for r in cur]
            delta = new_scope - cur_sc
            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                cur_sc = new_scope
            else:
                cur[ri][pos] = old_val
            T *= 0.9999
        return best_local, best_local_sc

    def beam_polish(rows, cur_scope, budget=3.0, beam_width=3):
        """Beam-search style polish: prioritize rows with largest entries, prune states > best_scope."""
        start = time.time()
        best = [r[:] for r in rows]
        best_sc = cur_scope
        beams = [[r[:] for r in rows]]
        
        while time.time() - start < budget and beams:
            new_beams = []
            for beam in beams:
                beam_sc = get_scope(beam)
                if beam_sc < best_sc:
                    best_sc = beam_sc
                    best = [r[:] for r in beam]
                
                # Sort rows by max entry descending (bottleneck priority)
                order = sorted(range(n_rows), key=lambda i: -max(beam[i]))
                
                for ri in order[:2]:  # Try top 2 bottleneck rows
                    if time.time() - start > budget:
                        break
                    row = beam[ri]
                    pos = len(row) - 1
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    
                    if old_val <= lower:
                        continue
                    
                    # Try shrinking
                    for new_val in range(lower, min(lower + 15, old_val)):
                        if new_val >= old_val:
                            continue
                        trial = [r[:] for r in beam]
                        trial[ri][pos] = new_val
                        sc = full_validate(trial)
                        if sc is not None and sc < best_sc:
                            new_beams.append(trial)
                            if len(new_beams) >= beam_width:
                                break
                    if len(new_beams) >= beam_width:
                        break
            if not new_beams:
                break
            # Keep only top beam_width by scope
            new_beams.sort(key=lambda b: get_scope(b))
            beams = new_beams[:beam_width]
        
        return best, best_sc

    # Phase 1: ILS with row reconstruction + local polish + targeted descent
    max_il_iter = 250
    for it in range(max_il_iter):
        if time.time() > deadline - 18:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            cur = kick(cur, num_kicks=rng.choice([1, 1, 2]))

        cur_scope = full_validate(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Local polish (row-rebuild against others)
        if time.time() < deadline - 15:
            polished = local_polish(cur)
            ps = full_validate(polished)
            if ps is not None and ps < best_scope:
                best_scope = ps
                best_rows = [r[:] for r in polished]
                cur = [r[:] for r in polished]

        # Targeted descent to squeeze scope further
        if time.time() < deadline - 10:
            desc_rows, desc_scope = targeted_descent(cur, budget=2.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
                cur = [r[:] for r in desc_rows]

    # Phase 2: fine-grained mark annealing to polish the best solution
    if time.time() < deadline - 8:
        anneal_rows, anneal_scope = mark_anneal(best_rows, best_scope, budget=6.0)
        if anneal_scope < best_scope:
            best_scope = anneal_scope
            best_rows = anneal_rows

    # Phase 3: beam-search style bottleneck polish
    if time.time() < deadline - 6:
        beam_rows, beam_scope = beam_polish(best_rows, best_scope, budget=3.0, beam_width=3)
        if beam_scope < best_scope:
            best_scope = beam_scope
            best_rows = beam_rows

    # Final targeted descent on the global best
    if time.time() < deadline - 5:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=4.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Final high-precision polish: very short SA with low temperature
    if time.time() < deadline - 3:
        final_rows, final_scope = mark_anneal(best_rows, best_scope, budget=2.0)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = final_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
