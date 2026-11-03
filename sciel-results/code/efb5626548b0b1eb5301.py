import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _row_rebuild(row, k, diff_counts, rng, max_scope):
    best_row = None
    best_score = float('inf')
    
    for attempt in range(12):
        row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.4:
                start = x + rng.randint(1, 4)
            
            # Try to find a value with no conflicts first, then allow 1 conflict
            best_t = None
            best_conflicts = float('inf')
            for t in range(start, max_scope + 1):
                conflicts = 0
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_t = t
                    if conflicts == 0:
                        break
            
            if best_t is not None:
                if best_conflicts == 0 or (attempt < 4 and best_conflicts <= 1 and rng.random() < 0.2):
                    row.append(best_t)
                    for a in row[:-1]:
                        d = best_t - a
                        if d > 0:
                            local_diffs.add(d)
                    x = best_t
                    found = True
        
            if not found:
                ok = False
                break
        if ok and len(row) == k + 1:
            score = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if diff_counts.get(d, 0) > 0:
                        score += 1
            if score < best_score:
                best_score = score
                best_row = list(row)
            if best_score == 0:
                break
    
    return best_row

def _polish(rows, k, rng, deadline):
    """
    Greedy polish: try to reduce the scope by shrinking elements from largest to smallest.
    We iterate through all elements in descending order of value.
    For each element, try to decrease it as much as possible without breaking validity.
    """
    rows = [list(r) for r in rows]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Get all (row_idx, pos_idx) sorted by value descending
        indices = []
        for ri, r in enumerate(rows):
            for pi in range(1, k + 1):
                indices.append((r[pi], ri, pi))
        indices.sort(reverse=True)
        
        for val, ri, pi in indices:
            if time.time() > deadline:
                break
            current_val = rows[ri][pi]
            if current_val != val:
                continue # Value changed by a previous move in this pass
            
            prev_val = rows[ri][pi-1]
            next_val = rows[ri][pi+1] if pi < k else float('inf')
            
            # Try to reduce current_val
            # The new value must be > prev_val and < next_val
            # We want the smallest possible value that doesn't create conflicts.
            # Conflicts are with differences in the same row and other rows.
            
            # To check validity of a specific change, we can temporarily remove the row's diffs,
            # change the value, and check if any new diff conflicts with existing diffs (from other rows or same row).
            # Since we are minimizing, we try values from low to high? No, we want to reduce, so we try from current_val-1 down to prev_val+1.
            # But checking every step is expensive. 
            # Better: calculate the set of "forbidden" differences for this position.
            
            # Let's just try a few random reductions or a binary search?
            # Given the small size, a linear scan from current_val-1 down to prev_val+1 is safe and fast enough for a few iterations.
            
            # Optimization: Only try to reduce if it can potentially reduce the global scope or just local conflicts.
            # Since we process in descending order, reducing a large value is good.
            
            old_row = list(rows[ri])
            # Get all diffs from other rows
            other_diffs = set()
            for r_idx, r in enumerate(rows):
                if r_idx == ri:
                    continue
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_diffs.add(r[j] - r[i])
            
            # Get diffs within the same row, excluding those involving position pi
            same_row_diffs_excl = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    if i == pi or j == pi:
                        continue
                    same_row_diffs_excl.add(old_row[j] - old_row[i])
            
            forbidden = other_diffs | same_row_diffs_excl
            
            # Try to find the smallest valid value for rows[ri][pi]
            # It must be > prev_val and < next_val
            # And (new_val - a) not in forbidden for all a < new_val in row
            # And (a - new_val) not in forbidden for all a > new_val in row
            
            best_new_val = None
            # We want to minimize the value, so start from low?
            # But we are trying to *reduce* the current value. 
            # If we just find the *minimum* valid value, it might be much smaller, which is great.
            # However, we must ensure the row remains strictly increasing.
            
            # Let's try values from prev_val + 1 upwards?
            # If we find a smaller value, we update.
            # But wait, if we change a middle element, it affects diffs with both left and right.
            
            # Let's just try to reduce the current value by 1, 2, ... up to a limit, or just find the min valid.
            # Finding the min valid is stronger.
            
            min_possible = prev_val + 1
            max_possible = min(current_val - 1, next_val - 1)
            
            if min_possible > max_possible:
                continue
                
            # Try all values in [min_possible, max_possible]?
            # That's too slow if the range is large.
            # But the range is bounded by the row's span.
            # For k=5, the span is small.
            
            # Heuristic: try a few random values in the range, plus the minimum.
            # Or just try to reduce by 1. If valid, keep going.
            
            # Let's try a binary search for the minimum valid value?
            # Validity is not monotonic.
            
            # Let's just try to reduce by 1 repeatedly.
            # This is slow.
            
            # Better approach for Polish:
            # Try to set rows[ri][pi] to the smallest valid value in [prev_val+1, current_val-1].
            # We can iterate from prev_val+1 to current_val-1 and check validity.
            # The range is at most current_val - prev_val.
            
            found = False
            for new_val in range(min_possible, current_val):
                # Check if new_val is valid
                valid = True
                # Check diffs with left elements
                for i in range(pi):
                    d = new_val - old_row[i]
                    if d in forbidden:
                        valid = False
                        break
                if not valid:
                    continue
                # Check diffs with right elements
                for j in range(pi + 1, len(old_row)):
                    d = old_row[j] - new_val
                    if d in forbidden:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also need to check that the new diffs don't conflict with *each other*?
                # No, diffs with left are distinct if elements are distinct.
                # Diffs with right are distinct if elements are distinct.
                # Diffs with left vs diffs with right?
                # (new_val - a_i) vs (a_j - new_val).
                # If new_val - a_i == a_j - new_val, then 2*new_val = a_i + a_j.
                # This is a conflict within the row itself.
                # So we must check that no left-diff equals any right-diff.
                
                left_diffs = set()
                for i in range(pi):
                    left_diffs.add(new_val - old_row[i])
                
                for j in range(pi + 1, len(old_row)):
                    d = old_row[j] - new_val
                    if d in left_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If valid, accept this new_val (which is smaller than current_val)
                rows[ri][pi] = new_val
                found = True
                improved = True
                break
            
            if found:
                # We found a smaller value. 
                # We should probably restart the loop or at least update the indices list?
                # Since we are iterating in descending order of the *original* values,
                # and we just reduced a value, the order might be slightly off, but it's fine for a few passes.
                pass
    
    return rows

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    collision_count = sum(max(0, c - 1) for c in diff_counts.values())
    
    def get_cost(scope, collisions):
        # Use a softer penalty to allow trading scope for collisions
        return scope + 20 * collisions

    current_cost = get_cost(current_scope, collision_count)
    best_cost = current_cost
    
    if polish:
        T = 0.3
        cooling = 0.99995
        min_T = 0.005
    else:
        T = 15.0
        cooling = 0.9995
        min_T = 0.1
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.20:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            
            # Remove old diffs
            for d in diff_counts:
                pass # We need to update counts
            old_diffs = []
            for idx_a in range(len(old_row)):
                for idx_b in range(idx_a + 1, len(old_row)):
                    d = old_row[idx_b] - old_row[idx_a]
                    old_diffs.append(d)
            
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                if diff_counts[d] == 1:
                    collision_count -= 1 # was 2, now 1 -> collision -1? 
                    # Wait. collision = sum(max(0, c-1)).
                    # If c goes 2->1, max(0,1)->max(0,0) = 1->0. Delta -1.
                    # If c goes 1->0, max(0,0)->max(0,-1) = 0->0. Delta 0.
                    # If c goes 3->2, max(0,2)->max(0,1) = 2->1. Delta -1.
                    # So if old_count > 1 and new_count > 0, collision decreases by 1 if old_count >= 2.
                    # Let's just recompute collision count for the affected diffs? No, that's expensive.
                    # Let's track collision count carefully.
            
            # Actually, let's just recompute collision count for the few diffs that changed?
            # Or simpler: keep a list of all diffs? No.
            
            # Let's just use a helper to update collision count.
            # For a diff d, if count changes from c_old to c_new:
            # delta_coll = max(0, c_new - 1) - max(0, c_old - 1)
            
            def update_collision(d, delta):
                nonlocal collision_count
                c_old = diff_counts.get(d, 0)
                c_new = c_old + delta
                if c_new < 0:
                    c_new = 0 # Should not happen
                diff_counts[d] = c_new
                if c_new == 0:
                    if d in diff_counts:
                        del diff_counts[d]
                
                old_coll = max(0, c_old - 1)
                new_coll = max(0, c_new - 1)
                collision_count += (new_coll - old_coll)

            for d in old_diffs:
                update_collision(d, -1)
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 3)
            if new_row is None:
                for d in old_diffs:
                    update_collision(d, 1)
                current_rows[i] = old_row
                continue
            
            new_diffs = []
            for idx_a in range(len(new_row)):
                for idx_b in range(idx_a + 1, len(new_row)):
                    d = new_row[idx_b] - new_row[idx_a]
                    new_diffs.append(d)
            
            for d in new_diffs:
                update_collision(d, 1)
                
            current_rows[i] = new_row
            new_scope = _scope(current_rows)
            new_cost = get_cost(new_scope, collision_count)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                for d in new_diffs:
                    update_collision(d, -1)
                for d in old_diffs:
                    update_collision(d, 1)
                current_rows[i] = old_row
                current_scope = _scope(current_rows) # Should be same as before
                
        elif r < 0.40:
            # Scope compression: try to reduce the max element
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            
            if max_row_idx < 0:
                continue
                
            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]
            
            prev_val = current_rows[i][j-1]
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            if new_val <= prev_val or new_val >= old_val:
                continue
            
            # Remove old row diffs
            old_row = list(current_rows[i])
            old_diffs = []
            for idx_a in range(len(old_row)):
                for idx_b in range(idx_a + 1, len(old_row)):
                    d = old_row[idx_b] - old_row[idx_a]
                    old_diffs.append(d)
            for d in old_diffs:
                update_collision(d, -1)
                
            current_rows[i][j] = new_val
            new_row = current_rows[i]
            new_diffs = []
            for idx_a in range(len(new_row)):
                for idx_b in range(idx_a + 1, len(new_row)):
                    d = new_row[idx_b] - new_row[idx_a]
                    new_diffs.append(d)
            for d in new_diffs:
                update_collision(d, 1)
                
            new_scope = _scope(current_rows)
            new_cost = get_cost(new_scope, collision_count)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                for d in new_diffs:
                    update_collision(d, -1)
                for d in old_diffs:
                    update_collision(d, 1)
                current_rows[i][j] = old_val
                current_scope = _scope(current_rows)
                
        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            if old_val == current_scope:
                new_val = rng.randint(lo, min(hi, old_val - 1))
            else:
                if rng.random() < 0.7:
                    span = hi - lo
                    delta = int(rng.gauss(0, max(1, span * 0.2)))
                    new_val = old_val + delta
                else:
                    new_val = rng.randint(lo, hi)
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            old_row = list(current_rows[i])
            old_diffs = []
            for idx_a in range(len(old_row)):
                for idx_b in range(idx_a + 1, len(old_row)):
                    d = old_row[idx_b] - old_row[idx_a]
                    old_diffs.append(d)
            for d in old_diffs:
                update_collision(d, -1)
                
            current_rows[i][j] = new_val
            new_row = current_rows[i]
            new_diffs = []
            for idx_a in range(len(new_row)):
                for idx_b in range(idx_a + 1, len(new_row)):
                    d = new_row[idx_b] - new_row[idx_a]
                    new_diffs.append(d)
            for d in new_diffs:
                update_collision(d, 1)
                
            new_scope = _scope(current_rows)
            new_cost = get_cost(new_scope, collision_count)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                for d in new_diffs:
                    update_collision(d, -1)
                for d in old_diffs:
                    update_collision(d, 1)
                current_rows[i][j] = old_val
                current_scope = _scope(current_rows)
            
        T = max(T * cooling, min_T)
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    for _ in range(4):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 8)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    for idx, cand in enumerate(candidates[:2]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.4
        else:
            sa_deadline = _DEADLINE
        iters = int((sa_deadline - time.time()) * 8000)
        iters = max(iters, 2000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)
        
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        extra_iters = int(remaining_time * 10000)
        if extra_iters > 500:
            s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _DEADLINE, polish=True)
            consider(rows)
            
            # Add Polish pass
            if time.time() < _DEADLINE - 5:
                polished_rows = _polish(_BEST[1], k, rng, _DEADLINE - 5)
                consider(polished_rows)
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
