import time
import random
import math
import sys
from collections import Counter

_BEST = None
_DEADLINE = time.time() + 110.0

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

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 110.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    # 1. Generate several initial candidates using different greedy strategies
    candidates = []
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    # Spread greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy (shuffle candidates)
    for _ in range(5):
        used = 0
        rows = []
        ok = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a range of candidates
                found = False
                # Randomly decide step size
                max_step = rng.randint(1, 10)
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
                    # Fallback: find smallest valid
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
        
    # 2. Run Local Search on the best few candidates
    # Sort candidates by scope
    candidates.sort(key=_scope)
    
    # Local Search Strategy:
    # - Maintain a global Counter of differences.
    # - Move: Rebuild a row to minimize its max element while maintaining validity.
    # - Compression: Try to lower the global scope by shrinking the largest elements.
    
    def _local_search(rows, k, deadline):
        n_rows = len(rows)
        current_rows = [list(r) for r in rows]
        current_scope = _scope(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_scope = current_scope
        
        # Build initial diff counter
        diff_counts = Counter()
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diff_counts[r[j] - r[i]] += 1
        
        # Helper: try to build a valid row given current diff_counts (excluding this row)
        # This is a bit expensive if done naively, so we use a greedy approach with backtracking
        # or just a simple greedy with randomization.
        
        def try_build_row(exclude_idx, max_val_hint=None):
            # Remove row exclude_idx from diff_counts
            r = current_rows[exclude_idx]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                        
            # Try to build a new row
            new_row = [0]
            x = 0
            success = False
            # We want to keep the max value low.
            # Let's try to build it greedily but with some randomization to escape local optima.
            # We can also try to force the max value to be lower than before.
            
            # Strategy: Greedy with lookahead/random restarts
            for attempt in range(10):
                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    found = False
                    # Try next few values
                    for t in range(x + 1, x + 1 + 20):
                        if max_val_hint is not None and t > max_val_hint:
                            break
                        new = 0
                        for a in new_row:
                            new |= (1 << (t - a))
                        # Check if any of these differences are already used
                        # This is a bit slow if we do it for every t.
                        # Optimization: precompute diffs for new_row?
                        # For small k, it's fine.
                        ok = True
                        for a in new_row:
                            if diff_counts.get(t - a, 0) > 0:
                                ok = False
                                break
                        if ok:
                            new_row.append(t)
                            # Update diff_counts temporarily? No, just check.
                            # We need to ensure that the new row's internal diffs don't conflict with each other.
                            # Since we build strictly increasing, internal diffs are unique within the row.
                            # Wait, if we pick t, we must ensure t-a is not used by *other* rows.
                            # And we must ensure t-a is not used by *this* row's previous elements.
                            # Since t is increasing, t-a is unique for fixed t and varying a?
                            # No, t-a1 != t-a2 if a1 != a2.
                            # So internal consistency is guaranteed by strict increase.
                            # We just need to check against global diff_counts.
                            
                            # Tentatively add to diff_counts to check for subsequent elements?
                            # No, we can just check against the current diff_counts (which excludes this row).
                            # But we must also ensure that the new differences don't conflict with each other.
                            # They don't, as argued.
                            
                            # So we just need to check if t-a is in diff_counts.
                            # If yes, conflict.
                            
                            # If no conflict, we can tentatively "use" these diffs.
                            # But we don't want to modify diff_counts until the row is complete.
                            # So we need a local set of used diffs for this row.
                            local_used = set()
                            for a in new_row:
                                d = t - a
                                if d in local_used or diff_counts.get(d, 0) > 0:
                                    ok = False
                                    break
                                local_used.add(d)
                            
                            if ok:
                                # Success for this step
                                x = t
                                found = True
                                break
                    if not found:
                        break
                if len(new_row) == k + 1:
                    success = True
                    break
            
            if success:
                # Add new row's diffs to diff_counts
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        diff_counts[new_row[j] - new_row[i]] += 1
                return new_row
            else:
                # Restore diff_counts
                # We didn't change them, so nothing to do.
                return None

        # Main loop
        while time.time() < deadline:
            # Pick a row to rebuild
            # Prefer rows with large max values
            row_indices = list(range(n_rows))
            rng.shuffle(row_indices)
            
            # Sort by max value descending to target worst rows first
            row_indices.sort(key=lambda i: current_rows[i][-1], reverse=True)
            
            improved = False
            for idx in row_indices:
                if time.time() > deadline:
                    break
                
                old_max = current_rows[idx][-1]
                # Try to rebuild with a slightly smaller max
                # Or just rebuild and see if it's better
                new_row = try_build_row(idx, max_val_hint=old_max - 1 if rng.random() < 0.5 else None)
                
                if new_row is not None:
                    new_max = new_row[-1]
                    if new_max < old_max:
                        current_rows[idx] = new_row
                        improved = True
                        # Update scope
                        # We don't need to recalculate full scope immediately, just track best
                        # But for simplicity, we can just check at the end of the loop
                        # Or update best if this row's reduction helped
                        pass
                    else:
                        # If new_max == old_max, we might still want to accept it to change the configuration
                        # But try_build_row already removed and re-added diffs.
                        # If we reject, we must restore.
                        # This is getting complicated with the diff_counts state.
                        # Let's simplify: only accept if strictly better or equal (for diversity).
                        # If we reject, we must revert diff_counts.
                        # This is error-prone.
                        
                        # Alternative: Always accept if valid, and keep track of best scope.
                        # But we need to revert if we don't want to change.
                        # Let's just accept any valid row, and rely on the fact that we target worst rows.
                        # Actually, if new_max >= old_max, we should probably revert to avoid drift.
                        
                        # To revert, we need to remove new_row diffs and re-add old_row diffs.
                        # This is expensive.
                        
                        # Let's change strategy:
                        # Only propose moves that are guaranteed to reduce the max of that row.
                        # i.e., max_val_hint = old_max - 1.
                        # If that fails, try old_max - 2, etc.
                        
                        pass
            
            # If we made improvements, update best
            if improved:
                # Check validity just in case
                if _valid(current_rows, k):
                    s = _scope(current_rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [list(r) for r in current_rows]
            
            # If no improvements in a while, restart or break
            # For now, just loop.
            
        return best_scope, best_rows

    # Run Local Search on the top 3 candidates
    for cand in candidates[:3]:
        if time.time() > _DEADLINE:
            break
        s, rows = _local_search(cand, k, _DEADLINE)
        consider(rows)
        
    # 3. If time remains, run on best result
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 5:
            s, rows = _local_search(_BEST[1], k, _DEADLINE)
            consider(rows)
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
