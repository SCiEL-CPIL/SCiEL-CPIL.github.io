"""Combined program: bitset CSP + randomized backtracking + greedy fallback."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _greedy_set(n_rows, k, seed=0):
    """Greedy construction: each row takes smallest next mark with unused diffs."""
    rng = random.Random(seed)
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

def _score(rows):
    return max(r[-1] for r in rows)

def _build_valid_rows(k, M):
    """
    For bound M, enumerate all valid rows: subsets of {0..M} of size k+1
    containing 0, strictly increasing, with all C(k+1,2) pairwise differences
    distinct within the row (always true for strictly increasing integers? No -
    differences can repeat within a row, e.g. {0,1,2} has diffs 1,2,1 -> repeat).
    So we filter: all pairwise differences must be distinct.
    Returns list of (row_tuple, diff_mask_int).
    """
    total_diffs_per_row = (k + 1) * k // 2
    valid = []
    # We need C(M, k) subsets (choosing k elements from 1..M, plus 0)
    # For small M this is tractable; for M ~ 112 and k=5, C(112,5) ~ 6e7 - too many.
    # So we only do this for small M, or use a smarter generation.
    # Alternative: generate rows incrementally.
    # For now, try all combinations if M is small enough.
    if M <= 60:
        for combo in combinations(range(1, M + 1), k):
            row = (0,) + combo
            diffs = set()
            ok = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            if ok:
                mask = 0
                for d in diffs:
                    mask |= (1 << d)
                valid.append((row, mask))
    else:
        # For larger M, use a randomized sampling approach
        # Generate random valid rows
        seen_masks = set()
        attempts = 0
        max_attempts = 50000
        while len(valid) < 5000 and attempts < max_attempts:
            attempts += 1
            # Random subset
            subset = sorted(random.sample(range(1, M + 1), k))
            row = (0,) + tuple(subset)
            diffs = set()
            ok = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
            if ok:
                mask = 0
                for d in diffs:
                    mask |= (1 << d)
                if mask not in seen_masks:
                    seen_masks.add(mask)
                    valid.append((row, mask))
    return valid

def _set_packing_backtrack(k, n_rows, M, deadline, seed):
    """
    Find n_rows valid rows with disjoint difference bitmasks, all within [0, M].
    Uses bitset representation for fast intersection checks.
    """
    start = time.time()
    valid_rows = _build_valid_rows(k, M)
    if len(valid_rows) < n_rows:
        return None
    
    # Sort valid rows by "difficulty": fewer compatible partners first (approximate)
    # For now, just use all
    n_valid = len(valid_rows)
    if n_valid < n_rows:
        return None
    
    best = None
    rng = random.Random(seed)
    
    # Shuffle for diversity
    indices = list(range(n_valid))
    rng.shuffle(indices)
    
    # Backtracking: select n_rows rows with disjoint masks
    selected = []
    used_mask = 0
    
    def backtrack(idx, depth):
        nonlocal used_mask, best
        if time.time() - start > deadline:
            return False
        if depth == n_rows:
            best = selected[:]
            return True
        # Try from idx to end
        for i in range(idx, n_valid):
            row, mask = valid_rows[indices[i]]
            if mask & used_mask:
                continue
            # Pruning: check if enough remaining rows can still be selected
            # (simple: just continue)
            selected.append(row)
            used_mask |= mask
            if backtrack(i + 1, depth + 1):
                return True
            selected.pop()
            used_mask &= ~mask
        return False
    
    # Try multiple shuffles
    for attempt in range(20):
        if time.time() - start > deadline:
            break
        if best is not None:
            break
        indices = list(range(n_valid))
        rng.shuffle(indices)
        selected.clear()
        used_mask = 0
        if backtrack(0, 0):
            break
    
    if best is not None:
        return [list(r) for r in best]
    return None

def _element_backtrack(n_rows, k, limit, deadline, seed):
    """
    Element-by-element backtracking with bitset for used diffs.
    limit is the max allowed value (M-1 for improvement).
    """
    start = time.time()
    rng = random.Random(seed)
    used_mask = 0
    rows = []
    
    def dfs(row_idx, marks, row_mask):
        nonlocal used_mask
        if time.time() - start > deadline:
            return None
        
        if len(marks) == k + 1:
            if row_idx == n_rows - 1:
                return rows + [marks[:]]
            else:
                # Check row internal consistency is guaranteed by construction
                # Update global used
                used_mask |= row_mask
                res = dfs(row_idx + 1, [0], 0)
                if res is not None:
                    return res
                used_mask &= ~row_mask
                return None
        else:
            last = marks[-1]
            # Try candidates from last+1 to limit
            # Use a heuristic: try larger gaps first for diversity
            candidates = list(range(last + 1, limit + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                # Compute new diffs
                new_diffs = [x - a for a in marks]
                new_mask = 0
                conflict = False
                for d in new_diffs:
                    bit = 1 << d
                    if new_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if conflict:
                    continue
                if new_mask & (used_mask | row_mask):
                    continue
                
                row_mask |= new_mask
                marks.append(x)
                res = dfs(row_idx, marks, row_mask)
                if res is not None:
                    return res
                marks.pop()
                row_mask &= ~new_mask
            return None
    
    res = dfs(0, [0], 0)
    return res

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    best_rows = None
    best_scope = float('inf')
    
    # 1. Get initial upper bound via Greedy (multiple seeds)
    for seed in range(30):
        if time.time() > deadline:
            break
        rows = _greedy_set(n_rows, k, seed)
        scope = _score(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    # 2. Try to improve using element-by-element backtracking with bitset
    #    Try bounds from best_scope-1 downward
    for target_M in range(best_scope - 1, max(best_scope - 10, 0), -1):
        if time.time() > deadline:
            break
        found = False
        for seed in range(50):
            if time.time() > deadline:
                break
            rows = _element_backtrack(n_rows, k, target_M, deadline - time.time(), seed)
            if rows is not None:
                scope = _score(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                    found = True
                    break
        if found:
            # Try to go even lower
            continue
    
    # 3. Try set-packing approach for smaller M
    #    This is useful when M is small enough to enumerate valid rows
    if best_scope <= 60:
        for target_M in range(max(best_scope - 5, 1), 0, -1):
            if time.time() > deadline:
                break
            rows = _set_packing_backtrack(k, n_rows, target_M, deadline - time.time(), 0)
            if rows is not None:
                scope = _score(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        best_rows = _greedy_set(n_rows, k, 0)
    
    return best_rows

# EVOLVE-BLOCK-END
