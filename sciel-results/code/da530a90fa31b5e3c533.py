"""Difference triangle set (7,5): minimize the largest entry.
Structural approach: CSP over the difference resource with degree-minimizing
variable ordering and randomized restarts."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _check_valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _greedy_rows(n_rows, k, used=None):
    """Greedy construction of n_rows rows (smallest-first)."""
    if used is None:
        used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows, used


def _degree_score(x, row, used):
    """Score a candidate x: count how many future difference slots it blocks.
    Lower score = better (leaves more room)."""
    new_diffs = {x - a for a in row}
    # Count how many of these diffs are 'small' (likely needed by future rows)
    # and how many total new diffs
    n_new = len(new_diffs)
    # Penalty for using small differences (they're more in demand)
    small_penalty = sum(1 for d in new_diffs if d <= 20)
    return n_new + small_penalty


def _backtrack_row(row, k, used, max_allowed, deadline, degree_mode=True):
    """Backtrack to find a valid row extension.
    row starts with [0]. Returns list or None."""
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        
        last = row[-1]
        candidates = []
        
        if degree_mode:
            # Collect candidates and score them
            for x in range(last + 1, min(last + 40, max_allowed + 1)):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                score = _degree_score(x, row, used)
                candidates.append((score, x))
            candidates.sort()
            # Try best candidates first, but with some randomness
            if random.random() < 0.3 and len(candidates) > 1:
                # Pick a random one from top 3
                idx = random.randint(0, min(2, len(candidates) - 1))
                candidates = [candidates[idx]] + [c for i, c in enumerate(candidates) if i != idx]
            for _, x in candidates:
                new_diffs = {x - a for a in row}
                row.append(x)
                used |= new_diffs
                if backtrack(pos + 1):
                    return True
                used -= new_diffs
                row.pop()
        else:
            # Simple smallest-first
            for x in range(last + 1, max_allowed + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
                if backtrack(pos + 1):
                    return True
                used -= new_diffs
                row.pop()
        
        return False
    
    if backtrack(1):
        return list(row)
    return None


def _build_rows_csp(n_rows, k, max_allowed, deadline, seed=None):
    """Build n_rows rows using CSP-style backtracking with degree ordering.
    Returns (rows, scope) or (None, None) if failed."""
    if seed is not None:
        random.seed(seed)
    
    used = set()
    rows = []
    
    for r in range(n_rows):
        row = [0]
        result = _backtrack_row(row, k, used, max_allowed, deadline, degree_mode=True)
        if result is None:
            return None, None
        rows.append(result)
        used |= _row_diffs(result)
    
    scope = max(max(r) for r in rows)
    return rows, scope


def _local_search(rows, k, deadline):
    """Iteratively try to replace each row with one that has a smaller max entry."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)
    no_improve_count = 0
    
    while time.time() < deadline - 3.0:
        improved = False
        order = sorted(range(len(best)), key=lambda i: -max(best[i]))
        
        for row_idx in order:
            if time.time() > deadline - 3.0:
                break
            current_max = max(best[row_idx])
            
            for target_max in range(current_max - 1, max(current_max - 15, 0), -1):
                if time.time() > deadline - 3.0:
                    break
                # Build the used set from other rows
                other_used = set()
                for i, r in enumerate(best):
                    if i != row_idx:
                        other_used |= _row_diffs(r)
                
                row = [0]
                result = _backtrack_row(row, k, other_used, target_max, deadline, degree_mode=True)
                if result is not None:
                    best[row_idx] = result
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        # Revert
                        row2 = [0]
                        result2 = _backtrack_row(row2, k, other_used, current_max, deadline, degree_mode=False)
                        if result2 is not None:
                            best[row_idx] = result2
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 4:
                # Random perturbation
                ridx = random.randint(0, len(best) - 1)
                other_used = set()
                for i, r in enumerate(best):
                    if i != ridx:
                        other_used |= _row_diffs(r)
                row = [0]
                result = _backtrack_row(row, k, other_used, max(best[ridx]), deadline, degree_mode=True)
                if result is not None and result != best[ridx]:
                    best[ridx] = result
                    no_improve_count = 0
                else:
                    break
        else:
            no_improve_count = 0
    
    return best, best_scope


def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy baseline
    greedy_rows, greedy_used = _greedy_rows(n_rows, k)
    best_rows = greedy_rows
    best_scope = max(max(r) for r in best_rows)
    
    # Phase 2: CSP-style construction with degree ordering
    # Try to build from scratch with decreasing max_allowed
    for target in range(best_scope - 1, max(best_scope - 25, 0), -1):
        if time.time() > deadline - 10.0:
            break
        rows, scope = _build_rows_csp(n_rows, k, target, deadline)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Phase 3: Randomized restarts with CSP
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 30:
        attempts += 1
        # Randomize the max_allowed slightly around current best
        target = best_scope - random.randint(0, 5)
        if target < 0:
            target = 0
        rows, scope = _build_rows_csp(n_rows, k, target, deadline, seed=random.randint(0, 10**9))
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Phase 4: Local search polish
    if time.time() < deadline - 5.0:
        local_rows, local_scope = _local_search(best_rows, k, deadline)
        if local_scope < best_scope:
            best_rows = local_rows
            best_scope = local_scope
    
    # Phase 5: Final targeted improvement - try to reduce scope by 1
    if time.time() < deadline - 5.0:
        for target in range(best_scope - 1, max(best_scope - 3, 0), -1):
            if time.time() > deadline - 5.0:
                break
            # Try to rebuild with this target
            for attempt in range(5):
                if time.time() > deadline - 5.0:
                    break
                rows, scope = _build_rows_csp(n_rows, k, target, deadline, seed=random.randint(0, 10**9))
                if rows is not None and scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                    break
    
    return best_rows
# EVOLVE-BLOCK-END
