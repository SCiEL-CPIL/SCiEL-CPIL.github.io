"""Difference triangle set (7,5) -- backtracking exact solver strategy."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 140.0
    # Small cases: return immediately
    if n_rows == 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # For very small problems, a quick greedy gives a valid baseline
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_score = _max_entry(baseline)
    
    # If trivial, return baseline
    if n_rows <= 1:
        return baseline
    
    # Try backtracking exact solver with increasing max_entry bounds
    # The counting bound: n_rows * k*(k+1)/2 differences must fit in [1, M]
    # So M >= n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs  # minimum possible max entry
    
    # Try bounds from lower_bound upward
    # For (7,5): total_diffs = 7*15 = 105
    # Known best: 112
    # We'll try a few bounds and use randomized backtracking for each
    
    # First, try a few deterministic-ish backtracking runs with the lower bound
    # and slightly above
    max_attempts = 20
    attempt = 0
    while time.time() < deadline - 10 and attempt < max_attempts:
        attempt += 1
        # Start from lower_bound, increase by 1 each attempt if not found
        bound = lower_bound + (attempt - 1)
        if bound > best_score - 1:
            break  # no point if we can't beat current best
        
        rows = _backtrack_solve(n_rows, k, bound, deadline, attempt)
        if rows is not None:
            score = _max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = rows
                # If we hit the lower bound, that's optimal
                if score <= lower_bound:
                    break
    
    # If backtracking didn't find anything better, try a local search polish
    if time.time() < deadline - 15:
        polished = _local_search_polish(best_rows, n_rows, k, deadline)
        if polished is not None:
            p_score = _max_entry(polished)
            if p_score < best_score:
                best_score = p_score
                best_rows = polished
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Greedy: pick each row to use the smallest available differences."""
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
    return rows


def _backtrack_solve(n_rows, k, max_bound, deadline, seed_offset):
    """
    Backtracking: build rows one by one. For each row, find a Sidon set
    (k+1 increasing ints starting at 0, max <= max_bound) whose differences
    don't conflict with already-used differences.
    Use randomized DFS with a time limit.
    """
    rng = random.Random(seed_offset * 7919 + 42)
    
    # Pre-generate candidate rows: all valid Sidon sets of size k+1 with max <= max_bound
    # This could be large, so we generate on the fly
    
    used_diffs = set()
    rows = []
    
    def dfs(row_idx):
        if time.time() > deadline - 10:
            return False
        if row_idx == n_rows:
            return True
        
        # Generate candidate rows for this position
        candidates = _generate_candidate_rows(k, max_bound, used_diffs, rng, limit=50)
        
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        for cand in candidates:
            cand_diffs = _get_diffs(cand)
            # Verify no conflict (should be guaranteed by generation, but double-check)
            if cand_diffs & used_diffs:
                continue
            # Try this row
            rows.append(cand)
            old_used = used_diffs.copy()
            used_diffs.update(cand_diffs)
            
            if dfs(row_idx + 1):
                return True
            
            # Backtrack
            rows.pop()
            used_diffs.clear()
            used_diffs.update(old_used)
        
        return False
    
    success = dfs(0)
    if success:
        return rows[:]
    return None


def _generate_candidate_rows(k, max_bound, used_diffs, rng, limit=50):
    """
    Generate up to 'limit' valid Sidon rows of length k+1, starting at 0,
    max <= max_bound, with no differences in used_diffs.
    """
    candidates = []
    # Try to build rows by choosing gaps
    # A row is [0, a1, a2, ..., ak] with 0 < a1 < a2 < ... < ak <= max_bound
    # All differences must be distinct and not in used_diffs
    
    # Strategy: random backtracking to build a row
    attempts = 0
    max_attempts = 200
    while len(candidates) < limit and attempts < max_attempts:
        attempts += 1
        row = [0]
        # Choose remaining k elements
        if _build_row_backtrack(row, k, max_bound, used_diffs, rng, depth=1):
            # Verify it's a valid Sidon set
            if _is_valid_sidon(row):
                # Check no conflicts with used_diffs
                diffs = _get_diffs(row)
                if not (diffs & used_diffs):
                    candidates.append(row[:])
    
    return candidates


def _build_row_backtrack(row, k, max_bound, used_diffs, rng, depth):
    """Recursively build a row by choosing the next element."""
    if len(row) == k + 1:
        return True
    
    last = row[-1]
    # Next element must be > last and <= max_bound
    # Also, the new differences must not conflict with existing row diffs or used_diffs
    
    # Get current diffs in the row
    cur_diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            cur_diffs.add(row[j] - row[i])
    
    # Candidate next values: last+1 to max_bound
    # To speed up, limit the range
    lo = last + 1
    hi = min(max_bound, last + 30)  # keep gaps reasonable
    
    candidates = list(range(lo, hi + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        # New differences introduced
        new_diffs = {x - a for a in row}
        # Check no internal conflict
        if new_diffs & cur_diffs:
            continue
        # Check no conflict with global used_diffs
        if new_diffs & used_diffs:
            continue
        # Also check x - x = 0 is not an issue (x > a for all a in row since x > last >= all a)
        
        row.append(x)
        if _build_row_backtrack(row, k, max_bound, used_diffs, rng, depth + 1):
            return True
        row.pop()
    
    return False


def _is_valid_sidon(row):
    """Check if row is a valid Sidon set (distinct pairwise differences)."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True


def _get_diffs(row):
    """Get the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _local_search_polish(rows, n_rows, k, deadline):
    """
    Local search to try to reduce the max entry further.
    Try to decrease individual elements while maintaining validity.
    """
    current_rows = [row[:] for row in rows]
    current_score = _max_entry(current_rows)
    row_diffs = [_get_diffs(row) for row in current_rows]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try to reduce each element in each row
        for row_idx in range(n_rows):
            for pos_idx in range(k, 0, -1):  # from last to first (excluding 0)
                old_val = current_rows[row_idx][pos_idx]
                # Try to reduce
                for new_val in range(old_val - 1, 0, -1):
                    if new_val <= current_rows[row_idx][pos_idx - 1]:
                        continue
                    if pos_idx < k and new_val >= current_rows[row_idx][pos_idx + 1]:
                        continue
                    
                    test_row = current_rows[row_idx][:]
                    test_row[pos_idx] = new_val
                    if not _is_valid_sidon(test_row):
                        continue
                    
                    new_diffs = _get_diffs(test_row)
                    conflict = False
                    for other_idx in range(n_rows):
                        if other_idx == row_idx:
                            continue
                        if new_diffs & row_diffs[other_idx]:
                            conflict = True
                            break
                    
                    if not conflict:
                        current_rows[row_idx][pos_idx] = new_val
                        row_diffs[row_idx] = new_diffs
                        new_score = _max_entry(current_rows)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        break  # only one reduction per position per pass
    
    return current_rows


def _max_entry(rows):
    if not rows:
        return 0
    return max(max(row) for row in rows)

# EVOLVE-BLOCK-END
