"""Improved difference triangle set construction via randomized backtracking with deadline guard."""
import time
import random
import sys

sys.setrecursionlimit(10000)


def _is_valid_solution(rows, k):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _backtrack_row(used, k, deadline, time_budget_frac=0.85):
    """
    Build one row of k+1 increasing integers starting at 0,
    choosing elements to minimize future constraint violations.
    Returns the row or None if deadline exceeded.
    """
    start_time = time.time()
    row = [0]
    
    def _extend(row, used, depth):
        if time.time() > deadline:
            return None
        if len(row) == k + 1:
            return list(row)
        
        last = row[-1]
        # Candidate next values: try a range of candidates
        # Use a heuristic: prefer smaller gaps that don't conflict
        best_candidate = None
        best_score = -1
        
        # Try candidates from last+1 up to a reasonable bound
        # The bound depends on how much room we have
        max_candidate = last + 50  # adaptive
        
        for x in range(last + 1, max_candidate + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            
            # Heuristic score: prefer candidates that leave more "room"
            # Score = -max(new_diffs) + number of unused small diffs
            # We want small max diff (compact) but also want to not block future
            score = -max(new_diffs)
            
            # Bonus for using small differences (they're harder to place later)
            small_count = sum(1 for d in new_diffs if d <= 15)
            score += small_count * 0.5
            
            if score > best_score:
                best_score = score
                best_candidate = x
        
        if best_candidate is None:
            # No candidate found in range, expand range
            for x in range(last + max_candidate + 1, last + max_candidate + 100):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                score = -max(new_diffs)
                if score > best_score:
                    best_score = score
                    best_candidate = x
        
        if best_candidate is None:
            return None
        
        row.append(best_candidate)
        used.update({best_candidate - a for a in row[:-1]})
        
        result = _extend(row, used, depth + 1)
        if result is not None:
            return result
        
        # Backtrack: undo
        row.pop()
        for a in row:
            used.discard(best_candidate - a)
        
        # Try next candidate
        # This is simplified; a full backtracking would try all candidates
        return None
    
    result = _extend(row, used, 0)
    return result


def _greedy_construct(n_rows, k):
    """
    Greedy construction with randomized restarts and biased candidate selection.
    Each row starts at 0 and picks the smallest next mark whose differences
    to earlier marks are all unused.
    """
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


def _randomized_search(n_rows, k, deadline, rng):
    """
    Randomized search with backtracking.
    Tries multiple random strategies to find a compact packing.
    """
    best_rows = None
    best_max = float('inf')
    
    # Strategy 1: Pure greedy
    rows = _greedy_construct(n_rows, k)
    if rows and _is_valid_solution(rows, k):
        m = max(max(r) for r in rows)
        if m < best_max:
            best_max = m
            best_rows = [list(r) for r in rows]
    
    # Strategy 2: Randomized greedy with bias
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        used = set()
        rows = []
        valid = True
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                # Try candidates with some randomness
                found = False
                # First try small gaps
                for trial in range(200):
                    if trial < 100:
                        # Bias toward small gaps
                        x = last + 1 + rng.randint(0, 15)
                    else:
                        # Larger gaps
                        x = last + 1 + rng.randint(1, 50)
                    
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        found = True
                        break
                
                if not found:
                    # Fall back to linear search
                    x = last
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
            
            rows.append(row)
        
        if valid and _is_valid_solution(rows, k):
            m = max(max(r) for r in rows)
            if m < best_max:
                best_max = m
                best_rows = [list(r) for r in rows]
    
    return best_rows if best_rows else _greedy_construct(n_rows, k)


def _backtracking_search(n_rows, k, deadline, rng):
    """
    Full backtracking search with constraint propagation.
    Selects next row element by minimizing future constraint violations.
    """
    best_rows = None
    best_max = float('inf')
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            success = _build_row_backtrack(row, used, k, deadline, rng)
            if not success:
                # Fall back to greedy for this row
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
            
            rows.append(row)
        
        if _is_valid_solution(rows, k):
            m = max(max(r) for r in rows)
            if m < best_max:
                best_max = m
                best_rows = [list(r) for r in rows]
    
    return best_rows if best_rows else _greedy_construct(n_rows, k)


def _build_row_backtrack(row, used, k, deadline, rng):
    """
    Build a single row using backtracking with heuristic ordering.
    Returns True if successful.
    """
    if time.time() > deadline:
        return False
    
    if len(row) == k + 1:
        return True
    
    last = row[-1]
    
    # Generate candidate values in heuristic order
    candidates = []
    # Small gaps first (more constrained)
    for gap in range(1, 30):
        x = last + gap
        new = {x - a for a in row}
        if not new & used:
            # Score: prefer smaller max diff
            candidates.append((max(new), x))
    
    # Sort by max diff (prefer compact)
    candidates.sort()
    
    # Add some randomization
    if candidates and rng.random() < 0.3:
        # Pick a random candidate from top half
        idx = rng.randint(len(candidates) // 2, len(candidates) - 1)
        _, x = candidates[idx]
        new = {x - a for a in row}
        row.append(x)
        used.update(new)
        
        if _build_row_backtrack(row, used, k, deadline, rng):
            return True
        
        row.pop()
        used.difference_update(new)
    
    # Try candidates in order
    for _, x in candidates:
        if time.time() > deadline:
            return False
        
        new = {x - a for a in row}
        row.append(x)
        used.update(new)
        
        if _build_row_backtrack(row, used, k, deadline, rng):
            return True
        
        row.pop()
        used.difference_update(new)
    
    # If no small gap worked, try larger gaps
    x = last
    while x < last + 200:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used.update(new)
            
            if _build_row_backtrack(row, used, k, deadline, rng):
                return True
            
            row.pop()
            used.difference_update(new)
    
    return False


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 120 * 0.85  # 85% of 120s to be safe
    
    rng = random.Random(42)
    
    # Start with greedy
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    # Try randomized search
    if time.time() < deadline:
        rows = _randomized_search(n_rows, k, deadline, rng)
        if rows:
            m = max(max(r) for r in rows)
            if m < best_max:
                best_max = m
                best_rows = [list(r) for r in rows]
    
    # Try backtracking search
    if time.time() < deadline:
        rows = _backtracking_search(n_rows, k, deadline, rng)
        if rows:
            m = max(max(r) for r in rows)
            if m < best_max:
                best_max = m
                best_rows = [list(r) for r in rows]
    
    return best_rows
