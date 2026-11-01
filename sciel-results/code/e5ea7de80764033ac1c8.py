"""
Difference triangle set (7,5) - find 7 rows of 6 marks minimizing max entry.

Strategy: Combine constraint-based backtracking with forward checking (from inspiration)
with a smarter candidate ordering and iterative deepening on the max-entry bound.
The key structural change is modeling the difference set as a shared resource pool
and using forward checking to prune infeasible branches early.

We use:
1. Iterative deepening on max_entry starting from the theoretical lower bound
2. Backtracking with forward checking (verify remaining positions can be filled)
3. Candidate ordering that prefers values preserving future options
4. Multiple random restarts with different random seeds for diversity
5. Deadline guard to ensure we return the best result before timeout
"""
import time
import random

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_row(used, k, max_x=None):
    """Build one row greedily: start at 0, pick smallest next mark with no diff conflicts."""
    row = [0]
    while len(row) < k + 1:
        base = row[-1]
        found = False
        upper = (max_x if max_x is not None else 500) + 1
        for x in range(base + 1, upper):
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
    return rows

def _backtrack_solve(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking with forward checking.
    Treat the set of used differences as a global resource pool.
    At each step, forward-check whether the remaining positions can still be filled.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = []
    node_count = [0]
    
    def solve(row_idx, pos, current_row):
        """Recursively build row row_idx, position pos."""
        if time.time() > deadline:
            return False
        
        node_count[0] += 1
        if node_count[0] % 10000 == 0 and time.time() > deadline:
            return False
        
        if pos == k + 1:
            # Row complete
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 1, [0])
        
        if pos == 0:
            return solve(row_idx, 1, [0])
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        # Forward check: can we still fill remaining positions?
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return False
        
        # Generate valid candidates
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            candidates.append(val)
        
        if not candidates:
            return False
        
        # Forward check: for each candidate, verify at least one valid next value exists
        if pos + 1 < k + 1:
            filtered = []
            for val in candidates:
                next_lo = val + 1
                next_hi = max_entry
                next_remaining = k + 1 - (pos + 1)
                if next_hi - next_lo + 1 < next_remaining:
                    continue
                has_next = False
                for next_val in range(next_lo, next_hi + 1):
                    next_new_diffs = {next_val - a for a in current_row + [val]}
                    if len(next_new_diffs) == len(current_row) + 1:
                        if not (next_new_diffs & (used_diffs | {val - a for a in current_row})):
                            has_next = True
                            break
                if has_next:
                    filtered.append(val)
            candidates = filtered
        
        if not candidates:
            return False
        
        # Order candidates: prefer smaller values to keep max low,
        # and prefer values whose differences are less "crowded"
        def score(val):
            new_diffs = {val - a for a in current_row}
            # Primary: smaller value is better
            s1 = val * 10
            # Secondary: prefer differences that are less crowded
            s2 = 0
            for d in new_diffs:
                # Count how many used diffs are nearby
                s2 += sum(1 for ud in used_diffs if abs(ud - d) <= 5)
            return s1 + s2
        
        candidates.sort(key=score)
        
        # Add some randomness to avoid getting stuck in local patterns
        if rng.random() < 0.25 and len(candidates) > 1:
            swap_idx = rng.randrange(len(candidates))
            candidates[0], candidates[swap_idx] = candidates[swap_idx], candidates[0]
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            new_diffs = {val - a for a in current_row}
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if solve(row_idx, pos + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    if solve(0, 0, []):
        return rows
    return None

def _iterative_deepening_search(n_rows, k, deadline):
    """
    Iterative deepening on max_entry with multiple random restarts.
    Start from the theoretical lower bound and increase until a solution is found.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    # Theoretical lower bound: need at least total_diffs_needed distinct differences
    # in [1, max_entry], so max_entry >= total_diffs_needed
    lower_bound = total_diffs_needed
    
    # Also, each row has k+1 entries starting at 0, so the max entry must be
    # at least k (for the last position if all gaps are 1)
    lower_bound = max(lower_bound, k)
    
    best_rows = None
    
    # Try increasing bounds
    for max_entry in range(lower_bound, lower_bound + 30):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for each bound
        for restart in range(3):
            if time.time() > deadline:
                break
            
            rng = random.Random(42 + restart * 1000 + max_entry)
            rows = _backtrack_solve(n_rows, k, max_entry, rng, deadline)
            
            if rows is not None:
                best_rows = rows
                return rows
    
    return best_rows

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences are globally distinct, minimizing the largest entry.
    
    Returns a list of n_rows rows.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin before 150s kill
    
    # First try greedy to get an initial solution
    greedy_result = _greedy_all(n_rows, k)
    
    # Try iterative deepening to find a better solution
    best_rows = None
    best_score = float('inf')
    
    if greedy_result is not None:
        best_rows = greedy_result
        best_score = _max_entry(greedy_result)
    
    # Try to improve with backtracking + iterative deepening
    improved = _iterative_deepening_search(n_rows, k, deadline)
    if improved is not None:
        improved_score = _max_entry(improved)
        if improved_score < best_score:
            best_rows = improved
            best_score = improved_score
    
    if best_rows is None:
        # Fallback: try a simple sequential greedy with larger bound
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
        best_rows = rows
    
    return best_rows
