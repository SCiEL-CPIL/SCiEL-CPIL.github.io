"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry (scope)."""
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    total_diffs = n_rows * k * (k + 1) // 2  # total distinct differences needed
    # Counting bound: need at least total_diffs distinct positive differences
    min_scope_lower = total_diffs  # scope >= total_diffs since diffs are 1..scope
    
    best_result = None
    best_scope = float('inf')
    
    # Try scopes from lower bound upward
    # Best known for (7,5) is 112, counting bound is 105
    # Try a range of scopes
    
    def solve_with_scope(scope):
        """Try to find a valid set with all entries <= scope."""
        nonlocal best_result, best_scope, deadline
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, start_mark):
            nonlocal used_diffs, rows, deadline
            
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                # Check all rows are valid
                return list(rows)
            
            # Symmetry breaking: first row must start with 0,1,... 
            # Actually, let's use a smarter approach
            # For the first row, we can fix it to a "canonical" form
            # to reduce search space
            
            if row_idx == 0:
                # Try different first rows
                # A good first row: 0, 1, 3, 7, 12, 18 (triangular-ish)
                # Or use greedy to get a good starting row
                candidate_rows = _generate_candidate_rows(scope, k, used_diffs)
                for cand in candidate_rows:
                    if time.time() > deadline:
                        return None
                    new_used = set()
                    valid = True
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d in used_diffs or d in new_used:
                                valid = False
                                break
                            new_used.add(d)
                        if not valid:
                            break
                    if valid:
                        rows.append(cand)
                        old_used = used_diffs
                        used_diffs = used_diffs | new_used
                        result = backtrack(row_idx + 1, [], 0)
                        if result is not None:
                            return result
                        used_diffs = old_used
                        rows.pop()
                return None
            else:
                # Build current row incrementally
                if len(current_row) == k + 1:
                    # Row complete, check validity
                    new_used = set()
                    for i in range(len(current_row)):
                        for j in range(i + 1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs or d in new_used:
                                return None
                            new_used.add(d)
                    rows.append(current_row)
                    old_used = used_diffs
                    used_diffs = used_diffs | new_used
                    result = backtrack(row_idx + 1, [], 0)
                    if result is not None:
                        return result
                    used_diffs = old_used
                    rows.pop()
                    return None
                
                # Try next mark
                if not current_row:
                    current_row = [0]
                    return backtrack(row_idx, current_row, 1)
                
                last = current_row[-1]
                for x in range(last + 1, scope + 1):
                    if time.time() > deadline:
                        return None
                    
                    # Check if adding x creates any conflict
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_diffs:
                        continue
                    if len(new_diffs) != len(current_row):
                        # Duplicate difference within this row
                        continue
                    
                    # Forward checking: for remaining positions, can we still fit?
                    # Simple check: we need (k+1 - len(current_row)) more marks
                    # Each new mark adds (len(current_row)+1) new differences
                    # Total remaining differences needed for this row:
                    remaining_marks = k + 1 - len(current_row)
                    # This is a rough check
                    
                    result = backtrack(row_idx, current_row + [x], x + 1)
                    if result is not None:
                        return result
                
                return None
        
        return backtrack(0, [], 0)
    
    def _generate_candidate_rows(scope, k, used_diffs):
        """Generate candidate first rows, ordered by likelihood of leading to solution."""
        candidates = []
        
        # Greedy row
        row = [0]
        x = 0
        local_used = set()
        while len(row) < k + 1:
            x += 1
            if x > scope:
                break
            new = {x - a for a in row}
            if not new & local_used and not new & used_diffs:
                row.append(x)
                local_used |= new
        if len(row) == k + 1:
            candidates.append(row)
        
        # Some structured candidates
        # Triangular numbers
        tri = [0]
        for i in range(1, k + 1):
            tri.append(tri[-1] + i)
        if tri[-1] <= scope:
            if tri not in candidates:
                candidates.append(tri)
        
        # Squares
        sq = [i * i for i in range(k + 1)]
        if sq[-1] <= scope:
            if sq not in candidates:
                candidates.append(sq)
        
        # Linear with varying step
        for step in range(1, 5):
            lin = [i * step for i in range(k + 1)]
            if lin[-1] <= scope:
                if lin not in candidates:
                    candidates.append(lin)
        
        # Sort by largest element (prefer smaller scope usage)
        candidates.sort(key=lambda r: r[-1])
        
        return candidates
    
    # Try scopes from a reasonable lower bound
    # For (7,5): lower bound 105, best known 112
    # Start from a bit below best known and go up
    scope_start = max(min_scope_lower, 100)
    scope_end = 130
    
    # First, try to find any valid solution with greedy to get a baseline
    greedy_result = _greedy_construct(n_rows, k)
    if greedy_result:
        best_result = greedy_result
        best_scope = max(max(r) for r in greedy_result)
    
    # Now try to improve with backtracking
    for scope in range(scope_start, scope_end + 1):
        if time.time() > deadline:
            break
        if scope >= best_scope:
            break
        
        result = solve_with_scope(scope)
        if result is not None:
            # Verify
            if _validate(result, n_rows, k):
                best_result = result
                best_scope = scope
                # Found a solution at this scope, try to go lower? No, we're going up.
                # Actually we want the minimum scope, so if we found one at this scope,
                # we should check if we can do better by trying lower scopes
                # But we're iterating from low to high, so this IS the best
                break
    
    if best_result is None:
        # Fallback to greedy
        best_result = _greedy_construct(n_rows, k)
    
    return best_result


def _greedy_construct(n_rows, k):
    """Greedy construction as baseline."""
    used, rows = set(), []
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


def _validate(rows, n_rows, k):
    """Validate that the rows satisfy all constraints."""
    if len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        # Check differences
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True
# EVOLVE-BLOCK-END
