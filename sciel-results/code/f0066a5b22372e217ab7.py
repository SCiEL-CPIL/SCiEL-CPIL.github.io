"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    
    Uses a time-sliced backtracking search over candidate scopes.
    """
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # The greedy solution gives an upper bound on the scope
    greedy_rows = _greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in greedy_rows)
    best_rows = greedy_rows
    
    # Try to improve from the greedy bound downwards
    # But first, let's try a smarter initial upper bound
    # For (4,7): known best is 134, counting bound is 112
    # We'll search from a reasonable lower bound up
    
    # Lower bound: at least n_rows * k * (k+1) / 2 distinct differences needed
    # Each difference is in [1, scope], so scope >= total_diffs
    # But more carefully: total distinct positive differences needed = n_rows * k*(k+1)/2
    # Each value d in [1, scope] can appear at most once total
    # So scope >= n_rows * k*(k+1) / 2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs_needed  # crude lower bound
    
    # Actually, a better lower bound: the sum of all differences in one row
    # with k+1 elements starting at 0 and ending at M is at least... 
    # For a single row, the minimum sum of all positive differences is when
    # the row is as "compact" as possible. But the differences must be distinct
    # globally, so we need at least total_diffs_needed distinct values in [1, scope].
    
    # Let's try scopes from lower_bound upward, but start from something reasonable
    # For (4,7): total_diffs = 4*7*8/2 = 112, which matches the counting bound
    
    # Strategy: try scopes in increasing order, use backtracking with time slicing
    # If we find a valid solution at scope S, try S-1, S-2, ... down to lower_bound
    
    # First, let's try to find any solution at or below the greedy bound
    # Then try to improve
    
    # We'll do a binary-search-like approach or just iterate
    # Given time constraints, let's try a few scopes
    
    # Try scopes from min(greedy_bound, some_reasonable_start) down
    # Actually, let's try from lower_bound up to greedy_bound, but with time limits
    
    # Better approach: try the greedy bound first (we know it works),
    # then try to reduce it
    
    # Let's try scopes: greedy_bound, greedy_bound-1, ..., but with time limits
    # And also try lower_bound, lower_bound+1, ... 
    
    # For efficiency, let's try a few key scopes
    # The known best for (4,7) is 134, greedy might give something higher
    
    # Let's implement a backtracking solver that tries to build rows one by one
    # for a given scope, with forward checking
    
    def _try_scope(scope, time_limit):
        """Try to construct n_rows rows within the given scope.
        Returns rows if successful, None if failed or timeout."""
        start_time = time.time()
        
        # Total distinct differences needed
        total_needed = n_rows * k * (k + 1) // 2
        
        # Use a bitset for used differences (1 to scope)
        # Represent as a Python integer for speed
        used_bits = 0  # bit d is set if difference d is used
        
        rows = []
        
        def _build_row(row_idx, remaining_scope_budget):
            """Build one row starting at 0 with k+1 elements, all <= scope.
            Returns the row if successful, None if failed."""
            nonlocal used_bits
            
            row = [0]
            
            def _backtrack(pos, last_val):
                """Build row[pos:] given row[0:pos] and last_val = row[pos-1]."""
                nonlocal used_bits
                
                if time.time() - start_time > time_limit:
                    return None
                
                if pos == k + 1:
                    return list(row)
                
                # Try next values from last_val+1 to scope
                # But we need enough room: need k+1-pos more elements
                # Each at least 1 apart, so last_val + (k+1-pos) <= scope
                min_next = last_val + 1
                max_next = scope - (k - pos)  # need k-pos more after this
                
                # Sort candidates by some heuristic - try smaller gaps first for compactness
                # Actually, for finding ANY solution, let's try in order
                for candidate in range(min_next, max_next + 1):
                    # Check if adding candidate creates new differences that conflict
                    new_diffs = []
                    conflict = False
                    for a in row:
                        d = candidate - a
                        if d > scope:
                            conflict = True
                            break
                        if (used_bits >> d) & 1:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if conflict:
                        continue
                    
                    # Tentatively use these differences
                    old_bits = used_bits
                    for d in new_diffs:
                        used_bits |= (1 << d)
                    
                    row.append(candidate)
                    result = _backtrack(pos + 1, candidate)
                    if result is not None:
                        return result
                    
                    row.pop()
                    used_bits = old_bits
                
                return None
            
            return _backtrack(1, 0)
        
        # Build all rows
        for r in range(n_rows):
            if time.time() - start_time > time_limit:
                return None
            
            row = _build_row(r, scope)
            if row is None:
                return None
            rows.append(row)
        
        return rows
    
    # Try to improve from the greedy bound
    # First, verify greedy works (it should)
    # Then try scope = greedy_bound - 1, -2, etc.
    # Also try lower_bound, lower_bound + 1, etc.
    
    # Let's try a range of scopes with time limits
    # Start from the lower bound and go up, OR start from greedy and go down
    # Going down from greedy is better since we know greedy works
    
    # Try scopes from min(greedy_bound, 200) down to lower_bound
    # But with time limits per scope attempt
    
    # Actually, let's be smarter. Try a few scopes:
    # 1. The greedy bound (known to work)
    # 2. A few below it
    # 3. The lower bound and a few above
    
    # For (4,7), lower_bound = 112, known best = 134
    # Let's try: 134, 133, 132, ..., down to maybe 120
    # And also 112, 113, 114, ... up to 134
    
    # Time-sliced approach: give each scope attempt a small time budget
    
    # Let's try scopes in this order:
    # First try to beat the greedy bound
    candidates = []
    
    # Add scopes from lower_bound to greedy_bound
    # But that's too many. Let's be selective.
    
    # Try the known best for (4,7) if applicable
    if n_rows == 4 and k == 7:
        # Known best is 134
        candidates.append(134)
        candidates.extend(range(133, 120, -1))
        candidates.extend(range(112, 120))
    else:
        # General case: try from lower_bound to greedy_bound
        # But limit the number of attempts
        step = max(1, (greedy_bound - lower_bound) // 20)
        candidates = list(range(lower_bound, greedy_bound + 1, step))
        candidates.append(greedy_bound)
        # Sort by closeness to lower_bound first (optimistic)
        # Actually, try from lower_bound up, since we want the minimum
        candidates = sorted(set(candidates))
    
    # Remove duplicates and sort
    candidates = sorted(set(candidates))
    
    # Try each candidate scope with a time limit
    time_per_scope = 5.0  # seconds per scope attempt
    
    for scope in candidates:
        if time.time() > deadline:
            break
        
        # Don't try scopes larger than current best (we already have greedy)
        if scope >= best_scope and best_rows is not None:
            continue
        
        remaining_time = deadline - time.time() - 5.0  # leave 5s safety
        if remaining_time < 1.0:
            break
        
        actual_time_limit = min(time_per_scope, remaining_time)
        
        result = _try_scope(scope, actual_time_limit)
        if result is not None:
            # Verify the result
            if _verify(result, n_rows, k):
                best_scope = scope
                best_rows = result
                # If we found a solution at the lower bound, we're done
                if scope == lower_bound:
                    break
                # Continue trying to find even smaller
            else:
                # Shouldn't happen, but skip
                pass
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes the smallest next mark whose differences
    to earlier marks are all unused."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _verify(rows, n_rows, k):
    """Verify that the rows satisfy all constraints."""
    if len(rows) != n_rows:
        return False
    
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    
    # Check strictly increasing
    for row in rows:
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
    
    return True
# EVOLVE-BLOCK-END
