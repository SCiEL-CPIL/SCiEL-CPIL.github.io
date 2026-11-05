import time
import random
import math
import sys

def _row_diffs_list(row):
    """Return list of all positive differences within a row."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))

def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks

def _solve_csp(n_rows, k, max_scope, rng, deadline):
    """
    Solves the problem by backtracking over the rows.
    For each row, we select a set of k+1 numbers (starting with 0) such that:
    1. The numbers are strictly increasing.
    2. The differences between any pair are unique within the row.
    3. The differences do not conflict with differences in previous rows.
    
    We use a recursive search for each row, filling positions 1 to k.
    Heuristic: Try smaller values first to keep the scope low, but use
    forward checking to ensure we don't paint ourselves into a corner.
    """
    num_elements = k + 1
    used_mask = 0
    rows = []
    
    # Pre-compute which differences are "expensive" (large) to avoid them early?
    # No, we want to minimize the MAX value. So we want to pack differences tightly.
    # The "tightest" packing of differences for a row of length k+1 is a Sidon set (B_2 sequence) 
    # or similar, but here differences must be distinct across ALL rows.
    
    def backtrack_row(row_idx, pos, current_row, used_mask):
        if time.time() > deadline:
            return False
        
        if pos == num_elements:
            rows.append(current_row)
            return True
        
        last_val = current_row[-1]
        
        # Determine range for next value
        # The next value must be > last_val.
        # Also, we need to leave room for (num_elements - pos - 1) more elements.
        # The minimum increment is 1.
        # So max_val for this position is max_scope - (num_elements - pos - 1)
        max_possible = max_scope - (num_elements - pos - 1)
        
        # We also need to ensure that the differences created don't conflict.
        # And we need to ensure that the remaining positions can be filled.
        
        # Candidate generation:
        # We want to try values that allow a valid completion.
        # A strong heuristic: Try values that create differences which are "small" 
        # but don't conflict. However, small differences might block future small differences.
        
        # Let's iterate from last_val + 1 to max_possible.
        # To improve speed, we can shuffle the candidates slightly or use a heuristic order.
        # Actually, for minimizing scope, we generally want to fill the "gaps" efficiently.
        
        # Optimization: If we are late in the row, the choices are constrained.
        
        # Generate candidates
        candidates = []
        for val in range(last_val + 1, max_possible + 1):
            # Quick check: differences with previous elements
            # If any difference is already used in used_mask, skip
            # This is O(k) per candidate.
            
            # We can pre-filter: 
            # The set of differences {val - current_row[i] for i in range(pos)}
            # must be disjoint from used_mask.
            
            # Let's just check inside the loop to keep it simple, but we can optimize.
            candidates.append(val)
            
        # Heuristic: Sort candidates by how many "future conflicts" they might cause?
        # Simpler: Try values in increasing order (greedy for min scope) but with some randomness
        # to escape local minima if the first path fails.
        
        # For CSP, deterministic order (smallest first) is usually best for finding a solution
        # within the tightest bound. If it fails, we increase the bound.
        
        for val in candidates:
            # Check differences
            new_diffs_mask = 0
            valid = True
            for i in range(pos):
                d = val - current_row[i]
                if (1 << d) & used_mask:
                    valid = False
                    break
                # Also check against other differences in the current row
                # Since we are building the row sequentially, we just need to check against
                # the differences already formed in this row.
                # Wait, the differences formed in this row are:
                # current_row[j] - current_row[i] for i < j < pos.
                # We can maintain a local mask for the current row's internal diffs.
                
                # Let's compute the mask of diffs for this new val against previous elements
                new_diffs_mask |= (1 << d)
            
            if not valid:
                continue
            
            # Check internal consistency of the new row
            # The new differences {val - current_row[i]} must be distinct from each other?
            # No, val - current_row[i] are all distinct because current_row is strictly increasing.
            # But they must be distinct from differences already in the row.
            # e.g., if row is [0, 3, 7], diffs are 3, 7, 4.
            # If we add 10: diffs are 10, 7, 3.
            # 7 and 3 are already in the row! So this is invalid.
            
            # So we need to check if new_diffs_mask intersects with the mask of existing diffs in current_row.
            
            # Let's compute existing diffs mask for current_row.
            # We can pass this as an argument to avoid recomputation.
            
            # For simplicity in this implementation, we recompute or pass it.
            # Let's modify the function signature to pass current_row_mask.
            
            # Actually, let's just check the set of diffs.
            existing_diffs = set()
            for i in range(pos):
                for j in range(i+1, pos):
                    existing_diffs.add(current_row[j] - current_row[i])
            
            new_diffs_set = {val - current_row[i] for i in range(pos)}
            
            if new_diffs_set & existing_diffs:
                continue
            
            # Check against global used_mask
            # We already checked (1 << d) & used_mask for each d.
            # But we should combine them into a mask for efficiency in the recursive call.
            
            # Construct new row mask
            new_row_mask = _row_mask(current_row + [val])
            
            # Check global conflict
            if new_row_mask & used_mask:
                continue
            
            # Check internal conflict (redundant if we did the set check, but safe)
            # The set check above ensures internal uniqueness.
            
            # Recurse
            if backtrack_row(row_idx, pos + 1, current_row + [val], used_mask | new_row_mask):
                return True
        
        return False

    # We need to be careful with the "existing_diffs" calculation inside the loop.
    # It's O(k^2) per candidate. With k=6, it's small.
    # But we can optimize by passing the current row's diff mask.
    
    # Let's rewrite the inner logic to be more efficient.
    
    def backtrack_row_opt(row_idx, pos, current_row, current_row_mask, used_mask):
        if time.time() > deadline:
            return False
        
        if pos == num_elements:
            rows.append(current_row)
            return True
        
        last_val = current_row[-1]
        max_possible = max_scope - (num_elements - pos - 1)
        
        # Try candidates
        # To improve success rate, we might want to try a few random candidates if greedy fails quickly?
        # No, for CSP, greedy (smallest first) is standard for finding the tightest solution.
        
        for val in range(last_val + 1, max_possible + 1):
            # Check if val creates a diff that is in used_mask
            # or in current_row_mask
            
            valid = True
            for i in range(pos):
                d = val - current_row[i]
                if (1 << d) & used_mask:
                    valid = False
                    break
                if (1 << d) & current_row_mask:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # If valid, compute new mask
            # new_row_mask = current_row_mask | (mask of diffs from val to current_row)
            
            # We can just call _row_mask on the new list, it's small.
            # Or compute incrementally.
            
            new_row_list = current_row + [val]
            new_row_mask = _row_mask(new_row_list)
            
            # Double check global
            if new_row_mask & used_mask:
                continue
            
            # Recurse
            if backtrack_row_opt(row_idx, pos + 1, new_row_list, new_row_mask, used_mask | new_row_mask):
                return True
        
        return False

    # Start with row 0
    # Row 0 starts with [0]
    # used_mask is 0
    # current_row_mask is 0 (no diffs yet)
    
    # We need to handle the fact that rows are added to `rows` only on success.
    # This is a depth-first search over the sequence of rows.
    
    # Optimization: Order of rows?
    # Does the order of rows matter? Yes, because used_mask accumulates.
    # To make the search easier, we should probably construct rows that are "dense" first?
    # Or rows that have many small differences?
    # Actually, since we are minimizing the MAX scope, and all rows share the same bound,
    # the order might not matter for the final scope, but it matters for the search space.
    
    # Let's try constructing rows in a fixed order, but if it fails, we can try shuffling the row order?
    # No, the rows are indistinguishable in the problem statement (just a set of rows).
    # So we can assign the "first" constructed row to any slot.
    
    # However, a common trick is to construct the "hardest" row first.
    # What is the hardest row? Probably one that uses many small numbers.
    
    # Let's just try the standard backtracking.
    
    # To speed up, we can use a "forward checking" heuristic:
    # Before trying a value, check if the remaining positions in the current row
    # can be filled. This is hard to do exactly.
    
    # Let's rely on the tight bound `max_possible` to prune.
    
    # One major issue: Python recursion limit and speed.
    # k=6, n_rows=7. Depth is 7*6 = 42. This is fine.
    
    # But the branching factor can be large.
    # We need to ensure `max_scope` is not too large.
    # We will call this function with increasing `max_scope` starting from the counting bound.
    
    if backtrack_row_opt(0, 1, [0], 0, 0):
        return rows
    
    return None

def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    # The counting bound is a lower bound on the scope.
    # We start searching from counting_bound upwards.
    
    best_rows = None
    best_scope = float('inf')
    
    # Try CSP solver with increasing scope
    # We limit the number of attempts to avoid timeout if it gets stuck.
    
    # For (7,5), counting bound is 105. Best known is 112.
    # So we should try 105, 106, ...
    
    # To handle potential timeouts in CSP, we allocate a time slice per scope attempt.
    # If it fails, we try the next scope.
    
    # We also keep a fallback greedy solution if CSP is too slow.
    
    # First, get a quick greedy solution to have a valid result ready.
    # The current program's greedy is okay, but let's use a simple one here as a fallback.
    
    def _quick_greedy():
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    fallback_rows = _quick_greedy()
    fallback_scope = _scope(fallback_rows)
    
    # Now try CSP
    # We will try scopes from counting_bound to min(fallback_scope, counting_bound + 20)
    
    # If CSP finds a solution with scope S, we return it.
    # If CSP fails for all scopes up to fallback_scope, we return the fallback.
    
    # To make CSP faster, we can use the fallback scope as an upper bound for the search.
    
    # Let's try scopes in order.
    # We give each scope attempt a time limit of, say, 10 seconds.
    # If it doesn't find a solution in 10s, we move to the next scope.
    # This is risky if the solution is complex, but for small n, it should be okay.
    
    # Actually, if we increase the scope, the search space grows.
    # It's better to try the tightest bound first with a generous time limit.
    
    # Let's try the counting_bound with a 30s limit.
    # Then counting_bound + 1 with 20s.
    # Etc.
    
    # But if counting_bound is infeasible (which it often is), we waste time.
    # We can use a quick heuristic to check feasibility?
    # No, just run it.
    
    # Let's structure the time budget:
    # 10s for fallback generation (already done).
    # Remaining 125s for CSP.
    
    # We will try scopes: counting_bound, counting_bound+1, ...
    # Time per scope: starts at 10s, increases slightly?
    # Or just split the remaining time among, say, 10 scope attempts.
    
    # Let's try up to 15 scope values.
    # Time per attempt = (DEADLINE - time.time() - 5) / 15
    
    max_scope_to_try = min(fallback_scope, counting_bound + 30)
    
    scopes_to_try = list(range(counting_bound, max_scope_to_try + 1))
    
    if not scopes_to_try:
        return fallback_rows
    
    num_attempts = len(scopes_to_try)
    # Ensure we don't try too many
    if num_attempts > 10:
        scopes_to_try = scopes_to_try[:10]
        num_attempts = 10
    
    time_per_attempt = (DEADLINE - time.time() - 10) / num_attempts
    if time_per_attempt < 1.0:
        time_per_attempt = 1.0
        
    for scope in scopes_to_try:
        if time.time() > DEADLINE - 10:
            break
            
        # Use a new RNG for each attempt to vary the search?
        # CSP is deterministic here (greedy order).
        # We can add randomness to the candidate order if we want.
        # But deterministic is safer for reproducibility.
        
        # Let's add a small randomization to the candidate order to avoid getting stuck
        # in a specific bad path if the first path is a dead end.
        # Actually, backtracking handles dead ends.
        
        # Let's just run it.
        
        rows = _solve_csp(n_rows, k, scope, random.Random(42), time.time() + time_per_attempt)
        
        if rows is not None and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            # Since we are trying scopes in increasing order, the first valid solution
            # is the optimal one for this specific search path.
            # But wait, a solution with scope `scope` might not be the minimal possible
            # if we didn't explore the search space fully?
            # No, if we find a valid solution with max entry <= scope, and we tried
            # smaller scopes and failed, then this is the best we can do with this method.
            # However, the CSP might have failed for smaller scopes due to search limitations
            # (time out), not infeasibility.
            # So we should keep searching if we have time?
            # No, if we found a solution with scope S, and we tried S-1 and failed (timeout),
            # it's possible S-1 was feasible.
            # But given the time constraint, we accept the first found valid solution
            # as a good upper bound.
            
            # Actually, we should break and return this solution if it's good.
            # But maybe a later scope attempt (larger) is not needed.
            # We want the SMALLEST scope.
            # So if we find a solution for scope S, we should try to see if S-1 is possible?
            # We already tried S-1. If it failed due to timeout, we don't know.
            # If it failed due to infeasibility, we are good.
            
            # Given the complexity, let's just return the first valid solution found.
            # It is a valid solution with scope S.
            # Is it the best? Probably close.
            
            # Let's do a quick local search to see if we can reduce the scope further?
            # The current program has local search. We can reuse that logic?
            # No, let's stick to the CSP result.
            
            # Break out of the loop
            break
    
    if best_rows is not None:
        return best_rows
    else:
        return fallback_rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
