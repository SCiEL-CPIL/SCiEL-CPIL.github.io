"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    # We'll use a randomized local search / simulated annealing approach
    # combined with greedy initialization.
    
    # First, get a baseline using the original greedy method
    baseline = _greedy_construct(n_rows, k)
    best_scope = _scope_of(baseline)
    best_rows = [list(r) for r in baseline]
    
    # Try to improve with randomized search
    # Strategy: 
    # 1. Generate candidate rows (Golomb rulers) with bounded scope
    # 2. Use randomized greedy set-packing to select non-conflicting rows
    # 3. Local search: try replacing rows with smaller-scope alternatives
    
    total_diffs = k * (k + 1) // 2  # number of differences per row
    max_diff = 134  # known best scope for (4,7)
    
    # For general n_rows, k, we need to be adaptive
    # Let's estimate a reasonable scope bound
    # Counting bound: sum of all differences must fit in 1..scope
    # Total differences needed: n_rows * k*(k+1)/2
    # So scope >= sqrt(2 * n_rows * k*(k+1)/2) roughly
    # But we need integer constraints, so use a better bound
    
    import math
    min_scope_count = 0
    # The counting bound: if scope is S, we can have at most S differences (1 to S)
    # We need n_rows * k*(k+1)/2 distinct differences
    # So S >= n_rows * k*(k+1)/2
    # Wait, that's not right. The differences are a_j - a_i where 0 <= a_0 < ... < a_k
    # The maximum difference is a_k - a_0 = a_k (since a_0 = 0)
    # So all differences are in 1..a_k
    # We need all n_rows * k*(k+1)/2 differences to be distinct
    # So a_k >= n_rows * k*(k+1)/2
    # For (4,7): 4 * 7*8/2 = 4*28 = 112. Yes, that's the counting bound.
    
    counting_bound = n_rows * k * (k + 1) // 2
    
    # Start scope search from counting_bound up to some upper bound
    # Use the baseline as an upper bound
    upper_bound = best_scope
    lower_bound = max(counting_bound, 0)
    
    # We'll try randomized construction with scope bounded by current best
    # and try to find a valid set with smaller scope
    
    rng = random.Random(42)
    
    # Try multiple random seeds and scope values
    best_found_scope = best_scope
    best_found_rows = [list(r) for r in baseline]
    
    # Try decreasing scope from best_scope down to counting_bound
    # But first, let's try to improve the baseline with local search
    
    # Local search on the baseline
    improved = _local_search(best_rows, n_rows, k, start_time, deadline, rng)
    if improved is not None:
        imp_scope = _scope_of(improved)
        if imp_scope < best_found_scope:
            best_found_scope = imp_scope
            best_found_rows = [list(r) for r in improved]
    
    # Now try randomized greedy with various scope bounds
    for scope_attempt in range(best_found_scope, counting_bound - 1, -1):
        if time.time() > deadline - 5.0:
            break
        if scope_attempt < counting_bound:
            break
        
        # Try to construct with this scope
        result = _randomized_greedy_construct(n_rows, k, scope_attempt, rng, 
                                               time.time(), deadline)
        if result is not None:
            res_scope = _scope_of(result)
            if res_scope < best_found_scope:
                best_found_scope = res_scope
                best_found_rows = [list(r) for r in result]
                # Try local search on this improved result
                improved2 = _local_search(best_found_rows, n_rows, k, 
                                           time.time(), deadline, rng)
                if improved2 is not None:
                    imp2_scope = _scope_of(improved2)
                    if imp2_scope < best_found_scope:
                        best_found_scope = imp2_scope
                        best_found_rows = [list(r) for r in improved2]
        
        # If we found a valid set at this scope, we can stop
        # because we're going downward
        if best_found_scope == scope_attempt:
            # Try to go lower
            pass
    
    # Final local search
    final_improved = _local_search(best_found_rows, n_rows, k, 
                                    time.time(), deadline, rng)
    if final_improved is not None:
        fin_scope = _scope_of(final_improved)
        if fin_scope < best_found_scope:
            best_found_scope = fin_scope
            best_found_rows = [list(r) for r in final_improved]
    
    return best_found_rows


def _scope_of(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(row) for row in rows)


def _greedy_construct(n_rows, k):
    """Original greedy method."""
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


def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _randomized_greedy_construct(n_rows, k, max_scope, rng, start_time, deadline):
    """
    Try to construct n_rows non-conflicting rows with scope <= max_scope.
    Uses randomized greedy with backtracking-like restarts.
    """
    # Generate a pool of candidate rows with scope <= max_scope
    # This is expensive, so we'll generate them lazily
    
    # First, let's try a simpler approach: randomized greedy selection
    # where we build each row one at a time with randomization
    
    for attempt in range(50):
        if time.time() > deadline - 3.0:
            break
        
        used = set()
        rows = []
        success = True
        
        for row_idx in range(n_rows):
            # Try to build a row with scope <= max_scope
            row = None
            for sub_attempt in range(200):
                if time.time() > deadline - 3.0:
                    success = False
                    break
                candidate = _random_row(k, max_scope, used, rng)
                if candidate is not None:
                    row = candidate
                    break
            
            if row is None:
                success = False
                break
            
            rows.append(row)
            used |= _row_diffs(row)
        
        if success and len(rows) == n_rows:
            # Verify
            all_diffs = set()
            for r in rows:
                d = _row_diffs(r)
                if d & all_diffs:
                    success = False
                    break
                all_diffs |= d
                if max(r) > max_scope:
                    success = False
                    break
            if success:
                return rows
    
    return None


def _random_row(k, max_scope, used, rng):
    """
    Try to generate a random row of k+1 integers starting at 0,
    with max value <= max_scope, whose differences don't conflict with used.
    """
    # Simple random approach: try random increasing sequences
    for _ in range(50):
        if rng.random() < 0.3:
            # Try a structured approach: start with greedy-like but with randomization
            row = [0]
            for i in range(k):
                # Try several random candidates
                found = False
                candidates = list(range(len(row), max_scope + 1))
                rng.shuffle(candidates)
                for cand in candidates[:20]:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        row.append(cand)
                        found = True
                        break
                if not found:
                    break
            if len(row) == k + 1 and max(row) <= max_scope:
                # Final check
                final_diffs = _row_diffs(row)
                if not final_diffs & used:
                    return row
        else:
            # Pure random: pick k random distinct values from 1..max_scope
            if max_scope < k:
                continue
            vals = rng.sample(range(1, max_scope + 1), k)
            vals.sort()
            row = [0] + vals
            diffs = _row_diffs(row)
            if not diffs & used:
                return row
    
    return None


def _local_search(rows, n_rows, k, start_time, deadline, rng):
    """
    Local search: try to reduce the scope by modifying individual rows.
    """
    current = [list(r) for r in rows]
    current_scope = _scope_of(current)
    best = [list(r) for r in current]
    best_scope = current_scope
    
    # Compute used diffs for current solution
    def compute_used(rows):
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        return used
    
    used = compute_used(current)
    
    max_iterations = 500
    iteration = 0
    
    while iteration < max_iterations and time.time() < deadline - 5.0:
        iteration += 1
        
        # Pick a random row to try to improve
        row_idx = rng.randrange(n_rows)
        
        # Try to replace this row with a smaller-scope alternative
        old_row = current[row_idx]
        old_diffs = _row_diffs(old_row)
        other_used = used - old_diffs
        
        best_alt = None
        best_alt_scope = max(old_row)
        
        # Try several random alternatives
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            alt = _random_row(k, best_alt_scope - 1, other_used, rng)
            if alt is not None:
                alt_scope = max(alt)
                if alt_scope < best_alt_scope:
                    best_alt = alt
                    best_alt_scope = alt_scope
        
        if best_alt is not None:
            # Accept the improvement
            current[row_idx] = best_alt
            used = compute_used(current)
            new_scope = _scope_of(current)
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(r) for r in current]
        
        # Occasionally try a random perturbation (swap two rows' marks)
        if rng.random() < 0.2 and n_rows >= 2:
            i, j = rng.sample(range(n_rows), 2)
            # Try to swap some marks between rows i and j
            # This is complex, so skip for now
    
    if best_scope < current_scope:
        return best
    return None


# EVOLVE-BLOCK-END
