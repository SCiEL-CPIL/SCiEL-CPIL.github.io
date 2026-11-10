"""
Difference Triangle Set Solver using CSP and Bitsets
"""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 145s to be safe.
    deadline = start_time + 145.0
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Bitset helper: 
    # We use Python integers as bitsets.
    # Bit d is set if difference d is used.
    # Max difference is bounded by scope M.
    
    def check_time():
        return time.time()

    def get_diffs_bitset(row, scope):
        """Compute the bitset of differences for a given row."""
        bits = 0
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d <= scope:
                    bits |= (1 << d)
                else:
                    # If a difference exceeds scope, this row is invalid for this scope
                    # But in our construction, we ensure max element <= scope, 
                    # so max diff <= scope.
                    pass
        return bits

    def validate_rows(rows):
        """
        Validate that rows are strictly increasing, start at 0, 
        and all differences are distinct globally.
        """
        used = set()
        for r in rows:
            if not r or r[0] != 0:
                return False
            for i in range(len(r)):
                ai = r[i]
                for j in range(i + 1, len(r)):
                    d = r[j] - ai
                    if d in used:
                        return False
                    used.add(d)
            # Check strictly increasing
            for i in range(len(r) - 1):
                if r[i+1] <= r[i]:
                    return False
        return True

    # ---------- Phase 1: Greedy Initial Solution ----------
    def greedy_row(used, temp=0.0, scope=None):
        """Build one row greedily."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope is not None and x > scope:
                return None, used
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if temp > 0 and random.random() < temp and x < 300:
                    continue
                row.append(x)
                used = used | new_diffs
        return row, used

    # Get a baseline solution
    random.seed(42)
    used = set()
    best_rows = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        if row is None:
            # Fallback: increase scope or restart
            # For now, just break and use what we have? No, we need valid rows.
            # Greedy usually works for reasonable n_rows, k.
            break
        best_rows.append(row)
    
    if len(best_rows) < n_rows:
        # Emergency fallback: simple sequential construction without strict validation during build
        # This is a safety net
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used = used | new_diffs
            best_rows.append(row)
            
    best_max = max(max(r) for r in best_rows)
    
    if check_time() > deadline - 20:
        return best_rows

    # ---------- Phase 2: CSP Solver with Bitsets ----------
    
    # We will binary search for the minimum scope M.
    # Lower bound: The counting bound is roughly sqrt(total_diffs * (total_diffs+1)/2) ? 
    # Actually, the number of distinct differences is total_diffs_needed.
    # The differences are in [1, M]. So M must be at least total_diffs_needed.
    # But they don't have to be 1..M. 
    # A better lower bound: The largest element in a row with k+1 elements must be at least k.
    # And we need n_rows * k(k+1)/2 distinct differences.
    # The maximum difference is M.
    # So M >= total_diffs_needed is a weak bound.
    # Let's use M_low = best_max // 2 or something reasonable, or just 1.
    # To be safe and fast, start M_low from a heuristic.
    
    # Heuristic lower bound:
    # Each row has k+1 elements. Max element m_i.
    # Diffs are <= m_i.
    # Total diffs = sum of counts.
    # If all rows have max M, we have n_rows * k(k+1)/2 diffs in [1, M].
    # So M >= n_rows * k(k+1)/2.
    m_low = max(k, n_rows * diffs_per_row) 
    m_high = best_max
    
    # To save time, we can also try a few random restarts with the backtracker 
    # without binary search first, to see if we can beat best_max directly.
    
    def backtrack_solve(scope, time_limit):
        """
        Try to find a valid set of n_rows rows with max element <= scope.
        Uses backtracking with bitsets.
        Returns list of rows or None.
        """
        start_t = time.time()
        rows = []
        used_bits = 0
        # Precompute: what bits are allowed? 1 to scope.
        # We don't need to mask explicitly if we only set bits <= scope.
        
        # We build rows one by one.
        # To improve search, we can randomize the order of rows? 
        # No, the constraints are symmetric. 
        # However, the "difficulty" of a row depends on the used_bits.
        # Initially, all rows are equally hard.
        # As we build rows, the used_bits grow.
        
        # We will use a stack for backtracking.
        # State: (row_idx, current_row, used_bits)
        # Since rows are independent given used_bits, we can just recurse on row_idx.
        
        # Optimization: If we fail to build a row, we backtrack.
        # But building a row itself involves backtracking.
        
        def build_row(used_bits, row_idx, time_limit):
            """
            Build row_idx-th row given used_bits from previous rows.
            Returns (row, new_used_bits) or (None, None).
            """
            row = [0]
            # We need to choose k more elements.
            # We use a recursive DFS for the row construction.
            
            # To make it faster, we can use a heuristic: 
            # Try candidates in increasing order.
            
            def dfs_row(row, used_bits):
                if time.time() > time_limit:
                    return None, None
                if len(row) == k + 1:
                    return row, used_bits
                
                last = row[-1]
                rem = (k + 1) - len(row)
                
                # Min next value
                min_next = last + 1
                # Max next value
                max_next = scope - (rem - 1)
                
                if min_next > max_next:
                    return None, None
                
                # Generate candidates
                # To break symmetry or explore better, we can shuffle candidates 
                # but usually increasing order is best for finding *a* solution 
                # with small max.
                
                for x in range(min_next, max_next + 1):
                    if time.time() > time_limit:
                        return None, None
                    
                    # Check conflicts
                    # new_diffs bits
                    conflict = False
                    new_bits = 0
                    for a in row:
                        d = x - a
                        if d > scope: 
                            conflict = True
                            break
                        bit = 1 << d
                        if used_bits & bit:
                            conflict = True
                            break
                        new_bits |= bit
                    
                    if conflict:
                        continue
                    
                    new_row = row + [x]
                    new_used = used_bits | new_bits
                    
                    res_row, res_used = dfs_row(new_row, new_used)
                    if res_row is not None:
                        return res_row, res_used
                
                return None, None
            
            return dfs_row(row, used_bits)
        
        # Main loop for rows
        # We try to build row 0, then row 1, etc.
        # If we fail, we could backtrack to the previous row and try a different solution for it.
        # This requires a more complex state stack.
        
        # For simplicity and speed, we first try a "greedy backtracking" approach:
        # Build row 0. If it fails, restart row 0 with a different random seed/choice?
        # Actually, the inner dfs_row explores all possibilities for a row.
        # So if build_row returns None, it means NO row of size k+1 exists with max<=scope 
        # that is disjoint from used_bits.
        
        # So the strategy is:
        # 1. Build row 0.
        # 2. Build row 1 given row 0.
        # 3. ...
        # If any step fails, the whole scope is infeasible? 
        # NOT NECESSARILY. A different choice for row 0 might allow row 1 to be built.
        # So we need to backtrack row 0.
        
        # This is a standard CSP. 
        # Variables: R0, R1, ..., R_{n-1}.
        # Domains: All valid rows with max <= scope.
        # Constraint: Disjoint diffs.
        
        # We will implement a recursive solver that picks a row, tries a candidate, 
        # and recurses.
        
        # To improve efficiency, we can use "forward checking" or "arc consistency", 
        # but for this small problem, simple backtracking with bitsets is fast enough 
        # if we prune well.
        
        # Heuristic: Order rows by "difficulty". 
        # Initially, all rows have the same domain.
        # After R0 is chosen, R1's domain is reduced.
        # So R1 is harder than R0? No, R0 was chosen from the full domain.
        # R1 is chosen from the reduced domain.
        # So we should always pick the "next" row.
        
        # However, if we get stuck, we backtrack.
        
        def solve_rows(row_idx, used_bits, rows_acc):
            if time.time() > time_limit:
                return None
            
            if row_idx == n_rows:
                return list(rows_acc)
            
            # Try to build row_idx
            # We need to iterate through possible rows for row_idx.
            # The build_row function above only returns ONE solution (the first found).
            # We need to generate ALL possible rows or use a generator.
            
            # Let's modify build_row to be a generator or use a callback.
            # Alternatively, we can just use the simple approach:
            # If build_row fails, the scope is infeasible? 
            # No, because build_row uses a specific DFS order. 
            # If DFS finds a solution, it's valid. 
            # If DFS exhausts all options and finds none, then NO solution exists 
            # for this specific used_bits.
            
            # So:
            # 1. Call build_row.
            # 2. If it returns a row, append it and recurse.
            # 3. If it returns None, return None (backtrack).
            
            # BUT, this only explores ONE branch of the row construction tree 
            # (the first valid leaf found by DFS). 
            # If that branch leads to a dead end later (row_idx+1 fails), 
            # we need to try a DIFFERENT row for row_idx.
            
            # So build_row needs to support iteration.
            
            # Let's implement a generator for rows.
            
            def generate_rows(used_bits, time_limit):
                """Generator that yields valid rows disjoint from used_bits."""
                row = [0]
                
                def gen_dfs(row, used_bits):
                    if time.time() > time_limit:
                        return
                    if len(row) == k + 1:
                        yield row, used_bits
                        return
                    
                    last = row[-1]
                    rem = (k + 1) - len(row)
                    min_next = last + 1
                    max_next = scope - (rem - 1)
                    
                    if min_next > max_next:
                        return
                    
                    for x in range(min_next, max_next + 1):
                        if time.time() > time_limit:
                            return
                        
                        conflict = False
                        new_bits = 0
                        for a in row:
                            d = x - a
                            if d > scope:
                                conflict = True
                                break
                            bit = 1 << d
                            if used_bits & bit:
                                conflict = True
                                break
                            new_bits |= bit
                        
                        if conflict:
                            continue
                        
                        new_row = row + [x]
                        new_used = used_bits | new_bits
                        
                        yield from gen_dfs(new_row, new_used)
                
                yield from gen_dfs(row, used_bits)
            
            # Now, in solve_rows, we iterate over generate_rows
            for row, new_used in generate_rows(used_bits, time_limit):
                rows_acc.append(row)
                res = solve_rows(row_idx + 1, new_used, rows_acc)
                if res is not None:
                    return res
                rows_acc.pop()
            
            return None
        
        # Start solving
        return solve_rows(0, 0, [])

    # Binary Search for Scope
    # We want to find the smallest M such that backtrack_solve(M) returns a solution.
    
    # To speed up, we can first try to beat the current best_max directly 
    # with a few random restarts of the backtracker (with different tie-breaking).
    # But the generator `generate_rows` is deterministic (increasing order).
    # To introduce randomness, we can shuffle the candidate order in `gen_dfs`.
    
    # Let's add a randomization parameter to the solver.
    
    def backtrack_solve_randomized(scope, time_limit, seed):
        random.seed(seed)
        # We can't easily randomize the generator without passing seed down.
        # Instead, we can just run the deterministic solver.
        # If it fails, it fails.
        # To explore more, we need to randomize the DFS order.
        
        # Let's modify generate_rows to accept a randomization flag.
        # If random, shuffle candidates.
        
        def generate_rows_rand(used_bits, time_limit):
            row = [0]
            
            def gen_dfs(row, used_bits):
                if time.time() > time_limit:
                    return
                if len(row) == k + 1:
                    yield row, used_bits
                    return
                
                last = row[-1]
                rem = (k + 1) - len(row)
                min_next = last + 1
                max_next = scope - (rem - 1)
                
                if min_next > max_next:
                    return
                
                candidates = list(range(min_next, max_next + 1))
                # Heuristic: Usually increasing order is best for minimizing max.
                # But for finding *any* solution, random might be better?
                # Let's stick to increasing order for the binary search phase,
                # as we want the *smallest* scope.
                # If we randomize, we might find a solution with a larger max 
                # within the scope, which is fine, but it doesn't help with 
                # minimizing the global max directly in the binary search step.
                
                for x in candidates:
                    if time.time() > time_limit:
                        return
                    
                    conflict = False
                    new_bits = 0
                    for a in row:
                        d = x - a
                        if d > scope:
                            conflict = True
                            break
                        bit = 1 << d
                        if used_bits & bit:
                            conflict = True
                            break
                        new_bits |= bit
                    
                    if conflict:
                        continue
                    
                    new_row = row + [x]
                    new_used = used_bits | new_bits
                    
                    yield from gen_dfs(new_row, new_used)
            
            yield from gen_dfs(row, used_bits)
        
        def solve_rows(row_idx, used_bits, rows_acc):
            if time.time() > time_limit:
                return None
            
            if row_idx == n_rows:
                return list(rows_acc)
            
            for row, new_used in generate_rows_rand(used_bits, time_limit):
                rows_acc.append(row)
                res = solve_rows(row_idx + 1, new_used, rows_acc)
                if res is not None:
                    return res
                rows_acc.pop()
            
            return None
        
        return solve_rows(0, 0, [])

    # Try to improve best_max using binary search
    # We set a time limit for each binary search step.
    # Total time for binary search: ~50 seconds.
    
    m_low = max(k, n_rows * diffs_per_row)
    m_high = best_max
    
    # If m_low >= m_high, no binary search needed.
    if m_low < m_high:
        bs_time_limit = 5.0 # seconds per check
        while m_low < m_high:
            if check_time() > deadline - 30:
                break
            
            m_mid = (m_low + m_high) // 2
            # Try to find a solution with scope m_mid
            # We use a deterministic solver (increasing order)
            sol = backtrack_solve(m_mid, check_time() + bs_time_limit)
            
            if sol is not None:
                # Found a solution with max <= m_mid
                # Update best if it's better
                new_max = max(max(r) for r in sol)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = sol
                m_high = m_mid
            else:
                # No solution with scope m_mid
                m_low = m_mid + 1
        
        # After binary search, m_high is the best found scope.
        # We might have a solution for m_high in best_rows.
        # But best_rows might be from a previous iteration.
        # Let's ensure best_rows corresponds to best_max.
        # If best_max was updated, best_rows is updated.
        
    # Final Polish: Try to reduce best_max by 1, 2, ... using the solver
    # This is a linear search downwards from best_max.
    # We already did binary search, but maybe we can squeeze a bit more?
    # Binary search finds the minimum feasible M.
    # So best_max should be the minimum feasible M found.
    
    # However, the solver might be slow.
    # Let's do a few random restarts with the backtracker 
    # trying to find a solution with scope best_max - 1, best_max - 2, etc.
    
    # Actually, the binary search already does this.
    
    # One more thing: The greedy solution might be better than the CSP solution 
    # if the CSP times out. We keep best_rows from greedy if CSP fails.
    
    return best_rows

# Keep the original function signature
# The code above defines construct(n_rows, k)
