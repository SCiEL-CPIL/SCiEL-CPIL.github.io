import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Total differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: we need at least total_diffs distinct positive differences
    # If max entry is M, differences are in [1, M], so M >= total_diffs
    lower_bound = total_diffs
    
    # Strategy:
    # 1. Use bitsets for difference tracking to speed up backtracking.
    # 2. Use a robust local search that can move any element, not just the last.
    # 3. Iteratively try to reduce the scope.

    def get_bitset_mask(limit):
        # Returns a bitmask with bits 0 to limit set
        # We use 1-indexed differences, so bit i corresponds to diff i
        # We need bits 1..limit set.
        # (1 << (limit + 1)) - 1 sets bits 0..limit
        # We want to ignore bit 0, but it's harmless if we never check it or set it.
        # Actually, diffs are >= 1. So we can just use a mask for 1..limit.
        # To check if diff d is used: (used_bits >> d) & 1
        # To set diff d: used_bits |= (1 << d)
        # To clear diff d: used_bits &= ~(1 << d)
        pass

    def backtrack_search(n_rows, k, target_scope, deadline, node_limit=200000):
        """
        Search for a solution where all entries <= target_scope.
        Uses bitsets for speed.
        """
        used_bits = 0
        rows = []
        node_count = 0
        max_diff = target_scope
        
        # Precompute masks for quick validity check?
        # For a candidate x in a row r, the new diffs are x - r[i].
        # We need to check if any of these diffs are in used_bits.
        # new_diffs_mask = sum(1 << (x - r[i]) for i in range(len(r)))
        # if used_bits & new_diffs_mask: invalid
        
        def backtrack(row_idx, current_row):
            nonlocal node_count, used_bits
            if time.time() > deadline or node_count > node_limit:
                return False
            node_count += 1
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Pruning: if we can't fit 'remaining' elements in [last+1, target_scope], fail
            # Smallest possible next elements: last+1, last+2, ...
            # Largest possible last element if we pick smallest: last + remaining
            if last + remaining > target_scope:
                return False
            
            # Candidates: x in [last+1, target_scope]
            # We try smaller x first to keep the row compact.
            # However, sometimes a larger x allows better packing later?
            # For now, smallest first is a good heuristic for minimizing max entry.
            
            # Optimization: Generate candidates in reverse? No, forward is better for scope.
            # But we can prune candidates that create diffs > target_scope?
            # Diff x - 0 = x <= target_scope. So x <= target_scope is already handled.
            
            # To speed up, we can iterate x from last+1 to target_scope.
            # Check validity using bitset.
            
            for x in range(last + 1, target_scope + 1):
                # Check if any new diff conflicts with used_bits
                # Diffs: x - a for a in current_row
                # Since current_row is sorted, diffs are x - current_row[-1], ..., x - 0
                # We can construct the mask.
                
                # Optimization: 
                # If x - current_row[0] > target_scope, we would have stopped earlier.
                # We just need to check if (used_bits & mask_of_new_diffs) == 0.
                
                mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    # d is in [1, target_scope]
                    mask |= (1 << d)
                
                if used_bits & mask:
                    continue
                
                # Accept x
                used_bits |= mask
                current_row.append(x)
                
                if backtrack(row_idx, current_row):
                    return True
                
                # Undo
                current_row.pop()
                used_bits ^= mask # XOR works because we know the bits are set
                
            return False
        
        if backtrack(0, [0]):
            return rows
        return None

    def check_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def local_search(rows, time_limit=10.0):
        """
        Try to reduce the max entry by local search.
        Move: pick a row, pick an element index j (j>0), change it to a smaller value.
        We must maintain strict increase.
        """
        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in best_rows)
        start = time.time()
        
        # Pre-compute used diffs? 
        # To speed up, we can maintain a global used set, but updating it during search is tricky.
        # Given the small size (8 elements per row, 4 rows), a full check is fast enough if we do it smartly.
        # However, we can optimize by checking only the affected row against the rest.
        
        def get_used_set_except_row(rows, exclude_idx):
            used = set()
            for i, r in enumerate(rows):
                if i == exclude_idx:
                    continue
                for j in range(len(r)):
                    for l in range(j + 1, len(r)):
                        used.add(r[l] - r[j])
            return used

        while time.time() - start < time_limit:
            improved = False
            # Try to improve each row
            for r_idx in range(len(best_rows)):
                if time.time() - start > time_limit:
                    break
                row = best_rows[r_idx]
                used_others = get_used_set_except_row(best_rows, r_idx)
                
                # We want to minimize the max entry of this row.
                # We can try to reduce elements from right to left.
                # If we reduce row[j], we must ensure row[j-1] < new_val < row[j+1] (if exists)
                # And new diffs don't conflict with used_others or other diffs in this row.
                
                # Strategy: Try to reduce the largest element first?
                # Or try to reduce any element that allows the row to be "shifted" left?
                
                # Let's try to reduce each element j from k down to 1.
                # If we can reduce row[j], we do it.
                # Note: reducing an earlier element might allow reducing later ones?
                # Yes. So we should probably scan from left to right?
                # If we reduce row[1], row[2] might now be reducible.
                # But if we reduce row[2], row[1] is unaffected.
                # Let's scan from right to left (largest index first) to secure the "tail".
                # Actually, to minimize the *max* entry, reducing the last element is most direct.
                # But reducing an earlier element might allow the last element to be reduced more in the next iteration.
                
                # Let's do a pass from right to left.
                for j in range(len(row) - 1, 0, -1):
                    current_val = row[j]
                    lower_bound = row[j-1] + 1
                    # Upper bound is current_val - 1 (to try smaller)
                    # Also must be < row[j+1] if j < len(row)-1
                    
                    # We want the smallest valid value? 
                    # No, we want *any* value smaller than current_val that is valid.
                    # Smaller is better for scope.
                    
                    # Find the smallest valid x in [lower_bound, current_val - 1]
                    # such that x is valid.
                    
                    # To be efficient, we can check x from lower_bound upwards?
                    # Or from current_val - 1 downwards?
                    # We want to minimize the value, so check from lower_bound upwards.
                    # But if we find a valid x, we set it and break?
                    # Yes, because any larger x would be worse for scope.
                    
                    found = False
                    for x in range(lower_bound, current_val):
                        # Check validity of x in this row
                        # Diffs involving x:
                        # 1. x - row[i] for i < j
                        # 2. row[l] - x for l > j
                        # These must not be in used_others.
                        # Also, these new diffs must not conflict with existing diffs in this row 
                        # that are NOT involving the old row[j].
                        
                        # This is getting complex to maintain incrementally.
                        # Let's just construct the candidate row and check against used_others + internal diffs.
                        
                        candidate_row = row[:]
                        candidate_row[j] = x
                        
                        # Check internal consistency of candidate_row
                        # 1. Strictly increasing? Yes, by construction of x range.
                        # 2. Diffs distinct?
                        
                        valid = True
                        internal_used = set()
                        
                        # Check diffs within candidate_row
                        for i in range(len(candidate_row)):
                            for l in range(i + 1, len(candidate_row)):
                                d = candidate_row[l] - candidate_row[i]
                                if d in internal_used:
                                    valid = False
                                    break
                                internal_used.add(d)
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        # Check against used_others
                        if internal_used & used_others:
                            continue
                        
                        # Valid!
                        best_rows[r_idx] = candidate_row
                        improved = True
                        found = True
                        # Update used_others? 
                        # used_others is unchanged.
                        # But the "internal" part of this row changed.
                        # We break the j loop to restart the scan for this row?
                        # Or just continue?
                        # If we reduced row[j], row[j-1] is unchanged. 
                        # We might be able to reduce row[j-1] now?
                        # Let's break the j loop and restart the row optimization.
                        break
                    
                    if found:
                        # Restart this row's optimization
                        break
            
            if not improved:
                break
        
        return best_rows

    # 1. Get initial solution
    # Start with a scope estimate
    # Lower bound is total_diffs. 
    # Greedy usually produces something close to lower_bound * 1.2
    initial_scope = int(total_diffs * 1.5) + 10
    rows = backtrack_search(n_rows, k, initial_scope, deadline, node_limit=50000)
    
    if rows is None:
        # Fallback: even larger scope
        rows = backtrack_search(n_rows, k, initial_scope + 50, deadline, node_limit=50000)
        
    if rows is None:
        # Emergency fallback: simple greedy (slow but robust)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
                    if x > 1000:
                        return [[0]] * n_rows # Should not happen
            rows.append(row)

    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Local search to improve
    best_rows = local_search(best_rows, time_limit=20.0)
    best_scope = max(max(r) for r in best_rows)
    
    # 3. Try to reduce scope
    # We know a solution exists for best_scope.
    # Try best_scope - 1, best_scope - 2, etc.
    
    # To avoid wasting time on impossible scopes, we can use a heuristic:
    # If we fail to find a solution for S, we might try S again later?
    # For now, just decrement.
    
    current_target = best_scope - 1
    
    while current_target >= lower_bound and time.time() < deadline:
        # Try to find a solution with max entry <= current_target
        # Use backtracking
        # To increase success rate, we can try a few random restarts or different heuristics?
        # The backtracking is deterministic (smallest first).
        # We can add a randomization factor: shuffle candidates?
        # But "smallest first" is best for scope minimization.
        
        # Let's try backtracking first.
        sol = backtrack_search(n_rows, k, current_target, deadline, node_limit=20000)
        
        if sol:
            # Verify
            if check_valid(sol):
                best_rows = sol
                best_scope = current_target
                # Polish the new solution
                best_rows = local_search(best_rows, time_limit=5.0)
                new_scope = max(max(r) for r in best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    # Update target to the new scope - 1?
                    # Or just continue decrementing from the new best_scope?
                    # If local search reduced it further, we should update current_target.
                    current_target = new_scope - 1
                else:
                    current_target -= 1
                continue
            
        # If backtracking failed, try local search on a perturbed version?
        # Or just give up on this scope?
        # Given time limits, if backtracking fails, it's likely hard.
        # Let's decrement and try the next smaller scope.
        current_target -= 1
        
        # Safety: if we've gone too low or time is running out, break
        if time.time() > deadline - 10:
            break
            
    # Final check
    if not check_valid(best_rows):
        # Fallback to initial greedy if something went wrong
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows

    return best_rows
