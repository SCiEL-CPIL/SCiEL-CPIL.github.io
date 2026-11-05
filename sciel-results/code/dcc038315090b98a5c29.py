import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    L = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Counting bound: need total_diffs distinct values in [1, M]
    # So M >= total_diffs
    M_min = total_diffs
    
    def row_from_gaps(gaps):
        """Convert gap list to position list starting at 0."""
        pos = [0]
        for g in gaps:
            pos.append(pos[-1] + g)
        return pos
    
    def diffs_from_row(row):
        """Get sorted list of all pairwise differences."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    def get_diff_mask(row):
        """Bitmask of differences for a row."""
        mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                mask |= (1 << (row[j] - row[i]))
        return mask
    
    def find_row_backtrack(gaps, used_mask, M, depth=0):
        """
        Backtracking search for a valid row given current gaps and used difference mask.
        gaps: list of gap values chosen so far
        used_mask: bitmask of all differences used so far (globally)
        M: maximum allowed difference
        Returns: list of k gaps, or None
        """
        if depth == k:
            # Verify: all differences must be <= M and not in used_mask
            # And all differences within this row must be distinct
            row = row_from_gaps(gaps)
            if row[-1] > M:
                return None
            # Check all pairwise differences
            row_diffs = set()
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d > M:
                        return None
                    if d in row_diffs:
                        return None
                    row_diffs.add(d)
                    if used_mask & (1 << d):
                        return None
            return list(gaps)
        
        # Choose gap at position depth
        # The gap must be >= 1
        # Upper bound: we need remaining positions to fit within M
        # Current position sum
        cur_pos = sum(gaps)
        remaining = k - depth
        # The last position will be cur_pos + sum(remaining gaps)
        # Each remaining gap >= 1, so last pos >= cur_pos + remaining
        # Also last pos <= M, so sum(remaining gaps) <= M - cur_pos
        # Each gap <= M - cur_pos (trivially), but tighter:
        # gap[depth] + ... + gap[k-1] <= M - cur_pos
        # gap[depth] <= M - cur_pos - (remaining - 1)  [since each subsequent gap >= 1]
        
        max_gap = M - cur_pos - (remaining - 1)
        if max_gap < 1:
            return None
        
        # Also, the gap must not create a difference that's already used
        # The new differences introduced by adding gap[depth] are:
        # gap[depth] itself, and gap[depth] + gap[i] for i < depth, etc.
        # Actually, all pairwise differences involving the new position.
        
        # To speed up: try gaps in a smart order
        # Prefer smaller gaps to keep scope low, but also need to avoid collisions
        
        # Generate candidate gaps
        # We'll try a subset of possible gaps
        candidates = []
        # Try small gaps first (to minimize scope)
        for g in range(1, min(max_gap + 1, 50)):
            candidates.append(g)
        # Also try some larger gaps
        if max_gap > 50:
            for g in range(50, max_gap + 1, max(1, (max_gap - 50) // 20)):
                candidates.append(g)
        
        random.shuffle(candidates)
        
        for g in candidates:
            new_gaps = gaps + [g]
            # Quick check: does this gap create any difference > M or collision?
            # Compute the position of the new element
            new_pos = cur_pos + g
            if new_pos > M:
                continue
            
            # Check all differences involving the new position
            valid = True
            row_so_far = row_from_gaps(new_gaps)
            n_so_far = len(row_so_far)
            for i in range(n_so_far - 1):
                d = row_so_far[-1] - row_so_far[i]
                if d > M:
                    valid = False
                    break
                if used_mask & (1 << d):
                    valid = False
                    break
            if not valid:
                continue
            
            # Also check that differences within the new row are distinct
            # (we'll do full check at the end, but can do partial here)
            # For now, recurse
            result = find_row_backtrack(new_gaps, used_mask, M, depth + 1)
            if result is not None:
                return result
        
        return None
    
    def find_row_greedy_with_backtrack(used_mask, M, max_backtrack_depth=3):
        """
        Greedy with limited backtracking to find a valid row.
        """
        gaps = []
        cur_pos = 0
        row_diffs = set()
        
        def backtrack(depth, gaps, cur_pos, row_diffs):
            if depth == k:
                # Final validation
                row = row_from_gaps(gaps)
                if row[-1] > M:
                    return None
                all_diffs = set()
                n = len(row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = row[j] - row[i]
                        if d > M or d in all_diffs or (used_mask & (1 << d)):
                            return None
                        all_diffs.add(d)
                return list(gaps)
            
            remaining = k - depth
            max_gap = M - cur_pos - (remaining - 1)
            if max_gap < 1:
                return None
            
            # Try candidates
            candidates = list(range(1, min(max_gap + 1, 40)))
            if max_gap > 40:
                candidates.extend(range(40, max_gap + 1, max(1, (max_gap - 40) // 15)))
            random.shuffle(candidates)
            
            for g in candidates:
                new_pos = cur_pos + g
                if new_pos > M:
                    continue
                
                # Check differences involving new position
                valid = True
                new_diffs = set()
                row_so_far = row_from_gaps(gaps + [g])
                n_so_far = len(row_so_far)
                for i in range(n_so_far - 1):
                    d = row_so_far[-1] - row_so_far[i]
                    if d > M:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                    if used_mask & (1 << d):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also check internal distinctness of all diffs so far
                # (partial check)
                all_so_far = set()
                for i in range(n_so_far):
                    for j in range(i + 1, n_so_far):
                        d = row_so_far[j] - row_so_far[i]
                        if d in all_so_far:
                            valid = False
                            break
                        all_so_far.add(d)
                if not valid:
                    continue
                
                result = backtrack(depth + 1, gaps + [g], new_pos, new_diffs)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(0, gaps, cur_pos, row_diffs)
    
    def solve_with_M(M):
        """
        Try to find n_rows valid rows with all differences <= M and globally distinct.
        Returns list of rows or None.
        """
        used_mask = 0
        rows = []
        
        for r in range(n_rows):
            # Find a valid row
            gaps = find_row_greedy_with_backtrack(used_mask, M)
            if gaps is None:
                # Try a few more times with different randomization
                for attempt in range(5):
                    gaps = find_row_greedy_with_backtrack(used_mask, M)
                    if gaps is not None:
                        break
            if gaps is None:
                return None
            
            row = row_from_gaps(gaps)
            # Update used_mask
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_mask |= (1 << (row[j] - row[i]))
            rows.append(row)
        
        return rows
    
    def local_search_reduce_scope(rows, time_limit):
        """
        Local search to reduce the scope (max entry) further.
        """
        rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        current_scope = best_scope
        
        # Precompute masks
        row_masks = []
        for row in rows:
            mask = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    mask |= (1 << (row[j] - row[i]))
            row_masks.append(mask)
        
        global_mask = 0
        for m in row_masks:
            global_mask |= m
        
        last_improvement = time.time()
        
        while time.time() < time_limit:
            if time.time() - last_improvement > 8.0:
                break
            
            ri = random.randrange(n_rows)
            # Try to reduce the last element (scope) by adjusting gaps
            # Or try to shift the entire row
            # Simple move: try to decrease one of the gaps
            
            # Get current gaps
            row = rows[ri]
            gaps = [row[i+1] - row[i] for i in range(k)]
            
            # Try to decrease a random gap
            gi = random.randrange(k)
            old_gap = gaps[gi]
            
            # Try decreasing the gap
            new_gap = old_gap - 1
            if new_gap < 1:
                continue
            
            new_gaps = gaps[:]
            new_gaps[gi] = new_gap
            new_row = row_from_gaps(new_gaps)
            
            # Check validity
            new_mask = 0
            valid = True
            n = len(new_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = new_row[j] - new_row[i]
                    if d > current_scope:
                        valid = False
                        break
                    if new_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
                if not valid:
                    break
            if not valid:
                continue
            
            # Check collision with other rows
            other_mask = global_mask & ~row_masks[ri]
            if new_mask & other_mask:
                continue
            
            # Valid move!
            new_scope = max(max(r) for r in rows[:ri] + [new_row] + rows[ri+1:])
            
            # Update state
            global_mask = (global_mask & ~row_masks[ri]) | new_mask
            row_masks[ri] = new_mask
            rows[ri] = new_row
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                last_improvement = time.time()
                current_scope = new_scope
        
        return best_rows, best_scope
    
    # Main: iterative deepening on M
    # Start from counting bound
    M = M_min
    best_rows = None
    best_scope = None
    
    # First, try to find a solution with M = M_min
    # If that fails, increase M gradually
    
    # But we need to be careful about time. Let's allocate time per M value.
    # Also, if M_min works, great. If not, we increase.
    
    # Time budget per M attempt
    # We'll try M_min, then M_min+1, M_min+2, etc.
    # But we need to ensure we finish within deadline.
    
    # Let's try a few M values with limited time each
    time_per_M = 15.0
    max_M_tries = 5
    
    for m_offset in range(max_M_tries):
        M_try = M_min + m_offset
        if time.time() > deadline - 20.0:
            break
        
        t_limit = min(deadline - 10.0, time.time() + time_per_M)
        
        rows = solve_with_M(M_try)
        if rows is not None:
            best_rows = rows
            best_scope = max(max(r) for r in rows)
            break
    
    # If we found a solution, try to improve with local search
    if best_rows is not None and time.time() < deadline - 5.0:
        improved, new_scope = local_search_reduce_scope(best_rows, time_limit=deadline - 2.0)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # If we didn't find a solution with iterative deepening, fall back to greedy
    if best_rows is None:
        # Fallback: pure greedy without M constraint
        used_mask = 0
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                valid = True
                new_mask = 0
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                if valid:
                    row.append(x)
                    used_mask |= new_mask
            best_rows.append(row)
    
    return best_rows
