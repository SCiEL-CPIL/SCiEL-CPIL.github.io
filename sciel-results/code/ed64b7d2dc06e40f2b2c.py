# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a hybrid approach:
    1. Greedy baseline.
    2. Local search (hill climbing) to minimize the global max entry by rebuilding rows.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # --- Helper Functions ---
    
    def greedy_construct(n_rows, k, used=None):
        """
        Construct rows greedily. If 'used' is provided, it is updated in place.
        Returns the list of rows.
        """
        if used is None:
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
    
    def get_max_entry(rows):
        return max(max(r) for r in rows) if rows else 0

    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True

    def rebuild_row(rows, idx, used, max_limit, rng):
        """
        Attempt to rebuild rows[idx] to reduce its maximum entry,
        while respecting the differences used by other rows.
        Returns the new row or None if failed.
        """
        # Calculate the set of differences used by ALL other rows
        # We pass 'used' which contains ALL diffs. We need to remove the current row's diffs.
        current_row = rows[idx]
        current_diffs = set()
        for i in range(len(current_row)):
            for j in range(i + 1, len(current_row)):
                current_diffs.add(current_row[j] - current_row[i])
        
        available_diffs = used - current_diffs
        
        # Try to build a new row within max_limit
        # We use a randomized greedy with backtracking-lite (just retrying with different seeds/order)
        # For efficiency, we just try a greedy approach with randomization in candidate order.
        
        # To minimize the max entry, we should prefer smaller candidates.
        # But we also need to ensure we can complete the row.
        
        best_new_row = None
        best_max = max_limit + 1
        
        # Try multiple random restarts for this single row rebuild
        # Since k is small (7), the search space for a single row is manageable with good heuristics.
        # We'll use a simple backtracking search for the single row.
        
        def build_single_row(max_entry, deadline):
            """
            Backtracking search for a single row of k+1 elements starting at 0,
            using only differences in 'available_diffs' and staying <= max_entry.
            """
            row = [0]
            
            def backtrack(last, remaining):
                if time.time() > deadline:
                    return False
                if len(row) == k + 1:
                    return True
                
                min_x = last + 1
                max_x = max_entry - (remaining - 1)
                
                if min_x > max_x:
                    return False
                
                # Generate candidates
                # Sort by x ascending to find compact solutions first
                candidates = list(range(min_x, max_x + 1))
                
                # Add some randomness to avoid getting stuck in the same local minimum repeatedly
                # But bias towards smaller x.
                # We can shuffle a portion or use weighted random.
                # For now, simple shuffle.
                rng.shuffle(candidates)
                
                for x in candidates:
                    # Check if x can be added
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in available_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Also check internal consistency of the new row (diffs must be unique within the row too)
                    # The 'available_diffs' check handles global conflicts.
                    # We must ensure new_diffs don't conflict with each other or existing row diffs.
                    # Since we are building sequentially, existing row diffs are already in 'row'.
                    # We need to check if any d in new_diffs is already in the row's internal diffs?
                    # No, the row's internal diffs are determined by the elements.
                    # If we pick x, the new diffs are x - a for a in row.
                    # Are these distinct? Yes, because a are distinct.
                    # Do they conflict with existing diffs in the row?
                    # If d = x - a1 and d' = x - a2, d != d' since a1 != a2.
                    # What about d = x - a1 and existing diff d_old = b - a?
                    # This is handled by the global 'available_diffs' check?
                    # Wait. 'available_diffs' is (Global Used - Current Row Diffs).
                    # So if a diff was used in the current row, it's NOT in available_diffs.
                    # So if x - a1 equals a diff already in the current row, it will be in 'used' but removed from 'available_diffs'?
                    # No, 'available_diffs' = used - current_diffs.
                    # So if d is in current_diffs, it is NOT in available_diffs.
                    # So the check `if d in available_diffs` will fail if d is already in the current row.
                    # This correctly enforces that all diffs in the new row are distinct from each other and from other rows.
                    
                    row.append(x)
                    # Update available_diffs? No, we are just checking validity.
                    # We don't modify the global state during backtracking.
                    
                    if backtrack(x, remaining - 1):
                        return True
                    
                    row.pop()
                
                return False
            
            if backtrack(0, k):
                return row
            return None
        
        # We want to find a row with max entry < current max.
        # Let's try max_limit = current_max - 1, then -2, etc.
        # Or just try to build a row with max_entry = current_max - 1.
        # If that fails, we might try to optimize differently.
        # For now, try to build a row strictly smaller than the current max.
        
        current_max = max(current_row)
        
        # Try to build a row with max entry = current_max - 1
        # If successful, we have improved the global max (if this was the max row).
        
        # We need to be careful with time.
        # Let's allocate a small time slice for this rebuild.
        
        # Actually, a simpler strategy for local search:
        # Pick a random row.
        # Try to rebuild it with a max entry limit of current_global_max - 1.
        # If it succeeds, update the row.
        
        limit = get_max_entry(rows) - 1
        if limit < k: # Can't even fit k+1 elements
            return None
            
        # Try to build a row within 'limit'
        # We do a few random restarts
        for _ in range(10):
            new_row = build_single_row(limit, time.time() + 0.5)
            if new_row:
                return new_row
        
        return None

    # --- Main Logic ---
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Initial Greedy
    rows = greedy_construct(n_rows, k)
    current_max = get_max_entry(rows)
    
    # 2. Local Search
    # We will iteratively try to rebuild rows to reduce the global max.
    # We track the best solution found.
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Calculate used diffs for the current state
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                used.add(r[j] - r[i])
    
    rng = random.Random(42)
    
    # Local search loop
    # We try to reduce the max entry.
    # We pick a row that has the current max entry (or a random one) and try to rebuild it.
    
    iterations = 0
    max_iterations = 10000
    
    while time.time() < start_time + time_limit - 5:
        iterations += 1
        
        # Pick a row to rebuild
        # Prefer rows that have the maximum entry
        max_val = get_max_entry(rows)
        candidates = [i for i, r in enumerate(rows) if max(r) == max_val]
        if not candidates:
            candidates = list(range(n_rows))
            
        idx = rng.choice(candidates)
        
        # Try to rebuild
        new_row = rebuild_row(rows, idx, used, max_val, rng)
        
        if new_row is not None:
            # Update used set
            old_row = rows[idx]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            used = (used - old_diffs) | new_diffs
            
            # Update row
            rows[idx] = new_row
            
            # Check if global max improved
            new_global_max = get_max_entry(rows)
            if new_global_max < best_max:
                best_max = new_global_max
                best_rows = [r[:] for r in rows]
                
                # If we hit the lower bound, we can stop
                if best_max == lb:
                    break
        else:
            # If rebuild failed, maybe we should try a different row or accept a worse max?
            # For now, just continue.
            pass
            
    # 3. Final Validation
    if not validate(best_rows, n_rows, k):
        # Fallback to greedy if something went wrong
        best_rows = greedy_construct(n_rows, k)
        
    return best_rows

# EVOLVE-BLOCK-END
