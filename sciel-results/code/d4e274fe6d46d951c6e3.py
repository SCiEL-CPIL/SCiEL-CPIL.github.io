import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    # M(M+1) >= 2*total_diffs
    # M >= (-1 + sqrt(1 + 8*total_diffs)) / 2
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
        
    def compute_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Phase 1: Pure greedy
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope
    
    # Phase 2: Randomized greedy with restarts
    # For each restart, build rows one at a time with some randomness in candidate selection
    while time.time() < deadline - 15:  # Leave 15s for local search
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates in a window
                candidates = []
                x = current_max + 1
                # Search window: slightly larger than necessary to allow flexibility
                max_x = current_max + max(50, k * 8)  
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Pick randomly from candidates, but bias towards smaller values
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    # Weight inversely proportional to value, but not too aggressively
                    # to allow some exploration
                    weights = [1.0 / (c - current_max) for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break
                
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            if not success:
                break
            rows.append(row)
        
        if success and is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows

    # Phase 3: Local search on the best solution found
    # This is the critical part where we use the efficient diff-set update
    if best_rows is not None and time.time() < deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_diffs = compute_diffs(current_rows)
        
        # Temperature for simulated annealing (if we implement it)
        # For now, we'll use a greedy local search that only accepts improvements
        # or random moves to escape plateaus, but prioritize scope reduction.
        
        # We will try to modify elements to reduce the scope.
        # A move is valid if it doesn't create duplicate differences.
        
        def try_improve_scope(rows, diffs):
            """
            Try to reduce the scope by moving elements to smaller values.
            Returns (new_rows, new_diffs, new_scope) if an improvement is found, else (None, None, None).
            """
            # Try to modify each row
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                # Try to modify each element (except 0)
                for i_idx in range(1, len(row)):
                    current_val = row[i_idx]
                    # We want to try smaller values
                    # But we must maintain strict increasing order
                    min_val = row[i_idx - 1] + 1
                    max_val = current_val - 1 # Only try smaller values to reduce scope
                    
                    # If current_val is already minimal, we can't reduce it
                    if max_val < min_val:
                        continue
                        
                    # Try values in a random order, but bias towards smaller ones (which is the direction we want)
                    # Actually, let's just try a few random smaller values
                    # Or try the smallest valid value?
                    # The smallest valid value might be far away, but it gives the best scope reduction.
                    # Let's try to find the smallest valid value first.
                    
                    # Efficient check:
                    # For a candidate value v, the new diffs involving row[i_idx] are:
                    # v - row[j] for j < i_idx
                    # row[j] - v for j > i_idx
                    # We need to ensure these are not in `diffs` (excluding the old diffs involving row[i_idx])
                    
                    old_diffs_involving_i = set()
                    for j in range(len(row)):
                        if j == i_idx: continue
                        if j < i_idx:
                            old_diffs_involving_i.add(row[i_idx] - row[j])
                        else:
                            old_diffs_involving_i.add(row[j] - row[i_idx])
                    
                    other_diffs = diffs - old_diffs_involving_i
                    
                    # Try candidate values
                    # We want the smallest v > row[i_idx-1]
                    # Let's iterate v from min_val upwards
                    # But this could be slow if the gap is large.
                    # Let's try a limited range or random sampling.
                    # Given the time budget, let's try a few random candidates in [min_val, current_val)
                    
                    # If the range is small, try all
                    if current_val - min_val < 20:
                        candidates = range(min_val, current_val)
                    else:
                        # Sample some candidates
                        candidates = [random.randint(min_val, current_val - 1) for _ in range(10)]
                        candidates.append(min_val) # Always try the smallest possible
                    
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        if v < min_val or v >= current_val:
                            continue
                            
                        # Check if v is valid
                        valid = True
                        new_diffs_set = set()
                        for j in range(len(row)):
                            if j == i_idx: continue
                            if j < i_idx:
                                d = v - row[j]
                                new_diffs_set.add(d)
                            else:
                                d = row[j] - v
                                new_diffs_set.add(d)
                                
                        # Check for conflicts with other_diffs
                        if new_diffs_set & other_diffs:
                            continue
                            
                        # Also check internal conflicts within new_diffs_set?
                        # No, because the row is strictly increasing, differences within the row are distinct.
                        # Wait, are they?
                        # If row is 0, 2, 4, diffs are 2, 4, 2. No, 4-2=2, 2-0=2. Conflict!
                        # So we must ensure the row itself has distinct differences?
                        # The problem statement says: "all differences between two entries of the same row are distinct over all rows".
                        # This implies differences within a row must be distinct?
                        # "distinct over all rows" usually means the set of all differences from all rows is a set (no duplicates).
                        # If a row has duplicate differences, those differences are not distinct.
                        # So yes, each row must be a Sidon set (B_2 sequence) or similar, where all pairwise differences are distinct.
                        # My greedy construction ensures this because I check `new_diffs & used`.
                        # So `new_diffs_set` must also be a set of distinct values?
                        # Actually, if `v` is in a strictly increasing row, the differences `v - row[j]` for `j < i_idx` are distinct?
                        # Yes, because `row[j]` are distinct.
                        # And `row[j] - v` for `j > i_idx` are distinct?
                        # Yes, because `row[j]` are distinct.
                        # Are there conflicts between `v - row[j1]` and `row[j2] - v`?
                        # `v - row[j1] = row[j2] - v` => `2v = row[j1] + row[j2]`.
                        # This is possible. So we must check that `new_diffs_set` has no duplicates?
                        # No, `new_diffs_set` is a set, so duplicates are removed.
                        # But if `v - row[j1] == row[j2] - v`, then that difference appears twice in the row?
                        # No, it's just one difference value.
                        # The requirement is that the *set* of all differences across all rows has no duplicates.
                        # So if `v - row[j1] == row[j2] - v`, that's fine as long as that value isn't used elsewhere.
                        # Wait, if `v - row[j1] == row[j2] - v`, then the difference value `d` is generated by two pairs in the same row.
                        # Does that count as a duplicate?
                        # "all differences ... are distinct".
                        # If the difference 5 appears from (2,7) and (3,8), is that a duplicate?
                        # Yes, the set of differences would contain 5 only once, but the *multiset* of differences would have two 5s.
                        # The problem says "distinct". This usually implies a set.
                        # However, standard difference triangle problems (like Golomb rulers) require all pairwise differences to be distinct.
                        # If a row is not a Golomb ruler, it has duplicate differences.
                        # So yes, each row must be a Golomb ruler (or Sidon sequence).
                        # So `new_diffs_set` size must be equal to the number of pairs involving `i_idx`.
                        # Number of pairs involving `i_idx` is `(i_idx) + (len(row) - 1 - i_idx) = len(row) - 1`.
                        # So if `len(new_diffs_set) < len(row) - 1`, there's a duplicate within the row.
                        
                        if len(new_diffs_set) != len(row) - 1:
                            valid = False
                            continue
                            
                        if valid:
                            # Apply change
                            new_rows = [r[:] for r in rows]
                            new_rows[r_idx][i_idx] = v
                            
                            # Update diffs
                            new_diffs = other_diffs | new_diffs_set
                            
                            new_scope = get_scope(new_rows)
                            if new_scope < current_scope:
                                return new_rows, new_diffs, new_scope
                            
            return None, None, None

        steps = 0
        # Use a time-based loop
        while time.time() < deadline - 2:
            improved = False
            # Try to improve scope
            new_rows, new_diffs, new_scope = try_improve_scope(current_rows, current_diffs)
            if new_rows is not None:
                current_rows = new_rows
                current_diffs = new_diffs
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    if best_scope <= lower_bound:
                        return best_rows
                improved = True
                
            if not improved:
                # If we can't improve scope, maybe we can make a random move to escape a local minimum?
                # Or just break if we've been stuck for a while?
                # Given the time constraint, let's just break if we fail to improve for a while?
                # Or try a different strategy: swap elements?
                # For now, let's just exit the local search if we can't improve scope.
                # The randomized greedy in Phase 2 provides diversity.
                break
            steps += 1

    return best_rows
