"""
Strategy: Structured Greedy Construction + Simulated Annealing Optimization

1. Construction: Instead of random sampling, we build rows sequentially.
   For each new element, we pick the smallest integer > previous element 
   that satisfies all difference constraints (both internal to the row and 
   global across rows). This "best-fit" approach packs elements tightly 
   at the start, leaving larger gaps for later elements or other rows.
   
2. Optimization: Simulated Annealing (SA) on the full configuration.
   - Objective: Minimize the sum of all elements (surrogate for max/scope).
     Minimizing the sum tends to compress the entire structure, often 
     reducing the max as well.
   - Moves: 
     a) Move a single element to a new valid position.
     b) Swap two elements from different rows (if valid).
     c) "Jumps" to escape local minima.
   - Cooling: Geometric cooling schedule.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return set of all positive differences within a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs

def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _scope(rows):
    return max(r[-1] for r in rows)

def _sum_scope(rows):
    """Sum of all elements, used as SA objective."""
    return sum(sum(r) for r in rows)

def _greedy_construct(n_rows, num_elements, max_scope, rng):
    """
    Construct rows sequentially.
    For each row, pick the smallest possible value for each position 
    that doesn't conflict with existing differences.
    """
    used_mask = 0  # Bitmask of used differences
    rows = []
    
    for r_idx in range(n_rows):
        row = [0]
        current_max = 0
        row_diffs = set()
        
        for pos in range(1, num_elements):
            # Try to find the smallest value v > current_max such that:
            # 1. All new diffs (v - u for u in row) are not in row_diffs
            # 2. All new diffs are not in used_mask
            found = False
            # We search up to max_scope. 
            # Optimization: We can start from current_max + 1.
            # Since we want to pack tightly, we check sequentially.
            # However, checking every integer up to max_scope might be slow if max_scope is large.
            # But for (7,5), max_scope ~112, num_elements=6. 
            # 5 positions * 112 checks * 7 rows * 15 diffs/check = manageable.
            
            # To speed up, we can pre-compute available values? 
            # Let's just iterate.
            
            # Heuristic: Sometimes picking the absolute smallest causes issues later.
            # We'll stick to smallest for now, as it's a "best-fit" packing.
            
            # To avoid O(N^2) in the worst case, we limit the search range slightly 
            # or rely on the fact that valid spots exist.
            
            # Actually, a pure "smallest" greedy can get stuck. 
            # Let's add a small randomization: try the smallest, if it fails, 
            # try a few random candidates from the feasible set?
            # No, let's stick to deterministic greedy first, then SA will fix it.
            
            # Wait, if we just pick the smallest, we might fill up small diffs 
            # too quickly, leaving no room for large diffs needed for large gaps.
            # The "smallest" heuristic is good for minimizing the sum of the row, 
            # but might be bad for global packing.
            
            # Let's try a hybrid: 
            # 1. Find all valid values in [current_max+1, max_scope].
            # 2. Pick the one that minimizes the "fragmentation" or just the smallest.
            
            # For simplicity and speed in the constructor, let's use the smallest valid value.
            # If we get stuck (no valid value), we return None.
            
            start_val = current_max + 1
            # We need to ensure we don't go beyond max_scope
            for v in range(start_val, max_scope + 1):
                # Check internal constraints
                new_diffs = {v - u for u in row}
                if new_diffs & row_diffs:
                    continue
                
                # Check global constraints
                # Convert new_diffs to mask? 
                # Since max_scope is small (< 200), we can use a set for used_diffs 
                # or a bitmask. Bitmask is faster for intersection.
                
                # Let's use a set for used_diffs in the constructor for clarity, 
                # but bitmask in SA for speed.
                # Actually, let's stick to bitmask for consistency.
                
                # Mask check:
                # m = 0
                # for d in new_diffs: m |= (1 << d)
                # if m & used_mask: continue
                
                # For speed, let's pre-calculate mask for v? No, v changes.
                
                conflict = False
                for d in new_diffs:
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    if d in row_diffs: # Already checked above, but safe
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Valid!
                row.append(v)
                row_diffs |= new_diffs
                for d in new_diffs:
                    used_mask |= (1 << d)
                current_max = v
                found = True
                break
            
            if not found:
                return None
        
        rows.append(row)
    
    return rows

def _simulated_annealing(rows, num_elements, max_scope, rng, time_limit):
    """
    Simulated Annealing to minimize the sum of elements (and thus scope).
    """
    start_time = time.time()
    current_rows = [r[:] for r in rows]
    current_cost = _sum_scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_cost = current_cost
    best_scope = _scope(current_rows)
    
    # Pre-compute row masks and global mask for fast validity checks
    row_masks = [_row_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m
    
    # Temperature schedule
    T = 50.0
    T_min = 0.1
    alpha = 0.995 # Cooling rate
    
    n_rows = len(current_rows)
    
    def get_mask(row):
        m = 0
        for d in _row_diffs(row):
            m |= (1 << d)
        return m

    def is_move_valid(row_idx, pos, new_val):
        """Check if setting row_idx[pos] to new_val is valid."""
        row = current_rows[row_idx]
        
        # Bounds
        if new_val <= row[pos-1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos+1]:
            return False
        
        # Internal diffs
        new_diffs = set()
        # Diffs involving new_val
        for i in range(pos):
            d = new_val - row[i]
            if d in new_diffs: return False
            new_diffs.add(d)
        for j in range(pos+1, len(row)):
            d = row[j] - new_val
            if d in new_diffs: return False
            new_diffs.add(d)
        
        # Diffs not involving new_val remain same, so we only need to check 
        # if new_diffs intersects with:
        # 1. Old diffs of this row (excluding those involving old_val)
        # 2. Global diffs of other rows
        
        # This is getting complex to compute "old diffs excluding old_val" quickly.
        # Simpler: Reconstruct the row, get all diffs, check set intersection.
        # Given small k=5, this is fast enough.
        
        test_row = row[:]
        test_row[pos] = new_val
        test_diffs = _row_diffs(test_row)
        
        # Check internal distinctness
        if len(test_diffs) != len(list(test_diffs)): # set is already unique, need to check count vs list
             # Actually _row_diffs returns a set. 
             # We need to ensure no duplicates in the generation process? 
             # No, if it's a set, it's unique. 
             # But we need to ensure that the *generation* didn't have duplicates?
             # If a - b == c - d, they are the same value in the set. 
             # The requirement is that all differences are distinct *values*.
             # So a set is the correct representation.
             # The constraint is just that the set size matches the number of pairs?
             # No, if a-b == c-d, then the set has 1 element for 2 pairs.
             # The problem says "all differences ... are distinct".
             # This implies a-b != c-d for any distinct pairs.
             # So yes, the size of the set of differences must be k(k-1)/2.
             
             num_pairs = len(test_row) * (len(test_row)-1) // 2
             if len(test_diffs) != num_pairs:
                 return False
        
        # Check global conflict
        # Get mask of test_row
        test_mask = get_mask(test_row)
        
        # Get mask of other rows
        other_mask = global_mask & ~row_masks[row_idx]
        
        if test_mask & other_mask:
            return False
        
        return True

    def apply_move(row_idx, pos, new_val):
        old_val = current_rows[row_idx][pos]
        current_rows[row_idx][pos] = new_val
        new_mask = get_mask(current_rows[row_idx])
        global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
        row_masks[row_idx] = new_mask
        return old_val

    def revert_move(row_idx, pos, old_val):
        current_rows[row_idx][pos] = old_val
        new_mask = get_mask(current_rows[row_idx])
        global_mask = (global_mask & ~row_masks[row_idx]) | new_mask
        row_masks[row_idx] = new_mask

    iterations = 0
    while T > T_min:
        if time.time() > time_limit:
            break
        
        # Pick a random move
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, len(current_rows[row_idx]) - 1)
        
        # Determine range for new_val
        low = current_rows[row_idx][pos - 1] + 1
        high = current_rows[row_idx][pos + 1] - 1 if pos < len(current_rows[row_idx]) - 1 else max_scope
        
        if low > high:
            T *= alpha
            continue
            
        # Bias towards smaller values to reduce sum
        # Sample from [low, high] but with a bias
        if rng.random() < 0.8:
            # Sample in lower half
            mid = (low + high) // 2
            new_val = rng.randint(low, min(high, mid))
        else:
            new_val = rng.randint(low, high)
            
        if new_val == current_rows[row_idx][pos]:
            T *= alpha
            continue
            
        if not is_move_valid(row_idx, pos, new_val):
            T *= alpha
            continue
            
        # Calculate cost change
        old_val = current_rows[row_idx][pos]
        delta_cost = new_val - old_val
        
        # Acceptance probability
        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
            apply_move(row_idx, pos, new_val)
            current_cost += delta_cost
            
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                best_cost = current_cost
                # Update max_scope for future moves? 
                # We can shrink max_scope to best_scope to force compression
                # But careful: if we shrink max_scope, we might invalidate current state?
                # No, current state is valid within best_scope.
                # However, moves that go above best_scope are now disallowed.
                # This is a good way to ratchet down the scope.
                max_scope = best_scope
        
        T *= alpha
        iterations += 1
        
    return best_rows, best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    import math
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    # Start with a scope that is likely feasible
    # Best known for (7,5) is 112. Counting bound is ~105.
    # Let's start from counting_bound + small margin.
    initial_scope = max(counting_bound, 100)
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy Construction
    # Try to construct with increasing scope until we find a valid one.
    for scope in range(initial_scope, 200):
        if time.time() > DEADLINE - 30:
            break
        rows = _greedy_construct(n_rows, num_elements, scope, rng)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            # If we found a solution, we can break and let SA optimize.
            # But maybe a larger scope greedy gives a better starting point for SA?
            # Usually, tighter is better for SA.
            break
    
    if best_rows is None:
        # Fallback: very loose construction
        for scope in range(200, 400):
            rows = _greedy_construct(n_rows, num_elements, scope, rng)
            if rows is not None:
                best_rows = rows
                best_scope = _scope(rows)
                break
        
        if best_rows is None:
            # Last resort: sequential simple greedy
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            best_rows = rows
            best_scope = _scope(rows)
            
    # Phase 2: Simulated Annealing
    # Run SA to minimize sum/scope
    while time.time() < DEADLINE - 10:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, best_scope, rng,
            time_limit=DEADLINE - 15
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
        else:
            # If no improvement, break? 
            # Or try a different random seed?
            # Let's just break if we hit the counting bound.
            if best_scope <= counting_bound:
                break
            # If we are stuck, we might try to restart SA with a higher temp?
            # But the function handles its own temp.
            # Let's just break to save time for validation.
            break

    # Phase 3: Final Validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
