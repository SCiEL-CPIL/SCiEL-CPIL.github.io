import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Total number of differences required
    # Each row has k+1 elements, so k*(k+1)/2 differences per row.
    # Total diffs = n_rows * k * (k+1) // 2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # The minimum possible scope is at least total_diffs + 1 (since diffs are 1..scope)
    # Actually, diffs are distinct, so the largest diff must be at least total_diffs.
    # The largest entry in a row is at least the largest diff in that row.
    # So scope >= total_diffs.
    
    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_diffs_mask(rows):
        """Compute a bitset of all differences used."""
        mask = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
        return mask

    def is_valid_move(rows, row_idx, elem_idx, new_val, used_mask):
        """
        Check if moving rows[row_idx][elem_idx] to new_val is valid.
        Assumes the element is removed from 'used_mask' logic externally or we check against the rest.
        Here we check if new_val creates any conflict with the current state 
        where the element at elem_idx is considered 'removed' for the purpose of 
        checking against other elements in the same row and all other rows.
        
        Actually, it's easier to:
        1. Remove the differences involving the old element.
        2. Check if the new differences involving new_val are free.
        3. If yes, apply. If no, revert.
        """
        row = rows[row_idx]
        old_val = row[elem_idx]
        
        # Calculate differences that are "freed" by removing old_val
        # and differences that are "added" by placing new_val.
        
        # We need to check if any of the new differences are in the used_mask 
        # (excluding the differences that were created by old_val itself).
        
        # Let's construct the set of differences involving old_val in this row
        old_diffs = set()
        for i in range(len(row)):
            if i == elem_idx: continue
            d = abs(row[i] - old_val)
            old_diffs.add(d)
            
        # The mask of all differences currently in use
        # We want to check if new_diffs intersect (used_mask & ~old_diffs_mask)
        
        # To be safe and simple, let's just recompute the mask for the whole set 
        # excluding this element, then check new diffs. 
        # But that's O(N^2). 
        # Let's use the mask approach.
        
        # Mask of differences to ignore (those created by old_val)
        ignore_mask = 0
        for d in old_diffs:
            ignore_mask |= (1 << d)
        
        # Effective used mask = used_mask & ~ignore_mask
        effective_used = used_mask & ~ignore_mask
        
        # Check new differences
        new_diffs = []
        for i in range(len(row)):
            if i == elem_idx: continue
            d = abs(row[i] - new_val)
            new_diffs.append(d)
            # Check if d is in effective_used
            if effective_used & (1 << d):
                return False
            # Also check if new_diffs have duplicates among themselves?
            # No, new_val is distinct from row[i], so diffs are distinct from 0.
            # But can two different i's produce the same d?
            # If |row[i1] - new_val| == |row[i2] - new_val|, then row[i1] == row[i2] or row[i1]+row[i2] == 2*new_val.
            # Since row is strictly increasing, row[i1] != row[i2].
            # If row[i1] < new_val < row[i2], then d1 = new_val - row[i1], d2 = row[i2] - new_val.
            # d1 == d2 => new_val - row[i1] == row[i2] - new_val => 2*new_val = row[i1] + row[i2].
            # This is possible. We need to check for internal duplicates in new_diffs too.
            
        # Check for internal duplicates in new_diffs
        if len(new_diffs) != len(set(new_diffs)):
            return False
            
        return True

    def greedy_construct(seed):
        rng = random.Random(seed)
        rows = [[0] for _ in range(n_rows)]
        used_mask = 0
        
        # We build rows one by one.
        # To allow symmetry breaking, we can shuffle the row order in which we build,
        # but since they are independent until the mask fills up, it's just the order.
        
        for r_idx in range(n_rows):
            row = rows[r_idx]
            for e_idx in range(1, k + 1):
                last = row[-1]
                found = False
                # Search for next element
                # Heuristic: try small values first, but with some randomness
                # Limit search space to avoid infinite loop
                max_search = last + 100 + rng.randint(0, 50)
                
                # Randomize start slightly? No, must be > last.
                candidates = list(range(last + 1, max_search))
                if rng.random() < 0.5:
                    rng.shuffle(candidates)
                
                for x in candidates:
                    # Check validity
                    valid = True
                    new_diffs = []
                    for prev_val in row:
                        d = x - prev_val
                        if used_mask & (1 << d):
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Check internal duplicates in new_diffs
                    if len(new_diffs) != len(set(new_diffs)):
                        continue
                        
                    # Accept
                    row.append(x)
                    for d in new_diffs:
                        used_mask |= (1 << d)
                    found = True
                    break
                
                if not found:
                    # If greedy fails, try a wider search or return None
                    # For now, return None to trigger restart
                    return None
                    
        return rows

    def local_search(rows, seed, max_iters=5000):
        rng = random.Random(seed)
        current_rows = [list(r) for r in rows]
        current_scope = get_scope(current_rows)
        used_mask = compute_diffs_mask(current_rows)
        
        best_rows = [list(r) for r in current_rows]
        best_scope = current_scope
        
        # Simulated Annealing parameters
        T = 1.0
        cooling_rate = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random row and a random element (index > 0)
            r_idx = rng.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            e_idx = rng.randint(1, len(row) - 1)
            old_val = row[e_idx]
            
            # Determine range for new value
            # Must be between previous element and next element
            min_val = row[e_idx - 1] + 1
            max_val = row[e_idx + 1] - 1 if e_idx + 1 < len(row) else old_val - 1
            
            if min_val > max_val:
                continue
            
            # Try a random new value in range
            # Bias towards smaller values to reduce scope
            new_val = rng.randint(min_val, max_val)
            
            if new_val == old_val:
                continue
            
            # Check validity
            # Remove old diffs from mask
            ignore_mask = 0
            for prev_val in row:
                if prev_val == old_val: continue
                d = abs(prev_val - old_val)
                ignore_mask |= (1 << d)
            
            effective_used = used_mask & ~ignore_mask
            
            valid = True
            new_diffs = []
            for prev_val in row:
                if prev_val == old_val: continue
                d = abs(prev_val - new_val)
                if effective_used & (1 << d):
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            if len(new_diffs) != len(set(new_diffs)):
                continue
            
            # Valid move. Calculate new scope
            # Update mask
            # Remove old, add new
            # Actually, we just swap in the row and recompute mask if accepted?
            # Or update mask incrementally.
            
            # Incremental update:
            # Remove old diffs
            for d in [abs(prev_val - old_val) for prev_val in row if prev_val != old_val]:
                used_mask &= ~(1 << d)
            
            # Add new diffs
            for d in new_diffs:
                used_mask |= (1 << d)
            
            # Apply move
            row[e_idx] = new_val
            
            new_scope = get_scope(current_rows)
            
            # Acceptance criterion
            if new_scope < current_scope:
                # Always accept improvement
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Simulated Annealing: accept with probability exp(-delta/T)
                delta = new_scope - current_scope
                if delta == 0:
                    # Neutral move, accept with 0.5 prob
                    if rng.random() < 0.5:
                        pass
                    else:
                        # Revert
                        row[e_idx] = old_val
                        for d in new_diffs:
                            used_mask &= ~(1 << d)
                        for prev_val in row:
                            if prev_val == old_val: continue
                            d = abs(prev_val - old_val)
                            used_mask |= (1 << d)
                else:
                    prob = pow(2.71828, -delta / T)
                    if rng.random() < prob:
                        current_scope = new_scope
                    else:
                        # Revert
                        row[e_idx] = old_val
                        for d in new_diffs:
                            used_mask &= ~(1 << d)
                        for prev_val in row:
                            if prev_val == old_val: continue
                            d = abs(prev_val - old_val)
                            used_mask |= (1 << d)
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        return best_rows, best_scope

    # Main Loop
    best_rows = None
    best_scope = float('inf')
    
    # 1. Try to get a valid solution via greedy
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        rows = greedy_construct(seed)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
            if best_scope <= 112: # Theoretical lower bound
                break

    if best_rows is None:
        # If greedy failed, we might need a more robust initial construction
        # But for 4,7 it should succeed.
        return None

    # 2. Local Search
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        improved_rows, improved_scope = local_search(best_rows, seed + 1000, max_iters=2000)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    return best_rows

if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
