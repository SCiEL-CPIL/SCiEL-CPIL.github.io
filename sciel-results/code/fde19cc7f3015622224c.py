import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Helper to get all differences in a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_global(rows):
        all_diffs = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # 1. Initial Construction: Robust Greedy with Randomization
    
    def gen_row(used_diffs, force_greedy=False):
        """
        Generates a row of k+1 integers starting at 0.
        Avoids conflicts with used_diffs.
        """
        row = [0]
        local_diffs = set()
        
        for _ in range(k):
            base = row[-1]
            valid_candidates = []
            
            # Scan for valid candidates
            # We look ahead a bit to find options
            limit = base + 200 
            for v in range(base + 1, limit):
                new_diffs = {v - a for a in row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & local_diffs:
                    continue
                valid_candidates.append(v)
                if len(valid_candidates) >= 10:
                    break
            
            if not valid_candidates:
                # Fallback greedy: find the very next valid one
                v = base + 1
                while True:
                    new_diffs = {v - a for a in row}
                    if not (new_diffs & used_diffs) and not (new_diffs & local_diffs):
                        break
                    v += 1
                valid_candidates = [v]
            
            # Selection strategy
            if force_greedy:
                chosen_val = valid_candidates[0]
            else:
                # Bias towards smaller values, but allow some randomness
                # Pick from the first few valid candidates
                pick_idx = random.randint(0, min(4, len(valid_candidates)-1))
                chosen_val = valid_candidates[pick_idx]
            
            row.append(chosen_val)
            # Update local_diffs
            for a in row[:-1]:
                local_diffs.add(chosen_val - a)
                
        return row

    def try_compress_row(row, forbidden_diffs):
        """
        Attempts to reduce the maximum value of `row` by greedily replacing elements
        from the end to the start with smaller valid values.
        Returns a new row if successful in reducing max, or the original row if not.
        """
        # Work on a copy
        new_row = row[:]
        improved = False
        
        # To compress effectively, we should try to reduce the largest element first.
        # However, changing an earlier element might allow later elements to be smaller too.
        # A simple heuristic: Try to reduce the last element. If successful, try the second last, etc.
        # But we must ensure the row remains strictly increasing and valid.
        
        # A more robust approach: 
        # For each position i from k down to 1:
        #   Try to find a value v < new_row[i] such that v > new_row[i-1]
        #   and the new diffs introduced by v are valid against forbidden_diffs 
        #   and against the local diffs of the rest of the row (which might change if we modify later elements? 
        #   Actually, if we process from right to left, the "later" elements are already fixed. 
        #   Wait, if we change element i, it affects diffs with j > i. 
        #   So we must re-validate the entire suffix? Or just check if the new diffs collide with 
        #   diffs from (0..i-1) and (i+1..k) and forbidden.
        
        # Let's simplify: 
        # 1. Calculate current local diffs.
        # 2. Try to reduce the last element. 
        #    - Remove its diffs from local_diffs.
        #    - Find smallest v > new_row[-2] such that new diffs (v - x for x in new_row[:-1]) are valid 
        #      against forbidden_diffs and the remaining local_diffs.
        #    - If found v < old_last, update row and local_diffs.
        # 3. Repeat for second last, etc.
        
        # This is O(k^3) roughly, but k is small (6).
        
        for i in range(k, 0, -1):
            old_val = new_row[i]
            # Remove diffs involving old_val
            # We need to be careful with local_diffs maintenance.
            # Let's just recompute local_diffs for the prefix new_row[:i] 
            # and check candidates for position i against that + forbidden.
            
            prefix = new_row[:i]
            prefix_diffs = set()
            for a_idx in range(len(prefix)):
                for b_idx in range(a_idx + 1, len(prefix)):
                    prefix_diffs.add(prefix[b_idx] - prefix[a_idx])
            
            # Find smallest v > prefix[-1]
            best_v = None
            # We want v < old_val to improve.
            # Scan from prefix[-1] + 1 up to old_val - 1
            for v in range(prefix[-1] + 1, old_val):
                new_diffs = {v - a for a in prefix}
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & prefix_diffs:
                    continue
                # Also need to check against diffs with elements AFTER i?
                # If we change new_row[i], it affects diffs with new_row[i+1..k].
                # Those diffs are (new_row[j] - v) for j > i.
                # We must ensure these are also valid.
                valid = True
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - v
                    if d in forbidden_diffs:
                        valid = False
                        break
                    # Check against prefix_diffs? No, d is diff between i and j.
                    # Check against other diffs involving i or j?
                    # The set of all diffs in the row must be unique.
                    # The diffs involving i are:
                    # 1. v - prefix[a] (checked above)
                    # 2. new_row[j] - v (checking now)
                    # We must ensure these new diffs don't collide with:
                    # - forbidden_diffs
                    # - prefix_diffs (diffs within 0..i-1)
                    # - diffs within i+1..k (suffix_diffs)
                    # - other diffs involving i (already being built)
                    
                    # This is getting complex. 
                    # Simpler check: 
                    # Construct the full candidate row with v at i.
                    # Check if ALL diffs in this candidate row are valid against forbidden_diffs.
                    # And check if all diffs in candidate row are unique.
                    
                    # Since k is small, let's just build the candidate row and check validity.
                    # But doing this for every v in range is slow.
                    # However, range is small (old_val - prefix[-1]).
                    
                    # Let's optimize: 
                    # We know prefix_diffs are valid.
                    # We know suffix_diffs (diffs within i+1..k) are valid (from original row).
                    # We just need to check:
                    # A. New diffs (v - prefix) vs forbidden, prefix_diffs, suffix_diffs, and each other.
                    # B. New diffs (suffix - v) vs forbidden, prefix_diffs, suffix_diffs, and each other.
                    # C. New diffs (v - prefix) vs New diffs (suffix - v).
                    
                    # Let's precompute suffix_diffs for the current new_row[i+1:]
                    # But this depends on i.
                    
                    # Given the small size, let's just do a full check for the candidate row.
                    # We can break early if we find a collision.
                    
                    candidate_row = new_row[:i] + [v] + new_row[i+1:]
                    # Check uniqueness and forbidden
                    all_diffs = set()
                    conflict = False
                    for a_idx in range(len(candidate_row)):
                        for b_idx in range(a_idx + 1, len(candidate_row)):
                            d = candidate_row[b_idx] - candidate_row[a_idx]
                            if d in forbidden_diffs:
                                conflict = True
                                break
                            if d in all_diffs:
                                conflict = True
                                break
                            all_diffs.add(d)
                    if conflict:
                        valid = False
                        break
                    if not valid:
                        continue
                if valid:
                    best_v = v
                    break
            
            if best_v is not None:
                new_row[i] = best_v
                improved = True
                # If we improved, we might be able to improve earlier indices further?
                # The loop continues to i-1, so it will try.
        
        if improved:
            return new_row
        return row

    # Initial set of rows
    # Use a mix of greedy and randomized
    used = set()
    rows = []
    for i in range(n_rows):
        # First row: pure greedy (smallest possible)
        if i == 0:
            row = gen_row(used, force_greedy=True)
        else:
            # Subsequent rows: randomized
            row = gen_row(used, force_greedy=False)
        
        # Update used
        row_diffs = get_row_diffs(row)
        used |= row_diffs
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in best_rows)
    best_diffs = set(used)
    
    # 2. Local Search with Row Replacement and Compression
    current_rows = [r[:] for r in best_rows]
    current_max = best_score
    
    # Precompute diffs for each row
    current_row_diffs = [get_row_diffs(r) for r in current_rows]
    
    t = 10.0
    cooling = 0.999
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if t < 0.1:
            t = 5.0
            no_improve_count = 0
            
        # Pick a row to modify
        idx = random.randrange(n_rows)
        
        # Calculate the set of "forbidden" diffs: all diffs from other rows
        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= current_row_diffs[i]
        
        # Strategy: 
        # 1. Try to compress the existing row.
        # 2. If that fails or doesn't improve, generate a new random row.
        
        new_row = current_rows[idx][:]
        new_row_diffs = get_row_diffs(new_row)
        
        # Attempt compression
        compressed_row = try_compress_row(new_row, forbidden)
        if compressed_row != new_row:
            new_row = compressed_row
            new_row_diffs = get_row_diffs(new_row)
            # Verify validity (compression should preserve it, but double check)
            if not (new_row_diffs & forbidden):
                # Calculate new global max
                temp_rows = current_rows[:]
                temp_rows[idx] = new_row
                new_max = max(max(r) for r in temp_rows)
                
                if new_max < current_max:
                    current_rows[idx] = new_row
                    current_row_diffs[idx] = new_row_diffs
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_rows = [r[:] for r in current_rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    t *= cooling
                    continue
                elif new_max == current_max:
                    if random.random() < 0.5: # Accept neutral compression
                        current_rows[idx] = new_row
                        current_row_diffs[idx] = new_row_diffs
                        no_improve_count += 1
                        t *= cooling
                        continue
        
        # If compression didn't help enough, try generating a new row
        # Use greedy with some randomness
        gen_type = random.random()
        if gen_type < 0.3:
            new_row = gen_row(forbidden, force_greedy=True)
        else:
            new_row = gen_row(forbidden, force_greedy=False)
            
        new_row_diffs = get_row_diffs(new_row)
        
        if new_row_diffs & forbidden:
            t *= cooling
            continue
            
        temp_rows = current_rows[:]
        temp_rows[idx] = new_row
        new_max = max(max(r) for r in temp_rows)
        
        if new_max < current_max:
            current_rows[idx] = new_row
            current_row_diffs[idx] = new_row_diffs
            current_max = new_max
            if current_max < best_score:
                best_score = current_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif new_max == current_max:
            if random.random() < 0.2:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
                no_improve_count += 1
        else:
            delta = new_max - current_max
            prob = pow(2.71828, -delta / t)
            if random.random() < prob:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
                current_max = new_max
                no_improve_count += 1
        
        t *= cooling
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Try a few random restarts or perturbations
            # For now, just reset temperature and counter
            t = 10.0
            no_improve_count = 0
            
    return best_rows
# EVOLVE-BLOCK-END
