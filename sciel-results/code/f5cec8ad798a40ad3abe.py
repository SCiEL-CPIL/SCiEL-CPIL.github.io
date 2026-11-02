import time
import random
import copy

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def get_diffs(row):
        """Calculate the set of positive differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_init(seed=None):
        """
        Generates an initial set of rows using a randomized greedy approach.
        """
        if seed is not None:
            random.seed(seed)
        
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try to find the next x
                # Standard greedy picks smallest. Randomized picks from a few smallest.
                candidates = []
                # We don't know the upper bound, so we increment x.
                # To speed up, we can just iterate x.
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        # Found a valid x
                        # With some probability, pick a larger x to diversify?
                        # No, for init, we want valid rows.
                        # Let's just pick the first one (greedy) to ensure validity and small scope.
                        # Randomization in init can be done by shuffling the order of checking?
                        # No, x is ordered.
                        # Let's add slight randomness: with prob 0.1, skip the first valid x.
                        if random.random() < 0.1 and x + 1 <= 1000: # Arbitrary limit
                             # Check if x+1 is also valid
                             new_diffs_next = {x + 1 - a for a in row}
                             if not new_diffs_next & used:
                                 x += 1
                                 new_diffs = new_diffs_next
                        row.append(x)
                        used |= new_diffs
                        break
            rows.append(row)
        return rows, used

    def randomized_greedy_row(k, limit, used_diffs, seed=None):
        """
        Tries to construct a row of length k+1 starting at 0, with max value <= limit,
        such that its diffs are disjoint from used_diffs.
        Returns the row or None.
        """
        if seed is not None:
            random.seed(seed)
            
        row = [0]
        # We need to fill k more elements.
        for _ in range(k):
            # Find valid candidates for the next element
            # The next element must be > row[-1] and <= limit
            valid_candidates = []
            last_val = row[-1]
            # We iterate x from last_val + 1 to limit
            # Optimization: If limit is large, this is slow. But k is small (e.g., 7),
            # and we are trying to *reduce* the limit, so limit is usually small (< 200).
            for x in range(last_val + 1, limit + 1):
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    valid_candidates.append(x)
                    # We can stop early if we have enough candidates for randomization
                    if len(valid_candidates) >= 10:
                        break
            
            if not valid_candidates:
                return None
            
            # Choose a candidate
            # Greedy: smallest. Randomized: pick from first few.
            if random.random() < 0.7:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)
            
            row.append(x)
            # Note: We do NOT update used_diffs here because we are just checking
            # if *this* row is valid against the *other* rows.
            # However, diffs within the row must also be distinct?
            # The problem says "all differences between two entries of the same row are distinct over all rows".
            # This implies diffs within a row are distinct (which is true for increasing integers)
            # AND diffs from different rows are distinct.
            # It does NOT require that diffs within the same row are distinct from each other?
            # "all differences ... are distinct over all rows"
            # Usually DTS definition: The set of differences from ALL rows is a set (no duplicates).
            # So, if row 1 has diff 5, row 2 cannot have diff 5.
            # Within row 1, can we have diff 5 twice?
            # Row: 0, 5, 10. Diffs: 5, 10, 5.
            # If "distinct over all rows" means the global set of differences has size sum(k_i), then no duplicates allowed anywhere.
            # The seed program uses `used` set which accumulates all diffs.
            # `if not new & used` ensures the new diffs don't conflict with ANY previous diffs, including those in the same row?
            # In seed: `new = {x - a for a in row}`. `row` contains previous elements.
            # So `new` contains diffs from x to all previous elements.
            # `used` contains diffs from all previous rows AND previous elements of current row?
            # Seed: `used |= new` inside the while loop.
            # So yes, the seed enforces that ALL differences generated so far (across all rows and within the current row) are unique.
            # So my `randomized_greedy_row` must also ensure that the new diffs don't conflict with diffs *within the same row* that were already generated.
            # Wait, `used_diffs` passed to `randomized_greedy_row` is the set of diffs from OTHER rows.
            # I need to also track diffs within the current row being built.
            
            # Let's refine:
            # The function should check against `used_diffs` (external) AND `internal_diffs` (within current row).
            
            # Let's restart the logic for this function.
            
        return None # Placeholder, will rewrite

    def randomized_greedy_row_v2(k, limit, used_diffs_external):
        """
        Constructs a row of length k+1, max <= limit, disjoint from used_diffs_external.
        Also ensures internal diffs are distinct (which is automatic for increasing ints if we check against internal set).
        """
        row = [0]
        internal_diffs = set()
        
        for _ in range(k):
            last_val = row[-1]
            valid_candidates = []
            
            # Iterate x
            # To make it faster, we can jump? No, k is small.
            for x in range(last_val + 1, limit + 1):
                new_diffs = {x - a for a in row}
                # Check against external
                if new_diffs & used_diffs_external:
                    continue
                # Check against internal
                if new_diffs & internal_diffs:
                    continue
                
                valid_candidates.append(x)
                if len(valid_candidates) >= 10:
                    break
            
            if not valid_candidates:
                return None
            
            # Select x
            if random.random() < 0.7:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)
            
            row.append(x)
            # Update internal diffs
            for a in row[:-1]: # All previous elements
                internal_diffs.add(x - a)
                
        return row

    # 1. Initial Solution
    best_rows, best_used = greedy_init(seed=42)
    best_scope = max(max(r) for r in best_rows)
    
    current_rows = copy.deepcopy(best_rows)
    current_used = set(best_used)
    
    # Pre-compute diffs for current rows for quick removal
    current_row_diffs = [get_diffs(r) for r in current_rows]
    
    # 2. Local Search
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # Try to improve each row
        # Shuffle order to be random
        indices = list(range(n_rows))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            r = current_rows[i]
            M_i = r[-1]
            
            # We only want to try to reduce the scope if this row contributes to the current max
            # Or just try to reduce any row.
            # If we reduce a row that isn't the max, it doesn't help the global scope immediately,
            # but it might unblock other rows.
            
            # Target: Find a new row with max < M_i
            target_limit = M_i - 1
            
            # If target_limit is too small to fit k+1 integers (0..k), skip
            if target_limit < k:
                continue
                
            # Remove current row's diffs from global set
            current_used -= current_row_diffs[i]
            
            # Try to find a better row
            new_row = randomized_greedy_row_v2(k, target_limit, current_used)
            
            if new_row is not None:
                new_diffs = get_diffs(new_row)
                # It should be disjoint by construction, but double check
                if not (new_diffs & current_used):
                    # Accept
                    current_rows[i] = new_row
                    current_row_diffs[i] = new_diffs
                    current_used |= new_diffs
                    
                    # Check global scope
                    new_scope = max(max(r) for r in current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = copy.deepcopy(current_rows)
                        improved = True
                    else:
                        # Even if global scope didn't drop, we improved a local row.
                        # This is good for exploration.
                        pass
            else:
                # Failed to find a smaller row, put old diffs back
                current_used |= current_row_diffs[i]
            
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Restart if stuck for a while
        if no_improve_count > 20:
            # Re-initialize with a new random seed
            new_rows, new_used = greedy_init(seed=random.randint(0, 10000))
            new_scope = max(max(r) for r in new_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = copy.deepcopy(new_rows)
                current_rows = copy.deepcopy(new_rows)
                current_used = set(new_used)
                current_row_diffs = [get_diffs(r) for r in current_rows]
                no_improve_count = 0
            else:
                # Keep current state, just reset counter to allow more exploration?
                # Or force a restart anyway to escape local minimum?
                # Let's just reset counter.
                no_improve_count = 0

    return best_rows
