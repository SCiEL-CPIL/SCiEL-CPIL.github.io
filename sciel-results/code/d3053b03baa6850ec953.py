import time
import random
import copy

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0,
    minimizing the maximum entry (scope) such that all pairwise differences
    are distinct across all rows.
    """
    start_time = time.time()
    # Leave 5 seconds margin for safety
    deadline = start_time + 115.0
    
    # Helper to compute used differences excluding a specific row index
    def get_used_diffs_excluding(rows, idx):
        used = set()
        for i, row in enumerate(rows):
            if i == idx:
                continue
            # Compute differences for this row
            for j in range(len(row)):
                aj = row[j]
                for l in range(j):
                    used.add(aj - row[l])
        return used

    # Helper to try and rebuild a single row with a specific max bound
    def try_rebuild_row(k_val, bound, forbidden_set, timeout=0.1):
        """
        Attempts to construct a valid row of length k_val+1 starting at 0,
        with max element <= bound, such that no difference is in forbidden_set
        and all internal differences are distinct.
        Returns the row list or None.
        """
        # Pruning: Check if bound is large enough to hold required distinct differences
        # Number of differences in a row of length k+1 is (k+1)*k/2
        needed_diffs = (k_val + 1) * k_val // 2
        # Max possible distinct differences in [1, bound] is bound
        # Actually, differences are values in 1..bound.
        # We need 'needed_diffs' distinct values from 1..bound.
        if bound < needed_diffs:
            return None
        
        path = [0]
        diffs = set()
        
        # Check if there are enough available differences in the range
        # This is a strong prune.
        # Available diffs = {1..bound} - forbidden_set
        # We need to pick 'needed_diffs' of them that are compatible with the structure.
        # Simple count check:
        # len({1..bound} - forbidden_set) >= needed_diffs
        # This is O(bound). bound is small (< 200).
        # But we can skip this for speed in the inner loop, or do it once.
        
        def backtrack(pos, last_val):
            if time.time() > start_time + timeout:
                return None
            if pos == k_val + 1:
                return path[:]
            
            candidates = []
            # Iterate possible next values
            for x in range(last_val + 1, bound + 1):
                valid = True
                new_diffs = []
                for y in path:
                    d = x - y
                    if d in forbidden_set or d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return None
                
            # Randomize to explore different structures
            random.shuffle(candidates)
            
            for x, new_diffs in candidates:
                path.append(x)
                diffs.update(new_diffs)
                
                res = backtrack(pos + 1, x)
                if res:
                    return res
                
                path.pop()
                for d in new_diffs:
                    diffs.discard(d)
            
            return None

        return backtrack(1, 0)

    # Initial Greedy Construction
    def greedy_init(n, k_val):
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            x = 0
            while len(row) < k_val + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Handle trivial cases
    if k == 0:
        return [[0] for _ in range(n_rows)]

    rows = greedy_init(n_rows, k)
    best_rows = copy.deepcopy(rows)
    best_scope = max(max(r) for r in rows)
    
    if best_scope == 0:
        return best_rows

    stagnant_count = 0
    last_improvement_time = start_time
    
    # Determine a lower bound for the scope
    # For Golomb rulers, length is approx n^2/2. Here n=k+1.
    # For k=5, n=6, GL=17.
    # We use a conservative lower bound.
    lower_bound = max(10, k * (k + 1) // 2)

    while time.time() < deadline:
        current_scope = max(max(r) for r in rows)
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = copy.deepcopy(rows)
            stagnant_count = 0
            last_improvement_time = time.time()
        else:
            stagnant_count += 1

        # Identify the row with the maximum element
        max_val = current_scope
        # Pick a random row that has the max element to vary search
        candidates_idx = [i for i, r in enumerate(rows) if max(r) == max_val]
        idx = random.choice(candidates_idx)
        cur_max = max(rows[idx])
        
        # Try to reduce the scope of this row
        reduced = False
        
        # Try bounds from cur_max - 1 down to lower_bound
        # Limit the number of attempts to avoid spending all time on one row
        attempts = 0
        max_attempts = 100
        
        for new_max in range(cur_max - 1, lower_bound - 1, -1):
            if time.time() > deadline - 1.0:
                break
            if attempts > max_attempts:
                break
                
            forbidden = get_used_diffs_excluding(rows, idx)
            
            # Adaptive timeout: shorter as we approach deadline
            time_left = deadline - time.time()
            timeout = 0.05 if time_left < 5 else 0.1
            
            new_row = try_rebuild_row(k, new_max, forbidden, timeout=timeout)
            
            if new_row:
                rows[idx] = new_row
                reduced = True
                break
            attempts += 1
        
        if not reduced:
            # If we couldn't reduce scope, try to perturb the structure
            # to potentially enable future reductions.
            if stagnant_count > 5:
                # Try to rebuild the current row with the SAME max, but different structure
                forbidden = get_used_diffs_excluding(rows, idx)
                time_left = deadline - time.time()
                timeout = 0.05 if time_left < 5 else 0.1
                
                new_row = try_rebuild_row(k, cur_max, forbidden, timeout=timeout)
                if new_row and new_row != rows[idx]:
                    rows[idx] = new_row
                    stagnant_count = 0
                else:
                    # Try to perturb a different row
                    other_candidates = [i for i in range(n_rows) if i != idx]
                    if other_candidates:
                        other_idx = random.choice(other_candidates)
                        other_max = max(rows[other_idx])
                        forbidden_other = get_used_diffs_excluding(rows, other_idx)
                        new_row_other = try_rebuild_row(k, other_max, forbidden_other, timeout=timeout)
                        if new_row_other and new_row_other != rows[other_idx]:
                            rows[other_idx] = new_row_other
                            stagnant_count = 0

    return best_rows
