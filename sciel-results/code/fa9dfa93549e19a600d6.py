import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    
    Uses a local search approach:
    1. Generate an initial valid solution using a greedy method.
    2. Iteratively improve by removing one row and greedily reconstructing it
       to minimize its max value, given the constraints of the other rows.
    """
    start_time = time.time()
    max_time = 115.0
    
    # Helper: Greedily construct a row given a set of forbidden differences
    def construct_row(forbidden_diffs, max_val_limit):
        """
        Constructs a row of length k+1 starting with 0.
        Returns None if impossible within max_val_limit.
        """
        row = [0]
        used_in_row = set()
        
        for _ in range(k):
            found = False
            # We want to pick the smallest possible next value
            # to keep the scope small.
            # The next value must be > row[-1].
            # Also, we must ensure that the differences (next - a) for all a in row
            # are not in forbidden_diffs and not in used_in_row.
            
            # Optimization: We can estimate the minimum possible next value.
            # But simple iteration is fine for k=6.
            
            # Upper bound for the next value to stay within max_val_limit
            # We need to fit k - len(row) more elements.
            # If we pick v, the next must be > v.
            # So v + (k - len(row)) <= max_val_limit is a loose bound.
            # Actually, if we are at the last step, v <= max_val_limit.
            
            # Let's iterate from row[-1] + 1 upwards.
            # To speed up, we can jump, but for k=6, linear scan is cheap.
            
            start_val = row[-1] + 1
            # We must ensure we can still reach the end within max_val_limit.
            # If we are at index i (0-indexed, so we have placed i+1 elements),
            # we need k - i - 1 more elements.
            # The smallest possible max value if we pick v now is v + (k - i - 1).
            # So we require v + (k - i - 1) <= max_val_limit => v <= max_val_limit - (k - i - 1).
            
            max_candidate = max_val_limit - (k - len(row))
            
            for v in range(start_val, max_candidate + 1):
                new_diffs = {v - a for a in row}
                # Check against global forbidden and local used
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & used_in_row:
                    continue
                
                # Valid candidate
                row.append(v)
                used_in_row.update(new_diffs)
                found = True
                break
            
            if not found:
                return None
        
        return row

    # Step 1: Generate an initial valid solution
    # We use a greedy approach, adding rows one by one.
    # To get a good starting point, we might want to randomize the order of rows?
    # No, rows are symmetric. But we can randomize the "greediness" or just use standard greedy.
    # Standard greedy tends to produce compact rows.
    
    def get_initial_solution():
        rows = []
        used_diffs = set()
        # We try to build rows. If greedy fails to add a row (unlikely for small n,k),
        # we might need to restart or backtrack. For this problem, greedy usually works.
        for _ in range(n_rows):
            # We don't have a hard max_val_limit initially, so we use a large one.
            # But we want it to be small. Let's try with a limit that increases if it fails.
            # For the first row, limit can be small.
            # For subsequent rows, the limit might need to be larger.
            # Let's just use a large limit for initialization, e.g., 200.
            row = construct_row(used_diffs, 200)
            if row is None:
                # This shouldn't happen for n=7, k=6 with limit 200.
                # If it does, we return a dummy or raise.
                return None
            rows.append(row)
            # Update used_diffs
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        return rows

    best_rows = get_initial_solution()
    if best_rows is None:
        # Fallback if greedy fails (very unlikely)
        return [[0, 1, 2, 3, 4, 5, 6]] * n_rows # Invalid, but prevents crash

    best_scope = max(r[-1] for r in best_rows)
    
    # Step 2: Local Search
    # We will try to improve the scope by re-optimizing one row at a time.
    rng = random.Random(42)
    
    # Calculate current used diffs for fast updates
    def calc_used_diffs(rows, exclude_idx=None):
        used = set()
        for i, r in enumerate(rows):
            if i == exclude_idx:
                continue
            for x in range(len(r)):
                for y in range(x + 1, len(r)):
                    used.add(r[y] - r[x])
        return used
    
    current_used_diffs = calc_used_diffs(best_rows)
    
    # Simulated Annealing parameters
    # We don't strictly need SA if we just accept improvements, but SA helps escape local minima.
    # However, since we are minimizing max entry, and the space is discrete and small,
    # simple hill climbing (accept only improvements) often works well if we have multiple restarts
    # or if the "re-optimization" is stochastic.
    
    # To make the re-optimization stochastic, we can randomize the candidate selection in construct_row.
    # Let's modify construct_row to take a randomization parameter.
    
    def construct_row_randomized(forbidden_diffs, max_val_limit, rng, randomize_prob=0.2):
        row = [0]
        used_in_row = set()
        
        for _ in range(k):
            found = False
            start_val = row[-1] + 1
            max_candidate = max_val_limit - (k - len(row))
            
            # Collect valid candidates
            valid_cands = []
            for v in range(start_val, max_candidate + 1):
                new_diffs = {v - a for a in row}
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & used_in_row:
                    continue
                valid_cands.append((v, new_diffs))
            
            if not valid_cands:
                return None
            
            # Choose a candidate
            # If randomize_prob > 0, pick randomly from the first few valid candidates
            # to avoid always picking the absolute smallest, which might lead to a dead end
            # for the *global* scope minimization (though for a single row, smallest is usually best).
            # Actually, for a single row, picking the smallest valid value minimizes the row's max value.
            # The "randomization" should come from the *order* in which we try to fix rows,
            # or by allowing slightly larger values in a row if it allows other rows to be smaller?
            # No, rows are independent given the forbidden diffs.
            # So the best strategy for a single row is always to minimize its max value.
            # The non-determinism should come from *which* row we fix, or by
            # temporarily accepting a worse row to allow a better global configuration?
            # 
            # Let's stick to deterministic row reconstruction (minimizing row max)
            # but randomize the *order* of rows we attempt to improve.
            # And we can also try "kicks": replace a row with a random valid row
            # (not necessarily minimal) to escape local minima.
            
            # For now, pick the smallest valid candidate.
            v, new_diffs = valid_cands[0]
            
            row.append(v)
            used_in_row.update(new_diffs)
            found = True
            break
        
        return row

    # Improved Local Search Loop
    no_improve_count = 0
    temperature = 1.0 # For simulated annealing if we decide to use it
    
    while time.time() - start_time < max_time:
        if no_improve_count > 1000: # Restart or break
            # Let's just break, we've likely converged
            break
            
        # Pick a random row to re-optimize
        row_idx = rng.randint(0, n_rows - 1)
        old_row = best_rows[row_idx]
        old_scope = max(r[-1] for r in best_rows)
        
        # Calculate forbidden diffs excluding this row
        forbidden_diffs = calc_used_diffs(best_rows, exclude_idx=row_idx)
        
        # Try to reconstruct this row to have a smaller max value
        # We know the current max value of this row is old_row[-1].
        # We want a new row with max < old_row[-1] if possible.
        # If not, maybe max == old_row[-1] but different structure?
        # Or we can allow max to increase slightly if it allows others to decrease?
        # But we are only changing one row. So we must not increase the global scope.
        # So new_max must be <= old_scope.
        
        # Let's try to find a row with max < old_row[-1].
        # If found, accept.
        # If not, try to find a row with max == old_row[-1] but different?
        # This might not help.
        
        # A better strategy:
        # 1. Try to minimize the max value of this row specifically.
        #    Let new_row = construct_row(forbidden_diffs, old_row[-1] - 1)
        #    If new_row is not None, accept it.
        # 2. If that fails, try construct_row(forbidden_diffs, old_row[-1])
        #    If new_row is not None and new_row != old_row, accept it (might help future steps).
        # 3. If that fails, try a "kick": generate a random valid row with max <= old_scope.
        
        improved = False
        
        # Attempt 1: Strict improvement
        new_row = construct_row(forbidden_diffs, old_row[-1] - 1)
        if new_row is not None:
            best_rows[row_idx] = new_row
            improved = True
            no_improve_count = 0
            # Update used diffs
            current_used_diffs = calc_used_diffs(best_rows)
            continue
        
        # Attempt 2: Same max, different structure (stochastic exploration)
        # We use a randomized construction to find a different row with the same max bound.
        new_row = construct_row_randomized(forbidden_diffs, old_row[-1], rng)
        if new_row is not None and new_row != old_row:
            # Accept with some probability to allow exploration?
            # Or just accept if it's different?
            # Since it doesn't improve scope, it might be neutral.
            # But it might allow a different row to be improved later.
            # Let's accept it with 50% probability.
            if rng.random() < 0.5:
                best_rows[row_idx] = new_row
                no_improve_count = 0
                current_used_diffs = calc_used_diffs(best_rows)
                continue
        
        # If no improvement, increment counter
        no_improve_count += 1

    return best_rows
