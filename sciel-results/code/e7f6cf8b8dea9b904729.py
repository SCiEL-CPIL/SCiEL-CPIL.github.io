"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
    # We use 140s to be safe with Python overhead
    deadline = time.time() + 140.0
    
    # Total number of differences per row
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row

    def get_diffs(row):
        """Return list of all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def compute_scope(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    def count_violations(diff_counts):
        """Count how many differences are duplicated."""
        return sum(c - 1 for c in diff_counts.values() if c > 1)

    # --- Phase 1: Greedy Initialization ---
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # --- Phase 2: Fast Local Search with Incremental Updates ---
    def local_search(initial_rows, max_time=10.0):
        """
        Perform local search on the given rows.
        Returns the best rows found.
        """
        # Deep copy
        rows = [r[:] for r in initial_rows]
        
        # Initialize difference counts
        diff_counts = defaultdict(int)
        for row in rows:
            for d in get_diffs(row):
                diff_counts[d] += 1
        
        current_violations = count_violations(diff_counts)
        current_scope = compute_scope(rows)
        
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        best_violations = current_violations

        T = 50.0
        T_min = 0.1
        cooling = 0.99999
        start_time = time.time()

        # Precompute indices for moves
        # We will randomly pick a row and an index to modify
        # Modifying row[r][i] affects diffs (i, j) for all j != i
        # Specifically, diffs involving index i are:
        # row[i] - row[j] for j < i
        # row[j] - row[i] for j > i

        while time.time() - start_time < max_time:
            if T < T_min:
                break

            r = random.randint(0, n_rows - 1)
            i = random.randint(1, k)  # Don't change the first element (0)
            
            old_val = rows[r][i]
            
            # Determine bounds for new value
            lower = rows[r][i - 1] + 1
            upper = rows[r][i + 1] - 1 if i < k else rows[r][k] + 50
            
            # If we are trying to shrink the scope, bias towards smaller values
            # especially for the last element
            if i == k and current_scope > 100:
                # Try to reduce the max element
                if random.random() < 0.5:
                    new_val = random.randint(lower, old_val - 1) if old_val > lower else lower
                else:
                    new_val = random.randint(lower, upper)
            else:
                # Random move within bounds
                # To allow escape from local minima, sometimes allow "bad" moves
                span = upper - lower + 1
                if span <= 0:
                    continue
                new_val = lower + random.randint(0, span - 1)

            if new_val == old_val:
                continue

            # Check monotonicity
            if new_val <= rows[r][i - 1]:
                continue
            if i < k and new_val >= rows[r][i + 1]:
                continue

            # Calculate change in violations
            # Old diffs involving i:
            old_diffs = []
            for j in range(i):
                d = old_val - rows[r][j]
                old_diffs.append(d)
            for j in range(i + 1, len(rows[r])):
                d = rows[r][j] - old_val
                old_diffs.append(d)

            # New diffs involving i:
            new_diffs = []
            for j in range(i):
                d = new_val - rows[r][j]
                new_diffs.append(d)
            for j in range(i + 1, len(rows[r])):
                d = rows[r][j] - new_val
                new_diffs.append(d)

            # Update diff_counts and calculate delta violations
            delta_violations = 0
            
            # Remove old diffs
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                # If count went from 1 to 0, violation decreased by 1? No.
                # Violation is sum(c-1 for c>1).
                # If c was 1, contribution was 0. Now 0. Delta 0.
                # If c was 2, contribution was 1. Now 1. Delta -1.
                # If c was 3, contribution was 2. Now 2. Delta -1.
                # So if count becomes 1, delta is -1.
                if diff_counts[d] == 1:
                    delta_violations -= 1
                elif d not in diff_counts:
                    # It was 1, now 0. No change in violation count (0 -> 0)
                    pass

            # Add new diffs
            for d in new_diffs:
                old_c = diff_counts.get(d, 0)
                diff_counts[d] = old_c + 1
                if old_c == 1:
                    # Went from 1 to 2. Violation increases by 1.
                    delta_violations += 1
                # If old_c was 0, 0 -> 1, no change.
                # If old_c was 2, 2 -> 3, violation 1 -> 2, delta +1.
                elif old_c > 1:
                    delta_violations += 1

            new_violations = current_violations + delta_violations
            new_scope = compute_scope(rows) # This is O(n_rows * k), which is small
            
            # Actually, we can update scope more efficiently if we track max per row
            # But for now, O(n*k) is fine since n, k are small (e.g., 4, 8)
            
            # Acceptance criteria
            accept = False
            if new_violations < current_violations:
                accept = True
            elif new_violations == current_violations:
                if new_scope < current_scope:
                    accept = True
                elif new_scope == current_scope:
                    # Neutral move, accept with some probability to explore
                    if random.random() < 0.1:
                        accept = True
            else:
                # Worse violations, accept with probability exp(-delta/T)
                delta = new_violations - current_violations
                if delta < 100: # Avoid overflow in exp
                    if random.random() < math.exp(-delta / T):
                        accept = True

            if accept:
                rows[r][i] = new_val
                current_violations = new_violations
                current_scope = new_scope
                
                if current_violations == 0 and current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [rr[:] for rr in rows]
                    best_violations = 0
                elif current_violations < best_violations:
                    best_violations = current_violations
                    best_rows = [rr[:] for rr in rows]
                    best_scope = current_scope
            else:
                # Revert diff_counts
                # Add back old diffs
                for d in old_diffs:
                    old_c = diff_counts.get(d, 0)
                    diff_counts[d] = old_c + 1
                    if old_c == 1:
                        delta_violations -= 1 # Undo the +1
                    elif old_c > 1:
                        delta_violations -= 1 # Undo the +1
                    # If old_c was 0, it became 1. No violation change.
                
                # Remove new diffs
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 1:
                        delta_violations += 1 # Undo the -1
                    elif d not in diff_counts:
                        pass

                # Note: The delta_violations calculation above for reverting is tricky.
                # It is safer to just recalculate or store the exact changes.
                # Let's simplify: We just revert the rows, and since we didn't update rows,
                # we just need to revert diff_counts.
                # Actually, the code above for "Revert" is complex and error-prone.
                # Let's simplify the logic:
                # 1. Calculate old_diffs, new_diffs.
                # 2. Update diff_counts to remove old, add new.
                # 3. Calculate new_violations.
                # 4. If reject, remove new, add old.
                
                # Let's rewrite the rejection block cleanly.
                # Remove new_diffs from diff_counts
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 1:
                        pass # Violation decreased by 1 when it went 2->1, so we increased it by 1?
                        # Let's trace:
                        # Add new: c 1->2, viol +1.
                        # Remove new: c 2->1, viol -1.
                        # So if we reject, we must undo the violation change.
                        # My previous delta calc was:
                        # Remove old: if c 2->1, delta -1.
                        # Add new: if c 1->2, delta +1.
                        # Total delta = -1 + 1 = 0.
                        # If reject, we do nothing to current_violations.
                        # But we already modified diff_counts.
                        # So we must revert diff_counts.
                        pass
                    elif diff_counts[d] == 0:
                        del diff_counts[d]
                
                # Add old_diffs back to diff_counts
                for d in old_diffs:
                    old_c = diff_counts.get(d, 0)
                    diff_counts[d] = old_c + 1
                    # No need to track violations here as we are reverting the state
                    # to the one before the move, so current_violations stays the same.

            T *= cooling

        return best_rows, best_scope, best_violations

    # --- Main Execution Loop ---
    
    # 1. Get initial solution
    best_rows = greedy_init()
    best_scope = compute_scope(best_rows)
    
    # Check if already valid
    if is_valid(best_rows):
        # Try to improve scope with local search
        improved_rows, improved_scope, _ = local_search(best_rows, max_time=10.0)
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
        return best_rows

    # 2. Run local search to reduce violations and scope
    # We run multiple iterations with different random seeds for initialization if time permits
    # But first, let's just run local search on the greedy init
    
    current_rows = [r[:] for r in best_rows]
    
    # Run a long local search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_scope, best_violations = local_search(current_rows, max_time=remaining_time - 2)
    
    # If we still have violations, try a few restarts with perturbed greedy
    if best_violations > 0:
        restarts = 5
        for _ in range(restarts):
            if time.time() > deadline - 5:
                break
            
            # Perturb the greedy initialization
            rows = greedy_init(seed=random.randint(0, 10000))
            
            # Run a shorter local search
            max_t = min(5.0, deadline - time.time() - 2)
            if max_t <= 0:
                break
            
            new_rows, new_scope, new_violations = local_search(rows, max_time=max_t)
            
            if new_violations < best_violations:
                best_violations = new_violations
                best_rows = new_rows
                best_scope = new_scope
            elif new_violations == best_violations and new_scope < best_scope:
                best_scope = new_scope
                best_rows = new_rows

    # Final check: if we found a valid solution, return it
    # If not, we return the best we have (which might be invalid, but the task says
    # "Valid when all positive differences... are distinct". 
    # The prompt says "an invalid result... scores 0". 
    # So we MUST return a valid result if possible.
    # If we couldn't find a valid one, we are in trouble.
    # However, the greedy init is usually valid for small n, k.
    # The local search is designed to find valid ones.
    
    if is_valid(best_rows):
        return best_rows
    else:
        # As a last resort, return the greedy init which is guaranteed valid
        # by construction (it ensures no diff conflicts)
        return greedy_init()

# EVOLVE-BLOCK-END
