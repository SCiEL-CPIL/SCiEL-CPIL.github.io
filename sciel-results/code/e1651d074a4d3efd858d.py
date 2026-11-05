"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0

    def all_diffs(rows):
        diffs = []
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.append(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

    # --- Fast Incremental Cost ---
    # We maintain a global count of differences across all rows.
    # Cost = sum(count[d] - 1 for d where count[d] > 1) + lambda * max_entry
    # To make it fast, we don't recompute from scratch. We update counts when a value changes.

    def greedy_init():
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

    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    
    # If greedy is already valid, we might just need to optimize scope slightly
    # But greedy usually has a large scope.

    # --- Multi-Start Local Search with Parallel-ish Strategy (Sequential but fast restarts) ---
    
    def compute_diff_counts(rows):
        counts = defaultdict(int)
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    counts[row[j] - row[i]] += 1
        return counts

    def compute_cost(rows, counts):
        violations = 0
        for c in counts.values():
            if c > 1:
                violations += c - 1
        m = max_entry(rows)
        # We want to minimize violations first, then scope.
        # Use a large multiplier for violations to prioritize validity.
        return violations * 100000 + m

    def update_counts_on_change(rows, counts, r, p, old_val, new_val):
        """
        Update the global difference counts when rows[r][p] changes from old_val to new_val.
        This requires removing old diffs involving (r, p) and adding new ones.
        """
        row = rows[r]
        
        # Remove old contributions
        for i in range(k + 1):
            if i == p:
                continue
            if i < p:
                d = old_val - row[i]
            else:
                d = row[i] - old_val
            counts[d] -= 1
            if counts[d] == 0:
                del counts[d]
        
        # Add new contributions
        for i in range(k + 1):
            if i == p:
                continue
            if i < p:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            counts[d] += 1

    def local_search(rows, counts, max_time):
        """
        Perform a fast local search to minimize cost.
        Uses random single-point moves and accepts if cost improves or with small probability (simulated annealing lite).
        """
        start_time = time.time()
        current_cost = compute_cost(rows, counts)
        best_cost_local = current_cost
        best_rows_local = [r[:] for r in rows]
        
        T = 10.0
        cooling = 0.999
        
        while time.time() - start_time < max_time:
            # Pick a random row and position
            r = random.randint(0, n_rows - 1)
            p = random.randint(1, k) # Position 0 is fixed at 0
            
            old_val = rows[r][p]
            
            # Determine range for new value
            # Must be strictly increasing
            lower = rows[r][p-1] + 1
            upper = rows[r][p+1] - 1 if p < k else rows[r][k] + 20
            
            if lower >= upper:
                continue
                
            # Random choice within range, biased towards current value for locality
            # Or just uniform
            if random.random() < 0.5:
                new_val = random.randint(lower, upper)
            else:
                # Small perturbation
                step = random.randint(-5, 5)
                new_val = old_val + step
                if new_val < lower or new_val > upper:
                    continue
            
            if new_val == old_val:
                continue
            
            # Apply change
            rows[r][p] = new_val
            update_counts_on_change(rows, counts, r, p, old_val, new_val)
            new_cost = compute_cost(rows, counts)
            
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost_local:
                    best_cost_local = current_cost
                    best_rows_local = [rr[:] for rr in rows]
                # Check if valid
                if best_cost_local < 100000: # Valid
                    # Try to reduce scope further? 
                    # For now, just record
                    pass
            else:
                # Revert
                rows[r][p] = old_val
                update_counts_on_change(rows, counts, r, p, new_val, old_val)
            
            T *= cooling
            
        return best_rows_local, best_cost_local

    # Run multiple restarts
    num_restarts = 50
    time_per_restart = 2.0 # Adjust based on budget
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomized initialization: start from greedy but perturb
        start_rows = [r[:] for r in best_rows]
        # Perturb some values
        for _ in range(random.randint(5, 20)):
            r = random.randint(0, n_rows - 1)
            p = random.randint(1, k)
            lower = start_rows[r][p-1] + 1
            upper = start_rows[r][p+1] - 1 if p < k else start_rows[r][k] + 50
            if lower < upper:
                start_rows[r][p] = random.randint(lower, upper)
        
        # Ensure validity of structure (increasing)
        for r in range(n_rows):
            for p in range(1, k+1):
                if start_rows[r][p] <= start_rows[r][p-1]:
                    start_rows[r][p] = start_rows[r][p-1] + 1
                if p < k and start_rows[r][p] >= start_rows[r][p+1]:
                    # This might break increasing order, so just re-sort? No, must be increasing.
                    # Simple fix: force increasing
                    for j in range(p+1, k+1):
                        if start_rows[r][j] <= start_rows[r][j-1]:
                            start_rows[r][j] = start_rows[r][j-1] + 1

        counts = compute_diff_counts(start_rows)
        
        # Run local search
        res_rows, res_cost = local_search(start_rows, counts, time_per_restart)
        
        # Check if better
        res_max = max_entry(res_rows)
        res_valid = is_valid(res_rows)
        
        if res_valid and res_max < best_max:
            best_max = res_max
            best_rows = res_rows
            
    # Final phase: Try to reduce scope of the best valid solution found
    # If best_rows is valid, we can try to "compress" it.
    # Strategy: Randomly pick a row, pick a position, try to decrease the value 
    # while maintaining increasing order and distinct differences.
    
    if is_valid(best_rows):
        # We have a valid solution. Try to minimize max entry.
        # Use a dedicated local search that ONLY accepts moves that reduce max_entry 
        # and maintain validity.
        
        def try_reduce_scope(rows, time_limit):
            start_time = time.time()
            current_max = max_entry(rows)
            counts = compute_diff_counts(rows)
            
            while time.time() - start_time < time_limit:
                # Find the row with the maximum entry
                max_row_idx = 0
                max_val = -1
                for r in range(n_rows):
                    if rows[r][-1] > max_val:
                        max_val = rows[r][-1]
                        max_row_idx = r
                
                # Try to reduce the last element of max_row_idx
                # Or any element that contributes to the max?
                # Let's try to reduce any element in the max row that is not 0.
                p = random.randint(1, k)
                old_val = rows[max_row_idx][p]
                
                # Try to decrease
                new_val = old_val - random.randint(1, 5)
                
                # Check constraints
                if new_val <= rows[max_row_idx][p-1]:
                    continue
                if p < k and new_val >= rows[max_row_idx][p+1]:
                    continue
                
                # Check difference conflicts
                # Remove old diffs
                temp_counts = defaultdict(counts)
                row = rows[max_row_idx]
                for i in range(k + 1):
                    if i == p: continue
                    d = abs(row[i] - old_val)
                    temp_counts[d] -= 1
                    if temp_counts[d] == 0:
                        del temp_counts[d]
                    
                # Add new diffs
                valid_move = True
                for i in range(k + 1):
                    if i == p: continue
                    d = abs(row[i] - new_val)
                    if temp_counts[d] > 0:
                        valid_move = False
                        break
                    temp_counts[d] += 1
                
                if valid_move:
                    # Accept
                    rows[max_row_idx][p] = new_val
                    counts = temp_counts
                    new_max = max_entry(rows)
                    if new_max < current_max:
                        current_max = new_max
                        # We found a better scope!
                        # We could break early or continue
                        pass
                # Else reject

            return rows

        # Allocate remaining time to this phase
        remaining = deadline - time.time() - 5
        if remaining > 5:
            best_rows = try_reduce_scope([r[:] for r in best_rows], remaining)
            if max_entry(best_rows) < best_max:
                best_max = max_entry(best_rows)

    return best_rows
# EVOLVE-BLOCK-END
