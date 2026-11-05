import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set using Simulated Annealing.
    n_rows: number of rows
    k: number of steps (so row length is k+1)
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Total number of differences per row is k*(k+1)/2
    # Total differences overall is n_rows * k*(k+1)/2
    # For n=4, k=8: 4 * 36 = 144 differences.
    
    def init_rows():
        rows = []
        for _ in range(n_rows):
            row = [0]
            current = 0
            for _ in range(k):
                # Random step, keeping it relatively small initially
                step = random.randint(1, 5)
                current += step
                row.append(current)
            rows.append(row)
        return rows

    def calc_initial_cost(rows):
        diff_counts = {}
        scope = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    if r[j] > scope:
                        scope = r[j]
        
        duplicates = 0
        for count in diff_counts.values():
            if count > 1:
                duplicates += count - 1
        
        # Weight for duplicates should be high enough to dominate scope
        # Max possible scope is likely < 500. Max duplicates is 144.
        # If we use weight 1000, 1 duplicate > 1000 scope.
        cost = duplicates * 1000 + scope
        return cost, diff_counts, scope, duplicates

    def get_diffs_for_index(row, j):
        """
        Returns list of (diff, sign) where sign is +1 if diff is added, -1 if removed?
        Actually, we just need the set of differences involving index j.
        """
        diffs = []
        for i in range(j):
            diffs.append(row[j] - row[i])
        for m in range(j+1, len(row)):
            diffs.append(row[m] - row[j])
        return diffs

    rows = init_rows()
    cost, diff_counts, scope, duplicates = calc_initial_cost(rows)
    
    best_rows = [r[:] for r in rows]
    best_cost = cost
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with new random init if stuck? Or just keep going with low T?
            # Let's restart occasionally to escape local minima
            if random.random() < 0.01:
                rows = init_rows()
                cost, diff_counts, scope, duplicates = calc_initial_cost(rows)
                T = 100.0
            else:
                break
        
        # Pick a random row and index
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick index 1..k (index 0 is always 0)
        j = random.randint(1, k)
        
        old_val = row[j]
        
        # Determine range for new value
        # Must be > row[j-1] and < row[j+1] (if j < k)
        lower = row[j-1] + 1
        upper = row[j+1] - 1 if j < k else 500 # Cap upper bound to keep scope reasonable
        
        if lower >= upper:
            # Try to expand if we are at the end? Or skip?
            # If j==k, we can increase it.
            if j == k:
                upper = old_val + random.randint(1, 10)
                if lower >= upper:
                    continue
            else:
                # Cannot fit a new value? Maybe swap or shift? 
                # For simplicity, skip this move if no space.
                # Or we can allow "expanding" the row by increasing a later element?
                # Let's just skip to avoid invalid states.
                continue
        
        new_val = random.randint(lower, upper)
        if new_val == old_val:
            continue
            
        # Calculate delta cost
        # 1. Remove old diffs
        old_diffs = get_diffs_for_index(row, j)
        delta_dupes = 0
        # We need to simulate the change in diff_counts to know delta duplicates
        # This is tricky with just a count. 
        # Let's do a full recalculation of the affected diffs' contribution to duplicates.
        
        # Actually, efficient delta calculation:
        # For each diff d in old_diffs:
        #   count_old = diff_counts[d]
        #   count_new = count_old - 1
        #   if count_old > 1: delta_dupes -= 1 (was contributing to dupes, now less)
        #   if count_new > 1: delta_dupes += 1 (still contributing)
        #   Wait, duplicates = sum(max(0, count-1)).
        #   Change in duplicates for a diff d when count changes from c to c-1:
        #   if c > 1: old_contrib = c-1, new_contrib = c-2. delta = -1.
        #   if c == 1: old_contrib = 0, new_contrib = 0. delta = 0.
        #   if c == 0: impossible.
        
        # So, for each old diff:
        #   if diff_counts[d] > 1: delta_dupes -= 1
        #   diff_counts[d] -= 1
        
        # For each new diff:
        #   if diff_counts[d] > 0: delta_dupes += 1 (because count goes from c to c+1, contrib goes from max(0,c-1) to max(0,c))
        #   Wait:
        #   c=0 -> 1: 0 -> 0. delta 0.
        #   c=1 -> 2: 0 -> 1. delta 1.
        #   c=2 -> 3: 1 -> 2. delta 1.
        #   So if diff_counts[d] >= 1: delta_dupes += 1.
        #   diff_counts[d] += 1
        
        # Let's verify:
        # Start: d has count 2. Contrib = 1.
        # Remove one: count 1. Contrib = 0. Delta = -1. Correct (count > 1 was true).
        # Start: d has count 1. Contrib = 0.
        # Remove one: count 0. Contrib = 0. Delta = 0. Correct (count > 1 was false).
        
        # Add one:
        # Start: d has count 0. Contrib = 0.
        # Add one: count 1. Contrib = 0. Delta = 0. Correct (count >= 1 was false).
        # Start: d has count 1. Contrib = 0.
        # Add one: count 2. Contrib = 1. Delta = 1. Correct (count >= 1 was true).
        
        delta_dupes = 0
        
        # Apply removals
        for d in old_diffs:
            c = diff_counts.get(d, 0)
            if c > 1:
                delta_dupes -= 1
            diff_counts[d] = c - 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Calculate new diffs
        # We need to update row[j] temporarily to get new diffs? 
        # Or just calculate them.
        new_diffs = []
        for i in range(j):
            new_diffs.append(new_val - row[i])
        for m in range(j+1, len(row)):
            new_diffs.append(row[m] - new_val)
            
        # Apply additions
        for d in new_diffs:
            c = diff_counts.get(d, 0)
            if c >= 1:
                delta_dupes += 1
            diff_counts[d] = c + 1
            
        # Calculate delta scope
        old_scope = scope
        new_scope = scope
        # The max value in the row might change
        # If new_val > old_scope, new_scope = new_val
        # If old_val was the unique max and new_val < old_val, we need to find the next max.
        # This is expensive. 
        # Approximation: 
        # If new_val > old_scope: new_scope = new_val
        # Else: 
        #    We need to check if old_val was the max. 
        #    If yes, we need to find the max of the row with new_val inserted.
        #    For speed, let's just use new_val if it's larger, otherwise keep old_scope.
        #    This might underestimate the scope reduction, but that's okay for SA.
        #    Actually, if we reduce the max, the cost should drop. 
        #    Let's do a cheap check:
        if new_val > old_scope:
            new_scope = new_val
        else:
            # Check if old_val was the max of this row
            # If yes, find new max of this row
            # row_max = max(row)
            # if row_max == old_val:
            #    # find new max
            #    # O(k)
            #    pass
            
            # To be safe and simple, let's just recalculate the row max if old_val was the max.
            # Since k is small (8), this is fine.
            if old_val == max(row):
                # Find new max
                row[j] = new_val
                new_row_max = max(row)
                # Check if this row's max was the global max
                # This is getting complicated. 
                # Let's just accept that scope calculation might be slightly off in delta.
                # We will recalculate exact cost every N steps or if cost seems to improve significantly.
                # For now, let's just use new_val if it's bigger, else keep old_scope.
                # This biases towards not shrinking the max, which is bad.
                
                # Better: Just calculate the new global max if old_val was the global max.
                # If old_val != global max, new_scope = old_scope (since new_val < old_val < global_max? No, new_val could be < global_max but > other rows).
                # Actually, if old_val != global_max, then global_max is in another row or another element.
                # So new_scope = old_scope.
                
                # If old_val == global_max:
                #    We need to find the new global max.
                #    This requires scanning all rows. O(n*k).
                #    Let's do it.
                row[j] = old_val # restore for now
                # Find new max of this row
                # Temporarily set
                row[j] = new_val
                max_this_row = max(row)
                # Find max of other rows
                max_other = 0
                for i in range(n_rows):
                    if i != r_idx:
                        if max(rows[i]) > max_other:
                            max_other = max(rows[i])
                new_scope = max(max_this_row, max_other)
                row[j] = old_val # restore
            
        delta_scope = new_scope - old_scope
        delta_cost = delta_dupes * 1000 + delta_scope
        
        # Acceptance probability
        if delta_cost < 0:
            # Accept
            row[j] = new_val
            scope = new_scope
            duplicates += delta_dupes
            cost += delta_cost
        else:
            # Boltzmann probability
            # p = exp(-delta_cost / T)
            # If delta_cost is large, p is small.
            try:
                p = pow(2.718281828, -delta_cost / T)
            except OverflowError:
                p = 0.0
                
            if random.random() < p:
                row[j] = new_val
                scope = new_scope
                duplicates += delta_dupes
                cost += delta_cost
            else:
                # Reject: Revert diff_counts
                # We already modified diff_counts. We need to revert.
                # This is a bit messy. 
                # Let's refactor: Calculate delta without modifying state, then modify if accepted.
                
                # REFACTOR:
                # 1. Calculate delta_dupes and delta_scope without modifying diff_counts.
                # 2. If accepted, modify diff_counts.
                
                # Since we already modified diff_counts in the code above, we must revert if rejected.
                # Revert additions:
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                # Revert removals:
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                # Note: The delta_dupes calculation relied on the state *before* modification.
                # The logic above:
                # Removal: if c > 1, delta -= 1.
                # Addition: if c >= 1, delta += 1.
                # This logic is correct for the delta.
                # So we just need to revert the counts.
                pass

        # Update best
        if cost < best_cost:
            best_cost = cost
            best_rows = [r[:] for r in rows]
            
        T *= cooling

    # Return the best valid solution found
    # Ensure best_rows is valid (it should be if best_cost duplicates == 0)
    # If best_cost has duplicates, we might need to run a repair or just return it (invalid scores 0).
    # We should try to ensure we return a valid one.
    # If duplicates > 0, the cost is high. 
    # Let's do a final check.
    
    # Verify best_rows
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    if is_valid(best_rows):
        return best_rows
    else:
        # Fallback to greedy if SA failed to find valid solution
        # (Unlikely given the weights, but safe)
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
