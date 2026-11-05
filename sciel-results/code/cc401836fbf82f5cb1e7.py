import random
import time
import sys
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for evaluator overhead
    
    # Parameters
    # For n_rows=7, k=5 (6 elements), scope ~112 is optimal.
    # We will try to find a solution with scope as small as possible.
    
    best_result = None
    best_scope = float('inf')
    
    def get_diffs(row):
        """Calculate all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def check_validity(rows):
        """Check if the set of rows is valid (no duplicate diffs)."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_diffs(row)
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True

    def get_scope(rows):
        """Get the largest entry in any row."""
        return max(r[-1] for r in rows)

    def greedy_construct():
        """
        Construct a valid set of rows using a greedy heuristic.
        We try to keep the scope small by choosing the smallest possible next element
        that doesn't conflict with existing differences.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            # We need k more elements
            for _ in range(k):
                found = False
                # Try small increments
                # Start from last element + 1
                # To keep scope small, we prefer small values.
                # However, small values create small diffs which are "scarce".
                # A simple greedy: try x from last+1 upwards.
                # To avoid infinite loop if stuck, we might need to jump.
                # But for DTS, greedy usually works if we restart.
                
                # Heuristic: Try values in a specific order.
                # Randomized greedy is better.
                # Here we do a simple ascending greedy.
                for x in range(row[-1] + 1, 500): 
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback: try larger jumps or restart logic
                    # For simplicity in this helper, return None to signal failure
                    return None
            rows.append(row)
        return rows

    def simulated_annealing(initial_rows, time_budget):
        """
        Improve a valid set of rows using Simulated Annealing.
        Objective: Minimize Scope (max entry).
        Constraint: All diffs must be unique.
        Penalty: If invalid, add a large penalty proportional to number of conflicts.
        """
        current_rows = [r[:] for r in initial_rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        # Temperature parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        used_diffs = set()
        for r in current_rows:
            used_diffs |= get_diffs(r)
            
        # We track conflicts to calculate penalty quickly
        # Conflict count: number of differences that appear more than once.
        diff_counts = {}
        for r in current_rows:
            for d in get_diffs(r):
                diff_counts[d] = diff_counts.get(d, 0) + 1
        conflict_count = sum(1 for c in diff_counts.values() if c > 1)
        
        def calculate_energy(rows, conflicts):
            scope = get_scope(rows)
            # We want to minimize scope.
            # If invalid, we must fix it.
            # Energy = Scope + Lambda * Conflicts
            # Lambda should be large enough to prioritize validity.
            # Max possible scope is ~200. Max conflicts is ~21*7=147.
            # If Lambda=10, 1 conflict = 10 scope units.
            return scope + 10.0 * conflicts

        current_energy = calculate_energy(current_rows, conflict_count)
        
        steps = 0
        while T > T_min and time.time() - start_time < time_limit - time_budget:
            steps += 1
            # Perturbation: Pick a random row and a random element (not 0), and change it.
            # Or swap two elements in a row (keeping sorted order).
            # Changing a single element is simpler.
            
            row_idx = random.randint(0, n_rows - 1)
            # Element indices 1..k (0 is fixed)
            elem_idx = random.randint(1, k)
            
            old_val = current_rows[row_idx][elem_idx]
            
            # Generate new value
            # Try to move towards smaller values (to reduce scope) or random small range
            # If we are at scope S, we might try S-1, S-2, etc.
            # Or just random in [last+1, current_scope]
            
            # Strategy: Try to replace with a value that might reduce scope
            # or just random local search.
            
            # Let's try a random value in a window around the old value, biased towards smaller
            # if it's the max element, or just random.
            
            # Constraint: Must be > previous element and < next element (if exists)
            low_bound = current_rows[row_idx][elem_idx - 1] + 1
            high_bound = current_rows[row_idx][elem_idx + 1] - 1 if elem_idx < k else current_scope
            
            # If high_bound < low_bound, we can't change this element without changing neighbors.
            # For simplicity, if stuck, skip or change a different element.
            if high_bound < low_bound:
                T *= cooling_rate
                continue
                
            # Sample new value
            # Bias towards smaller values to reduce scope
            # new_val = random.randint(low_bound, high_bound)
            # To encourage scope reduction:
            # If old_val is the current max of this row, try smaller.
            # Let's just do random uniform in [low_bound, min(high_bound, current_scope)]
            upper_try = min(high_bound, current_scope)
            if upper_try < low_bound:
                T *= cooling_rate
                continue
                
            new_val = random.randint(low_bound, upper_try)
            
            if new_val == old_val:
                continue
                
            # Calculate change in diffs
            row = current_rows[row_idx]
            old_diffs = set()
            new_diffs = set()
            
            # Diffs involving old_val
            for i in range(len(row)):
                if i == elem_idx: continue
                d = abs(row[i] - old_val)
                old_diffs.add(d)
                
            # Diffs involving new_val
            for i in range(len(row)):
                if i == elem_idx: continue
                d = abs(row[i] - new_val)
                new_diffs.add(d)
                
            # Update diff counts
            # Remove old_diffs
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            # Add new_diffs
            for d in new_diffs:
                if d not in diff_counts:
                    diff_counts[d] = 1
                else:
                    diff_counts[d] += 1
                    
            # Recalculate conflict count
            # This is O(N) where N is number of unique diffs. ~200.
            # Acceptable.
            new_conflict_count = sum(1 for c in diff_counts.values() if c > 1)
            
            # Apply change
            row[elem_idx] = new_val
            # Ensure row is sorted (it should be if we respected bounds, but just in case)
            # Actually, if we picked a value that violates order, we skipped it.
            # So row remains sorted.
            
            new_scope = get_scope(current_rows)
            new_energy = calculate_energy(current_rows, new_conflict_count)
            
            # Metropolis criterion
            delta_energy = new_energy - current_energy
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                # Accept
                current_energy = new_energy
                conflict_count = new_conflict_count
                current_scope = new_scope
                
                if new_conflict_count == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Reject: Revert changes
                row[elem_idx] = old_val
                
                # Revert diff counts
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs:
                    if d not in diff_counts:
                        diff_counts[d] = 1
                    else:
                        diff_counts[d] += 1
                # Note: We don't strictly need to recalculate conflict_count for rejection
                # because we revert to state with known conflict_count.
                
            T *= cooling_rate
            
        return best_rows, best_scope

    def polish(rows):
        """
        Try to reduce the scope by replacing the largest elements with smaller valid ones.
        """
        rows = [r[:] for r in rows]
        changed = True
        while changed:
            changed = False
            # Find the row with the largest max
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            max_val = rows[max_row_idx][-1]
            
            # Try to reduce max_val
            # Check if we can replace max_val with a smaller value
            # The value must be > rows[max_row_idx][-2]
            # And must not create duplicate diffs.
            
            row = rows[max_row_idx]
            prev_val = row[-2]
            
            # Calculate current diffs of this row
            row_diffs = get_diffs(row)
            
            # Try values from prev_val + 1 to max_val - 1
            for new_val in range(prev_val + 1, max_val):
                # Check if new_val is already in row (it shouldn't be if we are replacing the last one and it's unique)
                # Actually, we are replacing the last element.
                
                # Calculate new diffs
                # Diffs involving new_val: {new_val - a for a in row[:-1]}
                # Diffs NOT involving new_val: remain same.
                
                # We need to check if any new diff conflicts with diffs from OTHER rows
                # or if it conflicts with diffs from THIS row that don't involve the old last element.
                
                # Simpler: Just build the new row and check global validity.
                # Since n_rows is small, this is fast enough for a polish step.
                
                new_row = row[:-1] + [new_val]
                
                # Check if new_row has internal duplicates?
                # If new_row is sorted, internal diffs are unique unless there are repeated values (impossible) 
                # or specific structures. 
                # Actually, a set of integers can have duplicate differences.
                # e.g., [0, 1, 3, 4] -> diffs 1,3,4,2,3,1. Duplicate 1 and 3.
                # So we must check internal uniqueness too.
                
                internal_diffs = get_diffs(new_row)
                if len(internal_diffs) != len(new_row)*(len(new_row)-1)//2:
                    continue
                
                # Check global uniqueness
                # Temporarily remove old row diffs, add new row diffs, check for dupes with other rows.
                
                # This is O(N^2) roughly.
                # Let's just check against all other rows.
                
                valid = True
                for i in range(n_rows):
                    if i == max_row_idx: continue
                    other_diffs = get_diffs(rows[i])
                    if internal_diffs & other_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also check if internal_diffs has duplicates? 
                # We already checked len(internal_diffs) == total_pairs.
                
                # If valid, apply
                rows[max_row_idx] = new_row
                changed = True
                break # Restart loop to find new max
                
        return rows

    # Main Logic
    # 1. Generate initial solution
    # Use multiple random seeds for greedy to get diverse starts
    
    found_initial = False
    for seed in range(100):
        random.seed(seed)
        # Randomize the order of rows? No, rows are indistinguishable in terms of validity, 
        # but the greedy process depends on the order of filling.
        
        # Simple Greedy
        res = greedy_construct()
        if res:
            found_initial = True
            best_result = res
            best_scope = get_scope(res)
            break
            
    if not found_initial:
        # Fallback: Very large scope greedy
        # Or just return a trivial valid set if we can find one
        # For DTS, a solution always exists for large enough scope.
        # Let's try a wider search in greedy
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1]+1, 1000):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                        break
            rows.append(row)
        best_result = rows
        best_scope = get_scope(rows)

    # 2. Run Simulated Annealing
    # We will run SA for a portion of the time, then polish.
    sa_time_budget = time_limit * 0.8
    sa_end_time = start_time + sa_time_budget
    
    # Run SA from the best initial found
    # If we have multiple good initials, we could run SA on the best one.
    # Or run SA on a few different ones.
    
    # Let's run SA on the best initial found.
    if best_result:
        improved_rows, improved_scope = simulated_annealing(best_result, time_limit - sa_time_budget)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_result = improved_rows
            
    # 3. Polish
    if best_result:
        polished = polish(best_result)
        polished_scope = get_scope(polished)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_result = polished

    # 4. Final Check
    if best_result is None:
        # Should not happen
        return [[0, 1, 2, 3, 4, 5] for _ in range(n_rows)] # Invalid fallback, but prevents crash

    return best_result
