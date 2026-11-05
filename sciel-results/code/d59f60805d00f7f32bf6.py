import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Problem parameters
    # Each row has k+1 elements: a_0=0, a_1, ..., a_k
    # Total differences per row: D = k*(k+1)//2
    # Total differences in system: T = n_rows * D
    D = k * (k + 1) // 2
    T = n_rows * D
    
    # Lower bound for M (largest entry)
    # We need T distinct differences in [1, M]. So M >= T.
    # Also, the largest entry is at least k (since it's increasing by at least 1).
    # But T is usually much larger.
    lower_bound = T
    
    def validate(rows):
        """Check if the set of rows is valid and return the max entry."""
        used_diffs = set()
        max_val = 0
        for row in rows:
            # Check increasing
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return -1
            max_val = max(max_val, row[-1])
            # Check differences
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return -1
                    used_diffs.add(d)
        return max_val

    def greedy_construct():
        """
        Construct a valid solution using a greedy approach.
        This provides an initial upper bound.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                val = row[-1] + 1
                while True:
                    conflict = False
                    for i in range(len(row)):
                        d = val - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                    if not conflict:
                        break
                    val += 1
                row.append(val)
                for i in range(len(row) - 1):
                    used_diffs.add(val - row[i])
            rows.append(row)
        return rows

    # Initial solution
    best_rows = greedy_construct()
    best_score = validate(best_rows)
    
    if best_score == lower_bound:
        return best_rows

    # Local Search / Simulated Annealing
    # State: list of rows
    # Move: Pick a random row, pick a random element (except 0), 
    #       and either increment or decrement it slightly, or swap with another?
    #       Simpler: Pick a random row, and try to replace one of its elements 
    #       with a new random value that maintains increasing order, 
    #       then check validity.
    
    # A better move for difference sets:
    # Since differences are the constraint, maybe we should work with the differences directly?
    # No, the structure "increasing integers" is hard to map back to differences without solving for the row.
    
    # Let's stick to modifying the rows directly.
    # Objective: Minimize max_entry.
    # Penalty: If invalid, large penalty.
    
    def get_score(rows):
        """
        Calculate a score for the local search.
        Lower is better.
        If valid: max_entry
        If invalid: max_entry + penalty * (number of conflicts)
        """
        used_diffs = set()
        conflicts = 0
        max_val = 0
        for row in rows:
            # Check increasing
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    conflicts += 1
                    # If not increasing, differences might be weird, but we count it as conflict
            max_val = max(max_val, row[-1] if row[-1] > 0 else 0)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        conflicts += 1
                    else:
                        used_diffs.add(d)
        # Penalty should be large enough to prioritize validity over minimizing max
        # But we want to minimize max among valid solutions.
        # So: score = max_val + 1000 * conflicts
        return max_val + 1000 * conflicts

    # Simulated Annealing Parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.999
    max_iters = 200000 # Adjust based on time
    
    current_rows = [row[:] for row in best_rows]
    current_score = get_score(current_rows)
    
    best_rows_sa = [row[:] for row in current_rows]
    best_score_sa = current_score
    
    # Random seed for reproducibility in a run, but different across restarts?
    # We will do multiple restarts.
    
    num_restarts = 10
    time_per_restart = (deadline - time.time() - 5.0) / num_restarts if num_restarts > 0 else 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize for this restart
        # Option 1: Start from best found so far
        # Option 2: Start from a new random greedy solution (maybe with randomization in greedy)
        
        # Let's try starting from the best found so far, but perturbed.
        current_rows = [row[:] for row in best_rows_sa]
        
        # Perturb
        for _ in range(10):
            r_idx = random.randint(0, n_rows - 1)
            e_idx = random.randint(1, k) # Don't change 0
            # Change value by +/- 1..5
            delta = random.randint(-5, 5)
            if delta == 0: continue
            old_val = current_rows[r_idx][e_idx]
            new_val = old_val + delta
            # Ensure increasing
            if e_idx > 0 and new_val <= current_rows[r_idx][e_idx - 1]:
                continue
            if e_idx < k and new_val >= current_rows[r_idx][e_idx + 1]:
                continue
            if new_val < 0:
                continue
            current_rows[r_idx][e_idx] = new_val
            
        current_score = get_score(current_rows)
        
        if current_score < best_score_sa:
            best_score_sa = current_score
            best_rows_sa = [row[:] for row in current_rows]
        
        T = T_start
        iter_count = 0
        max_iters_this_restart = int(10000 * time_per_restart / 0.1) # Rough estimate
        if max_iters_this_restart < 100: max_iters_this_restart = 100
        
        while T > T_end and iter_count < max_iters_this_restart:
            if time.time() > deadline - 5.0:
                break
                
            # Generate neighbor
            new_rows = [row[:] for row in current_rows]
            r_idx = random.randint(0, n_rows - 1)
            e_idx = random.randint(1, k)
            
            # Strategy: Try to move element towards 0 or towards max?
            # Or just random small step.
            # To minimize max, we should bias towards smaller values.
            # But we need to maintain validity.
            
            # Let's try a random step in [-3, 3]
            step = random.randint(-3, 3)
            if step == 0:
                step = 1 if random.random() > 0.5 else -1
                
            old_val = new_rows[r_idx][e_idx]
            new_val = old_val + step
            
            # Constraints
            valid_move = True
            if new_val < 0:
                valid_move = False
            if e_idx > 0 and new_val <= new_rows[r_idx][e_idx - 1]:
                valid_move = False
            if e_idx < k and new_val >= new_rows[r_idx][e_idx + 1]:
                valid_move = False
            
            if not valid_move:
                # Try to fix by adjusting neighbors? No, just reject move.
                # Or try a different e_idx?
                # For now, just reject.
                iter_count += 1
                T *= cooling_rate
                continue
            
            new_rows[r_idx][e_idx] = new_val
            
            new_score = get_score(new_rows)
            
            delta_score = new_score - current_score
            
            if delta_score < 0:
                # Accept
                current_rows = new_rows
                current_score = new_score
                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_rows_sa = [row[:] for row in current_rows]
            else:
                # Metropolis criterion
                if random.random() < pow(2.718281828, -delta_score / T):
                    current_rows = new_rows
                    current_score = new_score
            
            iter_count += 1
            T *= cooling_rate
            
        # End of restart
        if best_score_sa < best_score:
            best_score = best_score_sa
            best_rows = [row[:] for row in best_rows_sa]
            
        if best_score == lower_bound:
            break
            
    return best_rows
