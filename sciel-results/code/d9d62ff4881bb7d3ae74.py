import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    # Safety margin: 120s limit, use 115s for search
    DEADLINE = t0 + 115.0
    row_len = k + 1
    
    # Pre-compute pairs for a row to speed up difference calculation
    row_pairs = [(i, j) for i in range(row_len) for j in range(i+1, row_len)]
    
    def get_row_diffs(row):
        """Calculate all positive differences for a single row."""
        return [row[j] - row[i] for i, j in row_pairs]
    
    def get_scope(rows):
        """Calculate the maximum entry in all rows."""
        return max(row[-1] for row in rows)
    
    def count_conflicts(counter):
        """Count the number of differences that appear more than once."""
        return sum(c - 1 for c in counter.values() if c > 1)
    
    def generate_initial_solution(seed, rand_factor=0.0):
        """
        Generate a valid initial solution.
        rand_factor > 0 introduces randomness in the greedy choice to diversify.
        """
        rng = random.Random(seed)
        matrix = []
        used_diffs = set()
        
        # Shuffle the order in which we build rows? No, rows are independent in construction 
        # but they compete for differences. 
        # To diversify, we can randomize the "greediness" of the choice.
        
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                # Determine the next value
                # Standard greedy: smallest valid
                # Randomized: pick from a window of valid values
                
                # Find the smallest valid value first
                candidate = last + 1
                valid_candidates = []
                
                # Search for a small window of valid candidates
                # If rand_factor is 0, we just take the first one.
                # If rand_factor > 0, we look ahead a bit.
                look_ahead = 1 + int(rand_factor * 10) 
                
                while True:
                    new_diffs = {candidate - a for a in row}
                    if not (new_diffs & used_diffs):
                        valid_candidates.append(candidate)
                        if len(valid_candidates) >= look_ahead:
                            break
                    candidate += 1
                    # Safety break
                    if candidate > 200: 
                        break
                
                if not valid_candidates:
                    # Fallback to pure greedy if window fails
                    candidate = last + 1
                    while True:
                        new_diffs = {candidate - a for a in row}
                        if not (new_diffs & used_diffs):
                            valid_candidates.append(candidate)
                            break
                        candidate += 1
                
                # Pick a candidate
                if rand_factor > 0 and len(valid_candidates) > 1:
                    # Bias towards smaller values but allow some randomness
                    # Weights: 10, 9, 8...
                    weights = [len(valid_candidates) - i for i in range(len(valid_candidates))]
                    chosen_val = rng.choices(valid_candidates, weights=weights)[0]
                else:
                    chosen_val = valid_candidates[0]
                
                last = chosen_val
                row.append(last)
                new_diffs = {last - a for a in row[:-1]} # Diffs with previous elements
                used_diffs |= new_diffs
            
            matrix.append(row)
        
        return matrix

    def build_row_avoiding(used_diffs, cap, rng, jitter=0.0):
        """
        Build a new row of length row_len starting at 0,
        avoiding differences in used_diffs.
        """
        row = [0]
        current_max = 0
        
        while len(row) < row_len:
            next_val = current_max + 1
            found = False
            # Limit search space to avoid infinite loops in tight caps
            # If cap is tight, this might fail quickly
            while next_val <= cap:
                new_diffs = {next_val - a for a in row}
                if not (new_diffs & used_diffs):
                    # If jitter > 0, sometimes skip a valid value to diversify
                    if jitter > 0 and rng.random() < jitter and next_val < cap:
                         next_val += 1
                         continue
                    row.append(next_val)
                    current_max = next_val
                    used_diffs = used_diffs | new_diffs
                    found = True
                    break
                next_val += 1
            
            if not found:
                return None
                
        return row

    # State variables
    best_matrix = None
    best_scope = float('inf')
    best_cost = float('inf')
    
    # We will run multiple restarts
    restart_count = 0
    max_restarts = 10 # Limit restarts to prevent infinite loop if stuck
    
    while time.time() < DEADLINE and restart_count < max_restarts:
        restart_count += 1
        seed = restart_count * 12345 + 67890
        rng = random.Random(seed)
        
        # 1. Initial Solution
        # Use a higher rand_factor for later restarts to explore more
        rand_factor = min(0.5, restart_count * 0.1)
        current_matrix = generate_initial_solution(seed, rand_factor)
        
        # Initialize difference counter
        diff_counter = Counter()
        for row in current_matrix:
            for d in get_row_diffs(row):
                diff_counter[d] += 1
                
        current_conflicts = count_conflicts(diff_counter)
        current_scope = get_scope(current_matrix)
        current_cost = current_scope + 1000 * current_conflicts
        
        # Update best if this initial solution is good
        if current_conflicts == 0 and current_scope < best_scope:
            best_scope = current_scope
            best_matrix = [r[:] for r in current_matrix]
            best_cost = current_cost

        # 2. Simulated Annealing / Local Search
        T = 50.0
        cooling = 0.9995
        min_T = 0.1
        
        steps_since_improvement = 0
        stagnation_limit = 5000 # Steps
        time_since_improvement = time.time()
        time_stagnation_limit = 5.0 # Seconds
        
        # We will run SA until deadline or stagnation triggers restart
        # But we must ensure we don't restart too frequently.
        
        while time.time() < DEADLINE:
            # Check stagnation
            if steps_since_improvement > stagnation_limit or (time.time() - time_since_improvement) > time_stagnation_limit:
                # If we are stuck, break out of SA loop to restart
                # But only if we haven't found a valid solution yet? 
                # No, if we found one, we keep it. If we are stuck in an invalid state, restart helps.
                # If we are stuck in a valid state but scope is not improving, restart might find a better one.
                break
            
            # Choose move type
            # 60% Single Entry Shift, 20% Row Rebuild, 20% Row Swap
            move_type = rng.random()
            r_idx = rng.randint(0, n_rows - 1)
            
            if move_type < 0.6:
                # --- Move Type 1: Shift a single entry ---
                i_idx = rng.randint(1, row_len - 1)
                
                old_val = current_matrix[r_idx][i_idx]
                
                lo = current_matrix[r_idx][i_idx-1] + 1
                if i_idx < row_len - 1:
                    hi = current_matrix[r_idx][i_idx+1] - 1
                else:
                    hi = current_scope + 10 
                
                if lo > hi:
                    continue
                
                if rng.random() < 0.7:
                    new_val = rng.randint(lo, min(old_val, hi))
                else:
                    new_val = rng.randint(lo, hi)
                    
                if new_val == old_val:
                    continue
                
                if i_idx > 0 and new_val <= current_matrix[r_idx][i_idx-1]:
                    continue
                if i_idx < row_len - 1 and new_val >= current_matrix[r_idx][i_idx+1]:
                    continue
                    
                old_row = current_matrix[r_idx]
                old_diffs = get_row_diffs(old_row)
                
                for d in old_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                
                new_row = old_row[:]
                new_row[i_idx] = new_val
                
                new_diffs = get_row_diffs(new_row)
                for d in new_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                    
                new_conflicts = count_conflicts(diff_counter)
                new_scope = get_scope(current_matrix)
                new_cost = new_scope + 1000 * new_conflicts
                delta = new_cost - current_cost
                
                if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                    current_matrix[r_idx] = new_row
                    current_conflicts = new_conflicts
                    current_scope = new_scope
                    current_cost = new_cost
                    
                    if new_conflicts == 0 and new_scope < best_scope:
                        best_scope = new_scope
                        best_matrix = [r[:] for r in current_matrix]
                        best_cost = new_cost
                        steps_since_improvement = 0
                        time_since_improvement = time.time()
                    else:
                        steps_since_improvement += 1
                else:
                    for d in new_diffs:
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    for d in old_diffs:
                        diff_counter[d] = diff_counter.get(d, 0) + 1
                    current_matrix[r_idx] = old_row
                    steps_since_improvement += 1
                    
            elif move_type < 0.8:
                # --- Move Type 2: Rebuild Row ---
                old_row = current_matrix[r_idx]
                old_diffs = get_row_diffs(old_row)
                
                for d in old_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                
                used_diffs = set(diff_counter.keys())
                
                cap = current_scope + rng.randint(0, 5)
                new_row = build_row_avoiding(used_diffs, cap, rng, jitter=0.1)
                
                if new_row is None:
                    for d in old_diffs:
                        diff_counter[d] = diff_counter.get(d, 0) + 1
                    steps_since_improvement += 1
                    continue
                
                new_diffs = get_row_diffs(new_row)
                for d in new_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                    
                new_conflicts = count_conflicts(diff_counter)
                new_scope = get_scope(current_matrix)
                new_cost = new_scope + 1000 * new_conflicts
                delta = new_cost - current_cost
                
                if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                    current_matrix[r_idx] = new_row
                    current_conflicts = new_conflicts
                    current_scope = new_scope
                    current_cost = new_cost
                    
                    if new_conflicts == 0 and new_scope < best_scope:
                        best_scope = new_scope
                        best_matrix = [r[:] for r in current_matrix]
                        best_cost = new_cost
                        steps_since_improvement = 0
                        time_since_improvement = time.time()
                    else:
                        steps_since_improvement += 1
                else:
                    for d in new_diffs:
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    for d in old_diffs:
                        diff_counter[d] = diff_counter.get(d, 0) + 1
                    current_matrix[r_idx] = old_row
                    steps_since_improvement += 1

            else:
                # --- Move Type 3: Swap Rows ---
                # Swapping rows doesn't change the set of differences, so it doesn't help with validity.
                # However, if we combine this with a "row index shuffle" in the representation, 
                # it doesn't help. 
                # Actually, swapping rows is useless for the objective (min max entry) and constraints (distinct diffs).
                # The set of differences is invariant to row permutation.
                # So this move is a no-op for cost. 
                # Let's replace this with "Row Shift" (add/subtract constant to a row? No, must start at 0).
                # Let's replace with "Row Reversal"? No, must increase.
                # Let's replace with "Random Row Perturbation": Pick a row and shift a random entry significantly.
                # Or just increase the probability of Rebuild.
                # For now, treat as Rebuild with higher jitter.
                old_row = current_matrix[r_idx]
                old_diffs = get_row_diffs(old_row)
                
                for d in old_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                
                used_diffs = set(diff_counter.keys())
                
                # Higher jitter to create very different row
                cap = current_scope + rng.randint(5, 15)
                new_row = build_row_avoiding(used_diffs, cap, rng, jitter=0.5)
                
                if new_row is None:
                    for d in old_diffs:
                        diff_counter[d] = diff_counter.get(d, 0) + 1
                    steps_since_improvement += 1
                    continue
                
                new_diffs = get_row_diffs(new_row)
                for d in new_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                    
                new_conflicts = count_conflicts(diff_counter)
                new_scope = get_scope(current_matrix)
                new_cost = new_scope + 1000 * new_conflicts
                delta = new_cost - current_cost
                
                if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                    current_matrix[r_idx] = new_row
                    current_conflicts = new_conflicts
                    current_scope = new_scope
                    current_cost = new_cost
                    
                    if new_conflicts == 0 and new_scope < best_scope:
                        best_scope = new_scope
                        best_matrix = [r[:] for r in current_matrix]
                        best_cost = new_cost
                        steps_since_improvement = 0
                        time_since_improvement = time.time()
                    else:
                        steps_since_improvement += 1
                else:
                    for d in new_diffs:
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    for d in old_diffs:
                        diff_counter[d] = diff_counter.get(d, 0) + 1
                    current_matrix[r_idx] = old_row
                    steps_since_improvement += 1

            # Cooling
            T *= cooling
            if T < min_T:
                T = min_T

    # If we never found a valid solution (shouldn't happen with greedy init), return the best attempt
    # But the greedy init is always valid.
    if best_matrix is None:
        # Fallback
        return generate_initial_solution(1, 0.0)
        
    return best_matrix
