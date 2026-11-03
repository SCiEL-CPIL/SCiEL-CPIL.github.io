import time
import random
import sys

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1
    
    # Pre-compute pairs for a row
    pairs = [(i, j) for i in range(row_len) for j in range(i+1, row_len)]
    n_pairs = len(pairs)
    
    def get_row_diffs(row):
        return [row[j] - row[i] for i, j in pairs]

    def calculate_initial_state(matrix):
        """
        Calculates the initial global state:
        - max_val: maximum entry in the matrix
        - used_diffs: set of all differences
        - conflict_count: number of duplicate differences
        """
        max_val = 0
        used_diffs = set()
        conflict_count = 0
        for r in range(n_rows):
            row = matrix[r]
            if row[-1] > max_val:
                max_val = row[-1]
            diffs = get_row_diffs(row)
            for d in diffs:
                if d in used_diffs:
                    conflict_count += 1
                else:
                    used_diffs.add(d)
        return max_val, used_diffs, conflict_count

    def greedy_initial():
        """
        Generates a valid initial solution using greedy approach.
        """
        matrix = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                last += 1
                new_diffs = {last - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last)
                    used_diffs |= new_diffs
            matrix.append(row)
        return matrix

    # Start with a valid greedy solution
    current_matrix = greedy_initial()
    current_max, current_diffs, current_conflicts = calculate_initial_state(current_matrix)
    
    best_matrix = [r[:] for r in current_matrix]
    best_max = current_max
    
    rng = random.Random(42)
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.999
    
    # We maintain the state incrementally.
    # current_max: max entry
    # current_diffs: set of all differences
    # current_conflicts: count of duplicate differences
    
    # To update efficiently when a row changes:
    # 1. Remove old diffs of the row from global set/count.
    # 2. Add new diffs of the row to global set/count.
    # 3. Update max_val if the last element of the changed row was the max or is now the max.
    
    def update_state(r_idx, old_row, new_row):
        """
        Updates global state variables after changing row r_idx from old_row to new_row.
        Returns new_max, new_diffs, new_conflicts.
        """
        nonlocal current_max, current_diffs, current_conflicts
        
        # Remove old differences
        old_diffs = get_row_diffs(old_row)
        for d in old_diffs:
            current_diffs.discard(d) # discard is safe, but we need to handle counts if we used a Counter.
            # Since we are using a set and a conflict count, this simple discard logic is flawed if there are duplicates.
            # We need a Counter or a dict of counts.
            pass
        
        # This incremental update with a set is tricky if there are conflicts (duplicates).
        # Given the small size (147 diffs), it might be faster to just recalculate from scratch 
        # but without the numpy overhead.
        # Let's stick to a full recalculation but optimized in pure Python.
        
        # Actually, let's just recalculate. 147 operations is very small.
        # The bottleneck in the previous code was likely the numpy conversion and function call overhead.
        
        new_max = 0
        new_diffs = set()
        new_conflicts = 0
        
        # Rebuild the matrix state for the calculation
        # We can just iterate over all rows
        for r in range(n_rows):
            row = current_matrix[r]
            if row[-1] > new_max:
                new_max = row[-1]
            for i, j in pairs:
                d = row[j] - row[i]
                if d in new_diffs:
                    new_conflicts += 1
                else:
                    new_diffs.add(d)
        
        return new_max, new_diffs, new_conflicts

    # Let's use a simpler approach: 
    # Keep current_matrix.
    # Calculate score from scratch each time but efficiently.
    
    def get_score(matrix):
        max_val = 0
        diffs = set()
        conflicts = 0
        for r in range(n_rows):
            row = matrix[r]
            if row[-1] > max_val:
                max_val = row[-1]
            for i, j in pairs:
                d = row[j] - row[i]
                if d in diffs:
                    conflicts += 1
                else:
                    diffs.add(d)
        # Penalty for conflicts is high
        return max_val + 1000 * conflicts, max_val, conflicts

    current_score, current_max, current_conflicts = get_score(current_matrix)
    
    while time.time() < DEADLINE:
        # Choose a random row
        r_idx = rng.randint(0, n_rows - 1)
        
        # Choose a move type
        # 1. Change the last element (most likely to reduce max)
        # 2. Change an internal element (to create space for reducing last element)
        
        if rng.random() < 0.5:
            # Try to reduce the last element
            i_idx = row_len - 1
            old_val = current_matrix[r_idx][i_idx]
            lo = current_matrix[r_idx][i_idx-1] + 1
            hi = old_val - 1 # Only allow reduction
            
            if lo > hi:
                continue
                
            new_val = rng.randint(lo, hi)
        else:
            # Change an internal element
            i_idx = rng.randint(1, row_len - 2)
            old_val = current_matrix[r_idx][i_idx]
            lo = current_matrix[r_idx][i_idx-1] + 1
            hi = current_matrix[r_idx][i_idx+1] - 1
            
            if lo > hi:
                continue
                
            new_val = rng.randint(lo, hi)
            
            # If we increased an internal element, we might need to increase the last one too?
            # No, we just pick a random value in the valid range.
            
        if new_val == old_val:
            continue
            
        old_row = current_matrix[r_idx][:]
        current_matrix[r_idx][i_idx] = new_val
        
        new_score, new_max, new_conflicts = get_score(current_matrix)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            current_score = new_score
            current_max = new_max
            current_conflicts = new_conflicts
            
            if current_conflicts == 0 and current_max < best_max:
                best_max = current_max
                best_matrix = [r[:] for r in current_matrix]
        else:
            current_matrix[r_idx] = old_row
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_matrix
