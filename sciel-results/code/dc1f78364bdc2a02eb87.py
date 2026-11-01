import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Target parameters
    n = k + 1
    total_diffs = n_rows * (n * (n - 1) // 2)
    
    # 1. Generate initial population using a fast greedy method
    # We use a simple greedy to get a valid starting point, even if suboptimal.
    best_rows = None
    best_score = float('inf')
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Initialize with a few greedy runs to find a decent baseline
    # Since greedy is deterministic for this problem structure, it yields one specific result.
    # We will randomize the initial state for SA by permuting or slightly perturbing, 
    # but it's safer to start with a valid configuration and then "break" it slightly or 
    # use a different random initialization.
    
    # Actually, for SA, we need to allow invalid states during search.
    # Let's initialize with random rows that are strictly increasing but might have conflicts.
    
    def random_initial_rows():
        rows = []
        # Estimate a reasonable upper bound for random init to avoid huge penalties initially
        # The counting bound is around 105 for (7,5). Greedy is usually close to optimal or slightly worse.
        # Let's pick a random scope slightly larger than the greedy result to start.
        greedy_rows = greedy_construct()
        greedy_max = max(max(r) for r in greedy_rows)
        
        # Use greedy as the best known so far
        best_rows = greedy_rows
        best_score = max(max(r) for r in greedy_rows)
        
        # Create a random perturbation of the greedy rows to start SA
        # Or just start SA from the greedy rows.
        # To make SA effective, we need a good neighborhood.
        # Representation: list of rows, each row is list of ints.
        
        return greedy_rows

    initial_rows = random_initial_rows()
    current_rows = [row[:] for row in initial_rows]
    current_score = max(max(r) for r in current_rows)
    best_rows = [row[:] for row in current_rows]
    best_score = current_score
    
    # SA Parameters
    T = 50.0 # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    
    # Helper to calculate score and validity
    def evaluate(rows):
        used = set()
        max_entry = 0
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
                    if row[j] > max_entry:
                        max_entry = row[j]
        # Score: We want to minimize max_entry, but conflicts are fatal.
        # Penalty approach: Score = max_entry + conflicts * penalty
        # If conflicts > 0, it's invalid.
        # To guide search towards valid solutions, we use a large penalty.
        penalty = 10000
        return max_entry + conflicts * penalty, conflicts, max_entry
    
    current_eval, current_conflicts, current_max = evaluate(current_rows)
    best_eval, best_conflicts, best_max = evaluate(best_rows)
    
    while time.time() < deadline and T > min_T:
        # Generate neighbor
        # Move: Pick a random row, pick a random element (index > 0), and change its value.
        # Or: Pick a random row and shift a suffix? 
        # Simplest: Pick random row r, random index i (1..k), add delta to row[r][i].
        # To keep strictly increasing, we must ensure row[r][i] > row[r][i-1] and row[r][i+1] > row[r][i] (if i+1 exists).
        
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick an index to modify. Index 0 is always 0.
        i_idx = random.randint(1, n - 1)
        
        # Determine valid range for new value
        min_val = row[i_idx - 1] + 1
        # Max val is tricky because it affects subsequent elements. 
        # If we change row[i_idx], we might need to adjust subsequent elements to maintain increase.
        # Simpler move: "Insert/Remove/Shift" is complex.
        # Let's use a "Delta" move on a single element, and if it breaks ordering, we propagate the change to the right to maintain ordering.
        
        old_val = row[i_idx]
        # Random delta
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Enforce lower bound
        if new_val < min_val:
            new_val = min_val
            
        # If we increase a value, subsequent values must be > new_val.
        # If we decrease a value, subsequent values are already > old_val >= new_val, so they are valid.
        # However, if we decrease, we might create a conflict with previous values? No, previous are < old_val.
        # Wait, if we decrease row[i_idx], it must be > row[i_idx-1]. We handled min_val.
        
        # Let's try a different move: "Swap" or "Shift Suffix".
        # Shift suffix: Pick row r, pick split point s (1..k-1). 
        # Add delta to row[r][s], row[r][s+1], ..., row[r][k].
        # This preserves internal differences of the suffix? No, differences within suffix remain same.
        # Differences between prefix and suffix change.
        # This is a good move.
        
        s_idx = random.randint(1, n - 1)
        delta = random.randint(-10, 10)
        
        # Check if shifting maintains strictly increasing order
        # row[s-1] < row[s] + delta
        # row[s] + delta < row[s+1] + delta (always true)
        # ...
        # row[k-1] + delta < row[k] + delta (always true)
        
        if s_idx > 0 and row[s_idx - 1] >= row[s_idx] + delta:
            # Invalid move, try small delta or skip
            # Adjust delta to be valid
            if row[s_idx] + delta <= row[s_idx - 1]:
                delta = row[s_idx - 1] - row[s_idx] + 1
                if delta <= 0:
                    continue # Can't fix with positive delta? If row[s-1] >= row[s], they are not increasing. But they are.
                    # So delta must be > row[s-1] - row[s].
        
        # Apply move
        for j in range(s_idx, n):
            current_rows[r_idx][j] += delta
            
        # Evaluate
        new_eval, new_conflicts, new_max = evaluate(current_rows)
        
        # Acceptance criterion
        if new_eval < current_eval:
            current_eval = new_eval
            current_conflicts = new_conflicts
            current_max = new_max
            if new_eval < best_eval:
                best_eval = new_eval
                best_conflicts = new_conflicts
                best_max = new_max
                best_rows = [row[:] for row in current_rows]
        else:
            # Probabilistic acceptance
            delta_eval = new_eval - current_eval
            if random.random() < pow(2.71828, -delta_eval / T):
                current_eval = new_eval
                current_conflicts = new_conflicts
                current_max = new_max
            else:
                # Revert
                for j in range(s_idx, n):
                    current_rows[r_idx][j] -= delta

        T *= cooling_rate

    # Return best valid solution found
    if best_conflicts == 0:
        return best_rows
    else:
        # If no valid solution found (unlikely given greedy start), return greedy
        return greedy_construct()

# Keep the original greedy function available if needed, but construct is the entry point.
# The prompt asks to keep function names and signatures.
# construct(n_rows, k) is the main one.
