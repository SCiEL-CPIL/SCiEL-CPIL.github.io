"""Rewritten program: structural change to randomized local search with deadline guard."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within each row are distinct across all rows.
    
    Strategy: 
    1. Start with a greedy construction (fast baseline).
    2. Apply randomized local search: 
       - Represent state as list of rows (each row is list of k+1 ints starting at 0).
       - Objective: maximize validity; among valid, minimize max entry.
       - Moves: 
         a) Shift a single row by adding/subtracting a small delta to all elements except 0,
            keeping strictly increasing and >=0.
         b) Swap two elements within a row (maintaining sorted order).
         c) Insert a new element in a row and remove another (to change row structure).
       - Accept if it improves (or equal with some probability) the objective.
    3. Use deadline guard to return best valid solution found.
    """
    deadline = time.time() + 110  # 110s safety margin out of 120s
    
    # Helper: check if a set of rows is valid (all diffs distinct)
    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def objective(rows):
        # Returns (valid, max_entry). Lower is better for valid solutions.
        # Invalid solutions are worse than any valid.
        if is_valid(rows):
            return (0, max_entry(rows))
        else:
            return (1, max_entry(rows))
    
    # Greedy baseline
    def greedy_construct():
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
    
    best_rows = greedy_construct()
    best_obj = objective(best_rows)
    
    # Local search
    # We'll try to improve by local moves.
    # State: list of rows, each row is a list of ints (sorted, starts at 0).
    
    # Random seed for reproducibility within a run
    random.seed(42)
    
    def random_move(rows):
        """Generate a random perturbation of the rows."""
        rows = [list(r) for r in rows]
        move_type = random.choice(['shift', 'swap', 'insert_remove'])
        
        if move_type == 'shift':
            # Shift one row by delta (except first element)
            if n_rows > 0:
                ridx = random.randrange(n_rows)
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                row = rows[ridx]
                # Try to shift elements 1..end by delta
                new_row = [row[0]] + [row[i] + delta for i in range(1, len(row))]
                # Check strictly increasing and all >= 0
                if all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)) and new_row[0] == 0:
                    rows[ridx] = new_row
                else:
                    # Revert
                    rows[ridx] = row
        
        elif move_type == 'swap':
            # Swap two elements in a row (maintain sorted)
            if n_rows > 0:
                ridx = random.randrange(n_rows)
                row = rows[ridx]
                if len(row) >= 2:
                    i, j = random.sample(range(1, len(row)), 2)  # don't swap index 0
                    row[i], row[j] = row[j], row[i]
                    row.sort()
                    # Ensure starts at 0
                    if row[0] != 0:
                        # Revert
                        rows[ridx] = [list(r) for r in [row]][0]  # this is tricky
                        # Actually, let's just revert properly
                        row[i], row[j] = row[j], row[i]
                        row.sort()
        
        elif move_type == 'insert_remove':
            # In a random row, remove a random element (not 0) and insert a new one
            if n_rows > 0:
                ridx = random.randrange(n_rows)
                row = rows[ridx]
                if len(row) > 1:
                    # Remove a random element (index 1..end)
                    rem_idx = random.randrange(1, len(row))
                    removed = row.pop(rem_idx)
                    # Insert a new element at a random position
                    ins_idx = random.randrange(len(row) + 1)
                    # New value: around the neighbors
                    if ins_idx == 0:
                        new_val = row[0] + random.randint(1, 5)
                    elif ins_idx == len(row):
                        new_val = row[-1] + random.randint(1, 5)
                    else:
                        lo, hi = row[ins_idx-1], row[ins_idx]
                        if hi - lo > 1:
                            new_val = random.randint(lo+1, hi-1)
                        else:
                            new_val = hi + random.randint(1, 3)
                    row.insert(ins_idx, new_val)
                    row.sort()
                    # Ensure starts at 0
                    if row[0] != 0:
                        # Revert: put back removed, remove new
                        row.pop(ins_idx)
                        row.insert(rem_idx, removed)
                        row.sort()
        
        return rows
    
    # Simulated annealing / hill climbing
    current_rows = [list(r) for r in best_rows]
    current_obj = objective(current_rows)
    
    # Temperature schedule
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        new_rows = random_move(current_rows)
        new_obj = objective(new_rows)
        
        # Compare objectives: (valid_flag, max_entry)
        # Lower is better.
        if new_obj < current_obj:
            current_rows = new_rows
            current_obj = new_obj
            if current_obj < best_obj:
                best_rows = [list(r) for r in current_rows]
                best_obj = current_obj
        else:
            # Accept with probability exp(-delta/T)
            delta = 0
            if new_obj[0] != current_obj[0]:
                # Different validity: penalize invalid heavily
                if new_obj[0] > current_obj[0]:
                    delta = 100  # invalid -> valid is good, valid -> invalid is bad
                else:
                    delta = -100
            else:
                delta = new_obj[1] - current_obj[1]
            
            if delta > 0:
                # Worse: accept with prob exp(-delta/T)
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_rows = new_rows
                    current_obj = new_obj
            else:
                # Equal or better (shouldn't happen since we checked <)
                current_rows = new_rows
                current_obj = new_obj
        
        T *= cooling
    
    return best_rows
# EVOLVE-BLOCK-END
