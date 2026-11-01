import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 145  # 5s safety margin
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            for d in get_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_solution():
        """Generate a valid solution using greedy approach."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def random_valid_solution():
        """Try to generate a random valid solution."""
        for attempt in range(100):
            used = set()
            rows = []
            valid = True
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Random jump
                    x += random.randint(1, 20)
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        valid = False
                        break
                    row.append(x)
                    used |= new_diffs
                if not valid:
                    break
                rows.append(row)
            if valid and len(rows) == n_rows:
                return rows
        return None
    
    # Start with greedy solution
    best_rows = greedy_solution()
    best_score = max_entry(best_rows)
    
    if time.time() > deadline:
        return best_rows
    
    # Local search: try to improve by modifying individual rows
    def try_improve(row, used_by_others, n_rows, k):
        """Try to find a better row given the differences used by other rows."""
        # Greedy with some randomness
        best_row = None
        best_max = float('inf')
        
        for _ in range(50):
            row = [0]
            x = 0
            valid = True
            for _ in range(k):
                # Try several random increments
                found = False
                for trial in range(30):
                    increment = random.randint(1, 15)
                    x += increment
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_by_others):
                        row.append(x)
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid:
                m = max(row)
                if m < best_max:
                    best_max = m
                    best_row = row[:]
        
        return best_row
    
    # Iterative local search
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        current_rows = best_rows[:]
        improved = False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            
            # Get differences used by all other rows
            used_by_others = set()
            for other_idx in range(n_rows):
                if other_idx != row_idx:
                    used_by_others |= get_diffs(current_rows[other_idx])
            
            # Try to improve this row
            new_row = try_improve(current_rows[row_idx], used_by_others, n_rows, k)
            
            if new_row is not None:
                # Check if this improvement doesn't conflict
                new_diffs = get_diffs(new_row)
                conflict = False
                for other_idx in range(n_rows):
                    if other_idx != row_idx:
                        if new_diffs & get_diffs(current_rows[other_idx]):
                            conflict = True
                            break
                
                if not conflict and max(new_row) < max(current_rows[row_idx]):
                    current_rows[row_idx] = new_row
                    improved = True
        
        if improved:
            current_score = max_entry(current_rows)
            if current_score < best_score:
                best_rows = current_rows
                best_score = current_score
    
    return best_rows
