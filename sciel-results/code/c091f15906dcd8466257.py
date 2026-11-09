import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within each row must be distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # For the specific task: n_rows=4, k=7 (8 entries per row)
    # But we should handle general cases.
    
    # The problem is equivalent to finding n_rows disjoint rulers of length k
    # such that all differences are unique across all rulers.
    
    # Strategy:
    # 1. Start with a greedy construction to get an initial solution.
    # 2. Use local search / simulated annealing to improve (reduce scope).
    
    def greedy_build():
        """Build rows greedily, trying to minimize the last element."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Try to build a row with minimal last element
            # Use backtracking with pruning
            def backtrack(row, start_val):
                if len(row) == k + 1:
                    return True
                for v in range(start_val, 200):  # upper bound guess
                    new_diffs = set()
                    conflict = False
                    for a in row:
                        d = v - a
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if not conflict:
                        row.append(v)
                        old_used = set(used_diffs)
                        used_diffs.update(new_diffs)
                        if backtrack(row, v + 1):
                            return True
                        used_diffs.clear()
                        used_diffs.update(old_used)
                        row.pop()
                return False
            
            # Try increasing starting search
            row = [0]
            found = False
            for max_scope in range(k, 300):
                used_diffs_snapshot = set(used_diffs)
                if backtrack(row, 1):
                    found = True
                    break
                used_diffs.clear()
                used_diffs.update(used_diffs_snapshot)
                row = [0]
            
            if not found:
                # Fallback: simple greedy
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used_diffs:
                        row.append(x)
                        used_diffs |= new
            
            rows.append(row)
        
        return rows
    
    # Initial greedy solution
    rows = greedy_build()
    best_scope = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    
    def compute_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    # Local search: try to reduce the scope by modifying individual rows
    def try_improve_row(row_idx, rows, used_diffs_other):
        """Try to find a better row for row_idx that reduces its max element."""
        current_max = max(rows[row_idx])
        current_row = list(rows[row_idx])
        
        # Collect differences from other rows
        other_diffs = set()
        for i, row in enumerate(rows):
            if i == row_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    other_diffs.add(row[b] - row[a])
        
        # Try to find a row with smaller max element
        # Backtracking search for a valid row with max < current_max
        def find_row_with_max(max_allowed):
            row = [0]
            def bt(start_val):
                if len(row) == k + 1:
                    return True
                for v in range(start_val, max_allowed + 1):
                    if v > max_allowed:
                        break
                    new_diffs = set()
                    conflict = False
                    for a in row:
                        d = v - a
                        if d in other_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if not conflict:
                        row.append(v)
                        old = set(other_diffs)
                        other_diffs.update(new_diffs)
                        if bt(v + 1):
                            return True
                        other_diffs.clear()
                        other_diffs.update(old)
                        row.pop()
                return False
            
            return bt(1)
        
        # Try to reduce max element
        for new_max in range(current_max - 1, k, -1):
            if time.time() > deadline:
                break
            row = [0]
            # Save and restore other_diffs
            saved_other_diffs = set(other_diffs)
            if find_row_with_max(new_max):
                # Restore
                other_diffs.clear()
                other_diffs.update(saved_other_diffs)
                return list(row)
            other_diffs.clear()
            other_diffs.update(saved_other_diffs)
        
        return None
    
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_rows):
            if time.time() > deadline:
                break
            new_row = try_improve_row(i, rows, None)
            if new_row is not None and max(new_row) < max(rows[i]):
                rows[i] = new_row
                improved = True
    
    # Simulated annealing for further improvement
    def neighbor(rows):
        """Generate a neighbor by modifying one row slightly."""
        new_rows = [list(r) for r in rows]
        row_idx = random.randint(0, n_rows - 1)
        row = new_rows[row_idx]
        
        # Try to shift some elements
        # Pick a random position and try to change it
        pos = random.randint(1, k)  # don't change 0
        old_val = row[pos]
        
        # Try a new value
        new_val = old_val + random.choice([-2, -1, 1, 2])
        if new_val <= 0 or new_val >= row[pos + 1] if pos < k else False:
            return None
        if pos < k and new_val >= row[pos + 1]:
            return None
        if pos > 0 and new_val <= row[pos - 1]:
            return None
        
        # Check if this change is valid
        # Remove old differences, add new ones
        old_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_diffs.add(row[j] - row[i])
        
        row[pos] = new_val
        new_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                new_diffs.add(row[j] - row[i])
        
        # Check for conflicts with other rows
        for i in range(n_rows):
            if i == row_idx:
                continue
            other_row = new_rows[i]
            for a in range(len(other_row)):
                for b in range(a + 1, len(other_row)):
                    d = other_row[b] - other_row[a]
                    if d in new_diffs:
                        row[pos] = old_val
                        return None
        
        return new_rows
    
    # Simple hill climbing with restarts
    current_rows = [list(r) for r in rows]
    current_scope = compute_scope(current_rows)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random perturbation
        perturbed = [list(r) for r in current_rows]
        for _ in range(3):
            row_idx = random.randint(0, n_rows - 1)
            row = perturbed[row_idx]
            pos = random.randint(1, k)
            old_val = row[pos]
            new_val = old_val + random.randint(-5, 5)
            if new_val <= 0:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            row[pos] = new_val
        
        if not is_valid(perturbed):
            continue
        
        perturbed_scope = compute_scope(perturbed)
        if perturbed_scope < current_scope:
            current_rows = perturbed
            current_scope = perturbed_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
    
    if compute_scope(best_rows) < best_scope:
        best_scope = compute_scope(best_rows)
    
    return best_rows
