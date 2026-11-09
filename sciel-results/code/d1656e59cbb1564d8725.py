import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Lower bound for max element
    # Total differences = n_rows * k * (k+1) / 2
    # For k=5, diffs per row = 15. Total = 105.
    # Min max element is at least 105 (if diffs are 1..105)
    # Actually, max element is sum of gaps.
    # The largest difference in a row is the max element.
    # So max element >= max difference used.
    # If we use diffs 1..105, max diff is 105.
    # So lower bound is 105.
    
    best_solution = None
    best_max_val = float('inf')
    
    def get_diffs(row):
        # row is list of k+1 ints, row[0]=0
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row, used_diffs):
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(len(row)-1)):
            return False
        diffs = get_diffs(row)
        if len(diffs) != len(row) * (len(row)-1) // 2:
            return False
        if diffs & used_diffs:
            return False
        return True

    def generate_random_row(used_diffs, max_val_hint=200):
        # Generate a random row that doesn't conflict
        for _ in range(1000):
            # Generate random gaps
            # Start with small gaps to keep max element small
            max_gap = max(10, max_val_hint // (k+1))
            gaps = [random.randint(1, max_gap) for _ in range(k)]
            # Sort gaps? No, gaps are just differences between consecutive elements.
            # But we want the row to be increasing.
            # row[0] = 0
            # row[i] = row[i-1] + gaps[i-1]
            row = [0]
            for g in gaps:
                row.append(row[-1] + g)
            
            # Check max element
            if row[-1] > max_val_hint * 2:
                continue
                
            if is_valid_row(row, used_diffs):
                return row
        return None

    def slide_down_solution(rows, used_diffs):
        # Local search to minimize max element
        # We try to reduce each element of each row
        changed = True
        iterations = 0
        while changed and time.time() < deadline:
            changed = False
            iterations += 1
            # Shuffle rows to avoid bias
            random.shuffle(rows)
            for i, row in enumerate(rows):
                # Try to reduce elements from largest to smallest
                for j in range(len(row) - 1, 0, -1):
                    # Try to decrease row[j] by 1
                    new_val = row[j] - 1
                    if new_val <= row[j-1]:
                        continue
                    
                    # Create new row
                    new_row = row[:]
                    new_row[j] = new_val
                    
                    # Check validity against other rows
                    # Remove current row's diffs from used_diffs
                    old_diffs = get_diffs(row)
                    temp_used = used_diffs - old_diffs
                    
                    # Check new row
                    new_diffs = get_diffs(new_row)
                    if new_diffs & temp_used:
                        continue
                    
                    # If valid, accept
                    row[j] = new_val
                    used_diffs = (used_diffs - old_diffs) | new_diffs
                    changed = True
                    break # Restart loop for this row
                    
        return rows, used_diffs

    def perturb_solution(rows, used_diffs):
        # Randomly increase some elements to escape local minima
        for i in range(n_rows):
            if random.random() < 0.5:
                # Pick a random element to increase
                j = random.randint(1, len(rows[i])-1)
                inc = random.randint(1, 10)
                old_val = rows[i][j]
                rows[i][j] = old_val + inc
                # Ensure increasing
                for m in range(j, len(rows[i])-1):
                    if rows[i][m] >= rows[i][m+1]:
                        rows[i][m+1] = rows[i][m] + 1
                # Update used_diffs
                old_diffs = get_diffs(rows[i])
                # This is tricky, we need to recompute
                # Let's just recompute all diffs for simplicity in perturbation
                pass
        
        # Recompute used_diffs
        new_used = set()
        for r in rows:
            new_used |= get_diffs(r)
        
        # Check if any conflicts were introduced (should not be if we just increased, but diffs change)
        # Actually, increasing elements changes diffs.
        # We need to ensure the new set of rows is valid.
        # If not, revert.
        if len(new_used) != n_rows * k * (k+1) // 2:
            # Conflict
            pass
        # This perturbation is risky and complex. 
        # Let's use a simpler restart strategy instead of complex perturbation.
        pass

    # Main loop: Multiple restarts
    while time.time() < deadline:
        # Generate initial solution
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            # Try to generate a row
            row = generate_random_row(used_diffs, max_val_hint=best_max_val if best_max_val < float('inf') else 200)
            if row is None:
                # Retry with larger hint
                row = generate_random_row(used_diffs, max_val_hint=300)
            if row is None:
                success = False
                break
            rows.append(row)
            used_diffs |= get_diffs(row)
        
        if not success:
            continue
            
        # Local search
        rows, used_diffs = slide_down_solution(rows, used_diffs)
        
        # Check result
        current_max = max(max(r) for r in rows)
        if current_max < best_max_val:
            best_max_val = current_max
            best_solution = [r[:] for r in rows]
            
        if best_max_val <= 105:
            break

    if best_solution is None:
        # Fallback to greedy if no solution found (should not happen)
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
        
    return best_solution
# EVOLVE-BLOCK-END
