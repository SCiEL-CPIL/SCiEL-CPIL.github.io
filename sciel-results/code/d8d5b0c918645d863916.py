import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    best_solution = None
    best_score = float('inf')
    
    def get_score(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
        
    def greedy_construct(randomize=False):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                # If randomize, we might skip some small values to explore different branches
                # But for a strong greedy, we stick to smallest valid.
                # To add randomness, we can shuffle the candidate order slightly or restart.
                # Let's stick to deterministic greedy for the base, and use restarts with different random seeds for local search.
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    found = True
                # Optimization: if x gets too large, we might want to backtrack, but greedy doesn't backtrack.
                # We rely on multiple restarts with different initial conditions if we modify the greedy.
                # For now, standard greedy.
            rows.append(row)
        return rows

    # The standard greedy is deterministic. To get different results, we need to modify the selection.
    # Let's implement a randomized greedy where we pick the next element from a set of candidates.
    def randomized_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Look ahead a bit? Or just pick smallest valid.
                # To make it randomized, we can add a small random offset to x before checking?
                # No, that breaks the "smallest" property which is usually good.
                # Instead, let's use a "lookahead" greedy or just run the deterministic greedy.
                # Wait, the deterministic greedy is the same every time.
                # We need to introduce variation.
                
                # Strategy: Pick the next element from a range [x_min, x_max] where x_min is the smallest valid.
                # If we pick a larger one, we might free up small differences for later rows?
                # Actually, picking the smallest valid is usually optimal for minimizing the max.
                
                # Let's try a different heuristic:
                # 1. Estimate the remaining space needed.
                # 2. If we are close to the limit, be more conservative.
                
                # For now, let's just run the deterministic greedy.
                # If it's too slow or not good enough, we'll add local search.
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Local Search: Try to reduce the max value.
    def local_search(rows):
        rows = [list(r) for r in rows]
        improved = True
        while improved:
            improved = False
            # Calculate used diffs
            used = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        used.add(r[j] - r[i])
                        
            # Try to reduce the largest element in each row
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                # Find the largest element
                max_val = row[-1]
                # Try to replace it with a smaller value
                # We need to check all positions, not just the last one, but the last one is the max.
                # Actually, we can try to reduce any element.
                # Let's try to reduce the last element first.
                for new_val in range(max_val - 1, 0, -1):
                    if new_val <= row[-2]: # Must be strictly increasing
                        continue
                    # Check if new_val is valid
                    # Remove old diffs involving row[-1]
                    old_diffs = {max_val - a for a in row[:-1]}
                    used -= old_diffs
                    
                    # Add new diffs
                    row[-1] = new_val
                    new_diffs = {new_val - a for a in row[:-1]}
                    if not new_diffs & used:
                        used |= new_diffs
                        improved = True
                        break
                    else:
                        row[-1] = max_val # Revert
                        used |= old_diffs
                # Also try to reduce other elements?
                # This might be complex. Let's stick to reducing the max for now.
                # A more powerful local search would try all elements.
                
        return rows

    # Run the greedy
    initial_rows = randomized_greedy()
    if is_valid(initial_rows):
        initial_rows = local_search(initial_rows)
        best_solution = initial_rows
        best_score = get_score(initial_rows)
        
    # If we have time, try to improve further?
    # For this specific small instance, the greedy + local search might be sufficient.
    # However, the greedy is deterministic.
    # To get a better result, we might need a smarter construction.
    
    # Let's try a "backtracking" approach for the first few rows to see if we can get a better base.
    # But given time constraints, let's stick to the current best found.
    
    if time.time() > deadline:
        return best_solution if best_solution else randomized_greedy()
        
    return best_solution
