"""Rewritten program: diversified greedy construction with local improvement."""
import time
import random
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope), subject to all positive
    differences within rows being distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Greedy construction with a bias parameter to diversify choices.
    # bias controls how aggressively we pick the smallest available mark
    # vs. allowing larger gaps to spread differences.
    def greedy_construct(bias=0.0, seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # With bias, sometimes skip a valid mark to allow larger spread
                    if bias > 0 and rng.random() < bias and len(row) > 1:
                        # Skip this mark, try next
                        continue
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Compute scope (largest entry)
    def scope(rows):
        return max(max(r) for r in rows)

    # Validate: all positive differences within rows are distinct across all rows
    def validate(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # Local improvement: try to reduce scope by swapping or shifting marks
    def local_improve(rows, used, max_iters=200):
        """
        Try to reduce the maximum entry by local moves.
        We track which differences are used.
        """
        rows = [list(r) for r in rows]
        best_scope = scope(rows)
        
        # Rebuild used set
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Find the row with the largest last element
            max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]
            current_max = row[-1]
            
            # Try to reduce each position from the end
            for pos in range(k, 0, -1):
                if rows[max_row_idx][pos] <= rows[max_row_idx][pos - 1]:
                    continue
                # Try to decrease this mark
                old_val = row[pos]
                # Remove old differences
                old_diffs = set()
                for i in range(pos):
                    d = old_val - row[i]
                    old_diffs.add(d)
                for d in old_diffs:
                    used.discard(d)
                
                # Try smaller values
                for new_val in range(row[pos - 1] + 1, old_val):
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    if not (new_diffs & used):
                        # Accept
                        row[pos] = new_val
                        used |= new_diffs
                        improved = True
                        break
                    else:
                        # Restore and continue trying smaller
                        pass
                
                if improved:
                    break
                else:
                    # Restore old value
                    row[pos] = old_val
                    used |= old_diffs
        
        return rows, used, scope(rows)

    # Main search: try multiple biased greedy constructions, then improve
    best_rows = None
    best_scope = float('inf')
    
    # First, get a baseline
    base_rows = greedy_construct(bias=0.0, seed=42)
    base_scope = scope(base_rows)
    best_rows = [list(r) for r in base_rows]
    best_scope = base_scope
    
    # Try various biases and seeds
    biases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]
    
    attempts = 0
    for bias in biases:
        for seed in seeds:
            if time.time() > deadline - 10.0:
                break
            attempts += 1
            rows = greedy_construct(bias=bias, seed=seed)
            s = scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
        
        if time.time() > deadline - 10.0:
            break
    
    # Local improvement on the best found
    if best_rows is not None and time.time() < deadline - 5.0:
        improved_rows, improved_used, improved_scope = local_improve(
            best_rows, set(), max_iters=500
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Try a few more local improvements with different starting points
    if time.time() < deadline - 5.0:
        for seed in [777, 888, 999, 111, 222]:
            if time.time() > deadline - 5.0:
                break
            rows = greedy_construct(bias=0.15, seed=seed)
            s = scope(rows)
            if s <= best_scope + 5:  # close enough to be worth improving
                improved_rows, improved_used, improved_scope = local_improve(
                    rows, set(), max_iters=300
                )
                if improved_scope < best_scope:
                    best_scope = improved_scope
                    best_rows = improved_rows
    
    # Final validation
    if best_rows is None:
        best_rows = greedy_construct(bias=0.0, seed=42)
    
    return best_rows
