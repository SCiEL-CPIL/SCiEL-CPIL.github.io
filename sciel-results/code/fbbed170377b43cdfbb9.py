import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within rows are distinct across all rows.
    
    Strategy: iterative local search with conflict resolution.
    """
    deadline = time.time() + 130  # safety margin before 150s limit
    num_differences = n_rows * k * (k + 1) // 2
    
    # Lower bound: need enough differences
    # Upper bound: start with a reasonable estimate
    # For k=6, n_rows=7: known best is 112, counting bound 105
    
    # Start with greedy construction
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)
    
    # Try local search to improve
    rows = best_rows[:]
    max_val = best_max
    
    # Iterative improvement
    while time.time() < deadline - 5:
        improved = False
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Pick a random row to modify
            ri = random.randrange(n_rows)
            old_row = rows[ri][:]
            
            # Try to rebuild this row with smaller max
            # Compute used differences excluding this row
            used_diffs = set()
            for j in range(n_rows):
                if j == ri:
                    continue
                row = rows[j]
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        used_diffs.add(row[b] - row[a])
            
            # Try to find a valid row with max <= max_val - 1
            new_max = max_val - 1
            new_row = _find_row_with_max_k(k, new_max, used_diffs)
            
            if new_row is not None:
                rows[ri] = new_row
                max_val = new_max
                improved = True
                break
            else:
                # Try random perturbation of the row
                if random.random() < 0.3:
                    # Slightly perturb the row and see if we can resolve
                    perturbed = _perturb_row(old_row, k, used_diffs)
                    if perturbed is not None:
                        rows[ri] = perturbed
                        # Recompute max_val
                        max_val = max(max(r) for r in rows)
                        if max_val < best_max:
                            best_rows = rows[:]
                            best_max = max_val
                            improved = True
                        break
        
        if not improved:
            # Try a different strategy: swap elements between rows
            success = _try_swap_improve(rows, k, deadline)
            if success:
                new_max_val = max(max(r) for r in rows)
                if new_max_val < best_max:
                    best_rows = rows[:]
                    best_max = new_max_val
            
            if random.random() < 0.2:
                # Occasional restart with random initialization
                rows = _random_construct(n_rows, k, max_val)
                max_val = max(max(r) for r in rows)
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row with smallest possible max."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _random_construct(n_rows, k, max_bound):
    """Random construction with constraint on max value."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current = 0
        attempts = 0
        while len(row) < k + 1 and attempts < 10000:
            attempts += 1
            # Randomly pick next value
            next_val = random.randint(current + 1, max_bound)
            new_diffs = {next_val - a for a in row}
            if not (new_diffs & used):
                row.append(next_val)
                used |= new_diffs
                current = next_val
        if len(row) < k + 1:
            # Fall back to greedy for this row
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows


def _find_row_with_max_k(k, max_val, used_diffs):
    """Find a valid row of k+1 elements starting at 0 with max <= max_val."""
    if max_val < k:
        return None
    
    # Backtracking search
    def backtrack(row, current_max):
        if len(row) == k + 1:
            return row[:]
        
        if current_max > max_val:
            return None
        
        # Try each possible next value
        start = row[-1] + 1
        for next_val in range(start, max_val + 1):
            new_diffs = {next_val - a for a in row}
            if not (new_diffs & used_diffs):
                result = backtrack(row + [next_val], max(current_max, next_val))
                if result is not None:
                    return result
        return None
    
    result = backtrack([0], 0)
    return result


def _perturb_row(old_row, k, used_diffs):
    """Perturb a row to find a different valid row."""
    # Try shifting some elements
    for _ in range(50):
        new_row = old_row[:]
        # Randomly change one or two elements
        for _ in range(random.randint(1, 2)):
            idx = random.randint(1, k)  # Don't change the 0
            new_row[idx] = old_row[idx] + random.randint(-3, 3)
        
        # Ensure strictly increasing
        valid = True
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i-1]:
                valid = False
                break
        if not valid:
            continue
        
        # Check differences
        row_diffs = set()
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                d = new_row[j] - new_row[i]
                if d in used_diffs or d in row_diffs:
                    valid = False
                    break
            if not valid:
                break
            row_diffs.update({new_row[j] - new_row[i] for j in range(i+1, len(new_row))})
        
        if valid:
            return new_row
    
    return None


def _try_swap_improve(rows, k, deadline):
    """Try to improve by swapping or adjusting elements between rows."""
    n_rows = len(rows)
    max_val = max(max(r) for r in rows)
    
    # Find the row with the largest element
    max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
    max_elem = max(rows[max_row_idx])
    
    if max_elem <= 0:
        return False
    
    # Compute used diffs excluding max_row_idx
    used_diffs = set()
    for j in range(n_rows):
        if j == max_row_idx:
            continue
        row = rows[j]
        for a in range(len(row)):
            for b in range(a+1, len(row)):
                used_diffs.add(row[b] - row[a])
    
    # Try to find a better row
    new_row = _find_row_with_max_k(k, max_elem - 1, used_diffs)
    if new_row is not None:
        rows[max_row_idx] = new_row
        return True
    
    return False

# EVOLVE-BLOCK-END
