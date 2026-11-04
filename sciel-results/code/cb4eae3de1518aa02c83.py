"""Difference triangle set (7,5) — optimized construction."""
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within each row are distinct across all rows.
    
    Uses a randomized backtracking / local search approach with
    multiple restarts and greedy heuristics.
    """
    if n_rows <= 0 or k <= 0:
        return []
    
    # Total number of differences needed: n_rows * (k+1)*k/2
    num_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on largest entry: need at least num_diffs distinct positive integers
    # The differences range from 1 to M where M is the largest entry.
    # So M must satisfy M*(M+1)/2 >= num_diffs
    low = 1
    while low * (low + 1) // 2 < num_diffs:
        low += 1
    # Add a small margin
    low = max(low, k)
    
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple random seeds and strategies
    rng = random.Random(42)
    
    # Strategy 1: Greedy with randomization and backtracking
    for attempt in range(200):
        # Slightly vary the lower bound to explore different spaces
        target_max = low + rng.randint(0, 10)
        
        rows = _construct_greedy_random(n_rows, k, target_max, rng)
        if rows is not None:
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope == low:
                    break  # Hit the lower bound, can't do better
    
    # Strategy 2: Local search / hill climbing on partial assignments
    if best_rows is None:
        # Fallback to simple greedy (original approach) as safety net
        best_rows = _construct_simple_greedy(n_rows, k)
    else:
        # Try to improve with local search
        improved = _local_search_improve(best_rows, n_rows, k, rng, max_iters=5000)
        if improved is not None:
            scope = max(max(row) for row in improved)
            if scope < best_scope:
                best_scope = scope
                best_rows = improved
    
    # Final validation
    if best_rows is None:
        # Absolute fallback
        best_rows = _construct_simple_greedy(n_rows, k)
    
    return best_rows


def _construct_simple_greedy(n_rows, k):
    """Original greedy approach as fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _construct_greedy_random(n_rows, k, target_max, rng):
    """
    Greedy construction with randomization.
    Tries to fit all rows within target_max.
    Returns list of rows or None if failed.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        # Try multiple random orders for placing the remaining k elements
        success = False
        
        for trial in range(50):
            row = [0]
            temp_used = set(used)
            
            # Randomize the order in which we pick positions
            # We'll greedily add elements, but with some randomness in tie-breaking
            placed = []
            current_row = [0]
            
            for _ in range(k):
                # Find all valid next positions
                valid_positions = []
                for x in range(1, target_max + 1):
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & temp_used):
                        valid_positions.append((x, new_diffs))
                
                if not valid_positions:
                    break
                
                # Pick randomly from valid positions, but bias toward smaller ones
                # to keep the row compact
                if rng.random() < 0.7:
                    # Prefer smaller positions
                    valid_positions.sort(key=lambda t: t[0])
                    # Pick from the first few
                    top_n = min(5, len(valid_positions))
                    chosen = rng.choice(valid_positions[:top_n])
                else:
                    chosen = rng.choice(valid_positions)
                
                x, new_diffs = chosen
                current_row.append(x)
                temp_used |= new_diffs
            
            if len(current_row) == k + 1:
                # Check that all differences are still unique when combined with global used
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        row_diffs.add(current_row[j] - current_row[i])
                
                if not (row_diffs & used):
                    used |= row_diffs
                    rows.append(current_row)
                    success = True
                    break
        
        if not success:
            return None
    
    return rows


def _local_search_improve(rows, n_rows, k, rng, max_iters=5000):
    """
    Try to reduce the scope by modifying rows.
    Simple hill climbing: try to replace a row with a better one.
    """
    if rows is None:
        return None
    
    best_rows = [list(r) for r in rows]
    best_scope = max(max(row) for row in best_rows)
    
    for _ in range(max_iters):
        # Pick a random row to modify
        idx = rng.randint(0, n_rows - 1)
        
        # Compute used differences excluding this row
        all_diffs = set()
        for i, row in enumerate(best_rows):
            if i == idx:
                continue
            for j in range(len(row)):
                for l in range(j + 1, len(row)):
                    all_diffs.add(row[l] - row[j])
        
        # Try to find a better row for position idx
        old_row = best_rows[idx]
        old_scope = max(old_row)
        
        # Try several random replacements
        for _ in range(20):
            new_row = _generate_random_row(k, old_scope, rng)
            if new_row is None:
                continue
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            if new_diffs & all_diffs:
                continue
            
            new_scope = max(new_row)
            if new_scope < old_scope:
                best_rows[idx] = new_row
                all_diffs |= new_diffs
                best_scope = max(max(r) for r in best_rows)
                break
    
    return best_rows


def _generate_random_row(k, max_val, rng):
    """Generate a random row of k+1 increasing integers starting at 0."""
    if max_val < k:
        return None
    
    # Choose k-1 random values from 1 to max_val-1 (leaving room for the last)
    if k == 1:
        return [0, rng.randint(1, max_val)]
    
    candidates = list(range(1, max_val))
    if len(candidates) < k - 1:
        return None
    
    chosen = sorted(rng.sample(candidates, k - 1))
    row = [0] + chosen
    # Optionally add a final value larger than the last chosen
    if rng.random() < 0.3 and max_val > chosen[-1]:
        row.append(rng.randint(chosen[-1] + 1, max_val))
    
    # Ensure strictly increasing
    for i in range(1, len(row)):
        if row[i] <= row[i-1]:
            return None
    
    return row
