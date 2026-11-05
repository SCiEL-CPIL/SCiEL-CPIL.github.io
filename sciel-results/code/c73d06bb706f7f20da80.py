"""
Difference triangle set (7,5) solver.

Strategy:
1. Use a Simulated Annealing approach on the set of rows.
2. Moves include:
   a. Element Shrinking: Reducing individual elements to minimize scope.
   b. Row Regeneration: Replacing an entire row with a new candidate that
      is compatible with the others. This allows the algorithm to escape
      local minima by changing the structural composition of the solution.
3. This structural change avoids the computational overhead of maximum clique
   and allows for more flexible exploration of the solution space.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _generate_row_greedy(used, k, rng, pool_size=3):
    """
    Greedily construct a row of k+1 integers starting at 0.
    At each step, pick from the top 'pool_size' smallest valid candidates.
    Returns (row, diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        # Pick randomly from candidates, weighted toward smaller
        if rng.random() < 0.5:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, frozenset(new_diffs)


def _generate_row_greedy_pure(used, k):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, frozenset(new_diffs)


def _local_search_scope(rows, k, rng, deadline):
    """
    Aggressive local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows with multiple strategies.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        row_indices = list(range(len(rows)))
        rng.shuffle(row_indices)
        
        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]
            
            # Try to reduce the last element
            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    row[-1] = new_last
                    used = (used - old_diffs) | new_diffs
                    improved = True
                    break
            
            # Try to reduce other elements
            for pos in range(len(row) - 2, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break
    
    return rows


def _simulated_annealing(rows, k, rng, deadline):
    """
    Simulated Annealing to optimize the set of rows.
    Moves:
    1. Shrink an element.
    2. Regenerate a row (replace with a new valid row).
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    temperature = 10.0
    cooling_rate = 0.95
    min_temp = 0.01
    
    while time.time() < deadline and temperature > min_temp:
        move_type = rng.random()
        
        if move_type < 0.7:
            # Move 1: Try to shrink a random element
            idx = rng.randint(0, len(rows) - 1)
            pos = rng.randint(1, len(rows[idx]) - 1)
            row = rows[idx]
            old_val = row[pos]
            
            # Try a few random smaller values
            for _ in range(5):
                new_val = rng.randint(row[pos - 1] + 1, old_val - 1) if old_val > row[pos - 1] + 1 else -1
                if new_val < 0 or new_val >= old_val:
                    continue
                
                old_diffs = set()
                for i in range(len(row)):
                    if i == pos: continue
                    if i < pos: old_diffs.add(old_val - row[i])
                    else: old_diffs.add(row[i] - old_val)
                
                new_diffs = set()
                for i in range(len(row)):
                    if i == pos: continue
                    if i < pos: new_diffs.add(new_val - row[i])
                    else: new_diffs.add(row[i] - new_val)
                
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    # Valid move
                    new_scope = max(max(r) for r in rows) # Approximation, exact calc below
                    # Calculate exact new scope
                    row[pos] = new_val
                    new_scope = _total_scope(rows)
                    delta = new_scope - current_scope
                    
                    if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                        current_scope = new_scope
                        used = (used - old_diffs) | new_diffs
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                    else:
                        row[pos] = old_val # Revert
                    break
        else:
            # Move 2: Regenerate a row
            idx = rng.randint(0, len(rows) - 1)
            old_row = rows[idx]
            old_diffs = _diffs_of_row(old_row)
            
            # Generate a new row that is compatible with the rest
            used_without_idx = set()
            for i, r in enumerate(rows):
                if i != idx:
                    used_without_idx |= _diffs_of_row(r)
            
            new_row, new_diffs = _generate_row_greedy(used_without_idx, k, rng, pool_size=5)
            
            if new_row is not None:
                # Check if it's actually compatible (should be by construction, but double check)
                if not (new_diffs & used_without_idx):
                    rows[idx] = new_row
                    # Recalculate used set
                    used = _all_diffs_used(rows)
                    new_scope = _total_scope(rows)
                    delta = new_scope - current_scope
                    
                    if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                    else:
                        rows[idx] = old_row # Revert
                        used = _all_diffs_used(rows)
        
        temperature *= cooling_rate
        
        # Occasionally reset temperature to explore more
        if rng.random() < 0.05:
            temperature = max(min_temp, temperature * 0.5)
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Initial construction
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        # Very simple fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Local search to reduce scope
    rows = _local_search_scope(rows, k, rng, deadline - 30)
    
    # Simulated Annealing to escape local minima and explore structure
    rows = _simulated_annealing(rows, k, rng, deadline - 5)
    
    # Final polish
    rows = _local_search_scope(rows, k, rng, deadline - 1)
    
    return rows

# EVOLVE-BLOCK-END
