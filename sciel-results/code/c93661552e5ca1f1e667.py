"""Difference triangle set (4,7) - structural rewrite using randomized greedy + local repair."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, subject to all pairwise differences
    within rows being distinct across all rows.
    
    Strategy: randomized greedy construction with multiple restarts,
    plus a local search / repair phase to try to reduce the scope.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    best_rows = None
    best_scope = float('inf')
    
    # Greedy construction with randomization
    def greedy_construct(n_rows, k, rng):
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try candidate positions, starting from a random offset
                # to introduce diversity
                start_x = row[-1] + 1
                # Randomize the starting point a bit
                offset = rng.randint(0, 5) if r > 0 else 0
                found = False
                for x in range(start_x + offset, start_x + offset + 200):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # If we couldn't find a spot, try a wider search
                    for x in range(start_x, start_x + 500):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                if not found:
                    # This row failed; return None to signal failure
                    return None
            rows.append(row)
        return rows
    
    # Local search: try to reduce the largest entry by moving elements
    def local_search(rows, n_rows, k, rng, max_iterations=500):
        """
        Try to reduce the scope by attempting to move elements to smaller positions.
        """
        best_scope = max(max(row) for row in rows)
        current_rows = [list(row) for row in rows]
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to improve each row
            for r in range(n_rows):
                if time.time() > deadline:
                    break
                row = current_rows[r]
                # Try to move each element (except 0) to a smaller position
                for i in range(len(row) - 1, 0, -1):  # from end to start
                    old_val = row[i]
                    # Try smaller values
                    for new_val in range(row[i-1] + 1, old_val):
                        # Check if new_val works with the rest of the row
                        new_row = row[:i] + [new_val] + row[i+1:]
                        # Check strictly increasing
                        if any(new_row[j] >= new_row[j+1] for j in range(len(new_row)-1)):
                            continue
                        # Compute new differences for this row
                        new_diffs = set()
                        for a in range(len(new_row)):
                            for b in range(a+1, len(new_row)):
                                new_diffs.add(new_row[b] - new_row[a])
                        # Check against all other rows' differences
                        valid = True
                        for r2 in range(n_rows):
                            if r2 == r:
                                continue
                            other_row = current_rows[r2]
                            for a in range(len(other_row)):
                                for b in range(a+1, len(other_row)):
                                    if (other_row[b] - other_row[a]) in new_diffs:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            # Also check within the same row (already done via new_diffs)
                            current_rows[r] = new_row
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try a random perturbation
                r = rng.randint(0, n_rows - 1)
                i = rng.randint(1, len(current_rows[r]) - 1)
                delta = rng.choice([-2, -1, 1, 2])
                old_val = current_rows[r][i]
                new_val = old_val + delta
                if new_val > current_rows[r][i-1] and (i == len(current_rows[r])-1 or new_val < current_rows[r][i+1]):
                    current_rows[r][i] = new_val
                    # Check validity
                    new_diffs = set()
                    row = current_rows[r]
                    for a in range(len(row)):
                        for b in range(a+1, len(row)):
                            new_diffs.add(row[b] - row[a])
                    valid = True
                    for r2 in range(n_rows):
                        if r2 == r:
                            continue
                        other_row = current_rows[r2]
                        for a in range(len(other_row)):
                            for b in range(a+1, len(other_row)):
                                if (other_row[b] - other_row[a]) in new_diffs:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        current_rows[r][i] = old_val
        
        new_scope = max(max(row) for row in current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
        return current_rows, best_scope
    
    # Main search loop
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(seed)
        rows = greedy_construct(n_rows, k, rng)
        if rows is None:
            continue
        
        scope = max(max(row) for row in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(row) for row in rows]
        
        # If we have a good candidate, try local search
        if scope <= best_scope and time.time() < deadline - 10:
            improved_rows, improved_scope = local_search(rows, n_rows, k, rng, max_iterations=200)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows
    
    # Return the best found, or fall back to greedy
    if best_rows is not None:
        return best_rows
    else:
        # Fallback: simple greedy
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
