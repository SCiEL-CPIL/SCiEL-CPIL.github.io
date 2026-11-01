import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry (the scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # For small cases, greedy is often optimal or near-optimal
    if n_rows * (k + 1) * k // 2 <= 500:
        # Use greedy with multiple random seeds and pick best
        best = _greedy_with_seeds(n_rows, k, n_seeds=20)
        if time.time() < deadline:
            # Try local search to improve
            improved = _local_search(best, n_rows, k, deadline)
            if improved is not None:
                best = improved
        return best

    # For larger cases, use a more sophisticated approach
    best = _greedy_with_seeds(n_rows, k, n_seeds=10)
    if time.time() < deadline:
        improved = _local_search(best, n_rows, k, deadline)
        if improved is not None:
            best = improved
    return best


def _greedy_with_seeds(n_rows, k, n_seeds=10):
    """Run greedy construction with different random orderings of rows, return best."""
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(n_seeds):
        rng = random.Random(seed)
        rows = []
        used = set()
        
        # Shuffle the order in which we build rows (but we build them in sequence)
        # Instead, vary the greedy choices by adding small random perturbations
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                # Try candidates in increasing order, but with slight randomization
                found = False
                while not found:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        found = True
                    # Occasionally skip a candidate to explore different structures
                    if seed > 0 and rng.random() < 0.05:
                        continue
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows[:]
    
    return best_rows


def _local_search(rows, n_rows, k, deadline):
    """
    Local search: try to reduce the scope by removing the largest element
    and re-inserting it at a smaller position, or by swapping elements.
    """
    if rows is None:
        return None
    
    current_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    start_time = time.time()
    
    # Collect all differences and their sources
    def get_all_diffs(rows):
        """Return set of all positive differences."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    # Try to reduce the scope by moving the largest element to a smaller value
    # We'll do this by trying to replace the last element of each row with a smaller unused value
    
    for iteration in range(10000):
        if time.time() > deadline - 5.0:
            break
        
        # Find the row with the maximum element
        max_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        row = rows[max_idx]
        last_val = row[-1]
        
        if last_val <= 1:
            continue
        
        # Try to replace last_val with a smaller value
        used = get_all_diffs(rows)
        
        # Remove the last element's contributions
        for i in range(len(row) - 1):
            used.discard(last_val - row[i])
        
        found = False
        for new_val in range(last_val - 1, 0, -1):
            if new_val <= row[-2]:
                break
            # Check if new_val can replace last_val
            new_diffs = {new_val - row[i] for i in range(len(row) - 1)}
            if not new_diffs & used:
                # Also need to check that new_val itself doesn't create issues
                # Actually we just need the differences to be unique
                row[-1] = new_val
                used |= new_diffs
                found = True
                break
        
        if found:
            new_scope = max(max(r) for r in rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        
        # Also try small random perturbations
        if random.random() < 0.3:
            # Pick a random row and try to reduce one of its internal elements
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # position in row (0 is fixed at 0)
            if pos < len(rows[r_idx]):
                old_val = rows[r_idx][pos]
                # Try to decrease it
                used = get_all_diffs(rows)
                # Remove contributions of old_val
                row = rows[r_idx]
                for i in range(len(row)):
                    if i != pos:
                        if row[pos] > row[i]:
                            used.discard(row[pos] - row[i])
                        else:
                            used.discard(row[i] - row[pos])
                
                found_smaller = False
                for new_val in range(old_val - 1, row[pos - 1] if pos > 0 else 0, -1):
                    new_diffs = set()
                    for i in range(len(row)):
                        if i != pos:
                            if new_val > row[i]:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                    if not new_diffs & used:
                        row[pos] = new_val
                        used |= new_diffs
                        found_smaller = True
                        break
                
                if found_smaller:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                    else:
                        # Revert if it didn't help
                        row[pos] = old_val
    
    if best_rows is not None and best_scope < current_scope:
        return best_rows
    return rows


def _validate(rows, n_rows, k):
    """Check if the rows are valid."""
    if len(rows) != n_rows:
        return False
    diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True
