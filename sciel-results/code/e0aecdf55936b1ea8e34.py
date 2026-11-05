import time
import sys

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = 115.0  # Safe margin before 120s kill
    
    def greedy_solve(max_scope):
        """
        Attempts to construct n_rows rows of k+1 integers
        starting at 0, strictly increasing, max <= max_scope,
        with all differences distinct.
        Returns list of rows or None if stuck.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            current_row = [0]
            # We need k more elements
            for _ in range(k):
                last_val = current_row[-1]
                found = False
                # Try smallest x > last_val up to max_scope
                # Optimization: iterate from last_val + 1
                for x in range(last_val + 1, max_scope + 1):
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs):
                        # Valid
                        current_row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(current_row)
        return rows

    best_result = None
    best_scope = float('inf')
    
    # Start from a reasonable lower bound.
    # Counting bound for (7, 5) is 105.
    # Let's start at 105 and go up.
    current_scope = 105
    
    while time.time() - start_time < deadline:
        result = greedy_solve(current_scope)
        if result:
            actual_max = max(r[-1] for r in result)
            if actual_max < best_scope:
                best_scope = actual_max
                best_result = result
                # Try to improve by searching for a smaller scope
                # Since greedy is deterministic, if it succeeded at current_scope,
                # it might have used less. We can try current_scope - 1 next?
                # But if it succeeded, it found a solution <= current_scope.
                # The actual max might be much smaller.
                # Let's set the next target to actual_max - 1 to try to tighten.
                current_scope = actual_max - 1
                if current_scope < 105:
                    break # Hit the theoretical lower bound
            else:
                # Found a solution but not better than best?
                # This shouldn't happen if we always decrease current_scope on success.
                break
        else:
            # Failed to find solution with this scope
            current_scope += 1
            # Safety check: if we go too high, stop
            if current_scope > 200:
                break

    if best_result is None:
        # Fallback: if even high scope failed (unlikely), do a very loose greedy
        # without scope limit (just keep going until valid)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            current_row = [0]
            for _ in range(k):
                last_val = current_row[-1]
                x = last_val + 1
                while True:
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs):
                        current_row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(current_row)
        best_result = rows

    return best_result
