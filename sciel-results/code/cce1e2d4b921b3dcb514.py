import time
import random
import math
import sys

_START = time.time()
_DEADLINE = _START + 145.0

def _elapsed():
    return time.time() - _START

def _dead():
    return time.time() >= _DEADLINE

def _diffs_of_row(row):
    """Return set of all positive differences in a row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d

def _scope(rows):
    return max(max(r) for r in rows)

def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True

class DiffTracker:
    """Fast difference tracking using a boolean array for O(1) membership."""
    def __init__(self, max_diff):
        self.max_diff = max_diff
        self.used = bytearray(max_diff + 1)
        self.count = 0

    def add(self, diffs):
        for d in diffs:
            if d <= self.max_diff and not self.used[d]:
                self.used[d] = 1
                self.count += 1

    def remove(self, diffs):
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                self.used[d] = 0
                self.count -= 1

    def check(self, diffs):
        """Return True if none of the diffs conflict with current used set."""
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                return False
        return True

    def copy(self):
        nt = DiffTracker(self.max_diff)
        nt.used = bytearray(self.used)
        nt.count = self.count
        return nt

def _row_diffs_incremental(row, new_x):
    """Given a row (list) and a new candidate x, compute the diffs that would be added."""
    return {new_x - a for a in row}

def _greedy_row_with_noise(tracker, k, max_mark, rnd, noise_level=0.0):
    """
    Greedy construction of a single row.
    'noise_level' introduces randomness: with prob noise_level, skip the smallest valid x.
    """
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        # Limit search depth for speed? No, we need to find a valid x.
        # But if max_mark is tight, this might fail.
        while x <= max_mark:
            # Apply noise
            if rnd.random() < noise_level:
                x += 1
                if x > max_mark:
                    break
                continue
            
            new_diffs = _row_diffs_incremental(row, x)
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            # Undo partial additions? No, we only add if found.
            # But we added to tracker inside the loop?
            # Ah, the tracker is shared. We must be careful.
            # If we return None, the caller must not have modified the tracker permanently 
            # or must undo.
            # In this function, we modify `tracker` as we go. 
            # If we fail, we have added diffs for the partial row.
            # We should revert those.
            # Let's restructure: collect row, check at end? No, incremental is faster.
            # If we fail, we must remove the diffs we added for the partial row.
            for i in range(len(row)-1, 0, -1):
                x_removed = row.pop()
                removed_diffs = {x_removed - a for a in row}
                tracker.remove(removed_diffs)
            return None
    return row

def _interleaved_greedy(n_rows, k, max_mark, rnd):
    """Interleaved greedy: extend the row with smallest current max at each step."""
    tracker = DiffTracker(max_mark + 1)
    rows = [[0] for _ in range(n_rows)]

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = _row_diffs_incremental(row, x)
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None, None

    if all(len(r) == k + 1 for r in rows):
        return rows, tracker
    return None, None

def _reconstruct_row(rows, idx, max_mark, rnd, noise_level=0.1):
    """
    Remove row idx, then try to reconstruct it using greedy with noise.
    Returns the new row or None.
    """
    old_row = rows[idx]
    old_diffs = _diffs_of_row(old_row)
    
    # Create a tracker with all other rows
    tracker = DiffTracker(max_mark + 1)
    for i, r in enumerate(rows):
        if i != idx:
            tracker.add(_diffs_of_row(r))
            
    # Try to build new row
    new_row = _greedy_row_with_noise(tracker, len(old_row)-1, max_mark, rnd, noise_level)
    
    if new_row is not None:
        return new_row
    return None

def _simulated_annealing(rows, n_rows, k, max_mark, time_budget, rnd):
    """
    Simulated Annealing on row-level moves.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [r[:] for r in rows]
    current_scope = best_scope
    
    T = 10.0
    cooling = 0.999
    deadline = time.time() + time_budget
    
    # Pre-compute max_mark for tracker (use current scope + some margin)
    tracker_max = max_mark + 10
    
    while time.time() < deadline and T > 0.01:
        ri = rnd.randint(0, n_rows - 1)
        
        # Try to reconstruct row ri
        new_row = _reconstruct_row(current_rows, ri, tracker_max, rnd, noise_level=0.2)
        
        if new_row is not None:
            # Check if new_row is valid with respect to others (it is, by construction)
            # Calculate new scope
            # We need to know the max of all other rows
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    if max(current_rows[i]) > other_max:
                        other_max = max(current_rows[i])
            
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope
            
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                # Accept
                current_rows[ri] = new_row
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Reject, current_rows remains unchanged (new_row was not assigned)
                pass
        else:
            # Reconstruction failed, do nothing
            pass
            
        T *= cooling
        
    return best_rows, best_scope

def _targeted_reduction(rows, n_rows, k, time_budget):
    """
    Systematically try to reduce the maximum entry of each row.
    """
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    deadline = time.time() + time_budget
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            # Try to reduce the last element first
            for pi in range(len(row) - 1, 0, -1):
                old_val = row[pi]
                # Compute forbidden diffs
                other_diffs = set()
                for r2 in best:
                    if r2 is not row:
                        other_diffs |= _diffs_of_row(r2)
                
                # Diffs within this row not involving pi
                row_diffs_other = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if i != pi and j != pi:
                            row_diffs_other.add(row[j] - row[i])
                
                forbidden = other_diffs | row_diffs_other
                
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    
                    new_diffs_pi = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs_pi.add(new_val - row[j])
                            else:
                                new_diffs_pi.add(row[j] - new_val)
                    
                    if not (new_diffs_pi & forbidden):
                        row[pi] = new_val
                        new_scope = _scope(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        break
                if improved:
                    break
    return best, best_scope

def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')
    
    # 1. Get a starting solution
    # Try interleaved greedy with a generous max_mark
    max_mark_start = 150
    rows, _ = _interleaved_greedy(n_rows, k, max_mark_start, rnd)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)
        
    if best_rows is None:
        # Fallback: simple greedy
        tracker = DiffTracker(200)
        for _ in range(n_rows):
            row = _greedy_row_with_noise(tracker, k, 200, rnd, 0.0)
            if row is None:
                return [] # Should not happen
            best_rows = best_rows or []
            best_rows.append(row)
        best_scope = _scope(best_rows)

    if _dead():
        return best_rows

    # 2. Targeted reduction
    remaining = _DEADLINE - time.time() - 5.0
    if remaining > 1.0:
        best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # 3. Simulated Annealing
    remaining = _DEADLINE - time.time() - 2.0
    if remaining > 2.0:
        # Use best_scope + margin for SA
        sa_max = best_scope + 20
        best_rows, best_scope = _simulated_annealing(
            best_rows, n_rows, k, sa_max, remaining, rnd
        )

    if _dead():
        return best_rows

    # 4. Final targeted reduction
    remaining = _DEADLINE - time.time() - 1.0
    if remaining > 0.5:
        best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    # Validation
    if not _valid(best_rows, n_rows, k):
        # Fallback to greedy if SA broke it (should not happen if logic is correct)
        rows, _ = _interleaved_greedy(n_rows, k, 200, rnd)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
