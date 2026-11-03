"""Rewritten program for the difference-triangle-set task (n_rows=4, k=7).

Design:
- Fast greedy construction (seed) to get a valid initial set.
- Local-search refinement: repeatedly try to shift individual elements to smaller
  values (pull-to-smallest), and use 2-element swap perturbations to escape local
  minima. Randomized restarts with different greedy orderings help.
- Deadline guard: track elapsed time and return the best valid result found.
"""
import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s (and 150s kill)
    
    def elapsed():
        return time.time() - start_time
    
    def greedy_build(order=None):
        """Greedy: build rows one by one, each row picks smallest next mark.
        If order is given, it's the order in which rows are built (list of row indices)."""
        used = set()
        rows = [None] * n_rows
        if order is None:
            order = list(range(n_rows))
        for r in order:
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows[r] = row
        return rows, used
    
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def is_valid(rows):
        used = set()
        for row in rows:
            # check strictly increasing and starts at 0
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def scope(rows):
        return max(max(r) for r in rows)
    
    # Initial greedy
    best_rows, _ = greedy_build()
    best_scope = scope(best_rows)
    
    # Local search: pull-to-smallest with 2-element swap perturbations
    rng = random.Random(42)
    
    def try_pull(rows, used):
        """Try to pull each element (except 0) to a smaller value. Returns (success, new_rows, new_used)."""
        for r_idx in range(n_rows):
            row = rows[r_idx]
            for pos in range(len(row) - 1, 0, -1):  # from right to left
                old_val = row[pos]
                # Try to move to smaller values
                for new_val in range(old_val - 1, 0, -1):
                    # Check strictly increasing
                    if pos > 0 and new_val <= row[pos - 1]:
                        continue
                    # Check differences with other elements in this row
                    new_diffs = set()
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        if d == 0:
                            valid = False
                            break
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    # Also need to remove old diffs and add new ones
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        old_diffs.add(abs(old_val - row[i]))
                    # Check that after removing old and adding new, no conflicts
                    temp_used = used - old_diffs
                    if new_diffs & temp_used:
                        continue
                    # Success
                    new_row = row.copy()
                    new_row[pos] = new_val
                    new_rows = [row.copy() if i != r_idx else new_row for i in range(n_rows)]
                    new_used = (used - old_diffs) | new_diffs
                    return True, new_rows, new_used
        return False, rows, used
    
    def try_swap_perturbation(rows, used, rng):
        """Swap two elements' values in the same row (if it keeps increasing order) or 
        perturb by moving one element to a random smaller position."""
        # Try moving a random element to a random smaller value
        for _ in range(50):
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # not the 0
            row = rows[r_idx]
            old_val = row[pos]
            # Pick a new value smaller than old_val
            if old_val <= 1:
                continue
            new_val = rng.randint(1, old_val - 1)
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            # Check validity
            old_diffs = set()
            for i in range(len(row)):
                if i != pos:
                    old_diffs.add(abs(old_val - row[i]))
            new_diffs = set()
            valid = True
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(new_val - row[i])
                if d == 0:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            temp_used = used - old_diffs
            if new_diffs & temp_used:
                continue
            new_row = row.copy()
            new_row[pos] = new_val
            new_rows = [row.copy() if i != r_idx else new_row for i in range(n_rows)]
            new_used = (used - old_diffs) | new_diffs
            return new_rows, new_used
        return None, None
    
    best_used = compute_used(best_rows)
    
    # Iterative improvement
    max_iters = 50000
    for it in range(max_iters):
        if elapsed() > deadline:
            break
        
        # Try pull-to-smallest
        success, new_rows, new_used = try_pull(best_rows, best_used)
        if success:
            new_scope = scope(new_rows)
            if new_scope <= best_scope:
                best_rows = new_rows
                best_used = new_used
                best_scope = new_scope
            continue
        
        # Try swap perturbation
        pert_rows, pert_used = try_swap_perturbation(best_rows, best_used, rng)
        if pert_rows is not None:
            pert_scope = scope(pert_rows)
            if pert_scope < best_scope:
                best_rows = pert_rows
                best_used = pert_used
                best_scope = pert_scope
            # After perturbation, try to pull
            for _ in range(10):
                if elapsed() > deadline:
                    break
                s2, nr2, nu2 = try_pull(best_rows, best_used)
                if s2:
                    ns2 = scope(nr2)
                    if ns2 <= best_scope:
                        best_rows = nr2
                        best_used = nu2
                        best_scope = ns2
                else:
                    break
    
    # Final: try greedy with different row orders
    for trial in range(20):
        if elapsed() > deadline:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows, used = greedy_build(order)
        sc = scope(rows)
        if sc < best_scope:
            best_rows = rows
            best_used = used
            best_scope = sc
            # Try to improve
            for _ in range(100):
                if elapsed() > deadline:
                    break
                s, nr, nu = try_pull(best_rows, best_used)
                if s:
                    ns = scope(nr)
                    if ns <= best_scope:
                        best_rows = nr
                        best_used = nu
                        best_scope = ns
                else:
                    break
    
    return best_rows
