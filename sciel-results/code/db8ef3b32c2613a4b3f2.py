import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s kill
    num_marks = k + 1
    diffs_per_row = num_marks * (num_marks - 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    def get_row_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))
    
    def is_valid(rows):
        used = set()
        for row in rows:
            d = get_row_diffs(row)
            if d & used:
                return False
            used |= d
        return True
    
    def greedy_row(used, num_marks, max_val, seed=None):
        """Build a row greedily avoiding used differences."""
        if seed is not None:
            random.seed(seed)
        row = [0]
        x = 0
        while len(row) < num_marks:
            found = False
            # Try a window of candidates
            candidates = list(range(x+1, min(x+1+200, max_val+1)))
            if seed is not None:
                random.shuffle(candidates)
            else:
                candidates.sort()
            for c in candidates:
                new_diffs = {c - a for a in row}
                if not new_diffs & used:
                    row.append(c)
                    used = used | new_diffs
                    x = c
                    found = True
                    break
            if not found:
                return None
        return row
    
    def backtrack_row(used, num_marks, max_val, depth=0):
        """Backtracking to build a row."""
        if time.time() > deadline:
            return None
        if depth >= num_marks:
            return None  # Should have been caught
        row = [0]
        return _backtrack_row_helper(row, used, num_marks, max_val, 0)
    
    def _backtrack_row_helper(row, used, num_marks, max_val, depth):
        if time.time() > deadline:
            return None
        if len(row) == num_marks:
            return list(row)
        start = row[-1] + 1
        # Limit search space
        end = min(start + 50, max_val + 1)
        for x in range(start, end):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            new_used = used | new_diffs
            result = _backtrack_row_helper(row + [x], new_used, num_marks, max_val, depth+1)
            if result is not None:
                return result
        return None
    
    def local_search(rows, used, max_iter=1000):
        """Try to reduce max entry via local perturbations."""
        best = list(rows)
        best_used = set()
        for r in best:
            best_used |= get_row_diffs(r)
        best_max = max(max(r) for r in best)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random row and try to improve it
            ri = random.randint(0, n_rows-1)
            old_row = best[ri]
            old_diffs = get_row_diffs(old_row)
            other_used = best_used - old_diffs
            
            # Try to rebuild row with smaller max
            new_row = None
            for attempt in range(20):
                row = [0]
                x = 0
                success = True
                while len(row) < num_marks:
                    found = False
                    # Try smaller values first, with some randomness
                    base = row[-1] + 1
                    limit = min(base + 30, best_max)
                    candidates = list(range(base, limit + 1))
                    random.shuffle(candidates)
                    for c in candidates:
                        new_diffs = {c - a for a in row}
                        if new_diffs & other_used:
                            continue
                        if new_diffs & (set(d for d in get_row_diffs(row) if True)):  # internal check
                            continue
                        row.append(c)
                        x = c
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if success and max(row) < max(old_row):
                    new_row = row
                    break
            
            if new_row is not None:
                new_diffs = get_row_diffs(new_row)
                if not new_diffs & other_used:
                    best[ri] = new_row
                    best_used = set()
                    for r in best:
                        best_used |= get_row_diffs(r)
                    if max(max(r) for r in best) < best_max:
                        best_max = max(max(r) for r in best)
        
        return best, best_max
    
    # Phase 1: Quick greedy baseline
    best_rows = None
    best_scope = float('inf')
    
    # Try deterministic greedy first
    used = set()
    rows = []
    success = True
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_marks:
            found = False
            for c in range(x+1, 500):
                new_diffs = {c - a for a in row}
                if not new_diffs & used:
                    row.append(c)
                    used = used | new_diffs
                    x = c
                    found = True
                    break
            if not found:
                success = False
                break
        if not success:
            break
        rows.append(row)
    
    if success:
        best_rows = rows
        best_scope = max(max(r) for r in rows)
    
    # Phase 2: Multi-start randomized greedy
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        success = True
        for r_idx in range(n_rows):
            row = greedy_row(used, num_marks, 300, seed=trial*100+r_idx)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= get_row_diffs(row)
        if success:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    # Phase 3: Local search refinement
    if best_rows is not None:
        used = set()
        for r in best_rows:
            used |= get_row_diffs(r)
        improved, improved_scope = local_search(best_rows, used, max_iter=5000)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved
    
    # Phase 4: More aggressive multi-start with backtracking
    for trial in range(50):
        if time.time() > deadline:
            break
        random.seed(1000 + trial)
        used = set()
        rows = []
        success = True
        for r_idx in range(n_rows):
            # Try backtracking with limited depth
            row = backtrack_row(used, num_marks, 250)
            if row is None:
                # Fall back to greedy
                row = greedy_row(used, num_marks, 250, seed=trial*10+r_idx)
                if row is None:
                    success = False
                    break
            rows.append(row)
            used |= get_row_diffs(row)
        if success:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                # Refine with local search
                used2 = set()
                for r in best_rows:
                    used2 |= get_row_diffs(r)
                improved, improved_scope = local_search(best_rows, used2, max_iter=2000)
                if improved_scope < best_scope:
                    best_scope = improved_scope
                    best_rows = improved
    
    if best_rows is None:
        # Should not happen, but fallback
        return [[0]] * n_rows
    
    return best_rows
