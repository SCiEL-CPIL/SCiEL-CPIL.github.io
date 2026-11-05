"""Program for difference triangle set (4,7) using a hybrid search approach."""
import random
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a greedy construction with backtracking and randomization to find
    good solutions within the time limit.
    """
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Try multiple strategies and keep the best
    best_rows = None
    best_max = float('inf')
    
    # Strategy 1: Pure greedy with different random seeds
    for seed in range(20):
        random.seed(seed)
        rows = _greedy_construct(n_rows, k, random)
        if rows is not None:
            max_val = max(max(r) for r in rows)
            if max_val < best_max:
                best_max = max_val
                best_rows = rows
    
    # Strategy 2: Greedy with local search / perturbation
    if best_rows is not None:
        for attempt in range(10):
            random.seed(100 + attempt)
            perturbed = _perturb_and_optimize(best_rows, n_rows, k, random)
            if perturbed is not None:
                max_val = max(max(r) for r in perturbed)
                if max_val < best_max:
                    best_max = max_val
                    best_rows = perturbed
    
    # Strategy 3: Backtracking search for small cases
    if n_rows <= 4 and k <= 8:
        bt_result = _backtracking_construct(n_rows, k, time_limit=30)
        if bt_result is not None:
            max_val = max(max(r) for r in bt_result)
            if max_val < best_max:
                best_max = max_val
                best_rows = bt_result
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _greedy_construct(n_rows, k, random)
    
    return best_rows


def _greedy_construct(n_rows, k, rng):
    """
    Greedy construction: for each row, pick the smallest next mark
    such that all new differences are unused.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows


def _perturb_and_optimize(rows, n_rows, k, rng):
    """
    Try to improve a solution by removing some marks and re-adding them greedily.
    """
    import copy
    best = copy.deepcopy(rows)
    best_max = max(max(r) for r in best)
    
    for trial in range(50):
        candidate = copy.deepcopy(best)
        
        # Pick a random row and remove its last few elements
        row_idx = rng.randint(0, n_rows - 1)
        num_remove = rng.randint(1, min(3, len(candidate[row_idx]) - 1))
        removed = candidate[row_idx][-num_remove:]
        candidate[row_idx] = candidate[row_idx][:-num_remove]
        
        # Rebuild the used set
        used = set()
        for r in candidate:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        
        # Try to re-add the removed marks or find better ones
        success = True
        for _ in range(num_remove):
            found = False
            # Try values around the original removed values
            candidates = set()
            for val in removed:
                for delta in range(-5, 6):
                    candidates.add(val + delta)
            # Also try greedy from current last value
            candidates.add(candidate[row_idx][-1] + 1)
            candidates.add(candidate[row_idx][-1] + 2)
            
            for x in sorted(candidates):
                if x <= candidate[row_idx][-1]:
                    continue
                new_diffs = {x - a for a in candidate[row_idx]}
                if not new_diffs & used:
                    candidate[row_idx].append(x)
                    used |= new_diffs
                    found = True
                    break
            
            if not found:
                # Try greedy
                x = candidate[row_idx][-1]
                while True:
                    x += 1
                    new_diffs = {x - a for a in candidate[row_idx]}
                    if not new_diffs & used:
                        candidate[row_idx].append(x)
                        used |= new_diffs
                        break
        
        if len(candidate[row_idx]) == k + 1:
            max_val = max(max(r) for r in candidate)
            if max_val < best_max:
                best = candidate
                best_max = max_val
    
    return best


def _backtracking_construct(n_rows, k, time_limit=30):
    """
    Backtracking search to find a valid configuration with minimal max value.
    Only feasible for small n_rows and k.
    """
    import time
    start_time = time.time()
    
    # Upper bound from greedy
    greedy = _greedy_construct(n_rows, k, random)
    if greedy is None:
        return None
    upper_bound = max(max(r) for r in greedy)
    
    # Try to find a solution with max value < upper_bound
    # Use DFS with pruning
    best_solution = [list(r) for r in greedy]
    best_max = upper_bound
    
    def dfs(row_idx, current_rows, used_diffs, max_val_so_far):
        nonlocal best_solution, best_max, start_time
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            max_val = max(max(r) for r in current_rows)
            if max_val < best_max:
                best_max = max_val
                best_solution = [list(r) for r in current_rows]
            return True
        
        # Build next row
        return _build_row_dfs(0, [0], used_diffs, current_rows, row_idx, max_val_so_far)
    
    def _build_row_dfs(pos, row, used_diffs, current_rows, row_idx, max_val_so_far):
        nonlocal best_solution, best_max, start_time
        
        if time.time() - start_time > time_limit:
            return False
        
        if pos == k + 1:
            # Row complete
            new_rows = current_rows + [row]
            max_val = max(max_val_so_far, row[-1])
            if max_val >= best_max:
                return False
            return dfs(row_idx + 1, new_rows, used_diffs, max_val)
        
        # Try next value
        start_x = row[-1] + 1 if row else 1
        # Limit search space
        limit = best_max if best_max < float('inf') else 200
        
        for x in range(start_x, min(limit + 1, 300)):
            if max_val_so_far > 0 and x > best_max:
                break
            
            new_diffs = set()
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Prune: if x already exceeds best_max, skip
            if x >= best_max and pos == k:
                continue
            
            if _build_row_dfs(pos + 1, row + [x], used_diffs | new_diffs, current_rows, row_idx, max(max_val_so_far, x)):
                return True
        
        return False
    
    dfs(0, [], set(), 0, 0)
    
    return best_solution
