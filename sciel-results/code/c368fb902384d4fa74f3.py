import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Try to find better solutions with randomized backtracking
    attempt = 0
    while time.time() < deadline - 10.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try multiple random restarts for this target scope
        found = False
        for _ in range(40):
            if time.time() > deadline - 10.0:
                break
            
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1
        
        if not found and target_scope <= lower_bound + 5:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
    
    # 4. Final polish: very fine-grained scope reduction
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    
    return best_solution


def greedy_construct(n_rows, k):
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


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search using bitset for fast difference checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1
    
    # Use a list of booleans for used differences (faster than set)
    max_diff = max_scope
    used_diffs = [False] * (max_diff + 1)
    
    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Find valid candidates
        valid_candidates = []
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
            
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Bias towards smaller values, with some randomness
        n_smallest = min(12, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            for d in new_diffs:
                used_diffs[d] = False
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                used_diffs[row[i2] - row[j]] = True
    
    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce scope by shifting elements down. Uses set for flexibility.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                for new_val in range(row[pos - 1] + 1, original):
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    if any(d <= 0 for d in new_diffs):
                        continue
                    
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    Final high-precision polish: try very small shifts (1-3 units) to squeeze scope.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    k_plus_1 = k + 1
    
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    # Multiple passes of fine-grained adjustments
    for _ in range(3):
        if time.time() > deadline - 1.0:
            break
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for row_idx in range(n_rows):
                row = best[row_idx]
                for pos in range(k_plus_1 - 1, 0, -1):
                    original = row[pos]
                    # Only try shifts of 1-3 units
                    for shift in range(1, min(4, original - row[pos - 1])):
                        new_val = original - shift
                        if new_val <= row[pos - 1]:
                            break
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, k_plus_1):
                            new_diffs.add(row[j] - new_val)
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(original - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - original)
                        
                        if new_diffs & (used_diffs - old_diffs):
                            continue
                        if any(d <= 0 for d in new_diffs):
                            continue
                        
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        improved = True
                        break
    
    return best
