import time
import random
import sys

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i):
            diffs.add(row[i] - row[j])
    return diffs

def _check_valid(rows, k):
    """Verify that all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for d in _row_diffs(row):
            if d in all_diffs:
                return False
            all_diffs.add(d)
    return True

def _greedy_construct(n_rows, k, M, rng=None):
    """
    Greedy construction with randomization.
    Tries to build n_rows rows with max entry <= M.
    Uses randomized candidate ordering to escape local optima.
    """
    if rng is None:
        rng = random.Random()
    
    used_diffs = set()
    rows = []
    
    for r_idx in range(n_rows):
        row = [0]
        row_diffs = set()
        success = False
        
        # Multiple attempts with different random seeds
        for attempt in range(50):
            row = [0]
            row_diffs = set()
            success = False
            
            for pos in range(1, k + 1):
                # Generate candidates
                last_val = row[-1]
                candidates = list(range(last_val + 1, M + 1))
                
                if not candidates:
                    break
                
                # Shuffle candidates with bias towards smaller values
                # Use a weighted shuffle: smaller values have higher priority
                weighted = []
                for c in candidates:
                    # Bias towards smaller values (1/c weight)
                    w = 1.0 / (c + 1)
                    weighted.append((w, c))
                
                # Weighted random selection
                total_w = sum(w for w, _ in weighted)
                if total_w == 0:
                    break
                
                r = rng.random() * total_w
                cumulative = 0
                chosen = None
                for w, c in weighted:
                    cumulative += w
                    if cumulative >= r:
                        chosen = c
                        break
                
                if chosen is None:
                    chosen = weighted[-1][1]
                
                # Check validity
                valid = True
                new_diffs = set()
                for a in row:
                    d = chosen - a
                    if d in used_diffs or d in row_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Pruning: check if enough diffs remain
                used_count = len(used_diffs) + len(row_diffs)
                remaining_needed = n_rows * k * (k + 1) // 2 - used_count
                available = M - used_count
                if available < remaining_needed:
                    continue
                
                row.append(chosen)
                row_diffs |= new_diffs
                success = True
            
            if success:
                break
        
        if not success:
            return None
        
        used_diffs |= row_diffs
        rows.append(row)
    
    return rows

def _backtrack_construct(n_rows, k, M, deadline, rng=None):
    """
    Backtracking search with pruning.
    More systematic but slower.
    """
    if rng is None:
        rng = random.Random()
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    used_diffs = set()
    rows = []
    
    def build_row(current_row, current_row_diffs, attempt_idx):
        if time.time() > deadline:
            return None
        
        if len(current_row) == k + 1:
            return current_row[:]
        
        last_val = current_row[-1]
        
        # Generate candidates
        candidates = list(range(last_val + 1, M + 1))
        
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        # Sort by heuristic: prefer values that leave more room
        # But keep some randomness
        
        for x in candidates:
            # Check validity
            valid = True
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d in used_diffs or d in current_row_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Pruning
            used_count = len(used_diffs) + len(current_row_diffs)
            remaining_needed = total_diffs_needed - used_count
            available = M - used_count
            if available < remaining_needed:
                continue
            
            next_row = current_row + [x]
            next_row_diffs = current_row_diffs | new_diffs
            
            result = build_row(next_row, next_row_diffs, attempt_idx)
            if result is not None:
                return result
        
        return None
    
    for r_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = build_row([0], set(), r_idx)
        if row is None:
            return None
        
        used_diffs |= _row_diffs(row)
        rows.append(row)
    
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    # Theoretical lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    best_solution = None
    best_M = float('inf')
    
    # Search M from lower bound to known best + margin
    # Known best is 134, counting bound is 112
    M_start = max(total_diffs_needed, 112)
    M_end = 135
    
    # Try multiple strategies for each M
    rng = random.Random(42)
    
    for M in range(M_start, M_end + 1):
        if time.time() > deadline:
            break
        
        # Strategy 1: Randomized greedy (fast, may find solutions)
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            solution = _greedy_construct(n_rows, k, M, rng)
            if solution and _check_valid(solution, k):
                max_entry = max(max(row) for row in solution)
                if max_entry <= M:
                    best_solution = solution
                    best_M = max_entry
                    return solution
        
        # Strategy 2: Backtracking (slower, more systematic)
        if time.time() < deadline - 10.0:
            solution = _backtrack_construct(n_rows, k, M, deadline, rng)
            if solution and _check_valid(solution, k):
                max_entry = max(max(row) for row in solution)
                if max_entry <= M:
                    best_solution = solution
                    best_M = max_entry
                    return solution
    
    # If we found a solution with a larger M, return it
    if best_solution:
        return best_solution
    
    # Fallback: simple greedy without M constraint
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
