import time
import random

def construct(n_rows, k):
    deadline = time.time() + 140.0
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    M_min = total_diffs - 1
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Backtracking search for small M
    # Only try if n_rows and k are small enough
    if n_rows <= 10 and k <= 6:
        for M in range(M_min, M_min + 10):
            if time.time() > deadline:
                break
            rows = backtrack_search(n_rows, k, M, deadline)
            if rows is not None:
                return rows
        
    # Strategy 2: Randomized Greedy
    # Run multiple times with different random seeds and window sizes
    attempts = 0
    while time.time() < deadline:
        rows, scope = randomized_greedy(n_rows, k, deadline)
        if rows is not None:
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
            if best_scope == M_min:
                break
        attempts += 1
        
    if best_rows is None:
        # Fallback to deterministic greedy
        rows, scope = deterministic_greedy(n_rows, k)
        best_rows = rows
        best_scope = scope
        
    return best_rows

def backtrack_search(n_rows, k, M, deadline):
    used = set()
    rows = []
    total_diffs_per_row = k * (k + 1) // 2
    
    def solve(row_idx, current_row):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        
        if len(current_row) == k + 1:
            # Calculate diffs
            diffs = set()
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    diffs.add(current_row[j] - current_row[i])
            
            if diffs & used:
                return False
            
            rows.append(current_row)
            used.update(diffs)
            
            if solve(row_idx + 1, [0]):
                return True
            
            # Backtrack
            used.difference_update(diffs)
            rows.pop()
            return False
        
        last = current_row[-1]
        # Pruning: Check if enough diffs are available
        remaining_rows = n_rows - row_idx - 1
        current_diffs_count = len(current_row) * (len(current_row) - 1) // 2
        remaining_diffs_needed = (remaining_rows * total_diffs_per_row) + (total_diffs_per_row - current_diffs_count)
        available_diffs = M - len(used)
        if available_diffs < remaining_diffs_needed:
            return False
            
        # Try next mark
        # Heuristic: Try smaller x first
        for x in range(last + 1, M + 1):
            new_diffs = {x - a for a in current_row}
            if not (new_diffs & used):
                if solve(row_idx, current_row + [x]):
                    return True
        return False

    if solve(0, [0]):
        return rows
    return None

def randomized_greedy(n_rows, k, deadline):
    used = set()
    rows = []
    # Randomize the window size for candidates
    C = random.randint(1, 20)
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            last = row[-1]
            candidates = []
            x = last
            # Safety limit to avoid infinite loop if no candidate found
            limit = 1000 
            while x < limit:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append(x)
                    if len(candidates) >= C:
                        break
                if time.time() > deadline:
                    return None, None
            
            if not candidates:
                return None, None
                
            x = random.choice(candidates)
            row.append(x)
            # Update used
            for a in row[:-1]:
                used.add(x - a)
                
        rows.append(row)
        
    max_val = max(max(r) for r in rows)
    return rows, max_val

def deterministic_greedy(n_rows, k):
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
    return rows, max(max(r) for r in rows)
