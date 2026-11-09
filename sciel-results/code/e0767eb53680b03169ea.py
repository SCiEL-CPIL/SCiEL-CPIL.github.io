"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    # Build up to k+1 elements in each row
    for step in range(1, k + 1):
        # Shuffle row order for diversity
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows

def _backtracking_solve(n_rows, k, max_scope, deadline=None):
    """
    Backtracking solver with constraint propagation.
    Tries to find a valid solution with max entry <= max_scope.
    Returns None if not found within time limit.
    """
    if deadline is None:
        deadline = time.time() + 5.0
    
    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # We need to assign each of the total_diffs differences a unique value in [1, max_scope]
    # But we also need consistency: for each row, a_j = a_i + d_ij
    
    # Strategy: build rows one by one, choosing values that maintain uniqueness
    # Use forward checking: track which differences are already used
    
    used_diffs = set()
    rows = []
    
    def solve_row(row_idx, current_row, depth):
        """Recursively build row row_idx starting from current_row (list of chosen values)."""
        if time.time() > deadline:
            return None
        
        if depth == k + 1:
            return list(current_row)
        
        # Determine lower bound for next element
        if depth == 0:
            lower = 0
        else:
            lower = current_row[-1] + 1
        
        # Upper bound
        upper = max_scope
        
        for val in range(lower, upper + 1):
            if time.time() > deadline:
                return None
            
            # Check if adding val creates any difference that's already used
            valid = True
            new_diffs = []
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Forward check: can we still complete the row?
                # Simple heuristic: ensure enough room for remaining elements
                remaining = k + 1 - depth - 1
                if remaining > 0 and val + remaining > max_scope:
                    continue
                
                # Try this value
                for d in new_diffs:
                    used_diffs.add(d)
                current_row.append(val)
                
                result = solve_row(row_idx, current_row, depth + 1)
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                
                if result is not None:
                    return result
        
        return None
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = solve_row(r, [], 0)
        if row is None:
            return None
        rows.append(row)
    
    return rows

def _optimize(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Optimized local search using difference counts.
    Returns the best rows found.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    # Helper to compute current scope
    def get_scope():
        return max(max(r) for r in current_rows)
    
    # Initialize difference counts
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 10
    diff_counts = [0] * (diff_size + 1)
    
    # Populate initial counts
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    # Local Search: Hill Climbing with multiple passes
    # We try to reduce the scope by moving elements to smaller positions.
    
    def try_move(r, pos, new_val):
        """Attempt to move current_rows[r][pos] to new_val. Returns True if successful."""
        row = current_rows[r]
        old_val = row[pos]
        
        # Check increasing constraint
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        # Calculate old and new differences
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        # Remove old diffs
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] -= 1
        
        # Check new diffs
        valid = True
        for d in new_diffs:
            if d >= len(diff_counts):
                diff_counts.extend([0] * (d - len(diff_counts) + 1))
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old diffs
            for d in old_diffs:
                diff_counts[d] += 1
            return False
        
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
            
        return True

    # Multiple passes of local search with different strategies
    strategies = [
        "right_to_left",   # Reduce from right to left
        "left_to_right",   # Reduce from left to right  
        "random_order",    # Random order of positions
        "largest_first",   # Target largest elements first
    ]
    
    improved = True
    pass_count = 0
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        pass_count += 1
        strategy = strategies[pass_count % len(strategies)]
        
        # Determine position order
        positions_by_row = {}
        for r in range(n_rows):
            row = current_rows[r]
            pos_list = list(range(1, len(row)))  # Skip position 0 (always 0)
            
            if strategy == "right_to_left":
                pos_list.sort(reverse=True)
            elif strategy == "left_to_right":
                pass  # Already in order
            elif strategy == "random_order":
                random.shuffle(pos_list)
            elif strategy == "largest_first":
                # Sort by current value descending
                pos_list.sort(key=lambda p: row[p], reverse=True)
            
            positions_by_row[r] = pos_list
        
        # Try moves
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            
            for pos in positions_by_row[r]:
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                
                # Lower bound
                lower_bound = row[pos - 1] + 1 if pos > 0 else 1
                
                # Try to find smallest valid value
                for new_val in range(lower_bound, old_val):
                    if try_move(r, pos, new_val):
                        improved = True
                        break
        
        # Update best
        current_scope = get_scope()
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [list(r) for r in current_rows]
    
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try backtracking solver with progressively larger max_scope
    # Start from a reasonable lower bound
    # Counting bound: need at least total_diffs distinct differences
    # But max_scope must be large enough to fit them
    total_diffs = n_rows * k * (k + 1) // 2
    # Simple lower bound: max_scope >= total_diffs / 2 (rough heuristic)
    # Better: use the known lower bound or start from a small value and increase
    
    # Try backtracking with various max_scope values
    bt_time_limit = min(20.0, deadline - start_time - 30.0)
    if bt_time_limit > 2.0:
        # Start from a reasonable estimate
        # For (7,5): known best is 112, counting bound is 105
        # Try a range of max_scope values
        for max_scope in [105, 108, 110, 112, 115, 120]:
            if time.time() > deadline - 30.0:
                break
            
            rows = _backtracking_solve(n_rows, k, max_scope, deadline=time.time() + 5.0)
            if rows is not None:
                sc = max(max(r) for r in rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [list(r) for r in rows]
                    # If we found a solution, try to improve with smaller max_scope
                    if sc <= max_scope:
                        # Found valid solution, can break or continue for better
                        if sc <= 105:
                            break
    
    # Phase 2: Greedy initialization with multiple seeds
    greedy_time_limit = deadline - start_time - 40.0
    if greedy_time_limit > 5.0 and (best_rows is None or best_scope > 115):
        max_restarts = min(20, int(greedy_time_limit / 2.0))
        for seed in range(max_restarts):
            if time.time() > deadline - 40.0:
                break
                
            rows = _interleaved_greedy(n_rows, k, seed=seed)
            sc = max(max(r) for r in rows)
            
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: simple greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Phase 3: Local search optimization
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        best_rows = _optimize(best_rows, n_rows, k, time_limit=time_remaining, seed=42)
    
    return best_rows
# EVOLVE-BLOCK-END
