"""Difference triangle set (4,7) - optimized construction."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all pairwise differences distinct."""
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Bitset for used differences (as Python int, bit i = difference i used)
    # Max difference we'll consider: start with a reasonable upper bound
    # Simple bound: total_diffs_needed + some margin
    max_diff_estimate = total_diffs_needed + k * (k + 1) // 2
    
    best_result = None
    best_scope = float('inf')
    
    def compute_scope(rows):
        return max(max(r) for r in rows)
    
    def check_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_build(used_mask, max_diff):
        """Build rows greedily given current used_mask. Returns (rows, used_mask) or None."""
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                for x in range(1, max_diff + 1):
                    # Check if adding x creates conflicts
                    new_diffs = []
                    conflict = False
                    for a in row:
                        d = x - a
                        if d <= 0 or d > max_diff:
                            conflict = True
                            break
                        if (used_mask >> d) & 1:
                            conflict = True
                            break
                        new_diffs.append(d)
                    if not conflict:
                        # Check internal conflicts within new_diffs
                        if len(set(new_diffs)) == len(new_diffs):
                            row.append(x)
                            for d in new_diffs:
                                used_mask |= (1 << d)
                            found = True
                            break
                if not found:
                    return None, used_mask
            rows.append(row)
        return rows, used_mask
    
    def build_row_greedy(used_mask, max_diff, start_val=0):
        """Build a single row greedily."""
        row = [0]
        while len(row) < k + 1:
            found = False
            for x in range(start_val + 1, max_diff + 1):
                new_diffs = []
                conflict = False
                for a in row:
                    d = x - a
                    if d <= 0 or d > max_diff:
                        conflict = True
                        break
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs.append(d)
                if not conflict:
                    if len(set(new_diffs)) == len(new_diffs):
                        row.append(x)
                        for d in new_diffs:
                            used_mask |= (1 << d)
                        found = True
                        break
            if not found:
                return None, used_mask
        return row, used_mask
    
    def local_search_improve(rows, used_mask, max_diff, time_limit):
        """Try to improve scope by removing and rebuilding rows."""
        nonlocal best_result, best_scope
        current_scope = compute_scope(rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_result = [r[:] for r in rows]
        
        # Try rebuilding each row with a tighter constraint
        for row_idx in range(n_rows):
            if time.time() > time_limit:
                break
            
            # Remove this row's differences from used_mask
            row = rows[row_idx]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used_mask &= ~(1 << d)
            
            # Try to rebuild this row with smaller max
            new_max = max_diff - 1
            improved = False
            for attempt_max in range(new_max, current_scope - 1, -1):
                if time.time() > time_limit:
                    break
                new_row, new_used = build_row_greedy(used_mask, attempt_max)
                if new_row is not None:
                    rows[row_idx] = new_row
                    used_mask = new_used
                    new_scope = compute_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_result = [r[:] for r in rows]
                    improved = True
                    max_diff = min(max_diff, new_scope)
                    break
            
            if not improved:
                # Restore original row
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        used_mask |= (1 << d)
    
    # Main search loop with multiple strategies
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Strategy: vary the max_diff bound and row construction order
        # Start with a tighter bound
        max_diff = max_diff_estimate
        
        # First, do a clean greedy build
        used_mask = 0
        rows, used_mask = greedy_build(used_mask, max_diff)
        
        if rows is None:
            # Try larger bound
            max_diff = int(max_diff * 1.2)
            rows, used_mask = greedy_build(0, max_diff)
            if rows is None:
                continue
        
        scope = compute_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_result = [r[:] for r in rows]
        
        # Try local improvement
        if time.time() < deadline - 5:
            local_search_improve(rows, used_mask, max_diff, deadline - 5)
        
        # If we've been stuck, try perturbation
        if attempt > 5 and best_result is not None:
            # Perturb: randomly change one row and rebuild
            if time.time() < deadline - 10:
                perturbed = [r[:] for r in best_result]
                used_mask2 = 0
                for row in perturbed:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            used_mask2 |= (1 << d)
                
                # Remove a random row
                remove_idx = random.randint(0, n_rows - 1)
                row = perturbed[remove_idx]
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        used_mask2 &= ~(1 << d)
                
                # Rebuild with some randomness
                new_row = [0]
                while len(new_row) < k + 1:
                    found = False
                    # Try values with some randomization
                    candidates = list(range(1, best_scope + 1))
                    random.shuffle(candidates)
                    for x in candidates:
                        new_diffs = []
                        conflict = False
                        for a in new_row:
                            d = x - a
                            if d <= 0 or d > best_scope:
                                conflict = True
                                break
                            if (used_mask2 >> d) & 1:
                                conflict = True
                                break
                            new_diffs.append(d)
                        if not conflict:
                            if len(set(new_diffs)) == len(new_diffs):
                                new_row.append(x)
                                for d in new_diffs:
                                    used_mask2 |= (1 << d)
                                found = True
                                break
                    if not found:
                        break
                
                if len(new_row) == k + 1:
                    perturbed[remove_idx] = new_row
                    if check_valid(perturbed):
                        new_scope = compute_scope(perturbed)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_result = [r[:] for r in perturbed]
    
    if best_result is None:
        # Fallback: simple greedy
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
    
    return best_result
# EVOLVE-BLOCK-END
