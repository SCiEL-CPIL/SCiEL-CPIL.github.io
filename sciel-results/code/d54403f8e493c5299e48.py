import time
import random
import bisect
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 140
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(r) for r in rows)
    
    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs
    
    def is_valid(rows):
        if not rows:
            return True
        diffs = get_all_diffs(rows)
        return len(diffs) == total_diffs

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def hill_climbing(rows, iterations=20000, timeout=None):
        if timeout is None:
            timeout = deadline
        current = [r[:] for r in rows]
        current_scope = get_scope(current)
        best = [r[:] for r in current]
        best_scope = current_scope
        
        # Precompute diffs for current to speed up? No, just recompute locally.
        
        for it in range(iterations):
            if time.time() > timeout:
                break
            
            ri = random.randint(0, len(current) - 1)
            pi = random.randint(1, k)
            
            old_val = current[ri][pi]
            lo = current[ri][pi - 1] + 1
            # If pi is the last element, hi can be larger
            if pi == k:
                hi = current_scope + 10 # Allow some expansion
            else:
                hi = current[ri][pi + 1] - 1
            
            if lo > hi:
                continue
            
            # Try a few random values in range
            # To be faster, try a few specific moves
            candidates = set()
            if hi - lo < 10:
                candidates = set(range(lo, hi + 1))
            else:
                for _ in range(5):
                    candidates.add(random.randint(lo, hi))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                
                new_row = current[ri][:]
                new_row[pi] = new_val
                
                # Check strictly increasing
                if new_row != sorted(new_row):
                    continue
                
                # Check if new row has internal duplicates (shouldn't if sorted strictly)
                # Check conflicts with other rows
                other_diffs = set()
                for idx, r in enumerate(current):
                    if idx == ri:
                        continue
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            other_diffs.add(r[j] - r[i])
                
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])
                
                if new_diffs & other_diffs:
                    continue
                
                # Check if new row itself has duplicates? 
                # Since it's strictly increasing, differences are positive.
                # But can a difference appear twice within the same row?
                # e.g. 0, 1, 2 -> diffs 1, 2, 1. Duplicate 1.
                # The problem requires "all positive differences ... are distinct over all rows".
                # So differences must be distinct globally, including within a row.
                if len(new_diffs) != k * (k + 1) // 2:
                    continue
                
                new_scope = max(max(r) for idx, r in enumerate(current) if idx != ri)
                new_scope = max(new_scope, max(new_row))
                
                if new_scope < best_scope:
                    best = [r[:] for r in current]
                    best[ri] = new_row
                    best_scope = new_scope
                    current = [r[:] for r in best]
                    current_scope = best_scope
                elif new_scope == current_scope:
                    # Accept to explore
                    current[ri] = new_row
                    # current_scope remains same? No, if max is in another row, it stays.
                    # If max was in this row and we reduced it, current_scope should drop?
                    # But we only accept if new_scope < best_scope for update.
                    # For exploration, we just move current.
                    # Recalculate current_scope
                    current_scope = get_scope(current)
                    
        return best, best_scope

    # Start with greedy
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # Run hill climbing
    if time.time() < deadline - 5:
        hc_rows, hc_scope = hill_climbing(best_rows, iterations=50000)
        if hc_scope < best_scope:
            best_rows = hc_rows
            best_scope = hc_scope
            
    # Final validity check
    if not is_valid(best_rows):
        # Fallback to a simpler valid construction if possible
        # Greedy is usually valid
        greedy_rows = greedy_construct()
        if is_valid(greedy_rows):
            best_rows = greedy_rows
            best_scope = get_scope(best_rows)
        else:
            # Return whatever we have, hoping it's valid or close
            pass

    return best_rows
