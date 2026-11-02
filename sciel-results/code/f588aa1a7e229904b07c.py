import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    best_rows = None
    best_scope = float('inf')
    
    def check_valid(x, row, used_set):
        """Check if x can be added to row without creating duplicate differences."""
        for a in row:
            d = x - a
            if d in used_set:
                return False, set()
        return True, set(x - a for a in row)
    
    def try_compact_rows(rows, used_set, max_iter=200):
        """Try to compact rows by removing and re-inserting elements."""
        nonlocal best_scope, best_rows
        scope = max(max(r) for r in rows)
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            improved = False
            # Pick a random row and try to shift its last element left
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            last = row[-1]
            if last <= 1:
                continue
            
            # Remove last element
            removed = row.pop()
            removed_diffs = set(removed - a for a in row)
            used_set -= removed_diffs
            
            # Try to re-insert a smaller value
            found = False
            for x in range(row[-1] + 1, last):
                valid, new_diffs = check_valid(x, row, used_set)
                if valid:
                    row.append(x)
                    used_set |= new_diffs
                    found = True
                    improved = True
                    break
            
            if not found:
                # Restore original
                row.append(removed)
                used_set |= removed_diffs
            
            # Also try to compact internal elements
            if not improved and len(row) > 2:
                pos = random.randint(1, len(row) - 2)
                elem = row[pos]
                if elem <= row[pos-1] + 1:
                    continue
                
                # Remove element at pos
                row.pop(pos)
                removed_diffs = set()
                for a in row:
                    if a < elem:
                        removed_diffs.add(elem - a)
                    else:
                        removed_diffs.add(a - elem)
                used_set -= removed_diffs
                
                # Try to re-insert a smaller value
                found = False
                lower = row[pos-1] + 1 if pos > 0 else 1
                upper = row[pos] + 1 if pos < len(row) else elem
                for x in range(lower, min(upper, elem)):
                    valid, new_diffs = check_valid(x, row, used_set)
                    if valid:
                        # Need to check that x maintains order
                        if (pos == 0 or x > row[pos-1]) and (pos == len(row) or x < row[pos]):
                            row.insert(pos, x)
                            used_set |= new_diffs
                            found = True
                            improved = True
                            break
                
                if not found:
                    row.insert(pos, elem)
                    used_set |= removed_diffs
            
            # Check if we improved scope
            current_scope = max(max(r) for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                
            if best_scope == lb:
                break
    
    def greedy_randomized(seed_offset=0):
        """Randomized greedy with various heuristics."""
        used_set = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            
            while len(row) < k + 1:
                last = row[-1]
                
                # Try values starting from last+1 with randomization
                found = False
                
                # Generate candidates with bias toward smaller values
                max_search = last + 60
                if row_idx == 0:
                    candidates = list(range(last + 1, min(last + max_search, lb + 50)))
                else:
                    candidates = list(range(last + 1, min(last + max_search, lb + 100)))
                
                # Shuffle with bias toward smaller values
                if random.random() < 0.6:
                    # Mostly try in order but with some perturbation
                    random.shuffle(candidates)
                    candidates = sorted(candidates, key=lambda x: x + random.random() * 8)
                else:
                    random.shuffle(candidates)
                
                for x in candidates:
                    valid, new_diffs = check_valid(x, row, used_set)
                    if valid:
                        row.append(x)
                        used_set |= new_diffs
                        found = True
                        break
                
                if not found:
                    # Try wider search
                    for x in range(last + 1, lb + 300):
                        valid, new_diffs = check_valid(x, row, used_set)
                        if valid:
                            row.append(x)
                            used_set |= new_diffs
                            found = True
                            break
                    if not found:
                        return None, float('inf')
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope, used_set
    
    def greedy_randomized_order(seed_offset=0):
        """Build rows in random order."""
        used_set = set()
        rows = [None] * n_rows
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for row_idx in row_order:
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                search_limit = min(last + 80, lb + 150)
                
                for x in range(last + 1, search_limit):
                    valid, new_diffs = check_valid(x, row, used_set)
                    if valid:
                        row.append(x)
                        used_set |= new_diffs
                        found = True
                        break
                
                if not found:
                    for x in range(last + 1, lb + 300):
                        valid, new_diffs = check_valid(x, row, used_set)
                        if valid:
                            row.append(x)
                            used_set |= new_diffs
                            found = True
                            break
                    if not found:
                        return None, float('inf'), None
            
            rows[row_idx] = row
        
        scope = max(max(r) for r in rows)
        return rows, scope, used_set
    
    # Run many randomized greedy attempts
    num_attempts = 300
    for i in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        random.seed(i * 7919 + 13)
        
        rows, scope, used_set = greedy_randomized(i)
        
        if rows is not None:
            # Try to compact
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
            
            # Local improvement pass
            if used_set is not None:
                try_compact_rows(rows, used_set, max_iter=50)
            
            if best_scope == lb:
                break
    
    # Try more with different strategies
    for i in range(150):
        if time.time() > deadline - 5:
            break
        
        random.seed(100000 + i * 104729)
        
        rows, scope, used_set = greedy_randomized_order(i)
        
        if rows is not None and all(r is not None for r in rows):
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
            
            if used_set is not None:
                try_compact_rows(rows, used_set, max_iter=50)
            
            if best_scope == lb:
                break
    
    return best_rows
