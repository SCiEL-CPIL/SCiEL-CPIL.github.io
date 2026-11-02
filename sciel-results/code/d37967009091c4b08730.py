import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    best_rows = None
    best_scope = float('inf')
    
    def greedy_randomized(seed_offset=0):
        """Randomized greedy with various heuristics."""
        used = 0  # bitmask for used differences
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            # Randomize the order in which we try to add elements
            # Strategy: try to keep the row compact but avoid used differences
            
            while len(row) < k + 1:
                last = row[-1]
                # Try values starting from last+1, but with some randomization
                # to avoid getting stuck in bad patterns
                
                # Determine a range to search
                # We want to try smaller values first (compact rows)
                # but occasionally jump to avoid local minima
                
                found = False
                # Try a window of values
                max_search = last + 50  # Look ahead up to 50
                if row_idx == 0:
                    # First row: be more systematic
                    candidates = list(range(last + 1, min(last + max_search, lb + 50)))
                else:
                    # Subsequent rows: more random
                    candidates = list(range(last + 1, min(last + max_search, lb + 100)))
                
                # Shuffle with bias toward smaller values
                if random.random() < 0.7:
                    # Mostly try in order but with some perturbation
                    random.shuffle(candidates)
                    # Re-sort but with some randomness in the first few
                    candidates = sorted(candidates, key=lambda x: x + random.random() * 5)
                else:
                    random.shuffle(candidates)
                
                for x in candidates:
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    # If no candidate in the window works, try a wider search
                    # This shouldn't happen often with good bounds
                    for x in range(last + 1, lb + 200):
                        valid = True
                        new_diffs = 0
                        for a in row:
                            d = x - a
                            if (used >> d) & 1:
                                valid = False
                                break
                            new_diffs |= (1 << d)
                        
                        if valid:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                    if not found:
                        return None, float('inf')
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope
    
    # Run many randomized greedy attempts
    num_attempts = 500
    for i in range(num_attempts):
        if time.time() > deadline - 5:
            break
        
        random.seed(i * 7919 + 13)  # Deterministic but varied seeds
        
        rows, scope = greedy_randomized(i)
        
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = rows
            
            # If we hit the lower bound, we can stop
            if best_scope == lb:
                break
    
    # Try a few more with different strategies
    for i in range(100):
        if time.time() > deadline - 5:
            break
        
        random.seed(100000 + i * 104729)
        
        # Alternative strategy: build rows in random order
        used = 0
        rows = [None] * n_rows
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        success = True
        for row_idx in row_order:
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Try values with randomization
                search_limit = min(last + 80, lb + 150)
                for x in range(last + 1, search_limit):
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    # Try wider
                    for x in range(last + 1, lb + 300):
                        valid = True
                        new_diffs = 0
                        for a in row:
                            d = x - a
                            if (used >> d) & 1:
                                valid = False
                                break
                            new_diffs |= (1 << d)
                        
                        if valid:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                    if not found:
                        success = False
                        break
            
            if not success:
                break
            rows[row_idx] = row
        
        if success and all(r is not None for r in rows):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope == lb:
                    break
    
    return best_rows
