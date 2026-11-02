import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 100.0 # Safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    # Run multiple iterations
    while time.time() < deadline:
        # 1. Generate a candidate using randomized greedy
        rows = randomized_greedy(n_rows, k)
        
        if rows is None:
            continue
            
        scope = max(max(row) for row in rows)
        
        # 2. Improve via local search
        rows = local_search_improve(rows, start_time, deadline)
        
        if rows is None:
            continue
            
        scope = max(max(row) for row in rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [row[:] for row in rows]
            
    # Fallback if no valid solution found (should not happen for small n, k)
    if best_rows is None:
        # Generate a trivial valid solution with large scope
        best_rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            current = 0
            for _ in range(k):
                while True:
                    current += 1
                    new_diffs = {current - a for a in row}
                    if not (new_diffs & used_diffs):
                        break
                row.append(current)
                used_diffs.update(new_diffs)
            best_rows.append(row)
            
    return best_rows

def randomized_greedy(n_rows, k):
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_val = row[-1]
            # Find valid candidates
            candidates = []
            x = last_val + 1
            # We don't need to search too far if we have enough candidates
            # But we need to find the smallest ones first
            # Limit search scope to avoid infinite loops, but 500 is safe for n=7, k=5
            while len(candidates) < 5: 
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    candidates.append(x)
                x += 1
                if x > 1000: break
            
            if not candidates:
                return None
            
            # Choose a candidate
            # Randomize to explore different branches
            # Weighted towards smaller values to keep scope tight
            r = random.random()
            if r < 0.5:
                chosen = candidates[0]
            elif r < 0.8:
                chosen = candidates[1] if len(candidates) > 1 else candidates[0]
            else:
                chosen = random.choice(candidates)
            
            row.append(chosen)
            used_diffs.update({chosen - a for a in row[:-1]})
            
        rows.append(row)
        
    return rows

def local_search_improve(rows, start_time, deadline):
    # Calculate global used diffs
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
                
    # We try to reduce any element in any row
    # To be efficient, we iterate multiple passes
    # Time limit check
    max_passes = 20
    
    for _ in range(max_passes):
        if time.time() > deadline - 5:
            break
            
        improved = False
        # Randomize order of rows and elements to avoid local minima bias
        row_indices = list(range(len(rows)))
        random.shuffle(row_indices)
        
        for idx in row_indices:
            row = rows[idx]
            # Try to reduce each element from largest to smallest (excluding 0)
            # We iterate from last to first
            for pos in range(len(row) - 1, 0, -1):
                v = row[pos]
                # v must be > row[pos-1]
                min_v = row[pos - 1] + 1
                
                if v <= min_v:
                    continue
                    
                # Get diffs involving this element v
                # Note: diffs are row[j] - row[i] for i < j
                # If we change row[pos], it affects:
                # 1. Diffs where pos is the larger index: {row[pos] - row[i] for i < pos}
                # 2. Diffs where pos is the smaller index: {row[j] - row[pos] for j > pos}
                
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(v - row[i])
                for j in range(pos + 1, len(row)):
                    old_diffs.add(row[j] - v)
                
                # Forbidden diffs are all used_diffs except those involving this element
                forbidden = used_diffs - old_diffs
                
                found = False
                # Try to reduce v
                for v_prime in range(v - 1, min_v - 1, -1):
                    # Check validity
                    new_diffs = set()
                    valid = True
                    for i in range(pos):
                        d = v_prime - row[i]
                        if d in forbidden or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    for j in range(pos + 1, len(row)):
                        d = row[j] - v_prime
                        if d in forbidden or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Update global set
                        used_diffs -= old_diffs
                        used_diffs |= new_diffs
                        row[pos] = v_prime
                        improved = True
                        found = True
                        break
                        
                if found:
                    # Restart row iteration? Or just continue?
                    # Continuing is faster.
                    pass
                    
        if not improved:
            break
            
    return rows
