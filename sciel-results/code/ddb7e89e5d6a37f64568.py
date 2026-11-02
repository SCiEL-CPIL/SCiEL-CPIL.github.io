<<<<<<< SEARCH
def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Local search that tries to reduce the maximum value by moving elements to smaller positions.
    Uses incremental diff tracking for speed.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Track current max
    curr_max = max(max(r) for r in best_rows)
    no_progress_count = 0
    
    while time.time() < deadline - 3:
        progress = False
        
        # Focus on high values first to reduce scope quickly
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        # Sort by value descending
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Skip if value is already small
            if old_val <= curr_max - 3:
                continue

            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                diffs_involving.add(abs(row[i] - old_val))
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        d = abs(row[i] - new_val)
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        if old_val == curr_max:
                            curr_max = max(max(r) for r in best_rows)
                        break
                
                if found:
                    no_progress_count = 0
                    continue
        
        if not progress:
            no_progress_count += 1
            if no_progress_count > 2:
                # Shake: try to move elements to different positions to escape local minimum
                # Use a more targeted approach: pick elements near the max and try various new values
                shake_attempts = 0
                for _ in range(5):
                    if shake_attempts >= 3:
                        break
                    
                    # Pick an element with high value
                    candidates = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)
                                  if best_rows[r_idx][pos] > curr_max - 10]
                    if not candidates:
                        candidates = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
                    
                    r_idx, pos = random.choice(candidates)
                    row = best_rows[r_idx]
                    old_val = row[pos]
                    
                    diffs_involving = set()
                    for i in range(n):
                        if i == pos: continue
                        diffs_involving.add(abs(row[i] - old_val))
                    forbidden_diffs = all_diffs - diffs_involving
                    
                    lower_bound = row[pos-1] + 1
                    upper_bound = min(250, row[pos+1] - 1) if pos + 1 < n else 250
                    
                    # Try multiple random values
                    tried = set()
                    for _ in range(100):
                        new_val = random.randint(lower_bound, upper_bound)
                        if new_val == old_val or new_val in tried:
                            continue
                        tried.add(new_val)
                        valid = True
                        new_diffs = set()
                        for i in range(n):
                            if i == pos: continue
                            d = abs(row[i] - new_val)
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            all_diffs.difference_update(diffs_involving)
                            all_diffs.update(new_diffs)
                            best_rows[r_idx][pos] = new_val
                            shake_attempts += 1
                            break
            else:
                break
                
    # Final polish: try to reduce each element as much as possible
    if time.time() < deadline - 1:
        # Go through all elements in order of value and try to reduce them
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 1:
                break
            
            row = best_rows[r_idx]
            old_val = row[pos]
            
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                diffs_involving.add(abs(row[i] - old_val))
            
            forbidden_diffs = all_diffs - diffs_involving
            
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        d = abs(row[i] - new_val)
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        break
                
    return best_rows
=======
def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Local search that tries to reduce the maximum value by moving elements to smaller positions.
    Uses incremental diff tracking for speed.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Track current max
    curr_max = max(max(r) for r in best_rows)
    no_progress_count = 0
    
    while time.time() < deadline - 3:
        progress = False
        
        # Focus on high values first to reduce scope quickly
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        # Sort by value descending
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Skip if value is already small
            if old_val <= curr_max - 3:
                continue

            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                diffs_involving.add(abs(row[i] - old_val))
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        d = abs(row[i] - new_val)
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        if old_val == curr_max:
                            curr_max = max(max(r) for r in best_rows)
                        break
                
                if found:
                    no_progress_count = 0
                    continue
        
        if not progress:
            no_progress_count += 1
            if no_progress_count > 2:
                # Shake: try to move elements to different positions to escape local minimum
                shake_attempts = 0
                for _ in range(5):
                    if shake_attempts >= 3:
                        break
                    
                    # Pick an element with high value
                    candidates = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)
                                  if best_rows[r_idx][pos] > curr_max - 10]
                    if not candidates:
                        candidates = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
                    
                    r_idx, pos = random.choice(candidates)
                    row = best_rows[r_idx]
                    old_val = row[pos]
                    
                    diffs_involving = set()
                    for i in range(n):
                        if i == pos: continue
                        diffs_involving.add(abs(row[i] - old_val))
                    forbidden_diffs = all_diffs - diffs_involving
                    
                    lower_bound = row[pos-1] + 1
                    upper_bound = min(250, row[pos+1] - 1) if pos + 1 < n else 250
                    
                    # Try multiple random values
                    tried = set()
                    for _ in range(100):
                        new_val = random.randint(lower_bound, upper_bound)
                        if new_val == old_val or new_val in tried:
                            continue
                        tried.add(new_val)
                        valid = True
                        new_diffs = set()
                        for i in range(n):
                            if i == pos: continue
                            d = abs(row[i] - new_val)
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            all_diffs.difference_update(diffs_involving)
                            all_diffs.update(new_diffs)
                            best_rows[r_idx][pos] = new_val
                            shake_attempts += 1
                            break
            else:
                break
                
    # Final polish: try to reduce each element as much as possible
    if time.time() < deadline - 1:
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 1:
                break
            
            row = best_rows[r_idx]
            old_val = row[pos]
            
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                diffs_involving.add(abs(row[i] - old_val))
            
            forbidden_diffs = all_diffs - diffs_involving
            
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        d = abs(row[i] - new_val)
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        break
                
    return best_rows
>>>>>>> REPLACE

