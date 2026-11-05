import time
import numpy as np

def construct(n_rows, k):
    deadline = time.time() + 115.0
    best_scope = float('inf')
    best_rows = None
    rng = np.random.default_rng()
    
    # Helper to check if a candidate x is valid for a row, given current used set
    # and the rest of the row (excluding the position being filled)
    # Here we assume we are appending to the end, so we check x - a for all a in row
    
    while time.time() < deadline:
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                valid_candidates = []
                # Look ahead window
                # If we look too far, we might pick a large x unnecessarily.
                # If we look too little, we might get stuck.
                # 20 seems reasonable.
                max_lookahead = 20
                while len(valid_candidates) < max_lookahead:
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        valid_candidates.append(x)
                    x += 1
                    if time.time() > deadline:
                        success = False
                        break
                
                if not success:
                    break
                if not valid_candidates:
                    success = False
                    break
                
                # Random selection with bias towards smaller
                n_cands = len(valid_candidates)
                if n_cands == 1:
                    chosen = valid_candidates[0]
                else:
                    weights = np.array([1.0 / (i + 1) for i in range(n_cands)])
                    weights /= weights.sum()
                    idx = rng.choice(n_cands, p=weights)
                    chosen = valid_candidates[idx]
                
                row.append(chosen)
                for a in row[:-1]:
                    used.add(chosen - a)
            
            if not success:
                break
            rows.append(row)
        
        if success:
            # Tightening Phase
            # Try to reduce each element to the smallest valid value
            # Iterate rows, and within row, from last to first
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for i in range(k, 0, -1):
                    old_val = row[i]
                    min_val = row[i-1] + 1
                    # Try to find smallest valid v in [min_val, old_val]
                    # We need to remove old diffs, try new, if fail restore
                    # Optimization: only try if old_val > min_val
                    if old_val == min_val:
                        continue
                    
                    old_diffs = set()
                    for j, a in enumerate(row):
                        if j == i: continue
                        if a < old_val:
                            old_diffs.add(old_val - a)
                        else:
                            old_diffs.add(a - old_val)
                    
                    for d in old_diffs:
                        used.remove(d)
                    
                    best_v = None
                    # Scan upwards from min_val to old_val - 1
                    # We want the smallest valid v
                    for v in range(min_val, old_val):
                        new_diffs = set()
                        conflict = False
                        for j, a in enumerate(row):
                            if j == i: continue
                            if a < v:
                                d = v - a
                            else:
                                d = a - v
                            if d in used:
                                conflict = True
                                break
                            new_diffs.add(d)
                        
                        if not conflict:
                            best_v = v
                            break
                    
                    if best_v is not None:
                        row[i] = best_v
                        for d in new_diffs:
                            used.add(d)
                    else:
                        # Restore
                        for d in old_diffs:
                            used.add(d)
            
            # Check final scope
            current_scope = max(max(r) for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                if current_scope <= 105:
                    break

    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows
        
    return best_rows
