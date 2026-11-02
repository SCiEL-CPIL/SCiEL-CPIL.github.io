import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all differences between two entries of the same row are distinct over all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 112
    n = k + 1  # number of entries per row
    
    # Use bitset for efficient difference tracking
    # Bit d is set if difference d is used
    MAX_DIFF = 2000
    
    best_solution = None
    best_score = float('inf')
    
    # Phase 1: Multi-start with randomized greedy
    n_starts = 12
    for s in range(n_starts):
        if time.time() > deadline - 50:
            break
        sol = _random_greedy(n_rows, k, s * 7919 + 13)
        if sol:
            score = max(max(r) for r in sol)
            if score < best_score:
                best_score = score
                best_solution = sol
    
    if best_solution is None:
        best_solution = _random_greedy(n_rows, k, 0)
        if not best_solution:
            return []
        best_score = max(max(r) for r in best_solution)
    
    # Phase 2: Iterated local search with simulated annealing
    curr_solution = [r[:] for r in best_solution]
    curr_score = best_score
    
    no_improve = 0
    temperature = 5.0
    min_temp = 0.5
    
    while time.time() < deadline - 10:
        # Perturb: pick a few random elements and try to move them
        candidate = [r[:] for r in curr_solution]
        n_perturb = random.randint(2, 5)
        
        for _ in range(n_perturb):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = candidate[r_idx]
            old_val = row[pos]
            
            # Try to find a different valid value
            new_val = _find_alt_value(candidate, r_idx, pos, n, MAX_DIFF)
            if new_val is not None and new_val != old_val:
                candidate[r_idx][pos] = new_val
        
        # Local search to minimize
        improved = _local_search(candidate, n_rows, k, deadline, MAX_DIFF)
        if improved:
            curr_score = max(max(r) for r in improved)
            
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                no_improve = 0
            else:
                # Accept with probability based on temperature
                delta = curr_score - best_score
                if delta < 0 or random.random() < (1.0 / (1.0 + delta / temperature)):
                    curr_solution = improved
                no_improve += 1
            
            # Cool down
            if no_improve >= 8:
                temperature = max(min_temp, temperature * 0.85)
                no_improve = 0
        
        # Every 15 iterations, do a more aggressive perturbation
        if no_improve == 7:
            candidate = [r[:] for r in best_solution]
            for _ in range(random.randint(4, 8)):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                row = candidate[r_idx]
                old_val = row[pos]
                new_val = _find_alt_value(candidate, r_idx, pos, n, MAX_DIFF)
                if new_val is not None and new_val != old_val:
                    candidate[r_idx][pos] = new_val
            improved = _local_search(candidate, n_rows, k, deadline, MAX_DIFF)
            if improved:
                cs = max(max(r) for r in improved)
                if cs < best_score:
                    best_score = cs
                    best_solution = improved
                    curr_solution = improved
                    no_improve = 0
    
    # Phase 3: Final squeeze
    if time.time() < deadline - 3:
        final = _final_squeeze([r[:] for r in best_solution], n_rows, k, deadline, MAX_DIFF)
        if final:
            fs = max(max(r) for r in final)
            if fs < best_score:
                best_score = fs
                best_solution = final
    
    return best_solution

def _random_greedy(n_rows, k, seed):
    """
    Build rows greedily with randomization.
    """
    rng = random.Random(seed)
    n = k + 1
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < n:
            found = False
            base = row[-1] + 1
            # Try sequential first
            for x in range(base, base + 400):
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
            
            if not found:
                # Randomized search
                for _retry in range(500):
                    x = base + rng.randint(0, 300)
                    if x > 1500:
                        continue
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
        rows.append(row)
    
    return rows

def _find_alt_value(rows, r_idx, pos, n, MAX_DIFF):
    """
    Find an alternative valid value for rows[r_idx][pos].
    """
    row = rows[r_idx]
    old_val = row[pos]
    
    # Compute all diffs in the system
    all_diffs = set()
    for r in rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Diffs involving old_val
    diffs_involving = set()
    for i in range(n):
        if i == pos:
            continue
        diffs_involving.add(abs(row[i] - old_val))
    
    forbidden = all_diffs - diffs_involving
    
    lower_bound = row[pos-1] + 1
    upper_bound = 400
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    if upper_bound < lower_bound:
        return None
    
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            d = abs(row[i] - new_val)
            if d in forbidden or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            valid_candidates.append(new_val)
            if len(valid_candidates) > 100:
                break
    
    if not valid_candidates:
        return None
    return random.choice(valid_candidates)

def _local_search(rows, n_rows, k, deadline, MAX_DIFF):
    """
    Local search to minimize the maximum value.
    Uses bitset for fast diff checking.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build diff bitset
    diff_bits = 0
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                d = r[j] - r[i]
                diff_bits |= (1 << d)
    
    no_progress = 0
    
    while time.time() < deadline - 3:
        progress = False
        
        # Process positions from highest value to lowest
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
            
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Diffs involving old_val
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                d = abs(row[i] - old_val)
                diffs_involving.add(d)
            
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
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                        # Check if d is forbidden
                        # d is forbidden if it's in diff_bits but NOT in diffs_involving
                        if d not in diffs_involving and ((diff_bits >> d) & 1):
                            valid = False
                            break
                    
                    if valid:
                        # Update diff_bits
                        for d in diffs_involving:
                            diff_bits &= ~(1 << d)
                        for d in new_diffs:
                            diff_bits |= (1 << d)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        break
        
        if not progress:
            no_progress += 1
            if no_progress > 2:
                # Shake
                for _ in range(4):
                    r_idx = random.randint(0, n_rows - 1)
                    pos = random.randint(1, k)
                    row = best_rows[r_idx]
                    old_val = row[pos]
                    new_val = _find_alt_value(best_rows, r_idx, pos, n, MAX_DIFF)
                    if new_val is not None and new_val != old_val:
                        # Update diff_bits
                        diffs_involving = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            diffs_involving.add(abs(row[i] - old_val))
                        new_diffs = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            new_diffs.add(abs(row[i] - new_val))
                        for d in diffs_involving:
                            diff_bits &= ~(1 << d)
                        for d in new_diffs:
                            diff_bits |= (1 << d)
                        best_rows[r_idx][pos] = new_val
                no_progress = 0
            else:
                break
    
    return best_rows

def _final_squeeze(rows, n_rows, k, deadline, MAX_DIFF):
    """
    Final pass to minimize each element.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    diff_bits = 0
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                d = r[j] - r[i]
                diff_bits |= (1 << d)
    
    for _pass in range(4):
        if time.time() > deadline - 2:
            break
        progress = False
        for r_idx in range(n_rows):
            for pos in range(k, 0, -1):
                if time.time() > deadline - 2:
                    break
                row = best_rows[r_idx]
                old_val = row[pos]
                
                diffs_involving = set()
                for i in range(n):
                    if i == pos:
                        continue
                    diffs_involving.add(abs(row[i] - old_val))
                
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
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                            if d not in diffs_involving and ((diff_bits >> d) & 1):
                                valid = False
                                break
                        
                        if valid:
                            for d in diffs_involving:
                                diff_bits &= ~(1 << d)
                            for d in new_diffs:
                                diff_bits |= (1 << d)
                            best_rows[r_idx][pos] = new_val
                            progress = True
                            break
        if not progress:
            break
    
    return best_rows
