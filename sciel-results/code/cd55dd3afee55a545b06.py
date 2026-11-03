"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all pairwise differences are distinct."""
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n_marks = k + 1
    n_diffs_per_row = n_marks * (n_marks - 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    lower_bound = total_diffs  # 105 for n_rows=7, k=5
    
    best_rows = None
    best_max = float('inf')
    
    def try_greedy(seed, max_limit, strategy='sequential'):
        """Try a greedy construction with given parameters."""
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            stuck = False
            while len(row) < n_marks:
                last = row[-1]
                found = False
                
                if strategy == 'sequential':
                    # Try marks in order from last+1
                    x = last + 1
                    max_x = min(max_limit, last + 200)
                    while x <= max_x:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                        x += 1
                elif strategy == 'randomized':
                    # Try a random subset of candidates
                    x = last + 1
                    max_x = min(max_limit, last + 100)
                    candidates = list(range(x, max_x + 1))
                    rng.shuffle(candidates)
                    for x in candidates:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                elif strategy == 'priority':
                    # Prioritize marks that create "rare" differences
                    x = last + 1
                    max_x = min(max_limit, last + 150)
                    candidates = list(range(x, max_x + 1))
                    # Score candidates by how "expensive" their differences are
                    scored = []
                    for x in candidates:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            # Prefer differences that are larger (less likely to conflict)
                            score = max(new_diffs)
                            scored.append((score, x, new_diffs))
                    scored.sort(reverse=True)  # Try largest differences first
                    for score, x, new_diffs in scored:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    stuck = True
                    break
            
            if stuck:
                return None
            rows.append(row)
        
        max_entry = max(max(r) for r in rows)
        if max_entry <= max_limit:
            return rows
        return None
    
    # Phase 1: Quick greedy attempts with tight limits
    for limit in [lower_bound, lower_bound + 1, lower_bound + 2, lower_bound + 3, lower_bound + 5, lower_bound + 7, lower_bound + 10]:
        if time.time() > deadline:
            break
        for seed in range(30):
            if time.time() > deadline:
                break
            for strategy in ['sequential', 'randomized']:
                rows = try_greedy(seed, limit, strategy)
                if rows is not None:
                    max_entry = max(max(r) for r in rows)
                    if max_entry < best_max:
                        best_max = max_entry
                        best_rows = rows
                    if best_max == lower_bound:
                        return best_rows
    
    # Phase 2: More relaxed limits with more seeds
    for limit in [lower_bound + 12, lower_bound + 15, lower_bound + 20, lower_bound + 25]:
        if time.time() > deadline:
            break
        for seed in range(50):
            if time.time() > deadline:
                break
            for strategy in ['sequential', 'randomized', 'priority']:
                rows = try_greedy(seed, limit, strategy)
                if rows is not None:
                    max_entry = max(max(r) for r in rows)
                    if max_entry < best_max:
                        best_max = max_entry
                        best_rows = rows
    
    # Phase 3: If we haven't found a good solution, do a more careful search
    if time.time() < deadline - 20 and best_max > lower_bound + 5:
        # Try building rows one at a time with backtracking on the last row
        for target_max in range(lower_bound, min(lower_bound + 15, deadline and 200)):
            if time.time() > deadline:
                break
            
            for seed in range(20):
                if time.time() > deadline:
                    break
                
                rng = random.Random(seed)
                used = set()
                rows = []
                success = True
                
                for row_idx in range(n_rows):
                    row = [0]
                    # For the last row, use more careful search
                    is_last = (row_idx == n_rows - 1)
                    search_width = 300 if is_last else 150
                    
                    while len(row) < n_marks:
                        last = row[-1]
                        found = False
                        x = last + 1
                        max_x = min(target_max, last + search_width)
                        
                        if is_last and len(row) >= n_marks - 2:
                            # For near-completion of last row, be more thorough
                            candidates = list(range(x, max_x + 1))
                            rng.shuffle(candidates)
                            for x in candidates:
                                new_diffs = {x - a for a in row}
                                if not new_diffs & used:
                                    row.append(x)
                                    used |= new_diffs
                                    found = True
                                    break
                        else:
                            while x <= max_x:
                                new_diffs = {x - a for a in row}
                                if not new_diffs & used:
                                    row.append(x)
                                    used |= new_diffs
                                    found = True
                                    break
                                x += 1
                        
                        if not found:
                            success = False
                            break
                    
                    if not success:
                        break
                    rows.append(row)
                
                if success:
                    max_entry = max(max(r) for r in rows)
                    if max_entry < best_max:
                        best_max = max_entry
                        best_rows = rows
                    if best_max == lower_bound:
                        break
            if best_max == lower_bound:
                break
    
    # Fallback: pure greedy without limit
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n_marks:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
    
    return best_rows

# EVOLVE-BLOCK-END
