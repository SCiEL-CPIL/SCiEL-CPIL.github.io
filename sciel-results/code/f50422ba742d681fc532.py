import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Each row has k+1 elements starting at 0
    total_diffs = n_rows * k * (k + 1) // 2
    
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    def greedy_row(used, max_val, strategy='compact'):
        """Build one row greedily given already-used differences."""
        row = [0]
        while len(row) < k + 1:
            last = row[-1]
            found = False
            
            if strategy == 'compact':
                start = last + 1
                end = min(last + 30, max_val)
            elif strategy == 'spread':
                start = last + 1
                end = min(last + 60, max_val)
            else:
                start = last + 1
                end = min(last + 40, max_val)
            
            candidates = list(range(start, end + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
            
            if not found:
                # Fallback: search wider
                for x in range(last + 1, max_val + 1):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
            
            if not found:
                return None
        
        return row
    
    def greedy_construct(seed_val, max_val, strategy='compact'):
        random.seed(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, max_val, strategy)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    # Phase 1: Find initial valid solution
    best_rows = None
    best_scope = float('inf')
    
    for s in range(50):
        if time.time() > deadline - 80:
            break
        for strat in ['compact', 'spread', 'mixed']:
            max_val = 130 + s
            rows = greedy_construct(s * 1000 + hash(strat) % 1000, max_val, strat)
            if rows and is_valid(rows):
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                break  # Found valid for this strategy, try next seed
    
    if best_rows is None:
        # Fallback: very generous greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)
        best_scope = get_scope(best_rows)
    
    # Phase 2: Local search with row regeneration
    current_rows = [list(r) for r in best_rows]
    current_scope = get_scope(current_rows)
    
    def regenerate_row(row_idx, used_without_row, max_val):
        """Regenerate a single row, removing its diffs from used first."""
        # Remove current row's diffs
        row = current_rows[row_idx]
        old_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_diffs.add(row[j] - row[i])
        
        new_used = used_without_row - old_diffs
        
        # Try to build new row
        new_row = greedy_row(new_used, max_val, 'compact')
        if new_row is None:
            new_row = greedy_row(new_used, max_val, 'spread')
        if new_row is None:
            return None
        
        return new_row
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9997
    
    while time.time() < deadline - 15:
        if T < T_min:
            T = 3.0
            cooling = 0.9999
        
        # Pick a row to regenerate
        r_idx = random.randint(0, n_rows - 1)
        
        # Compute used diffs without this row
        used_all = get_used_diffs(current_rows)
        row_diffs = set()
        row = current_rows[r_idx]
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                row_diffs.add(row[j] - row[i])
        used_without = used_all - row_diffs
        
        # Try to regenerate with tighter max
        target_max = max(current_scope - random.randint(0, 3), 115)
        new_row = regenerate_row(r_idx, used_without, target_max)
        
        if new_row is None:
            # Try with current scope
            new_row = regenerate_row(r_idx, used_without, current_scope)
        
        if new_row is not None:
            new_rows = [list(r) for r in current_rows]
            new_rows[r_idx] = new_row
            
            if is_valid(new_rows):
                new_scope = get_scope(new_rows)
                delta = new_scope - current_scope
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_rows = new_rows
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
        
        T *= cooling
    
    # Phase 3: Aggressive compaction
    if best_rows is not None:
        used_set = get_used_diffs(best_rows)
        scope = get_scope(best_rows)
        
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try to reduce the largest element
            # Find which row has the max
            max_row_idx = 0
            max_val = 0
            for i, r in enumerate(best_rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i
            
            row = best_rows[max_row_idx]
            if len(row) < 2 or row[-1] <= 1:
                break
            
            # Remove last element
            removed = row.pop()
            for a in row:
                used_set.discard(removed - a)
            
            found = False
            # Try smaller values
            for x in range(removed - 1, 0, -1):
                if x in row:
                    continue
                valid = True
                for a in row:
                    d = x - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    # Insert in sorted position
                    pos = 0
                    while pos < len(row) and row[pos] < x:
                        pos += 1
                    row.insert(pos, x)
                    for a in row:
                        if a != x:
                            used_set.add(x - a)
                    found = True
                    improved = True
                    break
            
            if not found:
                # Restore
                row.append(removed)
                row.sort()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used_set.add(row[j] - row[i])
            
            if improved:
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
        
        # Also try compaction on other rows
        for _ in range(3000):
            if time.time() > deadline - 1:
                break
            
            r_idx = random.randint(0, n_rows - 1)
            row = best_rows[r_idx]
            if len(row) < 2:
                continue
            
            # Try to reduce a random non-zero element
            p_idx = random.randint(1, len(row) - 1)
            old_val = row[p_idx]
            
            # Remove this element
            row.pop(p_idx)
            old_diffs = set()
            for a in row:
                d = abs(a - old_val)
                used_set.discard(d)
                old_diffs.add(d)
            
            found = False
            for x in range(old_val - 1, 0, -1):
                if x in row:
                    continue
                valid = True
                new_diffs = []
                for a in row:
                    d = abs(x - a)
                    if d in used_set:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < x:
                        pos += 1
                    row.insert(pos, x)
                    for d in new_diffs:
                        used_set.add(d)
                    found = True
                    break
            
            if not found:
                # Restore
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in old_diffs:
                    used_set.add(d)
            
            new_scope = get_scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
    
    # Final validation
    if best_rows is not None and is_valid(best_rows):
        return best_rows
    
    # Fallback: return a valid solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                x += 1
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    break
            row.append(x)
            used.update({x - a for a in row if a != x})
        rows.append(row)
    return rows
