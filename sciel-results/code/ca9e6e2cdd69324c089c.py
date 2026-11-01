import time
import random
import math
from collections import Counter

_DEADLINE = 145.0

def _row_diffs(row):
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d

def _build_counter(rows):
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return c

def _conflicts_from_counter(c):
    return sum(v - 1 for v in c.values() if v > 1)

def _scope(rows):
    return max(r[-1] for r in rows)

def _energy(rows, counter):
    conf = _conflicts_from_counter(counter)
    if conf > 0:
        return 1000 * conf + _scope(rows)
    else:
        return _scope(rows)

def _greedy_row_randomized(used, k, rng, rand_factor=0.1):
    row = [0]
    x = 0
    while len(row) < k + 1:
        valid_x = None
        candidates = []
        start_x = x + 1
        for trial_x in range(start_x, start_x + 2000):
            new = {trial_x - a for a in row}
            if not (new & used):
                candidates.append(trial_x)
                if len(candidates) > 10:
                    break
        if not candidates:
            trial_x = start_x
            while True:
                new = {trial_x - a for a in row}
                if not (new & used):
                    candidates.append(trial_x)
                    break
                trial_x += 1
        
        if rand_factor > 0 and len(candidates) > 1:
            x = rng.choice(candidates[:3])
        else:
            x = candidates[0]
            
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row

def _greedy_all_randomized(n_rows, k, rng, rand_factor=0.2):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row_randomized(used, k, rng, rand_factor)
        rows.append(row)
        used |= set(_row_diffs(row))
    return rows

def _targeted_descent(rows, k, deadline, rng):
    """
    Directly minimize the global maximum by attempting to reduce the largest elements.
    """
    t0 = time.time()
    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    cur_energy = _energy(cur_rows, cur_counter)
    
    best_rows = [list(r) for r in cur_rows]
    best_energy = cur_energy
    best_conf = _conflicts_from_counter(cur_counter)
    best_scope = _scope(cur_rows)
    
    n_rows = len(cur_rows)
    row_len = k + 1
    
    # We will repeatedly try to reduce the last element of the row with the largest last element.
    # If that fails or doesn't improve, we try other positions in that row, or other rows.
    
    while time.time() < deadline:
        # Find the row with the maximum last element
        max_idx = max(range(n_rows), key=lambda i: cur_rows[i][-1])
        max_row = cur_rows[max_idx]
        
        improved = False
        
        # Try to reduce elements in max_row, starting from the end
        for pos in range(row_len - 1, 0, -1):
            old_val = max_row[pos]
            lo = max_row[pos - 1] + 1
            hi = (max_row[pos + 1] - 1) if pos + 1 < row_len else old_val - 1
            
            if hi < lo:
                continue
                
            # Try values from hi down to lo
            # We want to find a value that is valid and reduces the scope (if pos is last) or at least maintains validity
            # For simplicity, we just try a few random values in [lo, hi] or scan downwards
            # Scanning downwards is better for minimization
            
            # To save time, we don't scan the whole range if it's huge.
            # But for the last element, the range is usually small if we've been optimizing.
            
            # Let's try a few candidates
            candidates = []
            if hi - lo < 50:
                candidates = list(range(hi, lo - 1, -1))
            else:
                # Random sample
                for _ in range(10):
                    candidates.append(rng.randint(lo, hi))
                # Always include hi and lo
                candidates.append(hi)
                candidates.append(lo)
                # Sort descending to try to reduce scope
                candidates.sort(reverse=True)
            
            for new_val in candidates:
                if new_val >= old_val:
                    continue
                    
                # Check validity
                # Remove old diffs involving pos
                old_d = []
                for j in range(row_len):
                    if j != pos:
                        if j < pos:
                            old_d.append(old_val - max_row[j])
                        else:
                            old_d.append(max_row[j] - old_val)
                            
                # Check if new diffs conflict with current counter
                # We need to check against (cur_counter - old_d) + new_d
                # Actually, simpler:
                # 1. Tentatively remove old_d from a copy of counter? No, too slow.
                # 2. Calculate new_d.
                # 3. For each d in new_d, check if cur_counter[d] - (1 if d in old_d else 0) > 0
                #    AND if d is not in new_d' (other new diffs) with count > (1 if d in old_d else 0) ... 
                #    This is getting complicated.
                
                # Let's just do the full counter update check for the few candidates.
                
                new_d = []
                for j in range(row_len):
                    if j != pos:
                        if j < pos:
                            new_d.append(new_val - max_row[j])
                        else:
                            new_d.append(max_row[j] - new_val)
                
                # Check conflicts
                # Create a temporary delta
                # If a diff d is in new_d, and in cur_counter, and not in old_d (or less times in old_d), it might be a conflict.
                
                # Efficient check:
                # Count occurrences in new_d
                new_d_counts = Counter(new_d)
                old_d_counts = Counter(old_d)
                
                valid = True
                for d, cnt in new_d_counts.items():
                    # Current count in global counter
                    cur_cnt = cur_counter.get(d, 0)
                    # Count removed from this row
                    rem_cnt = old_d_counts.get(d, 0)
                    # Net count after change
                    net_cnt = cur_cnt - rem_cnt + cnt
                    if net_cnt > 1:
                        valid = False
                        break
                
                if valid:
                    # Apply change
                    for d in old_d:
                        cur_counter[d] -= 1
                        if cur_counter[d] == 0:
                            del cur_counter[d]
                    for d in new_d:
                        cur_counter[d] = cur_counter.get(d, 0) + 1
                        
                    max_row[pos] = new_val
                    
                    new_scope = _scope(cur_rows)
                    new_conf = _conflicts_from_counter(cur_counter)
                    new_energy = _energy(cur_rows, cur_counter)
                    
                    if new_conf == 0 and new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in cur_rows]
                    
                    improved = True
                    break # Move to next position (lower index) or next row? 
                    # Since we reduced a value, the row might still have the max last element if pos != last.
                    # But we broke the loop for this pos. We should probably restart the row scan or just continue to next pos.
                    # Continuing to next pos is fine.
            
            if improved:
                # If we made an improvement, we might want to restart the row scan from the end
                # because the structure changed.
                # But for now, let's just continue to the next position.
                pass
        
        if not improved:
            # If we couldn't improve the max row, try to improve other rows?
            # Or try random moves?
            # Let's try a random move in a random row to escape local optima.
            r_idx = rng.randrange(n_rows)
            pos = 1 + rng.randrange(row_len - 1)
            row = cur_rows[r_idx]
            old_val = row[pos]
            lo = row[pos - 1] + 1
            hi = (row[pos + 1] - 1) if pos + 1 < row_len else old_val + 10
            
            if hi > lo:
                new_val = rng.randint(lo, hi)
                if new_val != old_val:
                    old_d = []
                    for j in range(row_len):
                        if j != pos:
                            if j < pos:
                                old_d.append(old_val - row[j])
                            else:
                                old_d.append(row[j] - old_val)
                    
                    new_d = []
                    for j in range(row_len):
                        if j != pos:
                            if j < pos:
                                new_d.append(new_val - row[j])
                            else:
                                new_d.append(row[j] - new_val)
                    
                    new_d_counts = Counter(new_d)
                    old_d_counts = Counter(old_d)
                    
                    valid = True
                    for d, cnt in new_d_counts.items():
                        cur_cnt = cur_counter.get(d, 0)
                        rem_cnt = old_d_counts.get(d, 0)
                        net_cnt = cur_cnt - rem_cnt + cnt
                        if net_cnt > 1:
                            valid = False
                            break
                    
                    if valid:
                        for d in old_d:
                            cur_counter[d] -= 1
                            if cur_counter[d] == 0:
                                del cur_counter[d]
                        for d in new_d:
                            cur_counter[d] = cur_counter.get(d, 0) + 1
                        row[pos] = new_val
                        
                        new_scope = _scope(cur_rows)
                        new_conf = _conflicts_from_counter(cur_counter)
                        new_energy = _energy(cur_rows, cur_counter)
                        
                        if new_conf == 0 and new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in cur_rows]

    return best_rows

def construct(n_rows, k):
    rng = random.Random(12345)
    
    # Generate multiple random greedy starts
    best_start = None
    best_start_energy = float('inf')
    
    for i in range(30):
        # Vary randomization factor
        rf = 0.05 * (i + 1)
        rows = _greedy_all_randomized(n_rows, k, rng, rf)
        counter = _build_counter(rows)
        conf = _conflicts_from_counter(counter)
        scope = _scope(rows)
        energy = 1000 * conf + scope
        
        if energy < best_start_energy:
            best_start_energy = energy
            best_start = [list(r) for r in rows]
            
    if not best_start:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        best_start = rows

    # Optimize with Targeted Descent
    t0 = time.time()
    optimized_rows = _targeted_descent(best_start, k, _DEADLINE, rng)
    
    # Final check
    if not _valid(optimized_rows):
        if _valid(best_start):
            return best_start
        else:
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows

    return optimized_rows

def _valid(rows):
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True
