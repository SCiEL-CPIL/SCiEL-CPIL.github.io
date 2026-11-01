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
        # Find smallest valid x
        valid_x = None
        # We can look ahead a bit
        # Simple approach: try x, x+1, ... until valid
        # Randomized: pick from the first few valids
        candidates = []
        start_x = x + 1
        # Limit search window for speed, but ensure we find one
        # In worst case, we might need to go far
        # But greedy usually finds it quickly
        for trial_x in range(start_x, start_x + 1000):
            new = {trial_x - a for a in row}
            if not (new & used):
                candidates.append(trial_x)
                if len(candidates) > 5:
                    break
        if not candidates:
            # Fallback to linear search if window was too small
            trial_x = start_x
            while True:
                new = {trial_x - a for a in row}
                if not (new & used):
                    candidates.append(trial_x)
                    break
                trial_x += 1
        
        # Pick randomly from candidates
        if rand_factor > 0 and len(candidates) > 1:
            x = rng.choice(candidates[:3]) # Pick from top 3
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

def _sa_optimize(rows, k, deadline, rng):
    t0 = time.time()
    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    cur_energy = _energy(cur_rows, cur_counter)
    
    best_rows = [list(r) for r in cur_rows]
    best_energy = cur_energy
    best_conf = _conflicts_from_counter(cur_counter)
    best_scope = _scope(cur_rows)
    
    # Initial temperature
    T = 10.0
    cooling = 0.9999
    
    n_rows = len(cur_rows)
    row_len = k + 1
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            T = 0.01
            
        # Pick a move
        r_idx = rng.randrange(n_rows)
        row = cur_rows[r_idx]
        pos = 1 + rng.randrange(row_len - 1) # 1 to k
        
        old_val = row[pos]
        
        # Determine valid range for new value
        lo = row[pos - 1] + 1
        hi = (row[pos + 1] - 1) if pos + 1 < row_len else old_val + 100 # Allow growth if needed
        
        # Propose new value
        # Bias towards smaller values to reduce scope, but allow some exploration
        # If we are feasible, we want to shrink. If infeasible, we want to fix conflicts.
        # Simple strategy: random uniform in [lo, hi] but biased towards lo if hi is large?
        # Let's just pick a random integer in [lo, hi]
        # To speed up convergence, we can restrict the step size
        # Step size proportional to T?
        max_step = 20
        if pos == row_len - 1:
             # Last element, can be larger
             max_step = 50
             
        new_val = old_val + rng.randint(-max_step, max_step)
        if new_val < lo:
            new_val = lo
        if new_val > hi:
            new_val = hi
            
        if new_val == old_val:
            continue
            
        # Calculate delta energy
        # This is expensive if done naively.
        # We need to update the counter.
        
        # Old diffs involving pos
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
                    
        # Update counter
        for d in old_d:
            cur_counter[d] -= 1
            if cur_counter[d] == 0:
                del cur_counter[d]
                
        for d in new_d:
            cur_counter[d] = cur_counter.get(d, 0) + 1
            
        new_conf = _conflicts_from_counter(cur_counter)
        new_scope = _scope(cur_rows) # Note: cur_rows not updated yet, so scope might be same or change if pos is last
        # Actually, scope is max of last elements. If pos < last, scope unchanged. If pos == last, scope might change.
        if pos == row_len - 1:
             new_scope = max(r[-1] for r in cur_rows) # Need to check if this row's last element changed
             # Wait, cur_rows[r_idx][pos] is still old_val.
             # So we must calculate new_scope correctly.
             temp_last = new_val
             other_scopes = [cur_rows[i][-1] for i in range(n_rows) if i != r_idx]
             new_scope = max([temp_last] + other_scopes)
        else:
             new_scope = _scope(cur_rows)
             
        new_energy = 1000 * new_conf + new_scope
        
        delta_E = new_energy - cur_energy
        
        # Metropolis
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            # Accept
            row[pos] = new_val
            cur_energy = new_energy
            
            # Update best
            if new_conf < best_conf or (new_conf == best_conf and new_scope < best_scope):
                best_conf = new_conf
                best_scope = new_scope
                best_rows = [list(r) for r in cur_rows]
                
        else:
            # Reject, revert counter
            for d in new_d:
                cur_counter[d] -= 1
                if cur_counter[d] == 0:
                    del cur_counter[d]
            for d in old_d:
                cur_counter[d] = cur_counter.get(d, 0) + 1
            # cur_rows[r_idx][pos] is unchanged, so no need to revert row
            
    return best_rows

def construct(n_rows, k):
    rng = random.Random(12345)
    
    # Generate multiple random greedy starts
    best_start = None
    best_start_energy = float('inf')
    
    for i in range(20):
        # Vary randomization factor
        rf = 0.1 * (i + 1)
        rows = _greedy_all_randomized(n_rows, k, rng, rf)
        counter = _build_counter(rows)
        conf = _conflicts_from_counter(counter)
        scope = _scope(rows)
        energy = 1000 * conf + scope
        
        if energy < best_start_energy:
            best_start_energy = energy
            best_start = [list(r) for r in rows]
            
    if not best_start:
        # Fallback to standard greedy
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

    # Optimize with SA
    t0 = time.time()
    optimized_rows = _sa_optimize(best_start, k, _DEADLINE, rng)
    
    # Final check
    if not _valid(optimized_rows):
        # Fallback to best_start if SA broke it (shouldn't happen if we track best correctly, but just in case)
        # Actually, best_rows in SA is always valid if best_conf==0.
        # If best_conf > 0, it's invalid.
        # The task requires valid rows.
        # If we can't find valid rows, we return the best valid one found?
        # The seed program returns greedy if invalid.
        # Greedy is always valid.
        # So if SA fails to find valid, we should return a valid greedy one.
        
        # Let's check if best_start was valid
        if _valid(best_start):
            return best_start
        else:
            # Generate a fresh greedy
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
