"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _check_row_valid(row, k):
    """Check if a single row is strictly increasing and starts at 0."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows):
    """
    Try to move position pi in row ri to new_val.
    Returns (cand_row, new_diffs, new_max) if valid, else None.
    """
    # Check strict increase within row
    if new_val <= cur[ri][pi - 1]:
        return None
    if pi < k and new_val >= cur[ri][pi + 1]:
        return None
    
    # Build candidate row
    cand_row = cur[ri][:]
    cand_row[pi] = new_val
    
    # Check row validity
    if not _check_row_valid(cand_row, k):
        return None
    
    # Compute new diffs for this row
    new_diffs = _row_diffs(cand_row)
    
    # Check for conflicts with other rows' diffs
    other_diffs = cur_used - cur_row_diffs[ri]
    if new_diffs & other_diffs:
        return None
    
    # Check internal uniqueness in new_diffs (Sidon set property)
    diff_list = _row_diffs_list(cand_row)
    if len(diff_list) != len(set(diff_list)):
        return None
    
    # Compute new max
    # We need max of all rows except ri, then max with new cand_row
    # For efficiency, precompute max of other rows
    # But since n_rows is small (7), just compute directly
    cand_max = max(max(r) for r in cur if r is not cand_row) if n_rows > 1 else 0
    cand_max = max(cand_max, max(cand_row))
    
    return (cand_row, new_diffs, cand_max)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 105  # safety margin before 120s limit
    random.seed(42)
    
    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)
    best_used = set()
    for r in best:
        best_used |= _row_diffs(r)
    
    # Local search with incremental difference tracking
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set(best_used)
    
    # Track diffs per row for efficient updates
    cur_row_diffs = [_row_diffs(r) for r in cur]
    
    # Precompute max of all rows except each row for faster max calculation
    # Actually, let's just compute on the fly since n_rows is small
    
    T = 30.0
    T_min = 0.1
    cool = 0.9998
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 30000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        step += 1
        
        # Pick a random row and a random position to perturb
        # Bias towards rows that have the max entry
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # positions 1..k (0 is fixed)
        old_val = cur[ri][pi]
        
        # Generate candidate moves: focus on reducing max, with some diversification
        candidates = []
        
        # Try decreasing values (to reduce max)
        for d in range(1, 15):
            new_val = old_val - d
            if new_val <= cur[ri][pi - 1]:
                break
            candidates.append(new_val)
        
        # Try small increases for diversification
        for d in range(1, 8):
            new_val = old_val + d
            if pi < k and new_val >= cur[ri][pi + 1]:
                break
            candidates.append(new_val)
        
        # Occasionally try larger jumps
        if random.random() < 0.3:
            for d in range(15, 30):
                new_val = old_val - d
                if new_val > cur[ri][pi - 1]:
                    candidates.append(new_val)
                new_val = old_val + d
                if pi < k and new_val < cur[ri][pi + 1]:
                    candidates.append(new_val)
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            result = _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows)
            if result is None:
                continue
            
            cand_row, new_diffs, cand_max = result
            
            # Acceptance
            delta_e = cand_max - cur_val
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if random.random() < 0.4:
                    accept = True
            else:
                # Simulated annealing: accept worse with probability
                if T > T_min:
                    prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                cur_used -= cur_row_diffs[ri]
                cur_used |= new_diffs
                cur[ri] = cand_row
                cur_row_diffs[ri] = new_diffs
                cur_val = cand_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                moved = True
                break
        
        if not moved:
            no_improve_steps += 1
        
        # Cool down
        T = max(T_min, T * cool)
        
        # Periodic restart: if stuck for too long, reset to best and restart SA
        if no_improve_steps > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs(r) for r in cur]
            T = 30.0
            no_improve_steps = 0
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best
# EVOLVE-BLOCK-END
