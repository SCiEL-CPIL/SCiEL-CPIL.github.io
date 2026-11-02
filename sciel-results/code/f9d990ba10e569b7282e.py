"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Compute the set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    """Compute list of all pairwise differences (to check for duplicates)."""
    lst = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            lst.append(row[j] - row[i])
    return lst

def _is_valid_row(row, forbidden, k):
    """
    Check if row is strictly increasing, starts at 0, has length k+1,
    has distinct internal diffs, and no collision with forbidden.
    """
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    
    diffs = _row_diffs_list(row)
    if len(diffs) != len(set(diffs)):
        return False
    
    diff_set = set(diffs)
    if diff_set & forbidden:
        return False
        
    return True

def _random_valid_row(forbidden_diffs, max_limit, k, rng, max_candidates=5):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    row = [0]
    used_in_row = set()
    
    for _ in range(k):
        candidates = []
        # We need to pick the next value x > row[-1]
        # such that {x - a for a in row} is disjoint from forbidden_diffs and used_in_row
        # and all diffs are distinct (which is guaranteed if x-a are distinct, i.e. a are distinct)
        
        # To find candidates, we scan upwards.
        # To keep it fast, we don't scan too far.
        start = row[-1] + 1
        # If max_limit is None, we might need to go higher, but let's bound the search window
        # to avoid infinite loops. If we can't find one, return None.
        
        # Heuristic: The gap shouldn't be too large.
        # Let's scan up to max_limit if provided, or start + 200 otherwise.
        # Actually, if max_limit is provided, we must stop there.
        
        end = max_limit if max_limit is not None else start + 300
        
        # Optimization: If the range is huge, this is slow.
        # But for k=6, n=7, max ~112, the range is small enough.
        
        for x in range(start, end):
            new_diffs = {x - a for a in row}
            if new_diffs & forbidden_diffs:
                continue
            if new_diffs & used_in_row:
                continue
            
            # Check if new_diffs are distinct from each other?
            # Since a are distinct, x-a are distinct. So yes.
            
            candidates.append(x)
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        # Pick a candidate
        # Weighted towards smaller values to keep the row compact?
        # Uniform is fine, but smaller is generally better for packing.
        # Let's use a simple bias: pick the smallest with high prob, or random.
        # Random gives more diversity for SA.
        chosen_x = rng.choice(candidates)
        
        row.append(chosen_x)
        used_in_row |= {chosen_x - a for a in row[:-1]}
        
    return row

def _greedy_init(n_rows, k, seed=42):
    """
    Build initial solution greedily.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _compute_all_used(rows):
    """Compute the union of all difference sets across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs_set(r)
    return used

def _global_max(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _max_except(rows, exclude_idx):
    """Return the maximum entry across all rows except the excluded one."""
    m = 0
    for i, r in enumerate(rows):
        if i != exclude_idx:
            if r[-1] > m:
                m = r[-1]
    return m

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Initialize with greedy
    rows = _greedy_init(n_rows, k, seed=42)
    best = [r[:] for r in rows]
    best_val = _global_max(best)
    best_used = _compute_all_used(best)
    
    # Current state
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    cur_used = set(best_used)

    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cool = 0.995
    
    no_improve_steps = 0
    max_no_improve = 50000
    rng = random.Random(12345)

    while time.time() < deadline and no_improve_steps < max_no_improve:
        # Pick a random row to replace
        ri = rng.randrange(n_rows)
        
        # Calculate forbidden diffs for this row
        forbidden = cur_used - cur_row_diffs[ri]
        
        # Try to find a new row with max <= cur_val (improvement or neutral)
        new_row = _random_valid_row(forbidden, cur_val, k, rng, max_candidates=8)
        
        if new_row is None:
            # Try with a slightly larger limit to allow moves that don't immediately improve
            new_row = _random_valid_row(forbidden, cur_val + 5, k, rng, max_candidates=8)
        
        if new_row is None:
            no_improve_steps += 1
            T = max(T_min, T * cool)
            continue
            
        # Efficient max calculation
        other_max = _max_except(cur, ri)
        new_max = max(other_max, new_row[-1])
        delta_e = new_max - cur_val

        accept = False
        if delta_e < 0:
            accept = True
        elif delta_e == 0:
            if rng.random() < 0.5:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            # FIX: Remove old diffs BEFORE updating cur_row_diffs
            old_diffs = cur_row_diffs[ri]
            cur_used -= old_diffs
            new_diffs = _row_diffs_set(new_row)
            cur_used |= new_diffs
            cur_row_diffs[ri] = new_diffs
            
            cur[ri] = new_row
            cur_val = new_max
            
            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in cur]
                best_used = set(cur_used)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        T = max(T_min, T * cool)

        # Periodic restart from best
        if no_improve_steps > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 5.0
            no_improve_steps = 0

    # Final polish: try to rebuild each row to minimize its max
    for _ in range(5):
        for ri in range(n_rows):
            forbidden = best_used - _row_diffs_set(best[ri])
            # Try to find a row with max < best_val
            new_row = _random_valid_row(forbidden, best_val - 1, k, rng, max_candidates=15)
            if new_row is not None:
                if _is_valid_row(new_row, forbidden, k):
                    other_max = _max_except(best, ri)
                    new_max = max(other_max, new_row[-1])
                    if new_max < best_val:
                        best[ri] = new_row
                        best_val = new_max
                        best_used = _compute_all_used(best)

    # Validate best
    if not _validate(best, n_rows, k):
        best = _greedy_init(n_rows, k, seed=42)

    return best

def _validate(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        # Check Sidon
        diff_list = _row_diffs_list(row)
        if len(diff_list) != len(set(diff_list)):
            return False
        for d in diff_list:
            if d in used:
                return False
            used.add(d)
    return True

# EVOLVE-BLOCK-END
