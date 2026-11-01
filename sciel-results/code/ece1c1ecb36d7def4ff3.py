"""Difference triangle set (7,5) - find 7 rows of 6 marks minimizing max entry.

Combined strategy:
1. Fast greedy initialization with randomization for diverse starting points.
2. Simulated Annealing with multi-restart: move individual marks to reduce global max,
   accepting worse moves probabilistically to escape local optima.
3. Targeted polishing phase focusing on rows/positions that contribute to the maximum.
4. Backtracking with forward checking for tight bound refinement.
"""
import time
import random
import copy
import math

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    return max(max(row) for row in rows)

def _compute_new_diffs(cand_row, pi):
    """
    Compute new diffs for a row after changing position pi.
    Returns (new_diffs_set, is_valid_internal).
    """
    n = len(cand_row)
    new_diffs = set()
    
    # Add diffs NOT involving pi (unchanged)
    for i in range(n):
        if i == pi:
            continue
        for j in range(i + 1, n):
            if j == pi:
                continue
            new_diffs.add(cand_row[j] - cand_row[i])
    
    # Add diffs involving pi
    for i in range(n):
        if i == pi:
            continue
        d = abs(cand_row[pi] - cand_row[i])
        if d in new_diffs:
            return None, False
        new_diffs.add(d)
    
    return new_diffs, True

def _greedy_all(n_rows, k, rng=None, max_x=500):
    """Build all rows greedily, each after the previous."""
    if rng is None:
        rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            base = row[-1]
            found = False
            # Try values in increasing order, with some randomization
            candidates = list(range(base + 1, min(base + 30, max_x) + 1))
            rng.shuffle(candidates)
            # Also try strictly increasing order first for determinism in early rows
            for x in sorted(candidates):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                # Fallback: try all values up to max_x
                for x in range(base + 1, max_x + 1):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
            if not found:
                return None
        rows.append(row)
    return rows

def _run_sa_chain(rows, n_rows, k, time_budget, T_start=100.0, T_min=0.5, cool=0.99995, rng=None):
    """
    Run a single SA chain for the given time budget.
    Returns (best_rows, best_val).
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = T_start
    
    no_improve = 0
    max_no_improve = 15000
    
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        candidates = []
        
        # Focus on decreasing the value (to reduce max)
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(80, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Also allow some increases to escape local optima
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
        max_increase = min(30, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        # Random jump
        if rng.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
            if low <= high:
                candidates.append(rng.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        rng.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            old_diffs = row_diffs[ri]
            new_diffs, valid_internal = _compute_new_diffs(cand_row, pi)
            
            if not valid_internal:
                continue
            
            # Check conflicts with other rows
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.3:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > 8000:
            # Restart from best
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = T_start
            no_improve = 0
    
    return best_rows, best_val

def _polish_targeted(rows, n_rows, k, time_budget, rng=None):
    """
    Targeted polishing: focus on reducing the max entry by modifying
    positions in rows that achieve the current maximum.
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Focus on rows that have the max value
        max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        if max_row_indices and rng.random() < 0.7:
            ri = rng.choice(max_row_indices)
        else:
            ri = rng.randrange(n_rows)
        
        # Focus on the last position (most likely to be the max)
        if ri in max_row_indices and rng.random() < 0.5:
            pi = k
        else:
            pi = rng.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(100, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try decreasing values
        candidates = list(range(old_val - 1, min_allowed - 1, -1))
        rng.shuffle(candidates)
        
        moved = False
        for new_val in candidates[:30]:
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            old_diffs = row_diffs[ri]
            new_diffs, valid_internal = _compute_new_diffs(cand_row, pi)
            
            if not valid_internal:
                continue
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            cand_row_max = max(cand_row)
            new_global_max = max(row_maxes[:ri] + [cand_row_max] + row_maxes[ri+1:])
            
            if new_global_max < current_val:
                # Accept improvement
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
            elif new_global_max == current_val:
                # Neutral move, accept with some probability
                if rng.random() < 0.2:
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    no_improve += 1
                    moved = True
                    break
        
        if not moved:
            no_improve += 1
    
    return best_rows, best_val

def _backtrack_with_forward_checking(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking with forward checking.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = []
    
    def solve(row_idx, pos, current_row):
        """Recursively build row row_idx, position pos."""
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            # Row complete
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 1, [0])
        
        if pos == 0:
            return solve(row_idx, 1, [0])
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        # Forward check: can we still fill remaining positions?
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return False
        
        # Generate candidates
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            
            # Forward check: after placing this value, can the next position be filled?
            if pos + 1 < k + 1:
                next_lo = val + 1
                next_hi = max_entry
                next_remaining = k + 1 - (pos + 1)
                if next_hi - next_lo + 1 < next_remaining:
                    continue
                # Check if at least one valid next value exists
                has_next = False
                for next_val in range(next_lo, min(next_lo + 50, next_hi + 1)):
                    next_new_diffs = {next_val - a for a in current_row + [val]}
                    if len(next_new_diffs) == len(current_row) + 1:
                        if not (next_new_diffs & (used_diffs | new_diffs)):
                            has_next = True
                            break
                if not has_next:
                    continue
            
            candidates.append(val)
        
        if not candidates:
            return False
        
        # Order candidates: prefer smaller values
        candidates.sort()
        
        # Add some randomness
        if rng.random() < 0.3 and len(candidates) > 1:
            swap_idx = rng.randrange(len(candidates))
            candidates[0], candidates[swap_idx] = candidates[swap_idx], candidates[0]
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            new_diffs = {val - a for a in current_row}
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if solve(row_idx, pos + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    if solve(0, 0, []):
        return rows
    return None

def construct(n_rows, k):
    deadline = time.time() + 140  # Safety margin from 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Get initial solution with greedy
    best_rows = _greedy_all(n_rows, k, rng)
    if best_rows is None or not _validate(best_rows):
        # Fallback: very simple construction
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1] + 1, 1000):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
    
    best_score = _max_entry(best_rows)
    
    # Phase 2: Multi-restart Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Run multiple SA chains with different random seeds
        sa_budget = min(remaining_time - 20, 60)
        n_restarts = max(2, int(sa_budget / 15))
        
        for restart in range(n_restarts):
            if time.time() > deadline - 25:
                break
            
            # Create a perturbed starting point
            start_rows = [r[:] for r in best_rows]
            if restart > 0:
                # Perturb some values
                for ri in range(n_rows):
                    for pi in range(1, k + 1):
                        if rng.random() < 0.3:
                            old = start_rows[ri][pi]
                            lo = start_rows[ri][pi-1] + 1
                            hi = (start_rows[ri][pi+1] - 1) if pi < k else old + 20
                            if lo <= hi:
                                new_val = rng.randint(lo, min(hi, old + 10))
                                cand_row = start_rows[ri][:]
                                cand_row[pi] = new_val
                                nd, valid = _compute_new_diffs(cand_row, pi)
                                if valid:
                                    all_u = _compute_all_used(start_rows)
                                    conflict = False
                                    for d in nd:
                                        if d in all_u and d not in _row_diffs_set(start_rows[ri]):
                                            conflict = True
                                            break
                                    if not conflict:
                                        start_rows[ri] = cand_row
            
            row_budget = max(5, sa_budget / n_restarts)
            rows, val = _run_sa_chain(start_rows, n_rows, k, row_budget, rng=rng)
            if rows is not None and _validate(rows):
                if val < best_score:
                    best_score = val
                    best_rows = copy.deepcopy(rows)
    
    # Phase 3: Targeted polishing
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        polish_budget = min(remaining_time - 10, 40)
        rows, val = _polish_targeted(best_rows, n_rows, k, polish_budget, rng=rng)
        if rows is not None and _validate(rows):
            if val < best_score:
                best_score = val
                best_rows = copy.deepcopy(rows)
    
    # Phase 4: Backtracking for tight bound (if time remains)
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        bt_budget = min(remaining_time - 10, 30)
        for bound in range(best_score, max(100, best_score - 20), -1):
            if time.time() > deadline - 10:
                break
            rows = _backtrack_with_forward_checking(n_rows, k, bound, rng, deadline - 5)
            if rows is not None and _validate(rows):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_rows = copy.deepcopy(rows)
                    break
    
    # Final validation
    if not _validate(best_rows):
        fresh_rows = _greedy_all(n_rows, k, random.Random(123))
        if fresh_rows is not None and _validate(fresh_rows):
            best_rows = fresh_rows
        else:
            # Emergency fallback
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    for x in range(row[-1] + 1, 10000):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            break
                best_rows.append(row)
    
    return best_rows
