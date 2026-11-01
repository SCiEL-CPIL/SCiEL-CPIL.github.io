import time
import random
import math

# EVOLVE-BLOCK-START

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
    """Compute the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_init(n_rows, k):
    """Initialize rows using greedy: pick smallest valid x for each position."""
    rows = []
    used = set()
    for r in range(n_rows):
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

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Track per-row max for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = 100.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    steps = 0
    total_steps = 0
    
    # Precompute diffs for each row
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    while time.time() < deadline and no_improve < max_no_improve:
        steps += 1
        total_steps += 1
        
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates biased toward decreasing values
        candidates = []
        
        # Decreasing candidates (primary direction to reduce max)
        min_allowed = current_rows[ri][pi - 1] + 1
        # Strongly bias toward values that reduce the global max
        if old_val >= current_val:
            # If this is a max-contributing position, try larger decreases
            for d in range(1, 50):
                new_v = old_val - d
                if new_v < min_allowed:
                    break
                candidates.append(new_v)
        else:
            for d in range(1, 30):
                new_v = old_val - d
                if new_v < min_allowed:
                    break
                candidates.append(new_v)
        
        # Increasing candidates (for exploration, less frequent)
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        for d in range(1, 15):
            new_v = old_val + d
            if new_v > max_allowed:
                break
            candidates.append(new_v)
        
        # Random jump (large perturbation for escaping local minima)
        if random.random() < 0.05:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            # Basic validity check for this position
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Build candidate row
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Compute new diffs for this row incrementally
            # Remove old diffs involving position pi, add new ones
            old_diffs = row_diffs[ri]
            
            # Compute which diffs are affected by changing position pi
            affected_old = set()
            for i in range(len(cand_row)):
                if i != pi:
                    affected_old.add(abs(cand_row[pi] - cand_row[i]))
            
            # New diffs for position pi
            new_pi_diffs = set()
            for i in range(len(cand_row)):
                if i != pi:
                    new_pi_diffs.add(abs(cand_row[pi] - cand_row[i]))
            
            # New diffs for the row = (old_diffs - affected_old) | new_pi_diffs
            new_diffs = (old_diffs - affected_old) | new_pi_diffs
            
            # Check internal validity: all diffs must be distinct
            # Since we're using a set, if len matches expected count, they're distinct
            expected_diff_count = len(cand_row) * (len(cand_row) - 1) // 2
            if len(new_diffs) != expected_diff_count:
                continue
            
            # Check against other rows
            # Collision exists if any d in new_diffs is in all_used but NOT in old_diffs
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Move is valid - calculate new global max efficiently
            cand_row_max = max(cand_row)
            new_row_maxes = row_maxes[:]
            new_row_maxes[ri] = cand_row_max
            new_val_max = max(new_row_maxes)
            
            delta = new_val_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:  # Allow some neutral moves
                    accept = True
            else:
                # Accept worse moves with probability based on temperature
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs
                all_used |= new_diffs
                row_diffs[ri] = new_diffs
                current_rows[ri] = cand_row
                row_maxes[ri] = cand_row_max
                current_val = new_val_max
                
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
        
        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            row_maxes = [max(r) for r in current_rows]
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            T = 80.0
            no_improve = 0

    # 3. Final polishing phase with tighter moves
    polish_deadline = min(deadline, start_time + 115.0)
    polish_T = 5.0
    polish_T_min = 0.1
    polish_cool = 0.999
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    row_maxes = [max(r) for r in current_rows]
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    polish_no_improve = 0
    polish_max_no_improve = 10000
    
    while time.time() < polish_deadline and polish_no_improve < polish_max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try small decreases in polish phase
        candidates = []
        min_allowed = current_rows[ri][pi - 1] + 1
        for d in range(1, 20):
            new_v = old_val - d
            if new_v < min_allowed:
                break
            candidates.append(new_v)
        
        # Small increases for exploration
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
        for d in range(1, 8):
            new_v = old_val + d
            if new_v > max_allowed:
                break
            candidates.append(new_v)
        
        if not candidates:
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            old_diffs = row_diffs[ri]
            
            affected_old = set()
            for i in range(len(cand_row)):
                if i != pi:
                    affected_old.add(abs(cand_row[pi] - cand_row[i]))
            
            new_pi_diffs = set()
            for i in range(len(cand_row)):
                if i != pi:
                    new_pi_diffs.add(abs(cand_row[pi] - cand_row[i]))
            
            new_diffs = (old_diffs - affected_old) | new_pi_diffs
            
            expected_diff_count = len(cand_row) * (len(cand_row) - 1) // 2
            if len(new_diffs) != expected_diff_count:
                continue
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            cand_row_max = max(cand_row)
            new_row_maxes = row_maxes[:]
            new_row_maxes[ri] = cand_row_max
            new_val_max = max(new_row_maxes)
            
            delta = new_val_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
                if polish_T > polish_T_min:
                    prob = math.exp(-delta / max(polish_T, 0.05))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                row_diffs[ri] = new_diffs
                current_rows[ri] = cand_row
                row_maxes[ri] = cand_row_max
                current_val = new_val_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    polish_no_improve = 0
                else:
                    polish_no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            polish_no_improve += 1
        
        polish_T = max(polish_T_min, polish_T * polish_cool)
        
        if polish_no_improve > 5000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            row_maxes = [max(r) for r in current_rows]
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            polish_T = 3.0
            polish_no_improve = 0

    return best_rows

# EVOLVE-BLOCK-END
