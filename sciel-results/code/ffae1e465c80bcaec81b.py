"""Difference triangle set (4,7) — simulated annealing with conflict-based repair."""
import time
import random
import sys

# EVOLVE-BLOCK-START


def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _max_entry(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _count_conflicts(rows):
    """Count how many duplicate differences exist across all rows."""
    from collections import Counter
    seen = Counter()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                seen[d] += 1
    conflicts = 0
    for d, cnt in seen.items():
        if cnt > 1:
            conflicts += cnt - 1
    return conflicts


def _greedy_build(n_rows, k, rng, bias=1.0):
    """Greedy build with optional randomness."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if bias >= 1.0 or rng.random() < bias:
                    row.append(x)
                    used |= new
                else:
                    if rng.random() > 0.3:
                        row.append(x)
                        used |= new
        rows.append(row)
    return rows


def _sa_optimize(rows, deadline, rng):
    """
    Simulated annealing on the full set of rows.
    Moves: pick a random row and a random position, then try to compact
    that element or the whole row by shifting elements down.
    Accepts invalid states with probability based on conflict count and temperature.
    """
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    # Current state
    current = [r[:] for r in rows]
    current_conflicts = _count_conflicts(current)
    current_scope = _max_entry(current)
    
    # Best state
    best = [r[:] for r in current]
    best_scope = current_scope
    best_conflicts = current_conflicts
    
    # Temperature schedule
    t_start = time.time()
    total_time = deadline - t_start
    t0 = 100.0  # initial temperature (in terms of conflict units)
    t_end = 0.01
    
    def temperature(elapsed_frac):
        # Exponential cooling
        return t0 * (t_end / t0) ** elapsed_frac
    
    steps = 0
    while time.time() < deadline:
        elapsed_frac = (time.time() - t_start) / max(total_time, 1e-9)
        if elapsed_frac >= 1.0:
            break
        T = temperature(elapsed_frac)
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Compact a single element: try to reduce row[r][p] to a smaller value
            r_idx = rng.randrange(n_rows)
            if len(current[r_idx]) <= 1:
                continue
            p = rng.randrange(1, len(current[r_idx]))
            old_val = current[r_idx][p]
            
            # Try values from old_val-1 down to current[r_idx][p-1]+1
            found = False
            for new_val in range(old_val - 1, current[r_idx][p - 1], -1):
                # Check if this creates a conflict within the row
                row = current[r_idx]
                row[p] = new_val
                # Check intra-row distinctness
                valid_intra = True
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if row[j] == row[i]:
                            valid_intra = False
                            break
                    if not valid_intra:
                        break
                if not valid_intra:
                    row[p] = old_val
                    continue
                # Check if this creates inter-row conflicts
                # Only check if new_val - row[i] or row[j] - new_val conflicts with other rows
                new_diffs = set()
                for i in range(len(row)):
                    if i != p:
                        d = abs(row[p] - row[i])
                        new_diffs.add(d)
                
                # Check against other rows
                conflict_count = 0
                for other_idx in range(n_rows):
                    if other_idx == r_idx:
                        continue
                    other_row = current[other_idx]
                    for i in range(len(other_row)):
                        for j in range(i + 1, len(other_row)):
                            d = other_row[j] - other_row[i]
                            if d in new_diffs:
                                conflict_count += 1
                
                if conflict_count == 0:
                    found = True
                    break
                else:
                    row[p] = old_val
                    # Accept with probability based on conflicts
                    if rng.random() < T / max(conflict_count, 1):
                        found = True
                        break
            
            if found:
                new_scope = _max_entry(current)
                new_conflicts = _count_conflicts(current)
                # Accept if better or with SA probability
                delta_scope = new_scope - current_scope
                delta_conf = new_conflicts - current_conflicts
                if delta_scope < 0 or delta_conf < 0:
                    current = [r[:] for r in current]
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                    if new_scope < best_scope and new_conflicts == 0:
                        best = [r[:] for r in current]
                        best_scope = new_scope
                        best_conflicts = 0
                elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                    current = [r[:] for r in current]
                    current_scope = new_scope
                    current_conflicts = new_conflicts
        
        elif move_type < 0.8:
            # Rebuild a single row greedily with randomness
            r_idx = rng.randrange(n_rows)
            old_row = current[r_idx][:]
            
            # Collect used diffs from other rows
            used_others = set()
            for i in range(n_rows):
                if i != r_idx:
                    used_others |= _all_diffs(current[i])
            
            # Greedy rebuild with randomness
            new_row = [0]
            x = 0
            max_x = old_row[-1] + 5  # Allow some expansion
            while len(new_row) < len(old_row) and x < max_x:
                x += 1
                new = {x - a for a in new_row}
                if not new & used_others:
                    if rng.random() < 0.8:
                        new_row.append(x)
                    else:
                        if rng.random() < 0.4:
                            new_row.append(x)
            
            if len(new_row) == len(old_row):
                # Check validity
                trial = current[:]
                trial[r_idx] = new_row
                new_conflicts = _count_conflicts(trial)
                new_scope = _max_entry(trial)
                
                delta_scope = new_scope - current_scope
                delta_conf = new_conflicts - current_conflicts
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best = trial
                    best_scope = new_scope
                    best_conflicts = 0
                    current = trial
                    current_scope = new_scope
                    current_conflicts = 0
                elif delta_scope < 0 or delta_conf < 0:
                    current = trial
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                    current = trial
                    current_scope = new_scope
                    current_conflicts = new_conflicts
        
        else:
            # Swap elements between two rows (if same length)
            r1 = rng.randrange(n_rows)
            r2 = rng.randrange(n_rows)
            if r1 == r2:
                continue
            if len(current[r1]) != len(current[r2]):
                continue
            # Try swapping the last element
            old1 = current[r1][-1]
            old2 = current[r2][-1]
            if old1 == old2:
                continue
            current[r1][-1] = old2
            current[r2][-1] = old1
            
            new_conflicts = _count_conflicts(current)
            new_scope = _max_entry(current)
            delta_scope = new_scope - current_scope
            delta_conf = new_conflicts - current_conflicts
            
            if new_conflicts == 0 and new_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = new_scope
                best_conflicts = 0
                current_scope = new_scope
                current_conflicts = new_conflicts
            elif delta_scope < 0 or delta_conf < 0:
                current_scope = new_scope
                current_conflicts = new_conflicts
            elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                current_scope = new_scope
                current_conflicts = new_conflicts
            else:
                # Reject
                current[r1][-1] = old1
                current[r2][-1] = old2
        
        steps += 1
        # Periodically check best
        if steps % 1000 == 0 and current_conflicts == 0:
            if current_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = current_scope
                best_conflicts = 0
    
    # Return best valid solution
    if best_conflicts == 0:
        return best
    # Fallback: return current if it's valid
    if current_conflicts == 0:
        return current
    # Otherwise return best even if invalid (shouldn't happen often)
    return best


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Start with multiple greedy baselines
    best = None
    best_score = float('inf')
    
    for seed in range(30):
        if time.time() > deadline - 10.0:
            break
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=rng.choice([1.0, 0.95, 0.9, 0.85]))
        if _is_valid(trial):
            sc = _max_entry(trial)
            if sc < best_score:
                best_score = sc
                best = trial
    
    if best is None:
        # Fallback: use first valid greedy
        best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
        best_score = _max_entry(best)
    
    # Run simulated annealing
    best = _sa_optimize(best, deadline, rng)
    
    return best

# EVOLVE-BLOCK-END
