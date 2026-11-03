"""Difference triangle set (7,5) solver with improved Simulated Annealing and local search."""
import time
import random
import math

# Deadline guard: return best result before 145s (5s safety margin under 150s kill)
_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    """Return the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_row(used, k, rng, max_val=None, diversity=0.05):
    """Build one row greedily, choosing the smallest available next mark.
    used: set of forbidden differences.
    max_val: optional upper bound for entries.
    diversity: probability of jumping ahead to create diversity.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        # Occasionally jump ahead to create diversity
        if rng.random() < diversity:
            x += rng.randint(1, 4)
            if max_val is not None and x > max_val:
                break
        
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_full(n_rows, k, rng, max_val=None, diversity=0.05):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val, diversity)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _sa_optimize(rows, k, rng, time_budget=60):
    """Simulated Annealing to minimize the scope (max entry) with incremental conflict tracking."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    current_rows = [list(r) for r in best_rows]
    
    # Initialize global diff counts and running conflict count
    diff_counts = {}
    current_conflicts = 0
    for row in current_rows:
        for d in _row_diffs(row):
            old_cnt = diff_counts.get(d, 0)
            diff_counts[d] = old_cnt + 1
            if old_cnt >= 1:
                current_conflicts += 1
    
    def get_scope():
        return _max_entry(current_rows)

    current_scope = get_scope()
    current_cost = current_scope + 1000 * current_conflicts
    
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temp = 8.0
    cooling = 0.99975
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Pick a random move
        move_type = rng.random()
        
        if move_type < 0.50:
            # Move: Rebuild a random row
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            
            # Remove old row diffs from counts, update conflicts incrementally
            old_diffs = _row_diffs(old_row)
            for d in old_diffs:
                old_cnt = diff_counts[d]
                if old_cnt > 1:
                    current_conflicts -= 1
                diff_counts[d] = old_cnt - 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Rebuild row with some randomness, trying to stay within current scope
            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 5)
            new_row = _greedy_row(used, k, rng, max_limit, diversity=0.08)
            
            if len(new_row) < k + 1:
                # Restore old row
                for d in old_diffs:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt >= 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                continue
            
            # Add new row diffs, update conflicts incrementally
            new_diffs = _row_diffs(new_row)
            for d in new_diffs:
                old_cnt = diff_counts.get(d, 0)
                if old_cnt >= 1:
                    current_conflicts += 1
                diff_counts[d] = old_cnt + 1
            
            current_rows[ri] = new_row
            
            # Evaluate cost
            new_scope = get_scope()
            new_cost = new_scope + 1000 * current_conflicts
            
            delta_cost = new_cost - current_cost
            
            # Acceptance criterion
            if delta_cost < 0 or (temp > 0.01 and rng.random() < math.exp(-delta_cost / temp)):
                current_scope = new_scope
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_conflicts = current_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject: Revert changes
                for d in new_diffs:
                    old_cnt = diff_counts[d]
                    if old_cnt > 1:
                        current_conflicts -= 1
                    diff_counts[d] = old_cnt - 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in old_diffs:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt >= 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                    
                current_rows[ri] = old_row

        elif move_type < 0.80:
            # Move: Shift an entry in a random row
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < 2:
                continue
            
            # Pick a random index > 0
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            # Try to reduce value to compress scope
            if rng.random() < 0.60:
                # Try to reduce
                reduction = rng.randint(1, 5)
                new_val = old_val - reduction
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            else:
                # Random small change
                delta = rng.randint(-5, 5)
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            
            if new_val == old_val:
                continue
                
            # Calculate diff changes
            old_diffs = _row_diffs(row)
            
            # Remove old diffs, update conflicts
            for d in old_diffs:
                old_cnt = diff_counts[d]
                if old_cnt > 1:
                    current_conflicts -= 1
                diff_counts[d] = old_cnt - 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Apply change
            row[idx] = new_val
            new_diffs = _row_diffs(row)
            
            # Add new diffs, update conflicts
            for d in new_diffs:
                old_cnt = diff_counts.get(d, 0)
                if old_cnt >= 1:
                    current_conflicts += 1
                diff_counts[d] = old_cnt + 1
            
            # Evaluate cost
            new_scope = get_scope()
            new_cost = new_scope + 1000 * current_conflicts
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (temp > 0.01 and rng.random() < math.exp(-delta_cost / temp)):
                current_scope = new_scope
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_conflicts = current_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject: Revert changes
                for d in new_diffs:
                    old_cnt = diff_counts[d]
                    if old_cnt > 1:
                        current_conflicts -= 1
                    diff_counts[d] = old_cnt - 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in old_diffs:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt >= 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                    
                row[idx] = old_val

        else:
            # Move: Shift a range of entries in a row
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < 3:
                continue
            
            # Pick start index (must be > 0 to keep row[0]=0)
            start_idx = rng.randint(1, len(row) - 1)
            
            # Pick end index
            end_idx = rng.randint(start_idx, len(row) - 1)
            
            # Pick a small delta
            delta = rng.randint(-3, 3)
            if delta == 0:
                continue
            
            # Check if the change maintains strictly increasing order
            # We need to check the boundaries
            if start_idx > 1:
                new_start = row[start_idx] + delta
                if new_start <= row[start_idx - 1]:
                    continue
            if end_idx < len(row) - 1:
                new_end = row[end_idx] + delta
                if new_end >= row[end_idx + 1]:
                    continue
            # Check that the shifted entries maintain order among themselves
            for i in range(start_idx + 1, end_idx + 1):
                if (row[i] + delta) <= (row[i-1] + delta):
                    continue
            # Check that first shifted entry is > previous
            if start_idx > 1 and (row[start_idx] + delta) <= row[start_idx - 1]:
                continue
            # Check that last shifted entry is < next
            if end_idx < len(row) - 1 and (row[end_idx] + delta) >= row[end_idx + 1]:
                continue
            
            # Also need to check that the gap between start_idx-1 and start_idx is maintained
            # and the gap between end_idx and end_idx+1 is maintained
            # Already done above.
            
            # Calculate diff changes
            old_diffs = _row_diffs(row)
            
            # Remove old diffs, update conflicts
            for d in old_diffs:
                old_cnt = diff_counts[d]
                if old_cnt > 1:
                    current_conflicts -= 1
                diff_counts[d] = old_cnt - 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Apply change
            for i in range(start_idx, end_idx + 1):
                row[i] += delta
            
            new_diffs = _row_diffs(row)
            
            # Add new diffs, update conflicts
            for d in new_diffs:
                old_cnt = diff_counts.get(d, 0)
                if old_cnt >= 1:
                    current_conflicts += 1
                diff_counts[d] = old_cnt + 1
            
            # Evaluate cost
            new_scope = get_scope()
            new_cost = new_scope + 1000 * current_conflicts
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (temp > 0.01 and rng.random() < math.exp(-delta_cost / temp)):
                current_scope = new_scope
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_conflicts = current_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject: Revert changes
                for d in new_diffs:
                    old_cnt = diff_counts[d]
                    if old_cnt > 1:
                        current_conflicts -= 1
                    diff_counts[d] = old_cnt - 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in old_diffs:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt >= 1:
                        current_conflicts += 1
                    diff_counts[d] = old_cnt + 1
                    
                for i in range(start_idx, end_idx + 1):
                    row[i] -= delta

        # Cool down
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    # Return the best valid solution found
    if _is_valid(best_rows, k):
        return best_rows
    else:
        if _is_valid(current_rows, k):
            return current_rows
        return best_rows

def _local_search_polish(rows, k, rng, time_budget=5):
    """Final high-precision polish: try multi-step entry reductions to reduce scope."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_cost = best_scope
    
    start_time = time.time()
    
    # Precompute diff sets for fast conflict checking
    all_diffs = set()
    for row in best_rows:
        for d in _row_diffs(row):
            all_diffs.add(d)
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 3:
            break
        
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            for idx in range(1, len(row)):
                old_val = row[idx]
                # Try reducing by 1, 2, 3, 4, 5
                for reduction in range(1, 6):
                    new_val = old_val - reduction
                    if new_val <= 0:
                        break
                    if idx > 1 and new_val <= row[idx - 1]:
                        break
                    if idx < len(row) - 1 and new_val >= row[idx + 1]:
                        break
                    
                    # Check if this change creates conflicts
                    old_diffs = _row_diffs(row)
                    row[idx] = new_val
                    new_diffs = _row_diffs(row)
                    
                    # Remove old diffs from all_diffs, add new ones, check for duplicates
                    temp_diffs = set(all_diffs)
                    for d in old_diffs:
                        temp_diffs.discard(d)
                    conflict = False
                    for d in new_diffs:
                        if d in temp_diffs:
                            conflict = True
                            break
                        temp_diffs.add(d)
                    
                    if not conflict:
                        # Accept the change
                        all_diffs = temp_diffs
                        new_scope = _max_entry(best_rows)
                        new_cost = new_scope
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_scope = new_scope
                            improved = True
                        break  # Stop trying larger reductions for this idx
                    else:
                        row[idx] = old_val
                        # Try larger reduction might work if smaller doesn't
                        continue
        
        if not improved:
            break
    
    if _is_valid(best_rows, k):
        return best_rows
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts to find a valid start
    best_rows = None
    best_score = float('inf')
    
    n_restarts = 300
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 20:
            break
        rng2 = random.Random(12345 + i * 7919)
        # Vary the diversity parameter
        div = 0.03 + 0.08 * (i % 10) / 10.0
        rows = _greedy_full(n_rows, k, rng2, diversity=div)
        if rows is None:
            continue
        if not _is_valid(rows, k):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy without randomization jumps
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _max_entry(best_rows)
    
    # Phase 2: Simulated Annealing (main optimization)
    if _elapsed() < _DEADLINE - 15:
        time_left = _DEADLINE - 15 - _elapsed()
        sa_time = min(95, time_left)
        improved_rows = _sa_optimize(best_rows, k, rng, time_budget=sa_time)
        
        if improved_rows and _is_valid(improved_rows, k):
            new_score = _max_entry(improved_rows)
            if new_score < best_score:
                best_rows = improved_rows
                best_score = new_score
    
    # Phase 3: Final high-precision polish
    if _elapsed() < _DEADLINE - 3:
        polish_time = min(10, _DEADLINE - 3 - _elapsed())
        if polish_time > 0:
            polished_rows = _local_search_polish(best_rows, k, rng, time_budget=polish_time)
            if polished_rows and _is_valid(polished_rows, k):
                new_score = _max_entry(polished_rows)
                if new_score < best_score:
                    best_rows = polished_rows
                    best_score = new_score
    
    # Final validation
    if not _is_valid(best_rows, k):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
