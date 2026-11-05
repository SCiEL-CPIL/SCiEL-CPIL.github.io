"""Difference triangle set (7,5): minimize the largest entry.
Strategy: multi-phase search combining:
1. Greedy baseline
2. Randomized backtracking with constraint propagation and forward checking
3. Local search with difference-level moves
4. Targeted polish to reduce scope by 1
"""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, seed=42):
    """Greedy construction with constraint propagation."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    break
                x += 1
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _backtrack_row(k, used, max_val, rng, deadline=None):
    """Try to build a single row of k+1 elements starting at 0, with all differences
    not in 'used', and all elements <= max_val. Returns row or None."""
    if deadline is not None and time.time() > deadline:
        return None
    
    row = [0]
    
    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        # Forward checking: ensure we can fit remaining elements
        remaining = k + 1 - idx
        # We need at least 'remaining' more elements, each at least 1 apart
        # So we need high - low + 1 >= remaining
        if high - low + 1 < remaining:
            return False
        
        # Try values in randomized order, but bias towards smaller values
        # to leave room for future elements
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
        # Sort by a heuristic: prefer values that minimize the maximum gap
        # Actually, let's just use random order but with a bias
        # Try smaller values first to be safe
        candidates.sort(key=lambda v: rng.random() + v * 0.1)
        
        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            row.append(val)
            used.update(new_diffs)
            if backtrack(idx + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_with_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using backtracking, each with elements <= max_val."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = _backtrack_row(k, used, max_val, rng, deadline=deadline)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        # Try to reduce each element that is at the current maximum
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope:
                    continue
                
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                # Try values from high down to low (prefer larger reductions)
                candidates = list(range(high, low - 1, -1))
                rng.shuffle(candidates)
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, len(row)):
                        old_diffs.add(row[i] - old_val)
                    
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, len(row)):
                        new_diffs.add(row[i] - new_val)
                    
                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue
                    
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
    
    return rows


def _polish_pass(rows, k, used, deadline, rng):
    """Final high-precision polish: try to reduce max by 1, then by 2, etc."""
    rows = [list(r) for r in rows]
    used = set(used)
    
    current_scope = _scope(rows)
    target = current_scope - 1
    
    while target >= 100 and time.time() < deadline:
        success = False
        
        # Multiple restarts with different random seeds
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            trial_rows = [list(r) for r in rows]
            trial_used = set(used)
            
            # For each row, try to reduce elements that exceed target
            row_changed = False
            for ri in range(len(trial_rows)):
                row = trial_rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    if old_val <= target:
                        continue
                    
                    low = row[pos - 1] + 1
                    high = min(target, row[pos + 1] - 1 if pos + 1 < len(row) else target)
                    
                    if low > high:
                        continue
                    
                    # Try all values from high down to low
                    for new_val in range(high, low - 1, -1):
                        if new_val >= old_val:
                            continue
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        
                        conflict = new_diffs & (trial_used - old_diffs)
                        if conflict:
                            continue
                        
                        row[pos] = new_val
                        trial_used -= old_diffs
                        trial_used |= new_diffs
                        row_changed = True
                        break
            
            if row_changed and _scope(trial_rows) <= target:
                if _valid(trial_rows):
                    rows = trial_rows
                    used = trial_used
                    success = True
                    break
        
        if success:
            current_scope = _scope(rows)
            target = current_scope - 1
        else:
            break
    
    return rows, used


def _simulated_annealing(rows, k, deadline, rng):
    """Simulated annealing to escape local minima."""
    rows = [list(r) for r in rows]
    used = _all_diffs(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    T = 1000.0
    cooling = 0.995
    min_T = 1.0
    
    while T > min_T and time.time() < deadline:
        # Pick a random row and position
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 10
        
        if low >= high:
            T *= cooling
            continue
        
        new_val = rng.randint(low, high)
        if new_val == old_val:
            T *= cooling
            continue
        
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for i in range(pos + 1, len(row)):
            old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for i in range(pos + 1, len(row)):
            new_diffs.add(row[i] - new_val)
        
        conflict = new_diffs & (used - old_diffs)
        
        if not conflict:
            # Accept the move
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        else:
            # Reject
            pass
        
        T *= cooling
    
    return best_rows, _scope(best_rows)


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Randomized backtracking with decreasing bounds
    if time.time() < deadline - 40.0:
        # Try to construct with tighter bounds
        for max_val in range(best_scope, max(105, best_scope - 20), -1):
            if time.time() > deadline - 40.0:
                break
            
            for attempt in range(10):
                if time.time() > deadline - 40.0:
                    break
                
                bt_rows = _construct_with_backtracking(n_rows, k, max_val, rng, deadline - 40.0)
                if bt_rows is not None:
                    scope = _scope(bt_rows)
                    if scope < best_scope:
                        best_rows = bt_rows
                        best_scope = scope
                        break
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 15.0:
        improved = _local_search_improve(best_rows, k, deadline - 15.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_rows, sa_scope = _simulated_annealing(best_rows, k, deadline - 5.0, rng)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 5: Final polish pass
    if time.time() < deadline - 1.0:
        used = _all_diffs(best_rows)
        polished, used = _polish_pass(best_rows, k, used, deadline - 1.0, rng)
        new_scope = _scope(polished)
        if new_scope < best_scope:
            best_rows = polished
            best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
