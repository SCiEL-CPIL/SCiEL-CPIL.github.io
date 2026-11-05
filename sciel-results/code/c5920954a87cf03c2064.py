"""Difference triangle set (7,5): minimize the largest entry.
CSP approach with degree-based heuristic: select row elements by minimizing
the "degree" of future constraint violations (fewest conflicts with remaining
difference slots), combined with backtracking and local search refinement."""
import time
import random
import math
import sys
from collections import defaultdict

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


def _degree_heuristic(row, pos, val, used, k, max_val):
    """Compute the 'degree' of future constraint violations for placing val at position pos.
    Lower degree means fewer conflicts with remaining difference slots."""
    new_diffs = {val - a for a in row}
    
    # Count how many of these new diffs are 'tight' (small values that will
    # likely conflict with future rows)
    tight_count = sum(1 for d in new_diffs if d <= 15)
    
    # Count how many potential future values would be blocked
    # For each existing element a in row, the diff val-a blocks certain future placements
    blocked_count = 0
    for a in row:
        d = val - a
        # This diff d means we can't place any future element at a + d (already used)
        # and also constrains other placements
        blocked_count += 1
    
    # Prefer values that spread out (larger gaps) to leave room for future rows
    if len(row) > 1:
        gap = val - row[-1]
        spread_bonus = -gap  # larger gap = lower score (better)
    else:
        spread_bonus = 0
    
    return tight_count * 2 + blocked_count + spread_bonus


def _build_row_csp(k, used, max_val, rng, deadline, row_order=None):
    """Build a single row using CSP with degree-based heuristic: at each position,
    pick the value that minimizes future constraint violations."""
    if time.time() > deadline:
        return None
    
    row = [0]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        # Generate candidate values
        if high - low > 50:
            candidates = set()
            # Near-low values
            for v in range(low, min(low + 20, high + 1)):
                candidates.add(v)
            # Spread values
            for _ in range(30):
                candidates.add(rng.randint(low, high))
            # Near-high values
            for v in range(max(low, high - 15), high + 1):
                candidates.add(v)
            candidates = list(candidates)
        else:
            candidates = list(range(low, high + 1))
        
        # Score candidates by degree heuristic
        scored = []
        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            score = _degree_heuristic(row, pos, val, used, k, max_val)
            scored.append((score, val, new_diffs))
        
        if not scored:
            return False
        
        # Try candidates in order of lowest degree (fewest future conflicts)
        scored.sort(key=lambda x: x[0])
        
        for score, val, new_diffs in scored:
            row.append(val)
            used.update(new_diffs)
            if backtrack(pos + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_simultaneous_csp(n_rows, k, max_val, rng, deadline):
    """Construct all rows simultaneously using CSP with degree heuristic.
    Rows are built in an order that balances difference-space usage."""
    used = set()
    rows = []
    
    # Build rows in a shuffled order to break symmetry
    order = list(range(n_rows))
    rng.shuffle(order)
    
    for r_idx in order:
        if time.time() > deadline:
            return None
        row = _build_row_csp(k, used, max_val, rng, deadline)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    # Restore original order
    rows_by_idx = [None] * n_rows
    for i, r_idx in enumerate(order):
        rows_by_idx[r_idx] = rows[i]
    
    return rows_by_idx


def _reconstruct_row_csp(rows, k, row_idx, max_val, rng, deadline):
    """Rebuild a single row from scratch using CSP, given the used differences
    of all other rows."""
    other_used = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        other_used |= _row_diffs(row)
    
    new_row = _build_row_csp(k, other_used, max_val, rng, deadline)
    if new_row is None:
        return None
    return new_row


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(50):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _csp_construct(n_rows, k, max_val, rng, deadline, n_attempts=5):
    """Try CSP construction multiple times with different random seeds and row orderings."""
    best_rows = None
    best_scope = float('inf')
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        rng_attempt = random.Random(rng.randint(0, 10**9))
        rows = _construct_simultaneous_csp(n_rows, k, max_val, rng_attempt, deadline)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    return best_rows, best_scope


def _simulated_annealing(rows, k, deadline, rng):
    """Simulated annealing with two move types:
    1. Single-element move (fast, local)
    2. Full row reconstruction (slow, can escape local minima)
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    current_scope = _scope(rows)
    T = 30.0
    cooling = 0.9997
    min_T = 0.05
    
    # Precompute per-row diff sets for fast update
    row_diffs = [_row_diffs(r) for r in rows]
    
    while time.time() < deadline and T > min_T:
        ri = rng.randint(0, len(rows) - 1)
        
        # 10% of the time, try full row reconstruction
        if rng.random() < 0.10:
            max_val = best_scope  # try to rebuild within current best scope
            new_row = _reconstruct_row_csp(rows, k, ri, max_val, rng, deadline)
            if new_row is not None:
                # Check it's valid (no conflicts with other rows)
                new_rd = _row_diffs(new_row)
                if not (new_rd & (used - row_diffs[ri])):
                    new_scope = max(max(r) for r in rows[:ri] + [new_row] + rows[ri+1:])
                    delta = new_scope - current_scope
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        rows[ri] = new_row
                        row_diffs[ri] = new_rd
                        used -= row_diffs[ri]
                        used |= new_rd
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in rows]
            T *= cooling
            continue
        
        # Single-element move
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 20
        
        if low >= high:
            T *= cooling
            continue
        
        new_val = rng.randint(low, high)
        if new_val == old_val:
            T *= cooling
            continue
        
        # Compute diff changes
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
            T *= cooling
            continue
        
        new_scope = max(max(row[:pos] + [new_val] + row[pos + 1:]),
                        max(max(r) for r in rows[:ri] + rows[ri+1:]))
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            row_diffs[ri] = _row_diffs(row)
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    return best_rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements
    and occasionally rebuilding entire rows using CSP."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    row_diffs = [_row_diffs(r) for r in rows]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        # Try to reduce elements at or near the current maximum
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope - 2:
                    continue
                
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                n_candidates = min(20, high - low + 1)
                candidates = rng.sample(range(low, high + 1), n_candidates)
                candidates.sort()
                
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
                    row_diffs[ri] = _row_diffs(row)
                    improved = True
                    break
        
        # Occasionally try full row reconstruction for the row with the max entry
        if time.time() < deadline and rng.random() < 0.3:
            max_row_idx = max(range(len(rows)), key=lambda i: max(rows[i]))
            new_row = _reconstruct_row_csp(rows, k, max_row_idx, current_scope - 1, rng, deadline)
            if new_row is not None:
                new_rd = _row_diffs(new_row)
                if not (new_rd & (used - row_diffs[max_row_idx])):
                    rows[max_row_idx] = new_row
                    used -= row_diffs[max_row_idx]
                    used |= new_rd
                    row_diffs[max_row_idx] = new_rd
                    improved = True
    
    return rows


def _difference_swap_mutation(rows, k, used, num_swaps=None, rng=None):
    """Perform difference-swap moves to reduce the max entry."""
    if rng is None:
        rng = random.Random()
    if num_swaps is None:
        num_swaps = rng.randint(1, 3)
    
    rows = [list(r) for r in rows]
    used = set(used)
    n_rows = len(rows)
    
    for _ in range(num_swaps):
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 20
        
        if low >= high:
            continue
        
        best_new_val = None
        best_score = float('inf')
        
        candidates = set()
        for _ in range(30):
            candidates.add(rng.randint(low, high))
        current_max = max(row)
        if pos == len(row) - 1:
            for dv in range(1, 8):
                candidates.add(max(low, current_max - dv))
        
        for new_val in candidates:
            if new_val == old_val or new_val < low or new_val > high:
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
            
            new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
            old_row_max = max(row)
            score = new_row_max - old_row_max
            if score < best_score:
                best_score = score
                best_new_val = new_val
        
        if best_new_val is not None:
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(best_new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - best_new_val)
            
            row[pos] = best_new_val
            used -= old_diffs
            used |= new_diffs
    
    return rows, used


def _randomized_csp(n_rows, k, deadline, rng, initial_max=120):
    """Repeatedly try to construct a valid set using randomized CSP,
    with decreasing max_val bounds."""
    best_rows = None
    best_scope = float('inf')
    
    for max_val in range(initial_max, 104, -1):
        if time.time() > deadline:
            break
        
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            rng_attempt = random.Random(rng.randint(0, 10**9))
            rows = _construct_simultaneous_csp(n_rows, k, max_val, rng_attempt, deadline)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    break
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: CSP construction with degree heuristic
    # Try CSP construction with decreasing max_val bounds
    for max_val in [115, 112, 110, 108, 106, 105]:
        if time.time() > deadline - 60.0:
            break
        csp_rows, csp_scope = _csp_construct(
            n_rows, k, max_val, rng, deadline - 60.0, n_attempts=8
        )
        if csp_rows is not None and csp_scope < best_scope:
            best_scope = csp_scope
            best_rows = csp_rows
            if best_scope <= 105:
                break
    
    # If CSP didn't find anything, fall back to greedy
    if best_rows is None:
        for seed_off in range(20):
            if time.time() > deadline - 60.0:
                break
            rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed_off)
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Phase 2: Randomized CSP with decreasing bounds
    if time.time() < deadline - 40.0 and best_rows is not None:
        csp_rows, csp_scope = _randomized_csp(
            n_rows, k, deadline - 40.0, rng, initial_max=best_scope
        )
        if csp_rows is not None and csp_scope < best_scope:
            best_rows = csp_rows
            best_scope = csp_scope
    
    # Phase 3: Simulated annealing (now includes CSP row-reconstruction moves)
    if time.time() < deadline - 15.0 and best_rows is not None:
        sa_rows = _simulated_annealing(best_rows, k, deadline - 15.0, rng)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Local search (now includes CSP row reconstruction)
    if time.time() < deadline - 5.0 and best_rows is not None:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 5: Final refinement with difference-swap mutations
    if time.time() < deadline - 2.0 and best_rows is not None:
        used = _all_diffs(best_rows)
        for _ in range(15):
            if time.time() > deadline - 2.0:
                break
            refined, used = _difference_swap_mutation(
                best_rows, k, used, num_swaps=rng.randint(2, 5), rng=rng
            )
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
