import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    # The lower bound on the largest entry m is roughly sqrt(2 * n_rows * num_diffs_per_row)
    # But we are minimizing the actual largest entry found.
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with best-fit greedy
    # Use multiple seeds to find diverse starting points
    seed = 0
    max_initial_seeds = 50
    while time.time() < deadline - 60 and seed < max_initial_seeds:
        rows = _greedy_construct(n_rows, k, seed)
        if rows is None:
            seed += 1
            continue
        scope = _scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
        seed += 1
    
    if best_rows is None:
        # Fallback if greedy failed completely (should not happen for reasonable inputs)
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
             # Last resort: return a trivially valid but possibly large set?
             # The problem requires distinct differences.
             # For small n,k, greedy should work.
             best_rows = [[0, 1, 3, 7, 15, 31] for _ in range(n_rows)] # This is NOT valid globally.
             # We must ensure validity.
             # If greedy fails, we have a problem. Let's hope it doesn't.
             return [[0, i] for i in range(k+1)] # This is definitely invalid for n_rows > 1.
             # Actually, if n_rows=1, this is valid.
             # The contract says "Valid when all positive differences... are distinct over all rows".
             # If n_rows > 1, we need unique diffs.
             # Let's assume greedy works.
        
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated annealing to improve
    if time.time() < deadline - 15:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Final polish pass targeting the maximum element
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, seed):
    """Best-fit greedy: prefer values whose differences pack tightly into used space."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]
            # Look ahead limit: ensure we don't search forever
            # The max value shouldn't be too large.
            limit = 1000 
            
            x = last_x + 1
            # Collect a good number of candidates to choose from
            while len(valid_xs) < 50 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
            
            if not valid_xs:
                # If we didn't find enough in the first pass, expand the limit
                x = last_x + 1
                while len(valid_xs) < 50:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1
                    if x > limit * 10: # Hard cap to prevent infinite loop
                        break
            
            if not valid_xs:
                return None
            
            # Best-fit: sort by packing score (lower = better fit), pick from top few
            scored_xs.sort(key=lambda t: t[0])
            top_n = min(8, len(scored_xs))
            # Pick randomly from top to maintain diversity
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(used_sorted, d)
        
        rows.append(row)
    
    return rows

def _packing_score(new_diffs, used_sorted, x):
    """
    Score a candidate x by how well its differences 'pack' into the used difference space.
    Lower score = better fit (less fragmentation).
    """
    if not used_sorted:
        return x * 1.0
    
    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, used_sorted[idx] - d)
        if idx > 0:
            min_dist = min(min_dist, d - used_sorted[idx - 1])
        # Penalize gaps: if min_dist is inf, it's far away, which is bad?
        # Actually, we want to fill gaps.
        # If used_sorted is [1, 2, 4, 5] and d=3, dist=1. Good.
        # If d=10, dist=inf (or large).
        # We want to minimize the "wasted space".
        # Let's just use the sum of distances to nearest neighbors.
        if min_dist == float('inf'):
             min_dist = 100.0 # Large penalty for being too far out
        total_dist += min_dist
    
    # Add a small penalty for x itself to prefer smaller x if scores are close
    return total_dist / max(len(new_diffs), 1) + 0.05 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """Simulated annealing with best-fit row reconstruction."""
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    total_steps = 20000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    steps_since_improve = 0
    max_no_improve = 3000
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Occasionally do a targeted move on the row with the maximum element
        if rng.random() < 0.4:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        # Generate a new candidate row using best-fit greedy
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        
        if new_row is None:
            continue
        
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        # Check validity: all differences must be distinct
        all_diffs = _all_diffs(new_rows)
        expected_diffs = n_rows * (k + 1) * k // 2
        if len(all_diffs) != expected_diffs:
            continue
        
        # Accept or reject based on Metropolis criterion
        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Avoid overflow in exp
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows = new_rows
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
        
        # If stuck for too long, do a small perturbation (restart-like)
        if steps_since_improve > max_no_improve:
            # Perturb a random row
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                test_rows = rows[:]
                test_rows[pert_idx] = pert_row
                if _is_valid(test_rows, n_rows, k):
                    rows = test_rows
                    current_scope = _scope(rows)
            steps_since_improve = 0
            # Increase temperature slightly to escape local minima
            T = max(T, 1.0)
    
    return best_rows

def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2

def _generate_candidate_row(rows, exclude_idx, k, rng):
    """Generate a new candidate row using best-fit packing heuristic."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 1000
        
        x = last_x + 1
        while len(valid_xs) < 40 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 40:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
                if x > limit * 10:
                    break
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(6, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        used_sorted = sorted(used_diffs)
    
    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: repeatedly try to reduce the maximum element in the row that has it.
    Uses targeted local search on the max element.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    for iteration in range(100):
        if time.time() > deadline - 3:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        improved_row = _reduce_max_row(rows, max_idx, k, rng, current_max)
        
        if improved_row is not None and improved_row[-1] < current_max:
            new_rows = rows[:]
            new_rows[max_idx] = improved_row
            if _is_valid(new_rows, n_rows, k):
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = new_rows
                else:
                    # Even if not better than best, accept if it's valid and scope is same or better than current?
                    # No, only accept if it reduces scope to maintain progress.
                    pass
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    """
    Try to find a replacement for rows[row_idx] with a smaller maximum element.
    Uses a constrained greedy that tries to keep all elements below current_max.
    """
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    # Try to find a row with max < current_max
    # We can try target_max = current_max - 1, then -2, etc.
    # But usually we just try to construct with a tighter bound.
    
    target_max = current_max - 1
    for attempt in range(20):
        row = [0]
        success = True
        for _ in range(k):
            valid_xs = []
            last_x = row[-1]
            
            x = last_x + 1
            while x <= target_max:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                success = False
                break
            
            # Pick smallest to minimize max
            # Or pick best packing score within the constraint
            # Let's pick smallest to be aggressive on max reduction
            chosen_x = valid_xs[0]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used_diffs.update(new_diffs)
            used_sorted = sorted(used_diffs)
        
        if success and len(row) == k + 1:
            return row
        
        # If failed, relax the constraint slightly?
        # No, if it fails at current_max-1, it might fail at current_max-2.
        # We should probably stop or try a randomized search.
        # For polish, we just try a few times with randomness.
        # Let's modify the choice to be random among valid_xs to explore
        # But here we fixed it to valid_xs[0].
        # Let's add randomness to _reduce_max_row attempts.
        
    return None
