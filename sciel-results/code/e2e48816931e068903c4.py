import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, rng=None):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    Returns rows and the initial all_used bitmask.
    rng: optional random.Random for tie-breaking (not used in pure greedy, but kept for interface).
    """
    rows = []
    used_bits = 0
    for r in range(n_rows):
        row = [0]
        x = 0
        row_bits = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_diffs_bits = 0
            for a in row:
                d = x - a
                if (row_bits >> d) & 1:
                    valid = False
                    break
                if (used_bits >> d) & 1:
                    valid = False
                    break
                new_diffs_bits |= (1 << d)
            if valid:
                row.append(x)
                row_bits |= new_diffs_bits
                used_bits |= new_diffs_bits
        rows.append(row)
    return rows, used_bits

def _row_diffs_bits(row):
    """Compute the bitmask of all positive differences in a row."""
    bits = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            bits |= (1 << d)
    return bits

def _build_static_diffs_bits(row, k):
    """
    For each position pi (1..k), compute the bitmask of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [0] * (k + 1)
    for pi in range(1, k + 1):
        s = 0
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                d = row[j] - row[i]
                s |= (1 << d)
        static[pi] = s
    return static

def _compute_new_diffs_bits_fast(row, pi, new_val, static_bits_pi):
    """
    Given a row with position pi replaced by new_val, compute the full diff bitmask
    using the precomputed static bits (diffs not involving pi) plus the k diffs
    involving pi. Returns (new_bits, valid_internal).
    """
    n = len(row)
    new_bits = static_bits_pi
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if (new_bits >> d) & 1:
            valid = False
            break
        new_bits |= (1 << d)
    return new_bits, valid

def _try_move(current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits, all_used_bits, current_val, k):
    """
    Attempt a single move: set current_rows[ri][pi] = new_val.
    Returns (accepted, new_bits, cand_row_max) if accepted, else (False, None, None).
    Does NOT mutate state; caller must apply.
    """
    if new_val <= current_rows[ri][pi - 1]:
        return False, None, None
    if pi < k and new_val >= current_rows[ri][pi + 1]:
        return False, None, None
    
    static_pi = static_diffs_bits[ri][pi]
    new_bits, valid_internal = _compute_new_diffs_bits_fast(
        current_rows[ri], pi, new_val, static_pi)
    
    if not valid_internal:
        return False, None, None
    
    old_bits = row_diffs_bits[ri]
    other_used_bits = all_used_bits & ~old_bits
    
    if new_bits & other_used_bits:
        return False, None, None
    
    if pi == k:
        cand_row_max = new_val
    else:
        cand_row_max = current_rows[ri][k]
    
    return True, new_bits, cand_row_max

def _apply_move(current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits, all_used_bits, k):
    """Apply a move that has been validated by _try_move."""
    new_bits = row_diffs_bits[ri]  # caller must have updated this
    all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
    current_rows[ri][pi] = new_val
    row_diffs_bits[ri] = new_bits
    static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
    return all_used_bits

def _random_valid_perturb(current_rows, row_diffs_bits, static_diffs_bits, all_used_bits, n_rows, k, n_moves, rng):
    """
    Perform n_moves random valid moves to perturb the solution.
    Returns True if at least one move was made.
    """
    made = False
    for _ in range(n_moves):
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else old_val + 40
        if min_allowed >= max_allowed:
            continue
        # Try a random candidate in range
        candidates = []
        # Prefer near current value for validity
        span = max_allowed - min_allowed + 1
        n_try = min(span, rng.randint(5, 15))
        for _ in range(n_try):
            candidates.append(rng.randint(min_allowed, max_allowed))
        rng.shuffle(candidates)
        for new_val in candidates:
            if new_val == old_val:
                continue
            accepted, new_bits, _ = _try_move(
                current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits,
                all_used_bits, 0, k)
            if accepted:
                all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
                current_rows[ri][pi] = new_val
                row_diffs_bits[ri] = new_bits
                static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
                made = True
                break
    return made

def _local_search(rows, n_rows, k, deadline, rng, time_budget=None):
    """
    Run SA + targeted shrink + final polish on a starting solution.
    Returns best_rows, best_val.
    """
    start_time = time.time()
    if time_budget is None:
        time_budget = max(5.0, deadline - start_time - 5.0)
    deadline_local = start_time + time_budget
    
    current_rows = [r[:] for r in rows]
    current_val = max(max(r) for r in current_rows)
    all_used_bits = 0
    for r in current_rows:
        all_used_bits |= _row_diffs_bits(r)
    row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
    static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    # Phase 1: Simulated Annealing
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline_local - 10.0 and no_improve < max_no_improve:
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        old_val = current_rows[ri][pi]
        
        candidates = []
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        if rng.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(rng.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        rng.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            accepted, new_bits, cand_row_max = _try_move(
                current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits,
                all_used_bits, current_val, k)
            
            if not accepted:
                continue
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
                current_rows[ri][pi] = new_val
                row_diffs_bits[ri] = new_bits
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
                
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
        
        if no_improve > 12000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used_bits = 0
            for r in current_rows:
                all_used_bits |= _row_diffs_bits(r)
            row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
            static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # Phase 2: Targeted shrink
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used_bits = 0
    for r in current_rows:
        all_used_bits |= _row_diffs_bits(r)
    row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
    static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    no_improve = 0
    max_no_improve_polish = 25000
    polish_deadline = min(deadline_local, start_time + time_budget - 5.0)
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        if max_row_indices and rng.random() < 0.8:
            ri = rng.choice(max_row_indices)
        else:
            ri = rng.randrange(n_rows)
        
        if ri in max_row_indices and rng.random() < 0.6:
            pi = k
        else:
            pi = rng.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        if max_decrease > 20:
            n_try = min(max_decrease, rng.randint(10, 20))
            weights = [d for d in range(1, max_decrease + 1)]
            try_decreases = rng.choices(weights, k=n_try)
            if max_decrease not in try_decreases:
                try_decreases.append(max_decrease)
        else:
            try_decreases = list(range(1, max_decrease + 1))
        
        candidates = [old_val - d for d in try_decreases]
        
        moved = False
        for new_val in candidates:
            accepted, new_bits, cand_row_max = _try_move(
                current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits,
                all_used_bits, current_val, k)
            
            if not accepted:
                continue
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.2
            else:
                accept = False
            
            if accept:
                all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
                current_rows[ri][pi] = new_val
                row_diffs_bits[ri] = new_bits
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                moved = True
                break
        
        if not moved:
            no_improve += 1
    
    # Phase 3: Final quick SA
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used_bits = 0
    for r in current_rows:
        all_used_bits |= _row_diffs_bits(r)
    row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
    static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    T = 0.15
    T_min = 0.01
    cool = 0.99999
    
    while time.time() < deadline_local:
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(25, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        n_try = min(max_decrease, rng.randint(5, 10))
        candidates = rng.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        moved = False
        for new_val in candidates:
            accepted, new_bits, cand_row_max = _try_move(
                current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits,
                all_used_bits, current_val, k)
            
            if not accepted:
                continue
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.1
            else:
                accept = False
            
            if accept:
                all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
                current_rows[ri][pi] = new_val
                row_diffs_bits[ri] = new_bits
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                moved = True
                break
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit (120s stated, use 110 for safety)
    
    rng = random.Random()
    
    # Phase 1: Multi-start greedy + local search
    best_rows = None
    best_val = float('inf')
    
    n_starts = 3
    for s in range(n_starts):
        if time.time() > deadline - 60.0:
            break
        rows, _ = _greedy_init(n_rows, k, rng)
        # Run local search with remaining budget split
        remaining = deadline - time.time() - 5.0
        if remaining < 10.0:
            break
        time_budget = min(30.0, remaining / (n_starts - s))
        candidate_rows, candidate_val = _local_search(rows, n_rows, k, deadline, rng, time_budget)
        if candidate_val < best_val:
            best_val = candidate_val
            best_rows = candidate_rows
    
    if best_rows is None:
        rows, _ = _greedy_init(n_rows, k, rng)
        best_rows = rows
        best_val = max(max(r) for r in best_rows)
    
    # Phase 2: Iterated Local Search (perturb + local search)
    while time.time() < deadline - 10.0:
        # Perturb the best solution
        current_rows = [r[:] for r in best_rows]
        all_used_bits = 0
        for r in current_rows:
            all_used_bits |= _row_diffs_bits(r)
        row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
        static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
        
        n_perturb = rng.randint(2, 4)
        _random_valid_perturb(current_rows, row_diffs_bits, static_diffs_bits, all_used_bits, n_rows, k, n_perturb, rng)
        
        # Local search on perturbed solution
        remaining = deadline - time.time() - 5.0
        if remaining < 10.0:
            break
        time_budget = min(20.0, remaining)
        candidate_rows, candidate_val = _local_search(current_rows, n_rows, k, deadline, rng, time_budget)
        
        if candidate_val < best_val:
            best_val = candidate_val
            best_rows = candidate_rows
        elif candidate_val == best_val and rng.random() < 0.3:
            best_rows = candidate_rows
    
    # Phase 3: Final polish on best
    remaining = deadline - time.time() - 2.0
    if remaining > 5.0 and best_rows is not None:
        final_rows, final_val = _local_search(best_rows, n_rows, k, deadline, rng, remaining)
        if final_val < best_val:
            best_val = final_val
            best_rows = final_rows
    
    return best_rows

# EVOLVE-BLOCK-END
