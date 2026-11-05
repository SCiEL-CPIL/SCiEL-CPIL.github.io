<<<<<<< SEARCH
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
=======
def _simulated_annealing(rows, k, used, deadline, rng):
    """Simulated annealing to minimize the scope."""
    rows = [list(r) for r in rows]
    used = set(used)
    current_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    T = 50.0  # Initial temperature
    T_min = 0.5
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        move = _sa_move(rows, k, used, rng, deadline)
        if move is None:
            T *= cooling
            continue
        
        new_rows, new_used, delta = move
        
        if delta <= 0:
            # Accept improving or neutral moves
            rows = new_rows
            used = new_used
            current_scope += delta
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Accept worsening moves with probability exp(-delta/T)
            if rng.random() < math.exp(-delta / T):
                rows = new_rows
                used = new_used
                current_scope += delta
        
        T *= cooling
    
    return best_rows, best_scope
>>>>>>> REPLACE

