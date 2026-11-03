def _final_polish(rows, k, deadline, rng):
    """Aggressive local search to compress scope further."""
    state = _SAState(rows, k)
    best_rows = [list(r) for r in state.rows]
    best_scope = state.max_val
    best_collisions, _ = state.get_cost()
    current_collisions, current_max = best_collisions, best_scope
    
    n_rows = state.n_rows
    row_len = state.row_len
    T = 1.0
    cooling = 0.99998
    min_T = 0.02
    
    while time.time() < deadline:
        r_idx = rng.randint(0, n_rows - 1)
        i_idx = rng.randint(1, row_len - 1)
        
        lower = state.rows[r_idx][i_idx - 1] + 1
        if i_idx < row_len - 1:
            upper = state.rows[r_idx][i_idx + 1] - 1
        else:
            upper = best_scope
        
        if lower > upper:
            continue
        
        # Prefer decreasing values for compression
        if state.rows[r_idx][i_idx] > lower:
            new_val = rng.randint(lower, state.rows[r_idx][i_idx] - 1)
        else:
            new_val = rng.randint(state.rows[r_idx][i_idx] + 1, upper)
            
        old_collisions, old_max = current_collisions, current_max
        old_cost = old_collisions * 1000 + old_max
        
        old_val, success = state.try_move(r_idx, i_idx, new_val)
        if not success:
            continue
        
        new_collisions, new_max = state.get_cost()
        new_cost = new_collisions * 1000 + new_max
        
        if new_cost < old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
            current_collisions, current_max = new_collisions, new_max
            if new_cost < best_collisions * 1000 + best_scope:
                best_collisions = new_collisions
                best_scope = new_max
                best_rows = [list(r) for r in state.rows]
        else:
            state.undo_move(r_idx, i_idx, old_val)
            # current_collisions, current_max remain unchanged as we reverted
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows
