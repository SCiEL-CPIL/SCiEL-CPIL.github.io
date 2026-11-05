<<<<<<< SEARCH
        # Compute new scope
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        # Accept or reject based on Metropolis criterion
        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            row_diffs[row_idx] = new_diffs
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                rows = new_rows
                row_diffs[row_idx] = new_diffs
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
=======
        # Compute new scope
        new_scope = max(new_row[-1], max(rows[i][-1] for i in range(n_rows) if i != row_idx))
        
        # Accept or reject based on Metropolis criterion with adaptive temperature
        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            row_diffs[row_idx] = new_diffs
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
            # Track improving deltas
            if delta < 0:
                delta_window.append(-delta)
                if len(delta_window) > window_size:
                    delta_window.pop(0)
                if delta_window:
                    avg_delta = sum(delta_window) / len(delta_window)
                    # Adapt temperature based on typical improvement size
                    T = max(T_end, min(T_start, avg_delta * 2.0 + T * 0.5))
        else:
            prob = math.exp(-delta / max(T, 0.1))
            if rng.random() < prob:
                rows = new_rows
                row_diffs[row_idx] = new_diffs
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        T *= cool_rate
        # Ensure T doesn't drop too fast
        T = max(T, T_end * 0.5)
>>>>>>> REPLACE

