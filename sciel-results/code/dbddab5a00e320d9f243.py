    current_score = deficit * 1000 + len(state)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Store state for rollback
        # Since we modify cov and state, we need to rollback if rejected.
        # Easier: Save old values.
        
        move_type = rng.random()
        old_len = len(state)
        
        if move_type < 0.7:
            # Replace
            if not state:
                continue
            pos = rng.randint(0, len(state) - 1)
            old_idx = state[pos]
            new_idx = rng.randint(0, total_words - 1)
            if old_idx == new_idx:
                continue
            
            # Apply
            for j in neighbors[old_idx]:
                cov[j] -= 1
            for j in neighbors[new_idx]:
                cov[j] += 1
            state[pos] = new_idx
            # Rollback data
            rb_pos = pos
            rb_old = old_idx
            rb_new = new_idx
            rb_type = 0
        elif move_type < 0.85:
            # Add
            new_idx = rng.randint(0, total_words - 1)
            for j in neighbors[new_idx]:
                cov[j] += 1
            state.append(new_idx)
            rb_pos = len(state) - 1
            rb_old = None
            rb_new = new_idx
            rb_type = 1
        else:
            # Remove
            if not state:
                continue
            pos = rng.randint(0, len(state) - 1)
            old_idx = state[pos]
            for j in neighbors[old_idx]:
                cov[j] -= 1
            state.pop(pos)
            rb_pos = pos
            rb_old = old_idx
            rb_new = None
            rb_type = 2
            
        new_deficit = np.sum(cov == 0)
        new_len = len(state)
        new_score = new_deficit * 1000 + new_len
        
        # Update best
        if new_deficit == 0:
            if new_len < best_valid_size:
                best_valid_size = new_len
                best_state = state.copy()
                
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
        else:
            # Rollback
            if rb_type == 0:
                # Revert Replace
                for j in neighbors[rb_new]:
                    cov[j] -= 1
                for j in neighbors[rb_old]:
                    cov[j] += 1
                state[rb_pos] = rb_old
            elif rb_type == 1:
                # Revert Add
                for j in neighbors[rb_new]:
                    cov[j] -= 1
                state.pop(rb_pos)
            else:
                # Revert Remove
                for j in neighbors[rb_old]:
                    cov[j] += 1
                state.insert(rb_pos, rb_old)
                
        T *= cooling
