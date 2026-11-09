import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 5s buffer

    def is_valid_extension(sums_set, suffix_sums, x):
        for s in suffix_sums:
            if (x + s) in sums_set:
                return False
        return True

    def update_sums(sums_set, suffix_sums, x):
        new_sums = set(sums_set)
        for s in suffix_sums:
            new_sums.add(x + s)
        return new_sums

    def update_suffix_sums(suffix_sums, x):
        return {x + s for s in suffix_sums}

    # State: (last_val, seq, sums_set, suffix_sums)
    # We sort by last_val to keep "open" sequences.
    
    beam = [(0, [], set(), {0})]
    best_seq = []
    best_len = 0
    W = 100  # Beam width
    
    while beam:
        if time.time() > DEADLINE:
            break
            
        next_beam = []
        
        for last_val, seq, sums_set, suffix_sums in beam:
            # Iterate candidates
            # Optimization: We can stop early if we find too many?
            # No, we need to evaluate all to prune correctly.
            
            for x in range(last_val + 1, 257):
                if is_valid_extension(sums_set, suffix_sums, x):
                    new_seq = seq + [x]
                    new_sums = update_sums(sums_set, suffix_sums, x)
                    new_suffix = update_suffix_sums(suffix_sums, x)
                    
                    next_beam.append((x, new_seq, new_sums, new_suffix))
        
        if not next_beam:
            break
            
        # Prune: Keep top W states with smallest last_val
        # If last_val is equal, we can break ties by sum set size (smaller is better?)
        # Or just keep the first W found (which are smallest x due to iteration order? No, we iterate x in range, so yes, smallest x first).
        # So if we just take the first W, we are keeping the smallest last_vals.
        # But we should sort to be sure if we added them in mixed order.
        # Since we iterate x from low to high for each state, and states in beam are sorted by last_val,
        # the generated next_beam is not globally sorted by last_val.
        # Example: State A (last=10) generates x=11, 12. State B (last=15) generates x=16, 17.
        # We want to keep the smallest last_vals globally.
        
        next_beam.sort(key=lambda state: state[0])
        
        beam = next_beam[:W]
        
        # Update best
        current_len = len(beam[0][1])
        if current_len > best_len:
            best_len = current_len
            best_seq = beam[0][1]
            
    return best_seq
