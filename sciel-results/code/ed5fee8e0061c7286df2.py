    def max_tri_free(AdjMask, N):
        best = [0]
        # Order vertices? Maybe sort by degree descending to find large sets faster?
        # Or just 0..N-1.
        
        def backtrack(remaining, current_mask, current_size):
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + remaining.bit_count() <= best[0]:
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in remaining? Or just first?
            # Let's pick the first one for simplicity, or one that restricts the most.
            # For now, just take the lowest set bit.
            v_bit = remaining & -remaining
            v = v_bit.bit_length() - 1
            rest = remaining ^ v_bit
            
            # Option 1: Include v
            # Check if current_mask U {v} is triangle-free
            # This requires that (current_mask & N[v]) is independent.
            if is_independent(current_mask & N[v]):
                backtrack(rest, current_mask | v_bit, current_size + 1)
            
            # Option 2: Exclude v
            backtrack(rest, current_mask, current_size)
            
        backtrack((1 << N) - 1, 0, 0)
        return best[0]
    