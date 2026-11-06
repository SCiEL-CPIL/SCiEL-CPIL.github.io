        if can_add(x, selected):
            # Branch 1: Include
            new_selected = selected | bit_x
            new_candidates = candidates & ~bit_x
            solve(new_selected, new_candidates, current_size + 1)
            
            # Branch 2: Exclude
            new_candidates = candidates & ~bit_x
            solve(selected, new_candidates, current_size)
        else:
            # Must Exclude
            new_candidates = candidates & ~bit_x
            solve(selected, new_candidates, current_size)
