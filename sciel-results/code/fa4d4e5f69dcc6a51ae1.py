"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    selected = []
    sums = set()
    
    def can_add(x, sel, sm):
        # Check if x can be added to sel while maintaining Sidon property
        # New sums: x + s for s in sel, and 2*x
        # These must be disjoint from sm
        # And distinct from each other (guaranteed if x not in sel and sel is set)
        for s in sel:
            if (x + s) in sm:
                return False
        if (2 * x) in sm:
            return False
        return True

    def get_valid_candidates(sel, sm):
        candidates = []
        sel_set = set(sel)
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if can_add(x, sel, sm):
                candidates.append(x)
        return candidates

    while True:
        if time.time() > deadline:
            break
            
        current_candidates = get_valid_candidates(selected, sums)
        
        if not current_candidates:
            break
            
        best_x = -1
        best_score = -1
        best_new_sums = None
        best_new_selected = None
        
        cand_set = set(current_candidates)
        
        for x in current_candidates:
            # Simulate adding x
            new_selected = selected + [x]
            new_sums = sums.union()
            
            # Generate new sums
            for s in selected:
                new_sums.add(x + s)
            new_sums.add(2 * x)
            
            # Check if this is a valid Sidon set (it is by construction of current_candidates)
            # Now determine how many other candidates are blocked
            # A candidate y is blocked if it is in current_candidates (and y != x)
            # but is NOT a valid candidate for new_selected
            
            # To be efficient, we can just check the other candidates
            blocked_count = 0
            for y in current_candidates:
                if y == x:
                    continue
                # Check if y can be added to new_selected
                if can_add(y, new_selected, new_sums):
                    # y is still valid, so not blocked
                    pass
                else:
                    blocked_count += 1
            
            # We want to maximize blocked_count
            # Tie-breaking: smaller x? Or larger x?
            # Smaller x might be better for "compactness" but doesn't strictly matter.
            # Let's stick to max blocked, then min x.
            if blocked_count > best_score or (blocked_count == best_score and (best_x == -1 or x < best_x)):
                best_score = blocked_count
                best_x = x
                best_new_sums = new_sums
                best_new_selected = new_selected

        if best_x == -1:
            # Should not happen if current_candidates is not empty
            break
            
        selected = best_new_selected
        sums = best_new_sums

    return selected
# EVOLVE-BLOCK-END
