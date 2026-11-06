<<<<<<< SEARCH
    # Try branch-and-bound from the best solution found
    if time.time() < deadline - 10:
        # Start B&B from empty set with all candidates
        bb_mask, bb_size = branch_and_bound(0, (1 << n) - 1, best_size, max_depth=12)
        if bb_size > best_size:
            best_size = bb_size
            best_mask = bb_mask
            
    # Also try B&B starting from the current best solution
    if time.time() < deadline - 5:
        # Candidates are elements not in best_mask
        candidates = ((1 << n) - 1) & ~best_mask
        bb_mask2, bb_size2 = branch_and_bound(best_mask, candidates, best_size, max_depth=8)
        if bb_size2 > best_size:
            best_size = bb_size2
            best_mask = bb_mask2
    
    # Final polish with more iterations
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, max_iters=100)
=======
    # Try branch-and-bound from the best solution found
    if time.time() < deadline - 8:
        # Start B&B from empty set with all candidates
        bb_mask, bb_size = branch_and_bound(0, (1 << n) - 1, best_size, max_depth=12)
        if bb_size > best_size:
            best_size = bb_size
            best_mask = bb_mask
            
    # Also try B&B starting from the current best solution
    if time.time() < deadline - 4:
        # Candidates are elements not in best_mask
        candidates = ((1 << n) - 1) & ~best_mask
        bb_mask2, bb_size2 = branch_and_bound(best_mask, candidates, best_size, max_depth=8)
        if bb_size2 > best_size:
            best_size = bb_size2
            best_mask = bb_mask2
    
    # Final high-precision polish with more iterations
    if time.time() < deadline - 1:
        best_mask = local_search(best_mask, max_iters=200)
        
        # Additional targeted polish: try removing each element and see if we can add more
        s_list = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
        for x in s_list:
            if time.time() > deadline - 1:
                break
            x_bit = 1 << (x - 1)
            mask_no_x = best_mask & ~x_bit
            # Try to add as many as possible
            new_mask = mask_no_x
            for y in range(1, n + 1):
                if y == x: continue
                y_bit = 1 << (y - 1)
                if not (new_mask & y_bit):
                    if can_add(y, new_mask):
                        new_mask |= y_bit
            if popcount(new_mask) > popcount(best_mask):
                best_mask = new_mask
>>>>>>> REPLACE

