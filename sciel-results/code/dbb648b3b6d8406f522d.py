<<<<<<< SEARCH
    # Final local search refinement with fast incremental FFT
    if time.time() < deadline - 10:
        h_refined, val_refined = _local_search_flip_fast(best_h, max_passes=8, deadline=deadline-5)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Additional perturbation-based local search
    if time.time() < deadline - 5:
        h_pert, val_pert = _local_search_perturb(best_h, max_passes=3, deadline=deadline-2, step_size=0.05)
        if val_pert < best_score:
            best_score = val_pert
            best_h = h_pert.copy()
    
    # Final fast flip pass if time remains
    if time.time() < deadline - 2:
        h_final, val_final = _local_search_flip_fast(best_h, max_passes=5, deadline=deadline-1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
=======
    # Block-flip local search first (explores sign patterns more broadly)
    if time.time() < deadline - 20:
        h_block, val_block = _local_search_block(best_h, max_passes=4, block_size=4, deadline=deadline-10)
        if val_block < best_score:
            best_score = val_block
            best_h = h_block.copy()
    
    # Single-element flip refinement
    if time.time() < deadline - 10:
        h_refined, val_refined = _local_search_flip_fast(best_h, max_passes=6, deadline=deadline-5)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Final block-flip pass with different block size
    if time.time() < deadline - 5:
        h_block2, val_block2 = _local_search_block(best_h, max_passes=3, block_size=8, deadline=deadline-2)
        if val_block2 < best_score:
            best_score = val_block2
            best_h = h_block2.copy()
    
    # Final fast flip pass if time remains
    if time.time() < deadline - 2:
        h_final, val_final = _local_search_flip_fast(best_h, max_passes=4, deadline=deadline-1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
>>>>>>> REPLACE
