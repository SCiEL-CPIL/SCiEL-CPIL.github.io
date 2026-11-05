<<<<<<< SEARCH
    # Final cleanup: ensure we return the best found
    # Note: best_blocks is a list of indices into all_blocks
    # We need to return the actual blocks
    return [list(all_blocks[i]) for i in best_blocks]
=======
    # Final cleanup: ensure we return the best found
    # Note: best_blocks is a list of indices into filtered_blocks
    # We need to return the actual blocks
    return [list(filtered_blocks[i]) for i in best_blocks]
>>>>>>> REPLACE
