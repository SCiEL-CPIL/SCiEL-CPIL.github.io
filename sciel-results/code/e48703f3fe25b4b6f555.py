<<<<<<< SEARCH
        # Check if we have a complete covering
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

            if best_count <= 335:
                break

    if best_blocks is None:
=======
        # Check if we have a complete covering
        if uncovered_count == 0 and len(blocks) < best_count:
            # Try to remove redundant blocks (local search refinement)
            refined_blocks = _remove_redundant_blocks(blocks, v, t, sub_bitmasks, n_subs, deadline, rng)
            if len(refined_blocks) < best_count:
                best_count = len(refined_blocks)
                best_blocks = [list(b) for b in refined_blocks]

            if best_count <= 335:
                break

    if best_blocks is None:
>>>>>>> REPLACE

