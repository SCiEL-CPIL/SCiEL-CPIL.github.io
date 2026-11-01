    # Final Polish: Remove any remaining redundant blocks
    changed = True
    while changed and time_left():
        changed = False
        # Rebuild counts? Or just check?
        # We have `counts` from the last iteration.
        # But if we just broke out of the while loop, counts are consistent with current_blocks.
        # Wait, if we reverted, counts are consistent.
        # If we accepted, counts are consistent.
        # So counts is valid for current_blocks.
        
        for i in range(len(current_blocks) - 1, -1, -1):
            block = current_blocks[i]
            b_idx = k_block_to_idx[block]
            # Check if all t-subsets of this block are covered by OTHER blocks
            # i.e., counts[t_idx] > 1 for all t_idx in block_to_t_subs[b_idx]
            # If counts[t_idx] == 1, it means this block is the only one covering it.
            redundant = True
            for t_idx in block_to_t_subs[b_idx]:
                if counts[t_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                current_blocks.pop(i)
                for t_idx in block_to_t_subs[b_idx]:
                    counts[t_idx] -= 1
                current_size -= 1
                changed = True
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = list(current_blocks)
    
    # Convert tuples to lists for output
    return [list(b) for b in best_blocks]
