    while covered != full_mask and active_blocks:
        best_block = -1
        best_gain = 0
        for b in active_blocks:
            gain = (block_masks[b] & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b
        # ...
    