    # Final validation and cleanup
    # Ensure blocks are distinct
    seen = set()
    valid_blocks = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            valid_blocks.append(b)
            
    return valid_blocks
