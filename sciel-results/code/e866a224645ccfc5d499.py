    current_blocks = best_init
    best_blocks = current_blocks.copy()
    best_len = len(current_blocks)

    # ... helper functions defined ...

    def try_2_for_1():
        # ...
        if len(current_blocks) < best_len:
            nonlocal best_len, best_blocks
            best_len = len(current_blocks)
            # ...
