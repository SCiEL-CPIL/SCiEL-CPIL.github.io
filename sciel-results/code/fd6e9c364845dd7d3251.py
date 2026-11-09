    fixed_set = set(fixed)
    # ...
        i = random.randint(0, 19)
        old_val = current_S[i]
        if old_val in fixed_set:
            continue # Don't swap fixed elements
        # ...
