    # Check seed set
    seed_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    # Verify seed set validity
    sums = {0}
    valid_seed = True
    for x in seed_set:
        vx = v[x]
        can_add = True
        for s in sums:
            if (s + vx) in sums:
                can_add = False
                break
        if not can_add:
            valid_seed = False
            break
        new_sums = [s + vx for s in sums]
        sums.update(new_sums)
        
    if valid_seed:
        best_set = seed_set[:]
        best_size = len(seed_set)
