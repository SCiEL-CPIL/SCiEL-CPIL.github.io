    # Precompute valid submasks for each mask
    # valid_submasks[mask] = list of valid submasks of mask (including mask itself if valid)
    # To save memory, we only store them if needed?
    # 3^14 is ~4.7M integers. 4.7M * 8 bytes ~ 37 MB. Feasible.
    
    valid_submasks = [[] for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        # Iterate submasks of mask
        sub = mask
        while sub:
            if valid_type[sub] != 0:
                valid_submasks[mask].append(sub)
            sub = (sub - 1) & mask
