<<<<<<< SEARCH
    # Phase 1: Lattice enumeration with multiple outer orientations
    oangles_phase1 = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0]
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oangle in oangles_phase1:
            if time.time() > deadline:
                break
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            center, side = _compute_outer_for_positions(full_xy, [0.0]*11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle
=======
    # Phase 1: Lattice enumeration with multiple outer orientations
    oangles_phase1 = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0, 7.5, -7.5, 22.5, -22.5]
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oangle in oangles_phase1:
            if time.time() > deadline:
                break
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            center, side = _compute_outer_for_positions(full_xy, [0.0]*11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle

    # Phase 1b: Fixed 1-2-3-2-1 topology with global rotation sweep
    # This is a rigid cluster: center + 6 ring1 + 4 specific ring2
    # The 4 ring2 hexes that form a compact shape: (2,0), (0,-2), (-2,2), (-1,-1) -- need to check
    # Actually, for 1-2-3-2-1, use a linear strip pattern
    # In axial coords, a 1-2-3-2-1 cluster:
    # Row 0 (1): (0, 0)
    # Row 1 (2): (1, 0), (0, -1)  -- neighbors
    # Row 2 (3): (2, 0), (1, -1), (0, -2)
    # Row 3 (2): (2, -1), (1, -2)
    # Row 4 (1): (2, -2)
    # That's 1+2+3+2+1 = 9... need 11. Let me use the standard 11-hex cluster.
    # Actually the best 11-hex cluster is center + 6 ring1 + 4 from ring2.
    # The most compact 4 from ring2 are those that minimize the bounding hex.
    # Let's just try all C(14,4) = 1001 combos which we already do above.
    # Instead, add a fixed compact topology and sweep rotation:
    if time.time() < deadline - 20:
        # Fixed compact topology: center + ring1 + 4 best ring2 positions
        # The 4 ring2 positions that are "corners" of a compact shape:
        # (2,0), (2,-2), (-2,2), (-2,-2) -- these are 4 of the 6 "far" ring2 positions
        # Actually let's use: (2,0), (0,-2), (-2,2), (-1,-1) is not ring2
        # Ring2 positions: (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,2)
        # Compact choice: (2,0), (2,-2), (-2,2), (-2,-2) -- 4 corners
        fixed_axial = pos_axial + [(2,0), (2,-2), (-2,2), (-2,-2)]
        fixed_xy = [axial_to_xy(q, r) for q, r in fixed_axial]

        # Sweep global rotation of the cluster relative to outer
        for cluster_rot in range(0, 60, 5):
            if time.time() > deadline:
                break
            # Rotate all positions around origin
            rot_rad = math.radians(cluster_rot)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            rotated_xy = []
            for x, y in fixed_xy:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated_xy.append((rx, ry))

            for oangle in [0.0, 15.0, 30.0, -15.0, -30.0]:
                if time.time() > deadline:
                    break
                center, side = _compute_outer_for_positions(rotated_xy, [0.0]*11, oangle)
                if side < best_side:
                    best_side = side
                    best_positions = rotated_xy[:]
                    best_angles = [0.0] * 11
                    best_center = center
                    best_oangle = oangle
>>>>>>> REPLACE
