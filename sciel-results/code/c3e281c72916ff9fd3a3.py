<<<<<<< SEARCH
    # Phase 1: Search over 4 ring-2 positions (495 combinations)
    ring2_indices = list(range(7, all_count))

    # Sort ring-2 by distance from origin to prioritize compact configurations
    ring2_with_dist = [(i, math.sqrt(all_xy[i][0]**2 + all_xy[i][1]**2)) for i in ring2_indices]
    ring2_with_dist.sort(key=lambda x: x[1])
    ring2_sorted = [i for i, d in ring2_with_dist]

    core_indices = list(range(7))

    # Evaluate all 495 combinations of 4 ring-2 positions
    for combo in itertools.combinations(ring2_sorted, 4):
        if time.time() > deadline - 15.0:
            break

        selected_indices = core_indices + list(combo)
        side, full_xy, center, angle = evaluate_config(selected_indices)

        if side < best_side:
            best_side = side
            best_xy = full_xy
            best_center = center
            best_angle = angle
=======
    # Phase 1: Search over 4 ring-2 positions (495 combinations)
    ring2_indices = list(range(7, all_count))

    # Sort ring-2 by distance from origin to prioritize compact configurations
    ring2_with_dist = [(i, math.sqrt(all_xy[i][0]**2 + all_xy[i][1]**2)) for i in ring2_indices]
    ring2_with_dist.sort(key=lambda x: x[1])
    ring2_sorted = [i for i, d in ring2_with_dist]

    core_indices = list(range(7))

    # Pre-evaluate: for each ring-2 index, compute a quick bounding box size
    # to prioritize evaluating more promising combos first
    # Use a fast proxy: sum of distances from centroid
    def quick_score(indices):
        """Fast proxy score for a configuration (lower is better)."""
        pts = [all_xy[i] for i in indices]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        # Max distance from centroid
        max_d = 0.0
        for p in pts:
            d = math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)
            if d > max_d:
                max_d = d
        return max_d

    # Generate all 495 combos and sort by quick score
    all_combos = list(itertools.combinations(ring2_sorted, 4))
    combo_scores = [(quick_score(core_indices + list(c)), c) for c in all_combos]
    combo_scores.sort(key=lambda x: x[0])

    # Evaluate combos in order of quick score
    for score, combo in combo_scores:
        if time.time() > deadline - 15.0:
            break

        selected_indices = core_indices + list(combo)
        side, full_xy, center, angle = evaluate_config(selected_indices)

        if side < best_side:
            best_side = side
            best_xy = full_xy
            best_center = center
            best_angle = angle
>>>>>>> REPLACE

