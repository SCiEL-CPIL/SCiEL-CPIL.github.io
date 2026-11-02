<<<<<<< SEARCH
    # --- If we found something, also try small perturbations of the best layout ---
    if best_overall_hexes is not None:
        # Try rotating the entire inner assembly
        base_hexes = best_overall_hexes
        all_v = np.vstack(base_hexes)
        centroid = all_v.mean(axis=0)
        
        for rot_delta in np.linspace(-0.1, 0.1, 5):
            if time.time() > deadline - 5:
                break
            try:
                new_hexes = []
                for v in base_hexes:
                    rel = v - centroid
                    c, s = np.cos(rot_delta), np.sin(rot_delta)
                    R = np.array([[c, -s], [s, c]])
                    new_v = rel @ R.T + centroid
                    new_hexes.append(new_v)
                if check_disjoint(new_hexes):
                    op, side = optimize_outer(new_hexes)
                    if side < best_overall_side:
                        best_overall_side = side
                        best_overall_hexes = new_hexes
                        best_overall_outer = op
            except Exception:
                pass
=======
    # --- If we found something, also try small perturbations of the best layout ---
    if best_overall_hexes is not None:
        # Try rotating the entire inner assembly
        # Note: Rotating the assembly changes the relative orientation of inner hexes to outer.
        # Since we treat inner as rigid body, we just rotate the whole set of vertices.
        # The "angle" of individual inner hexagons also rotates by rot_delta.
        base_hexes = best_overall_hexes
        all_v = np.vstack(base_hexes)
        centroid = all_v.mean(axis=0)
        
        for rot_delta in np.linspace(-0.1, 0.1, 5):
            if time.time() > deadline - 5:
                break
            try:
                new_hexes = []
                for v in base_hexes:
                    rel = v - centroid
                    c, s = np.cos(rot_delta), np.sin(rot_delta)
                    R = np.array([[c, -s], [s, c]])
                    new_v = rel @ R.T + centroid
                    new_hexes.append(new_v)
                if check_disjoint(new_hexes):
                    op, side = optimize_outer(new_hexes)
                    if side < best_overall_side:
                        best_overall_side = side
                        best_overall_hexes = new_hexes
                        best_overall_outer = op
                        best_inner_rot = best_inner_rot + rot_delta
            except Exception:
                pass
>>>>>>> REPLACE

