import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)

    start_time = time.time()
    time_limit = 575.0  # Safety margin before 630s kill

    def _make_densified(path):
        """Build the full densified path: start + path + end, with linear interpolation."""
        full = [(0.0, 0.0, 0.0)]
        for p in path:
            full.append((float(p[0]), float(p[1]), float(p[2])))
        full.append((-2.0, 0.0, 90.0))
        dens = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                dens.append((
                    p1[0] + (p2[0] - p1[0]) * t,
                    p1[1] + (p2[1] - p1[1]) * t,
                    p1[2] + (p2[2] - p1[2]) * t,
                ))
        dens.append(full[-1])
        return dens

    def _placement_mask(cx, cy, ca_deg):
        """Boolean mask of grid points inside corridor placement (cx, cy, ca_deg)."""
        ca = np.deg2rad(ca_deg)
        cos_a = np.cos(ca)
        sin_a = np.sin(ca)
        dx = x_flat - cx
        dy = y_flat - cy
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return in_arm1 | in_arm2

    def _segment_indices(dens, seg_start_idx):
        """Return (start, end) slice indices in dens for the segment starting at seg_start_idx."""
        # dens[0] = start, dens[1..] = interpolated + end
        # Segment from full_path[i] to full_path[i+1] corresponds to:
        # We need to figure out which dens indices belong to this segment.
        # Actually, it's easier to just recompute all masks for a given path.
        pass

    def evaluate_path(path):
        """Full evaluation: returns (area_score, densified_list, per_placement_masks)."""
        dens = _make_densified(path)
        mask = np.ones(n_points, dtype=bool)
        masks = []
        for (cx, cy, ca_deg) in dens:
            m = _placement_mask(cx, cy, ca_deg)
            mask &= m
            masks.append(m)
            if not np.any(mask):
                return 0.0, dens, masks
        return float(np.sum(mask)) * 0.0001, dens, masks

    def evaluate_path_fast(path, prev_dens=None, prev_masks=None, changed_idx=None):
        """
        Fast evaluation: if only one control point changed, recompute only affected segments.
        Returns (area_score, dens, masks).
        """
        n_ctrl = len(path)
        dens = _make_densified(path)
        
        if prev_dens is None or changed_idx is None:
            return evaluate_path(path)
        
        # Determine which dens indices are affected by changing path[changed_idx]
        # The densified path has segments between consecutive full_path points.
        # full_path = [start, p0, p1, ..., p_{n-1}, end]
        # Changing p_i affects segment (start->p0) if i==0, segment (p_i->p_{i+1}) always,
        # and segment (p_{i-1}->p_i) if i>0.
        # 
        # We need to map segment indices to dens index ranges.
        # Let's precompute the segment-to-dens-index mapping from the old dens.
        # Actually, the dens structure depends on the path (angle differences), so
        # changing an angle can change the number of interpolation steps.
        # For safety, if angles change significantly, do full recompute.
        
        # Check if the angle at changed_idx changed (affecting m for adjacent segments)
        old_p = prev_dens  # We stored the old densified list
        # This is getting complex. Let's just do full recompute but with early exit.
        # The incremental speedup is nice-to-have; the main gain is from the polish phase.
        return evaluate_path(path)

    # Initial path: hand-tuned starting point
    initial_path = [
        (-0.1, 0.1, 15.0),
        (-0.2, 0.2, 20.0),
        (-0.3, 0.3, 34.0),
        (-0.3, 0.3, 40.0),
        (-0.4, 0.3, 50.0),
        (-0.5, 0.2, 55.0),
        (-0.6, 0.2, 75.0),
        (-0.7, 0.3, 86.0),
    ]

    best_score, best_dens, best_masks = evaluate_path(initial_path)
    best_path = initial_path[:]
    current_path = initial_path[:]

    # Phase 1: Random local search with adaptive step sizes
    np.random.seed(42)
    epsilon_xy = 0.15
    epsilon_a = 3.0
    no_improve = 0
    max_no_improve = 30
    phase1_time_limit = 400.0  # Spend ~400s on random search

    while time.time() - start_time < min(time_limit, phase1_time_limit):
        idx = np.random.randint(len(current_path))
        p = current_path[idx]

        new_x = p[0] + np.random.normal(0, epsilon_xy)
        new_y = p[1] + np.random.normal(0, epsilon_xy)
        new_a = p[2] + np.random.normal(0, epsilon_a)
        new_a = np.clip(new_a, 0.0, 90.0)

        new_p = (new_x, new_y, new_a)
        test_path = current_path[:]
        test_path[idx] = new_p

        score, dens, masks = evaluate_path(test_path)

        if score > best_score:
            best_score = score
            best_path = test_path
            current_path = test_path
            best_dens = dens
            best_masks = masks
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                epsilon_xy *= 0.85
                epsilon_a *= 0.85
                no_improve = 0
                if epsilon_xy < 0.01 and epsilon_a < 0.5:
                    current_path = best_path[:]
                    epsilon_xy = 0.3
                    epsilon_a = 5.0

    # Phase 2: Coordinate-descent polish
    # For each control point, try small deterministic perturbations in x, y, a
    polish_xy = 0.05
    polish_a = 1.0
    polish_iters = 5
    polish_deadline = time_limit - 10.0  # Leave 10s buffer

    for it in range(polish_iters):
        if time.time() > polish_deadline:
            break
        improved = False
        for idx in range(len(best_path)):
            if time.time() > polish_deadline:
                break
            p = best_path[idx]
            best_local = best_score
            best_local_p = p
            # Try perturbations in x
            for dx in (-polish_xy, -polish_xy*0.5, polish_xy*0.5, polish_xy):
                test_p = (p[0]+dx, p[1], p[2])
                test_path = best_path[:]
                test_path[idx] = test_p
                s, d, m = evaluate_path(test_path)
                if s > best_local:
                    best_local = s
                    best_local_p = test_p
            # Try perturbations in y
            for dy in (-polish_xy, -polish_xy*0.5, polish_xy*0.5, polish_xy):
                test_p = (p[0], p[1]+dy, p[2])
                test_path = best_path[:]
                test_path[idx] = test_p
                s, d, m = evaluate_path(test_path)
                if s > best_local:
                    best_local = s
                    best_local_p = test_p
            # Try perturbations in a
            for da in (-polish_a, -polish_a*0.5, polish_a*0.5, polish_a):
                new_a = np.clip(p[2]+da, 0.0, 90.0)
                test_p = (p[0], p[1], new_a)
                test_path = best_path[:]
                test_path[idx] = test_p
                s, d, m = evaluate_path(test_path)
                if s > best_local:
                    best_local = s
                    best_local_p = test_p
            if best_local > best_score:
                best_path[idx] = best_local_p
                best_score = best_local
                improved = True
        if not improved:
            # Shrink polish steps
            polish_xy *= 0.6
            polish_a *= 0.6
            if polish_xy < 0.005 and polish_a < 0.1:
                break

    # Phase 3: Smooth boundary-nudging (epigraph-inspired)
    # For the best path, find the most constraining placement (the one that removes
    # the most points from the intersection). Nudge nearby control points away from
    # that placement's constraint.
    if time.time() < polish_deadline - 5:
        # Recompute best
        best_score, best_dens, best_masks = evaluate_path(best_path)
        
        # Find the placement that contributes most to the constraint
        # (i.e., the one with the fewest points inside, among those that are active)
        current_mask = np.ones(n_points, dtype=bool)
        for m in best_masks:
            current_mask &= m
        
        # For each placement, count how many points it excludes that are still in the intersection
        # Actually, the most constraining placement is the one whose mask has the smallest
        # overlap with the final intersection. Let's find the placement that, if removed,
        # would add the most points back.
        
        # Simpler: for each control point, try a small outward nudge based on the
        # gradient of the constraint. Since we don't have gradients, use random nudge
        # biased away from the most-constraining placement.
        
        # Find the most constraining placement index
        # A placement is "constraining" if it excludes points that would otherwise be in.
        # Approximate: the placement with the smallest number of points inside,
        # weighted by how many of those points are in the final intersection.
        
        final_mask = np.ones(n_points, dtype=bool)
        for m in best_masks:
            final_mask &= m
        
        # For each placement, compute how many final-intersection points it excludes
        # (i.e., points in final_mask that are NOT in this placement's mask)
        # But final_mask is the intersection, so every placement contains all final_mask points.
        # Instead, find the placement that is "tightest": the one where the fewest
        # non-intersection points are excluded (i.e., it's closest to the boundary).
        
        # Alternative: for each placement, compute the minimum distance from
        # final-intersection points to the corridor boundary in that placement's frame.
        # The placement with the smallest such minimum distance is the most constraining.
        
        # This is expensive. Let's just do a few random nudges of the control points
        # that are closest to the most-constraining placement.
        
        # Identify the most-constraining placement by checking which placement, when
        # slightly relaxed (shifted), would allow more points.
        # Approximation: the placement whose center is closest to any final-intersection point.
        
        # For simplicity, just do a final random polish with very small steps
        np.random.seed(123)
        small_xy = 0.03
        small_a = 0.5
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            idx = np.random.randint(len(best_path))
            p = best_path[idx]
            new_x = p[0] + np.random.normal(0, small_xy)
            new_y = p[1] + np.random.normal(0, small_xy)
            new_a = np.clip(p[2] + np.random.normal(0, small_a), 0.0, 90.0)
            test_path = best_path[:]
            test_path[idx] = (new_x, new_y, new_a)
            s, d, m = evaluate_path(test_path)
            if s > best_score:
                best_score = s
                best_path = test_path
                best_dens = d
                best_masks = m

    return np.array(best_path)
