# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    """
    deadline = time.time() + 580.0  # safety margin before 600s limit

    # Sofa grid points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    W, H = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    W_pts = np.column_stack([X.ravel(), Y.ravel()])  # (N, 2)
    N = len(W_pts)

    # Pre-compute base corridor check for a placement (x, y, a_deg)
    # A point p is inside if R(-a)(p - (x,y)) is in the base corridor
    # Base: {0<=u<=1, -3<=v<=1} UNION {-3<=u<=1, 0<=v<=1}

    def inside_mask(x, y, a_deg):
        a_rad = np.deg2rad(a_deg)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        dx = W_pts[:, 0] - x
        dy = W_pts[:, 1] - y
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return m1 | m2

    def full_score(path_rows):
        """Compute the sofa area (raw value) for a given set of rows."""
        # Build full path: [start] + rows + [end]
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        
        # Build all placements with densification
        placements = [start]
        for i in range(len(path_rows)):
            a1 = placements[-1][2]
            a2 = path_rows[i][2]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for j in range(1, m + 1):
                t = j / m
                placements.append([
                    placements[-1][0] + t * (path_rows[i][0] - placements[-1][0]),
                    placements[-1][1] + t * (path_rows[i][1] - placements[-1][1]),
                    a1 + t * (a2 - a1),
                ])
        # Add end
        a1 = placements[-1][2]
        a2 = end[2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for j in range(1, m + 1):
            t = j / m
            placements.append([
                placements[-1][0] + t * (end[0] - placements[-1][0]),
                placements[-1][1] + t * (end[1] - placements[-1][1]),
                a1 + t * (a2 - a1),
            ])
        
        # Compute intersection mask
        mask = np.ones(N, dtype=bool)
        for x, y, a in placements:
            mask &= inside_mask(x, y, a)
        return np.sum(mask) * 0.0001

    def fast_score_active(path_rows, active_mask):
        """Fast score using active mask (only check points that survived so far)."""
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([-2.0, 0.0, 90.0])
        
        # Build all placements
        placements = [start]
        for i in range(len(path_rows)):
            a1 = placements[-1][2]
            a2 = path_rows[i][2]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for j in range(1, m + 1):
                t = j / m
                placements.append([
                    placements[-1][0] + t * (path_rows[i][0] - placements[-1][0]),
                    placements[-1][1] + t * (path_rows[i][1] - placements[-1][1]),
                    a1 + t * (a2 - a1),
                ])
        a1 = placements[-1][2]
        a2 = end[2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for j in range(1, m + 1):
            t = j / m
            placements.append([
                placements[-1][0] + t * (end[0] - placements[-1][0]),
                placements[-1][1] + t * (end[1] - placements[-1][1]),
                a1 + t * (a2 - a1),
            ])
        
        # Apply each placement mask to active points only
        active = active_mask.copy()
        for x, y, a in placements:
            if not np.any(active):
                break
            a_rad = np.deg2rad(a)
            ca, sa = np.cos(a_rad), np.sin(a_rad)
            dx = W_pts[active, 0] - x
            dy = W_pts[active, 1] - y
            u = ca * dx + sa * dy
            v = -sa * dx + ca * dy
            m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            local = m1 | m2
            active_idx = np.where(active)[0]
            active[active_idx[~local]] = False
        return np.sum(active) * 0.0001

    # Initial path: 89 rows, each 1 degree apart (1 to 89)
    # Start with a reasonable arc
    def initial_path():
        rows = []
        for i in range(89):
            a = (i + 1) * 1.0
            # Arc: inner corner traces a curve
            t = a / 90.0
            x = -0.1 - 0.5 * t
            y = 0.1 + 0.2 * np.sin(np.pi * t)
            rows.append([x, y, a])
        return np.array(rows, dtype=np.float64)

    best_path = initial_path()
    best_score = full_score(best_path)
    
    # Coordinate descent optimization
    # For each row k (0..88), try perturbing x, y
    step = 0.05
    n_passes = 0
    while time.time() < deadline - 10:
        improved = False
        for k in range(len(best_path)):
            if time.time() > deadline - 10:
                break
            a_k = best_path[k, 2]
            # Try perturbations in x and y
            for dx in [step, -step]:
                if time.time() > deadline - 10:
                    break
                new_row = best_path[k].copy()
                new_row[0] += dx
                trial = best_path.copy()
                trial[k] = new_row
                s = fast_score_active(trial, np.ones(N, dtype=bool))
                if s > best_score + 1e-10:
                    best_path = trial
                    best_score = s
                    improved = True
            for dy in [step, -step]:
                if time.time() > deadline - 10:
                    break
                new_row = best_path[k].copy()
                new_row[1] += dy
                trial = best_path.copy()
                trial[k] = new_row
                s = fast_score_active(trial, np.ones(N, dtype=bool))
                if s > best_score + 1e-10:
                    best_path = trial
                    best_score = s
                    improved = True
        n_passes += 1
        if not improved:
            # Reduce step size
            step *= 0.7
            if step < 0.005:
                # Try some random restarts
                for _ in range(3):
                    if time.time() > deadline - 10:
                        break
                    rp = initial_path()
                    noise = np.random.normal(0, 0.1, (len(rp), 2))
                    rp[:, :2] += noise
                    rs = fast_score_active(rp, np.ones(N, dtype=bool))
                    if rs > best_score:
                        best_path = rp
                        best_score = rs
                        step = 0.05
                if step < 0.005:
                    break
        # After a few passes, reduce step
        if n_passes % 3 == 0 and step > 0.01:
            step *= 0.7

    return best_path


# EVOLVE-BLOCK-END
