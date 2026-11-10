# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) array for the moving sofa problem.
    """
    start_time = time.time()
    deadline = start_time + 595  # safety margin before 630s kill

    # Sofa grid points
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    sofa_X, sofa_Y = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    sofa_x_flat = sofa_X.ravel()
    sofa_y_flat = sofa_Y.ravel()
    n_sofa = len(sofa_x_flat)

    def _densify(path):
        """Add the evaluator's start/end and interpolate between consecutive rows."""
        full = np.vstack([[[0.0, 0.0, 0.0]], path, [[-2.0, 0.0, 90.0]]])
        result = [full[0].copy()]
        for i in range(len(full) - 1):
            p1, p2 = full[i], full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m):
                t = k / m
                result.append(p1 * (1 - t) + p2 * t)
            result.append(p2.copy())
        return np.array(result)

    def _count_inside(path):
        """Count sofa points inside all placements in the densified path."""
        full = _densify(path)
        inside = np.ones(n_sofa, dtype=bool)
        for px, py, pa in full:
            if not np.any(inside):
                return 0
            a_rad = np.deg2rad(pa)
            c, s = np.cos(a_rad), np.sin(a_rad)
            dx = sofa_x_flat - px
            dy = sofa_y_flat - py
            u = c * dx + s * dy
            v = -s * dx + c * dy
            in1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside &= (in1 | in2)
        return np.sum(inside)

    def _score(path):
        return _count_inside(path) * 0.0001

    # ---- Path models ----

    def _arc_path(cx, cy, n_pts):
        """Circular arc from (0,0,0) to (-2,0,90) with center (cx, cy)."""
        R = np.sqrt(cx * cx + cy * cy)
        if R < 1e-10:
            return None
        a0 = np.arctan2(0.0 - cy, 0.0 - cx)
        a1 = np.arctan2(0.0 - cy, -2.0 - cx)
        sweep_direct = (a1 - a0) % (2 * np.pi)
        if sweep_direct > np.pi:
            sweep = sweep_direct - 2 * np.pi
        else:
            sweep = sweep_direct
        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        for t in t_vals:
            angle = a0 + t * sweep
            x = cx + R * np.cos(angle)
            y = cy + R * np.sin(angle)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    def _straight_path(n_pts):
        """Straight line from (0,0,0) to (-2,0,90)."""
        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        for t in t_vals:
            x = -2.0 * t
            y = 0.0
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    def _bump_path(amp, n_pts):
        """Sinusoidal bump in y."""
        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        for t in t_vals:
            x = -2.0 * t
            y = amp * np.sin(np.pi * t)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    def _two_arc_path(c1x, c1y, c2x, c2y, n1, n2):
        """
        Two-segment path: arc1 from (0,0,0) to junction, then arc2 from junction to (-2,0,90).
        Arc1 has center (c1x, c1y), arc2 has center (c2x, c2y).
        The junction is chosen so that arc1 ends at some point and arc2 starts there,
        with angles interpolating 0->90.
        """
        # Arc1: from (0,0) with center (c1x, c1y)
        R1 = np.sqrt(c1x * c1x + c1y * c1y)
        if R1 < 1e-10:
            return None
        # Arc2: from (-2,0) with center (c2x, c2y)
        R2 = np.sqrt((c2x + 2.0) ** 2 + c2y * c2y)
        if R2 < 1e-10:
            return None

        # We need to find a junction point that lies on both arcs.
        # Arc1: points at distance R1 from (c1x, c1y)
        # Arc2: points at distance R2 from (c2x, c2y)
        # Intersection of two circles
        dx = c2x - c1x
        dy = c2y - c1y
        d = np.sqrt(dx * dx + dy * dy)
        if d < 1e-10:
            return None
        if d > R1 + R2 or d < abs(R1 - R2):
            return None

        # Circle intersection
        a = (R1 * R1 - R2 * R2 + d * d) / (2 * d)
        h2 = R1 * R1 - a * a
        if h2 < 0:
            return None
        h = np.sqrt(h2)

        mx = c1x + a * dx / d
        my = c1y + a * dy / d

        # Two intersection points
        ix1 = mx + h * (-dy) / d
        iy1 = my + h * (dx) / d
        ix2 = mx - h * (-dy) / d
        iy2 = my - h * (dx) / d

        # Choose the intersection that gives a sensible path (angle increasing 0->90)
        # We'll try both and pick the one that works better, or just pick the first
        # For now, try both and return the better one
        best = None
        for (jx, jy) in [(ix1, iy1), (ix2, iy2)]:
            # Angle from c1 to (0,0) and to (jx, jy)
            a1_start = np.arctan2(0.0 - c1y, 0.0 - c1x)
            a1_end = np.arctan2(jy - c1y, jx - c1x)
            sweep1 = (a1_end - a1_start) % (2 * np.pi)
            if sweep1 > np.pi:
                sweep1 -= 2 * np.pi

            # Angle from c2 to (jx, jy) and to (-2,0)
            a2_start = np.arctan2(jy - c2y, jx - c2x)
            a2_end = np.arctan2(0.0 - c2y, -2.0 - c2x)
            sweep2 = (a2_end - a2_start) % (2 * np.pi)
            if sweep2 > np.pi:
                sweep2 -= 2 * np.pi

            # Total sweep should be ~90 degrees worth of angle change
            # The angle parameter goes from 0 to 90 linearly with the path
            # We need the geometric sweeps to be consistent
            # Just build the path and check
            t1_vals = np.linspace(0, 1, n1 + 1)
            t2_vals = np.linspace(0, 1, n2 + 1)
            path = []
            for t in t1_vals:
                angle = a1_start + t * sweep1
                x = c1x + R1 * np.cos(angle)
                y = c1y + R1 * np.sin(angle)
                a_deg = 90.0 * t
                path.append([x, y, a_deg])
            for t in t2_vals[1:]:
                angle = a2_start + t * sweep2
                x = c2x + R2 * np.cos(angle)
                y = c2y + R2 * np.sin(angle)
                a_deg = 90.0 * t
                path.append([x, y, a_deg])
            p = np.array(path, dtype=np.float64)
            if best is None or len(p) > 0:
                # We'll score both and pick the better
                if best is None:
                    best = p
                else:
                    # Keep both, we'll evaluate separately
                    best = None  # signal to try both
                    break
            # Actually, let's just return the first valid one for simplicity in grid search
            return p
        if best is not None:
            return best
        return None

    # ---- Search ----

    best_score = -1
    best_path = None
    best_cx = -1.0
    best_cy = 1.0

    def _try_path(path, cx=None, cy=None):
        nonlocal best_score, best_path, best_cx, best_cy
        if path is None or len(path) < 1:
            return
        if not np.all(np.isfinite(path)):
            return
        s = _score(path)
        if s > best_score:
            best_score = s
            best_path = path.copy()
            if cx is not None:
                best_cx = cx
            if cy is not None:
                best_cy = cy

    # Phase 1: Coarse search over arc centers
    if time.time() < deadline - 200:
        cx_vals = np.linspace(-3, 1, 25)
        cy_vals = np.linspace(-3, 3, 25)
        for cy in cy_vals:
            if time.time() > deadline - 200:
                break
            for cx in cx_vals:
                if time.time() > deadline - 200:
                    break
                path = _arc_path(cx, cy, 15)
                _try_path(path, cx, cy)

    # Phase 1b: Bump paths
    if time.time() < deadline - 180:
        amp_vals = np.linspace(-2, 2, 31)
        for amp in amp_vals:
            if time.time() > deadline - 180:
                break
            path = _bump_path(amp, 9)
            _try_path(path)

    # Phase 1c: Straight
    if time.time() < deadline - 170:
        path = _straight_path(9)
        _try_path(path)

    # Phase 2: Coarse two-arc search
    if time.time() < deadline - 120:
        # Search over centers for arc1 and arc2
        c1x_vals = np.linspace(-3, 1, 12)
        c1y_vals = np.linspace(-3, 3, 12)
        c2x_vals = np.linspace(-5, -1, 12)
        c2y_vals = np.linspace(-3, 3, 12)
        for c1y in c1y_vals:
            if time.time() > deadline - 120:
                break
            for c1x in c1x_vals:
                if time.time() > deadline - 120:
                    break
                for c2y in c2y_vals:
                    if time.time() > deadline - 120:
                        break
                    for c2x in c2x_vals:
                        if time.time() > deadline - 120:
                            break
                        path = _two_arc_path(c1x, c1y, c2x, c2y, 5, 5)
                        _try_path(path)

    # Phase 3: Fine search around best single arc
    if best_path is not None and time.time() < deadline - 60:
        cx_fine = np.linspace(best_cx - 0.5, best_cx + 0.5, 21)
        cy_fine = np.linspace(best_cy - 0.5, best_cy + 0.5, 21)
        for cy in cy_fine:
            if time.time() > deadline - 60:
                break
            for cx in cx_fine:
                if time.time() > deadline - 60:
                    break
                path = _arc_path(cx, cy, 20)
                _try_path(path, cx, cy)
        
        # Phase 3b: Very Fine Search around the current best
        if best_path is not None and time.time() < deadline - 30:
             cx_vf = np.linspace(best_cx - 0.15, best_cx + 0.15, 15)
             cy_vf = np.linspace(best_cy - 0.15, best_cy + 0.15, 15)
             for cy in cy_vf:
                 if time.time() > deadline - 30:
                     break
                 for cx in cx_vf:
                     if time.time() > deadline - 30:
                         break
                     path = _arc_path(cx, cy, 30)
                     _try_path(path, cx, cy)

    # Phase 4: Local refinement of best path
    if best_path is not None and time.time() < deadline - 15:
        current = best_path.copy()
        n = len(current)
        steps = [0.05, 0.03, 0.02, 0.01, 0.005]
        for step in steps:
            if time.time() > deadline - 15:
                break
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                for i in range(1, n - 1):
                    if time.time() > deadline - 15:
                        break
                    for dx in [-step, 0.0, step]:
                        if time.time() > deadline - 15:
                            break
                        for dy in [-step, 0.0, step]:
                            if time.time() > deadline - 15:
                                break
                            if dx == 0.0 and dy == 0.0:
                                continue
                            trial = current.copy()
                            trial[i, 0] += dx
                            trial[i, 1] += dy
                            s = _score(trial)
                            if s > best_score:
                                best_score = s
                                best_path = trial
                                current = trial
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # Phase 5: Optimize number of points for best arc-like path
    if best_path is not None and time.time() < deadline - 10:
        for n_pts in [7, 8, 10, 11, 12, 14, 16, 18, 20]:
            if time.time() > deadline - 10:
                break
            path = _arc_path(best_cx, best_cy, n_pts)
            _try_path(path, best_cx, best_cy)

    if best_path is None:
        best_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    if not np.all(np.isfinite(best_path)):
        best_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    return best_path


# EVOLVE-BLOCK-END
