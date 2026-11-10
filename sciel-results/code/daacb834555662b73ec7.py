# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) array for the moving sofa problem.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

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

    def _arc_path(cx, cy, n_pts):
        """Generate a circular arc path from (0,0,0) to (-2,0,90) with center (cx, cy)."""
        # Find radius from center to (0,0)
        R = np.sqrt(cx * cx + cy * cy)
        if R < 1e-10:
            return None
        # Angles from center to start and end
        a0 = np.arctan2(0.0 - cy, 0.0 - cx)
        a1 = np.arctan2(0.0 - cy, -2.0 - cx)
        
        # We need the sweep that goes from a0 to a1 in the "right" direction
        # The path should have angle going from 0 to 90 degrees
        # Determine sweep direction based on geometry
        sweep_direct = (a1 - a0) % (2 * np.pi)
        # Choose the shorter arc or the one that makes sense geometrically
        if sweep_direct > np.pi:
            sweep = sweep_direct - 2 * np.pi
        else:
            sweep = sweep_direct
            
        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        valid = True
        for t in t_vals:
            angle = a0 + t * sweep
            x = cx + R * np.cos(angle)
            y = cy + R * np.sin(angle)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    def _bump_path(amp, n_pts):
        """Generate a path with sinusoidal bump in y."""
        t_vals = np.linspace(0, 1, n_pts + 2)
        path = []
        for t in t_vals:
            x = -2.0 * t
            y = amp * np.sin(np.pi * t)
            a_deg = 90.0 * t
            path.append([x, y, a_deg])
        return np.array(path, dtype=np.float64)

    # --- Phase 1: Coarse search over 2D arc center (cx, cy) ---
    best_score = -1
    best_cx = -1.0
    best_cy = 1.0
    best_path = None

    # Coarse grid over center positions
    cx_vals = np.linspace(-3, 1, 25)
    cy_vals = np.linspace(-3, 3, 25)
    
    for cy in cy_vals:
        if time.time() > deadline - 180:
            break
        for cx in cx_vals:
            if time.time() > deadline - 180:
                break
            path = _arc_path(cx, cy, 9)
            if path is None:
                continue
            s = _score(path)
            if s > best_score:
                best_score = s
                best_cx = cx
                best_cy = cy
                best_path = path

    # Also try bump paths
    amp_vals = np.linspace(-2, 2, 31)
    for amp in amp_vals:
        if time.time() > deadline - 180:
            break
        path = _bump_path(amp, 9)
        s = _score(path)
        if s > best_score:
            best_score = s
            best_path = path

    # --- Phase 2: Fine search around best center ---
    if time.time() < deadline - 90:
        cx_fine = np.linspace(best_cx - 0.5, best_cx + 0.5, 21)
        cy_fine = np.linspace(best_cy - 0.5, best_cy + 0.5, 21)
        for cy in cy_fine:
            if time.time() > deadline - 90:
                break
            for cx in cx_fine:
                if time.time() > deadline - 90:
                    break
                path = _arc_path(cx, cy, 9)
                if path is None:
                    continue
                s = _score(path)
                if s > best_score:
                    best_score = s
                    best_cx = cx
                    best_cy = cy
                    best_path = path

    # --- Phase 3: Very fine search ---
    if time.time() < deadline - 45:
        cx_fine2 = np.linspace(best_cx - 0.15, best_cx + 0.15, 15)
        cy_fine2 = np.linspace(best_cy - 0.15, best_cy + 0.15, 15)
        for cy in cy_fine2:
            if time.time() > deadline - 45:
                break
            for cx in cx_fine2:
                if time.time() > deadline - 45:
                    break
                path = _arc_path(cx, cy, 9)
                if path is None:
                    continue
                s = _score(path)
                if s > best_score:
                    best_score = s
                    best_cx = cx
                    best_cy = cy
                    best_path = path

    # --- Phase 4: Optimize number of points ---
    if time.time() < deadline - 25:
        for n_pts in [7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20]:
            if time.time() > deadline - 25:
                break
            path = _arc_path(best_cx, best_cy, n_pts)
            if path is None:
                continue
            s = _score(path)
            if s > best_score:
                best_score = s
                best_path = path

    # --- Phase 5: Local refinement with smaller steps ---
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

    if best_path is None:
        best_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    if not np.all(np.isfinite(best_path)):
        best_path = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    return best_path


# EVOLVE-BLOCK-END
