"""Root finder for 3 fixed 2D systems using Shrink-and-Search derivative-free method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a derivative-free Shrink-and-Search strategy:
      1. Locate a root's bounding box via grid scan + adaptive shrinking.
      2. Refine with 2D ternary / coordinate descent on ||F(x)||^2.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    call_count = 0
    max_calls = 490  # hard cap well under 500

    def time_up():
        return (time.time() - start_time) > deadline

    def Fw(x):
        nonlocal call_count
        if call_count >= max_calls or time_up():
            return None
        call_count += 1
        try:
            v = np.asarray(F(x), dtype=float).reshape(-1)
            if v.size != 2:
                return None
            if not np.all(np.isfinite(v)):
                return None
            return v
        except Exception:
            return None

    def g(x):
        """Scalar objective ||F(x)||^2, or +inf on failure."""
        v = Fw(x)
        if v is None:
            return np.inf
        return float(np.dot(v, v))

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != 2:
        x0 = np.array([0.0, 0.0])

    best_x = x0.copy()
    best_g = g(x0)
    if best_g <= 1e-16:  # 1e-8^2
        return best_x

    # ---------- Phase 1: Shrink-and-Search to locate a root box ----------
    # Build a grid over a box centered at x0, then shrink to the cell with min g.
    # Expand the search box if no good cell is found.
    center = x0.copy()
    half = 1.0  # initial half-width
    n_grid = 7   # 7x7 = 49 points per scan

    def scan_box(center, half, n):
        """Return (min_g, min_pt, min_cell) over an n x n grid."""
        xs = np.linspace(center[0] - half, center[0] + half, n)
        ys = np.linspace(center[1] - half, center[1] + half, n)
        min_g = np.inf
        min_pt = None
        min_cell = None
        for i in range(n):
            for j in range(n):
                if time_up() or call_count >= max_calls:
                    return min_g, min_pt, min_cell
                pt = np.array([xs[i], ys[j]])
                gv = g(pt)
                if gv < min_g:
                    min_g = gv
                    min_pt = pt.copy()
                    min_cell = (xs[i - 1] if i > 0 else pt[0] - 2 * half / (n - 1),
                                xs[i] if i < n - 1 else pt[0] + 2 * half / (n - 1),
                                ys[j - 1] if j > 0 else pt[1] - 2 * half / (n - 1),
                                ys[j] if j < n - 1 else pt[1] + 2 * half / (n - 1))
        return min_g, min_pt, min_cell

    # Try a few scales for the box to find one containing a root.
    best_scan_g = np.inf
    best_scan_pt = None
    best_box = None  # (xlo, xhi, ylo, yhi)

    for scale in [1.0, 0.25, 0.0625, 4.0, 16.0, 0.015625, 64.0]:
        if time_up() or call_count >= max_calls - 40:
            break
        half = scale
        mg, mp, cell = scan_box(center, half, n_grid)
        if mp is not None and mg < best_scan_g:
            best_scan_g = mg
            best_scan_pt = mp
            best_box = cell
        if mg <= 1e-16:
            break
        # If we found a very good point, stop expanding
        if mg < 1e-4:
            break

    if best_box is None and best_scan_pt is not None:
        # Fallback: tiny box around best point
        h = 1e-3
        best_box = (best_scan_pt[0] - h, best_scan_pt[0] + h,
                    best_scan_pt[1] - h, best_scan_pt[1] + h)

    # ---------- Phase 2: Adaptive refinement within the box ----------
    # Use nested coordinate-descent ternary search on g over the box.
    if best_box is not None:
        xlo, xhi, ylo, yhi = best_box
        # Ensure a valid box
        if xhi <= xlo:
            xlo, xhi = best_scan_pt[0] - 1e-3, best_scan_pt[0] + 1e-3
        if yhi <= ylo:
            ylo, yhi = best_scan_pt[1] - 1e-3, best_scan_pt[1] + 1e-3

        cur_x, cur_y = (xlo + xhi) / 2.0, (ylo + yhi) / 2.0
        cur_g = g(np.array([cur_x, cur_y]))
        if cur_g < best_g:
            best_g = cur_g
            best_x = np.array([cur_x, cur_y])
        if best_g <= 1e-16:
            return best_x

        # Coordinate-descent ternary refinement, alternating axes.
        # Each pass shrinks the interval on one axis by a factor.
        for it in range(60):
            if time_up() or call_count >= max_calls - 5:
                break
            if best_g <= 1e-16:
                return best_x

            # Ternary search along x (y fixed)
            for _ in range(8):
                if time_up() or call_count >= max_calls - 5:
                    break
                m1 = xlo + (xhi - xlo) / 3.0
                m2 = xhi - (xhi - xlo) / 3.0
                g1 = g(np.array([m1, cur_y]))
                g2 = g(np.array([m2, cur_y]))
                if g1 < g2:
                    xhi = m2
                else:
                    xlo = m1
                if xhi - xlo < 1e-12:
                    break

            # Ternary search along y (x fixed)
            for _ in range(8):
                if time_up() or call_count >= max_calls - 5:
                    break
                m1 = ylo + (yhi - ylo) / 3.0
                m2 = yhi - (yhi - ylo) / 3.0
                g1 = g(np.array([cur_x, m1]))
                g2 = g(np.array([cur_x, m2]))
                if g1 < g2:
                    yhi = m2
                else:
                    ylo = m1
                if yhi - ylo < 1e-12:
                    break

            cur_x = (xlo + xhi) / 2.0
            cur_y = (ylo + yhi) / 2.0
            cur_g = g(np.array([cur_x, cur_y]))
            if cur_g < best_g:
                best_g = cur_g
                best_x = np.array([cur_x, cur_y])
                if best_g <= 1e-16:
                    return best_x

            # Stop shrinking if box is tiny
            if (xhi - xlo) < 1e-14 and (yhi - ylo) < 1e-14:
                break

    # ---------- Phase 3: Final local nudge if still not converged ----------
    if best_g > 1e-16:
        # Try a few small perturbations around best_x
        for r in [1e-4, 1e-5, 1e-6, 1e-3, 1e-2]:
            if time_up() or call_count >= max_calls - 2:
                break
            for ang in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi / 4]:
                if time_up() or call_count >= max_calls - 2:
                    break
                pt = best_x + r * np.array([np.cos(ang), np.sin(ang)])
                gv = g(pt)
                if gv < best_g:
                    best_g = gv
                    best_x = pt.copy()
                    if best_g <= 1e-16:
                        return best_x

    return best_x
# EVOLVE-BLOCK-END
