"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21
DEADLINE = 120.0  # seconds; return best before this
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


def _hex_lattice_positions(n, aspect):
    """Generate n circle centers in a hexagonal close-packed pattern,
    normalized so the bounding box has the given aspect ratio (W/H).
    Returns (centers, W, H) where W + H = 2 (perimeter 4)."""
    # Build rows of a hexagonal lattice
    # Row 0: k0 circles, row 1: k1, etc., alternating offset
    # For 21 circles, try several row configurations
    # 21 = 5+4+5+4+3? Let's just do a standard hex packing:
    # rows with counts like [4,5,4,5,3] or [5,4,5,4,3] or [3,5,5,5,3]
    # Simpler: fill a hexagonal grid and take first n.
    
    # Generate a large hex grid and take first n points
    pts = []
    row = 0
    while len(pts) < n + 10:
        count = 12  # enough per row
        for i in range(count):
            x = i + (0.5 if row % 2 else 0.0)
            y = row * np.sqrt(3) / 2.0
            pts.append((x, y))
        row += 1
    pts = np.array(pts[:n])
    
    # Normalize to unit bounding box
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w = xmax - xmin
    h = ymax - ymin
    if w < 1e-9: w = 1.0
    if h < 1e-9: h = 1.0
    pts = (pts - np.array([xmin, ymin])) / np.array([w, h])
    
    # Now scale to aspect ratio: W/H = aspect, W + H = 2
    # If current bounding box is 1x1, we want W = 2*aspect/(1+aspect), H = 2/(1+aspect)
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    pts = pts * np.array([W, H])
    
    return pts, W, H


def _is_valid(circles, W, H, tol=1e-9):
    """Check all circles are inside [0,W]x[0,H] and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    # Containment
    if np.any(x < -tol) or np.any(y < -tol):
        return False
    if np.any(x + r > W + tol) or np.any(y + r > H + tol):
        return False
    # Non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx*dx + dy*dy < (r[i] + r[j])**2 - tol:
                return False
    return True


def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))


def _optimize(circles, W, H, max_iters=50000, step=0.01):
    """Iterative relaxation: grow radii and resolve overlaps.
    circles: (n,3) array of [x, y, r]. Modifies in place and returns a copy."""
    circles = circles.copy()
    n = len(circles)
    best_sum = _sum_radii(circles)
    best_circles = circles.copy()
    
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Try to grow all radii slightly
        grow = step * 0.1
        r_new = r + grow
        
        # Resolve overlaps: for each overlapping pair, push apart
        converged = True
        for _ in range(3):  # inner relaxation passes
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r_new[i] + r_new[j]
                    if dist < min_dist - 1e-12:
                        if dist < 1e-12:
                            # coincident, push in random direction
                            dx = 0.01
                            dy = 0.01
                            dist = np.sqrt(dx*dx+dy*dy)
                        overlap = min_dist - dist
                        # push each apart by half overlap
                        fx = dx / dist * overlap * 0.5
                        fy = dy / dist * overlap * 0.5
                        x[i] += fx
                        y[i] += fy
                        x[j] -= fx
                        y[j] -= fy
                        moved = True
            # Clamp to rectangle
            for i in range(n):
                if x[i] < r_new[i]:
                    x[i] = r_new[i]
                if x[i] > W - r_new[i]:
                    x[i] = W - r_new[i]
                if y[i] < r_new[i]:
                    y[i] = r_new[i]
                if y[i] > H - r_new[i]:
                    y[i] = H - r_new[i]
            if not moved:
                break
        if moved:
            converged = False
        
        circles[:, 0] = x
        circles[:, 1] = y
        circles[:, 2] = r_new
        
        s = _sum_radii(circles)
        if s > best_sum:
            best_sum = s
            best_circles = circles.copy()
        
        if converged and it > 100:
            # no overlap, can keep growing
            pass
        if it % 1000 == 0 and it > 0:
            # check validity
            if _is_valid(circles, W, H):
                pass
            else:
                # revert to best
                circles = best_circles.copy()
                r = circles[:, 2]
    
    return best_circles


def _try_aspect(aspect, max_iters=50000):
    """Try packing with given aspect ratio. Returns best circles and sum."""
    centers, W, H = _hex_lattice_positions(N, aspect)
    # Initial radius: small
    r0 = 0.05
    circles = np.column_stack([centers[:, 0], centers[:, 1], np.full(N, r0)])
    
    # Optimize
    result = _optimize(circles, W, H, max_iters=max_iters)
    if not _is_valid(result, W, H):
        return None, 0.0
    return result, _sum_radii(result)


def construct():
    """Return array of shape (21, 3): [x, y, r] for 21 disjoint circles
    in a rectangle of perimeter at most 4, maximizing sum of radii."""
    best_circles = None
    best_sum = 0.0
    best_WH = (1.0, 1.0)
    
    # Try several aspect ratios
    aspects = [1.0, 1.2, 1.4, 1.6, 1.73, 1.8, 2.0, 0.8, 0.7, 0.6]
    
    for aspect in aspects:
        if _time_left() < 5.0:
            break
        circles, s = _try_aspect(aspect, max_iters=30000)
        if circles is not None and s > best_sum:
            best_sum = s
            best_circles = circles
            best_WH = _hex_lattice_positions(N, aspect)[1:]
    
    if best_circles is None:
        # Fallback: seed-like square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
