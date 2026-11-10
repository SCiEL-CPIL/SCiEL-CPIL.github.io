"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

DEADLINE = 140.0  # safety margin before 150s kill
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


def _compute_radius(centers):
    """Return the largest common radius: min of half min-pair-dist and min wall-dist."""
    if len(centers) < 2:
        return 0.0
    d = cKDTree(centers)
    pairs = d.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        diffs = centers[pairs[:, 0]] - centers[pairs[:, 1]]
        min_pair = np.min(np.linalg.norm(diffs, axis=1))
    wall = np.min(0.5 - np.abs(centers), axis=1)
    return min(0.5 * min_pair, wall)


def _hex_initial():
    """Hexagonal (triangular) lattice packing for N circles in the unit square."""
    # For a hex lattice with circle radius r, horizontal spacing = 2r,
    # vertical spacing = 2r * sqrt(3)/2 = r*sqrt(3).
    # We'll build a lattice that fits N points, then scale to fit the square.
    # Approx: for N points in a hex lattice, rows alternate.
    # Try to find a good lattice by choosing rows.
    # Number of points in a hex lattice with R rows: roughly R * (R+1)/2 for triangular,
    # but we want a rectangular-ish region. Let's pick columns and rows.
    # For a hex packing filling a square, the optimal is roughly sqrt(N) per side.
    # We'll generate a hex lattice with spacing 1 (in lattice units), then scale.
    best = None
    # Try different numbers of rows
    for rows in range(13, 16):
        # In a hex lattice, alternate rows are offset by half.
        # cols per row: try a few
        for cols in range(13, 16):
            pts = []
            for i in range(rows):
                offset = 0.5 if i % 2 == 1 else 0.0
                for j in range(cols):
                    x = j + offset
                    y = i * np.sqrt(3) / 2.0
                    pts.append((x, y))
            pts = np.array(pts)
            if len(pts) < N:
                continue
            pts = pts[:N]
            # Scale to fit in [-0.5, 0.5]^2
            xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
            ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
            sx = 1.0 / (xmax - xmin) if xmax > xmin else 1.0
            sy = 1.0 / (ymax - ymin) if ymax > ymin else 1.0
            s = min(sx, sy)
            pts = (pts - pts.mean(axis=0)) * s
            # Check all inside
            if np.any(np.abs(pts) > 0.5):
                # nudge inward
                pts = pts * 0.99
            r = _compute_radius(pts)
            if best is None or r > best[0]:
                best = (r, pts.copy())
    if best is None:
        # fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best = (_compute_radius(grid[:N]), grid[:N].copy())
    return best[1]


def _relax(centers, max_iter=200000):
    """Force-directed relaxation: push apart close pairs, pull toward center if near wall."""
    centers = centers.astype(np.float64).copy()
    n = len(centers)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    # Adaptive step size
    step = 0.01
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        # Build tree and find close pairs
        tree = cKDTree(centers)
        # Use a radius that captures interactions; start with current min dist * 1.5
        r = _compute_radius(centers)
        search_r = max(2.0 * r, 0.05)
        pairs = tree.query_pairs(search_r, output_type='ndarray')
        if len(pairs) == 0:
            break

        forces = np.zeros_like(centers)
        for a, b in pairs:
            d = centers[a] - centers[b]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                # random nudge
                forces[a] += np.random.randn(2) * 1e-6
                forces[b] -= np.random.randn(2) * 1e-6
                continue
            # Repulsive force, stronger when very close
            f = 1.0 / (dist * dist + 1e-12)
            forces[a] += (d / dist) * f
            forces[b] -= (d / dist) * f

        # Wall repulsion: push away from boundaries
        for i in range(n):
            for axis in range(2):
                if centers[i, axis] > 0.5 - r:
                    forces[i, axis] -= 1.0 / (max(0.5 - centers[i, axis], 1e-9) ** 2)
                elif centers[i, axis] < -0.5 + r:
                    forces[i, axis] += 1.0 / (max(centers[i, axis] + 0.5, 1e-9) ** 2)

        # Normalize forces
        norm = np.linalg.norm(forces, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        forces = forces / norm

        # Update positions with small step
        new_centers = centers + step * forces
        # Clamp to square
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)

        new_r = _compute_radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers
            # slow down step
            step *= 0.999
        else:
            # reject, reduce step
            step *= 0.9
            if step < 1e-8:
                break

    return best_centers, best_r


def construct():
    centers = _hex_initial()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    # Multiple restarts with perturbations
    rng = np.random.default_rng(42)
    attempts = 0
    while _time_left() > 10.0:
        # Perturb the best solution
        perturb = best_centers + rng.normal(0, 0.005, size=best_centers.shape)
        perturb = np.clip(perturb, -0.5 + 1e-9, 0.5 - 1e-9)
        relaxed, r = _relax(perturb, max_iter=50000)
        if r > best_r:
            best_r = r
            best_centers = relaxed.copy()
        attempts += 1
        if attempts > 20:
            break

    return best_centers
# EVOLVE-BLOCK-END
