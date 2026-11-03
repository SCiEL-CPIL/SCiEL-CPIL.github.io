"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    # A unit regular hexagon (side 1) has circumradius 1.
    # For a set of equal-orientation unit hexagons centered at p_i,
    # the minimal enclosing regular hexagon (same orientation, center c)
    # has side length = max_i ( max_k (v_k . (p_i - c)) + 1 )
    # where v_k are the 6 inward normals (unit) of the hexagon sides.
    # The enclosing hexagon of side s has support function s * (apothem factor).
    # Regular hexagon side s: apothem = s * sqrt(3)/2, circumradius = s.
    # The distance from center to a side (apothem) = s*sqrt(3)/2.
    # For a point at distance d from center in direction u, the hexagon must
    # satisfy: for all 6 side normals n_k, n_k . (p - c) <= apothem = s*sqrt(3)/2.
    # So s >= max_k ( n_k . (p-c) ) * 2/sqrt(3).
    # The inner hexagon extends 1 (its circumradius) in every direction,
    # so effectively each center is "inflated" by 1 in the support sense:
    # s >= ( max_k n_k . (p_i - c) + 1 ) * 2/sqrt(3).
    #
    # The 6 outward normals of a regular hexagon (flat-top vs pointy-top).
    # Use pointy-top: vertices at angles 30,90,150,210,270,330 deg.
    # Outward normals at 0,60,120,180,240,300 deg.
    normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                        for a in [0, 60, 120, 180, 240, 300]])

    def min_side_for_centers(centers, c):
        # centers: (N,2), c: (2,)
        diffs = centers - c  # (N,2)
        # For each inner hexagon, the max over normals of (n . d)
        proj = diffs @ normals.T  # (N,6)
        maxproj = proj.max(axis=1)  # (N,)
        return ((maxproj + 1.0) * 2.0 / math.sqrt(3.0)).max()

    def objective(x):
        centers = x[:2*N].reshape(N, 2)
        c = x[2*N:2*N+2]
        return min_side_for_centers(centers, c)

    def feasible(x):
        centers = x[:2*N].reshape(N, 2)
        c = x[2*N:2*N+2]
        s = min_side_for_centers(centers, c)
        # Check pairwise non-overlap: for equal-orientation unit hexagons,
        # distance between centers must be >= 2 (Minkowski sum is a hexagon
        # of side 2, whose inradius is 2*sqrt(3)/2 = sqrt(3) ... actually
        # the minimum distance for non-overlap of two equal-orientation
        # unit hexagons is 2 along the normal directions, but the true
        # minimum center distance for non-overlap is 2*apothem = sqrt(3)
        # when offset along a normal, but 2 when offset along a vertex direction.
        # To be SAFE, require center distance >= 2 (guarantees non-overlap
        # in all directions since the hexagon's max extent from center is 1).
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2.0 - 1e-9:
                    return s + (2.0 - d) * 10.0
        return s

    # Hexagonal lattice initialization: 12 hexagons in a compact cluster.
    # Hex grid: 3-4-3-2 or a 4x3 arrangement. Use a hexagonal packing.
    # Place 12 centers in a hexagonal lattice with spacing 2.
    def hex_lattice_init():
        # Rows of a hexagonal close packing.
        # Use a pattern: rows with counts [3,4,3,2] = 12
        centers = []
        dx = 2.0
        dy = math.sqrt(3)  # vertical spacing between rows in hex lattice
        rows = [3, 4, 3, 2]
        # center the whole thing
        all_pts = []
        for r, count in enumerate(rows):
            y = r * dy
            x0 = -(count - 1) * dx / 2.0
            for k in range(count):
                x = x0 + k * dx
                if r % 2 == 1:
                    x += dx / 2.0
                all_pts.append((x, y))
        # center
        arr = np.array(all_pts)
        arr -= arr.mean(axis=0)
        return arr

    centers0 = hex_lattice_init()
    x0 = np.concatenate([centers0.flatten(), [0.0, 0.0]])

    best_x = x0.copy()
    best_val = objective(x0)

    # Nelder-Mead polish from multiple perturbations
    rng = np.random.default_rng(0)
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        if n_restarts == 1:
            x_start = x0
        else:
            # perturb the best solution
            pert = rng.normal(0, 0.3, size=x0.shape)
            x_start = best_x + pert
        try:
            res = minimize(feasible, x_start, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-7,
                                    'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        if time.time() > deadline - 5.0:
            break

    # Reconstruct output: inner hexagons with fixed angle 0, center, side, rotation
    centers = best_x[:2*N].reshape(N, 2)
    c = best_x[2*N:2*N+2]
    side = min_side_for_centers(centers, c)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
    return inner, (float(c[0]), float(c[1])), float(side), 0.0
# EVOLVE-BLOCK-END
