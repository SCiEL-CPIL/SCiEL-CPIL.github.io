"""Unit regular hexagons inside a regular hexagon – compact packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin before 150 s kill

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def hex_verts(theta, scale=1.0):
        """Vertices of a regular hexagon centred at origin, rotated by theta."""
        angles = theta + np.arange(6) * np.pi / 3.0
        return np.column_stack([np.cos(angles), np.sin(angles)]) * scale

    def hex_support(theta, p):
        """Maximum dot product of hexagon vertices with direction p."""
        v = hex_verts(theta)
        return np.max(v @ p)

    def outer_verts(side, rot):
        """Vertices of outer regular hexagon centred at origin."""
        return hex_verts(rot, side)

    def point_in_hex(px, py, side, rot):
        """Check if point (px,py) is inside the outer hexagon."""
        # Transform point into hexagon frame
        ang = -rot
        cx, sx = np.cos(ang), np.sin(ang)
        lx =  cx * px + sx * py
        ly = -sx * px + cx * py
        # Hexagon with "flat top" orientation has support in 6 directions
        # For a regular hexagon of side s, the inradius is s*sqrt(3)/2
        # Distance from center to each side
        r = side * np.sqrt(3.0) / 2.0
        for k in range(6):
            a = k * np.pi / 3.0
            d = lx * np.cos(a) + ly * np.sin(a)
            if abs(d) > r + 1e-9:
                return False
        return True

    def separation(v1, th1, v2, th2, margin=1e-6):
        """
        Minimum required separation between two unit hexagons
        at positions v1, v2 with rotations th1, th2.
        Uses Minkowski sum of two hexagons; the hexagons are disjoint
        iff |v2-v1| >= support of (hex1 rotated + hex2 reflected) in direction (v2-v1)/|v2-v1|.
        """
        d = v2 - v1
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            return 2.0 * 1.0  # both at same spot, need at least 2*radius
        u = d / dist
        # support of hex1 in direction u
        s1 = hex_support(th1, u)
        # support of hex2 in direction -u  (reflected)
        s2 = hex_support(th2, -u)
        return s1 + s2 + margin

    def is_valid(pos, theta, side, rot, eps=1e-6):
        """Check that all inner hexagons fit inside outer and are pairwise disjoint."""
        # Check containment: each vertex of each inner hex must be inside outer
        for i in range(len(pos)):
            verts = pos[i] + hex_verts(theta[i])
            for v in verts:
                if not point_in_hex(v[0], v[1], side, rot):
                    return False
        # Check pairwise separation
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                req = separation(pos[i], theta[i], pos[j], theta[j])
                if np.linalg.norm(pos[i] - pos[j]) < req - eps:
                    return False
        return True

    # ------------------------------------------------------------------
    # Objective: minimise side length given positions and angles
    # ------------------------------------------------------------------
    def required_side(pos, theta, rot):
        """Minimum side of outer hexagon (centred at origin, rotated by rot) that contains all inner hexagons."""
        # For each inner hex, compute support in each of 6 outward normals of outer hex
        max_s = 0.0
        for k in range(6):
            a = rot + k * np.pi / 3.0
            nx, ny = np.cos(a), np.sin(a)
            for i in range(len(pos)):
                v = hex_verts(theta[i])
                s_inner = np.max(v @ np.array([nx, ny]))
                d = pos[i] @ np.array([nx, ny])
                # outer hex: max dot in this direction = side * sqrt(3)/2
                # need: d + s_inner <= side * sqrt(3)/2
                s = (d + s_inner) / (np.sqrt(3.0) / 2.0)
                # also negative side
                s2 = (-d - s_inner) / (np.sqrt(3.0) / 2.0)
                max_s = max(max_s, s, s2)
        return max_s

    # ------------------------------------------------------------------
    # Initial placement: honeycomb lattice
    # ------------------------------------------------------------------
    def initial_honeycomb():
        """Place hexagons in a compact honeycomb cluster."""
        positions = []
        angles = []
        # Honeycomb: two basis vectors
        a1 = np.array([2.0, 0.0])
        a2 = np.array([1.0, np.sqrt(3.0)])
        # Generate a diamond-shaped cluster
        max_r = 3
        for dx in range(-max_r, max_r + 1):
            for dy in range(-max_r, max_r + 1):
                if abs(dx) + abs(dy) + abs(dx + dy) <= 2 * max_r:
                    p = dx * a1 + dy * a2
                    if len(positions) < N:
                        positions.append(p)
                        angles.append(0.0)
        # If we have too many, keep the N closest to origin
        if len(positions) > N:
            idx = np.argsort([np.linalg.norm(p) for p in positions])[:N]
            positions = [positions[i] for i in idx]
            angles = [angles[i] for i in idx]
        return np.array(positions), np.array(angles)

    pos0, ang0 = initial_honeycomb()

    # ------------------------------------------------------------------
    # Optimisation: minimise side length
    # ------------------------------------------------------------------
    def objective(x):
        """x = [x0,y0,x1,y1,...,x_{N-1},y_{N-1}, th0,th1,...,th_{N-1}]"""
        pos = x[:2*N].reshape(N, 2)
        thetas = x[2*N:]
        # Pairwise separation penalty
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                req = separation(pos[i], thetas[i], pos[j], thetas[j], margin=0.0)
                d = np.linalg.norm(pos[i] - pos[j])
                if d < req:
                    pen += (req - d) ** 2
        side = required_side(pos, thetas, 0.0)
        return side + 10.0 * pen

    # Multi-start with small perturbations
    best_side = np.inf
    best_x = None
    n_starts = 30
    rng = np.random.default_rng(42)

    for s in range(n_starts):
        if time.time() > DEADLINE:
            break
        # Perturb initial positions
        if s == 0:
            pos_s = pos0.copy()
            ang_s = ang0.copy()
        else:
            pos_s = pos0 + rng.normal(0, 0.3, (N, 2))
            ang_s = ang0 + rng.uniform(-0.2, 0.2, N)

        x0 = np.concatenate([pos_s.ravel(), ang_s])
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if time.time() > DEADLINE - 10:
                break
            side = objective(res.x)
            if side < best_side:
                best_side = side
                best_x = res.x.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Polish: tighten with a second pass
    # ------------------------------------------------------------------
    if best_x is not None and time.time() < DEADLINE - 15:
        try:
            res2 = minimize(objective, best_x, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            side2 = objective(res2.x)
            if side2 < best_side:
                best_side = side2
                best_x = res2.x.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Extract result
    # ------------------------------------------------------------------
    if best_x is None:
        # Fallback: return a valid but larger packing
        pos = pos0
        thetas = ang0
        side = required_side(pos, thetas, 0.0) + 0.5
        inner = [[pos[i][0], pos[i][1], np.degrees(thetas[i])] for i in range(N)]
        return inner, (0.0, 0.0), side, 0.0

    pos = best_x[:2*N].reshape(N, 2)
    thetas = best_x[2*N:]
    side = required_side(pos, thetas, 0.0)

    # Add small safety margin
    side += 0.001

    inner = [[pos[i][0], pos[i][1], np.degrees(thetas[i])] for i in range(N)]
    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
