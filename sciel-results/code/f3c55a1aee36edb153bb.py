"""Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.

Strategy: Langevin / molecular-dynamics style simulation.
  - 12 rigid unit hexagons (center + rotation) with hard-core repulsive forces.
  - A time-dependent, shrinking hexagonal container.
  - Damped integrator so the system settles into a dense, collision-free equilibrium.
  - Local refinement pass (small random nudges + re-relaxation) to escape local minima.
  - Final exact minimal enclosing side computed via half-plane / SAT support.
"""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START

# ---- geometry helpers -------------------------------------------------------

def _hex_corners(cx, cy, ang, side=1.0):
    a = ang
    c = side * np.cos(np.array([a, a + np.pi/3, a + 2*np.pi/3,
                                a + np.pi, a + 4*np.pi/3, a + 5*np.pi/3]))
    s = side * np.sin(np.array([a, a + np.pi/3, a + 2*np.pi/3,
                                a + np.pi, a + 4*np.pi/3, a + 5*np.pi/3]))
    return np.column_stack((cx + c, cy + s))

def _sat_min_dist(a, b):
    """Minimum separating distance between two convex hexagons (>=0 if disjoint).
    Returns a scalar: negative means overlap (penetration depth)."""
    min_dist = float('inf')
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p0 = poly[i]
            p1 = poly[(i + 1) % n]
            edge = p1 - p0
            L = math.hypot(edge[0], edge[1])
            if L < 1e-12:
                continue
            # outward normal (right of edge direction for CCW polygon)
            nx = edge[1] / L
            ny = -edge[0] / L
            d0 = np.dot(p0, [nx, ny])
            proj0 = np.dot(a, [nx, ny])
            proj1 = np.dot(b, [nx, ny])
            lo0, hi0 = proj0.min(), proj0.max()
            lo1, hi1 = proj1.min(), proj1.max()
            # signed gap: how far apart the two projections are
            if hi0 <= lo1:
                gap = lo1 - hi0
            elif hi1 <= lo0:
                gap = lo0 - hi1
            else:
                gap = 0.0
            if gap < min_dist:
                min_dist = gap
    return min_dist

def _penetration(a, b):
    """Positive penetration depth (0 if disjoint) via SAT."""
    d = _sat_min_dist(a, b)
    return max(0.0, -d)

def _hex_inside_container(cx, cy, ang, S, Srot):
    """Check hexagon (center cx,cy, rotation ang) is inside regular hexagon
    centered at origin with side S rotated by Srot. True if fully inside."""
    corners = _hex_corners(cx, cy, ang)
    # container half-planes: 6 outward normals
    for k in range(6):
        th = Srot + k * math.pi / 3
        nx = math.cos(th)
        ny = math.sin(th)
        support = np.max(corners[:, 0] * nx + corners[:, 1] * ny)
        if support > S * math.sqrt(3.0) / 2.0 + 1e-9:
            return False
    return True

def _min_side_for_hexagons(positions, angles):
    """Minimal outer side length (centered at origin, any rotation) that contains
    all hexagons. We search over container rotation in steps and compute the
    required S for that rotation, take the minimum."""
    best = float('inf')
    # container rotation search
    for Srot_deg in np.linspace(0.0, 60.0, 13):
        Srot = math.radians(Srot_deg)
        req = 0.0
        for cx, cy, ang in zip(positions, angles):
            corners = _hex_corners(cx, cy, ang)
            for k in range(6):
                th = Srot + k * math.pi / 3
                nx = math.cos(th)
                ny = math.sin(th)
                support = np.max(corners[:, 0] * nx + corners[:, 1] * ny)
                s_req = 2.0 * support / math.sqrt(3.0)
                if s_req > req:
                    req = s_req
        if req < best:
            best = req
    return best

# ---- simulation -------------------------------------------------------------

def _step(positions, angles, velocities, ang_vels, dt, S, Srot,
          repulse_k, damp, temp):
    """One Langevin step. Returns new positions, angles, velocities, ang_vels."""
    n = len(positions)
    pos = positions.copy()
    ang = angles.copy()
    vel = velocities.copy()
    angvel = ang_vels.copy()

    corners = [_hex_corners(pos[i, 0], pos[i, 1], ang[i]) for i in range(n)]

    # forces
    fx = np.zeros(n)
    fy = np.zeros(n)
    tq = np.zeros(n)

    # hard-core repulsion between hexagons
    for i in range(n):
        for j in range(i + 1, n):
            # cheap bounding-box prefilter
            cx1, cy1 = pos[i, 0], pos[i, 1]
            cx2, cy2 = pos[j, 0], pos[j, 1]
            if abs(cx1 - cx2) > 2.0 or abs(cy1 - cy2) > 2.0:
                continue
            pen = _penetration(corners[i], corners[j])
            if pen <= 0.0:
                continue
            dx = cx2 - cx1
            dy = cy2 - cy1
            d = math.hypot(dx, dy)
            if d < 1e-9:
                dx, dy, d = 1e-9, 0.0, 1e-9
            ux, uy = dx / d, dy / d
            f = repulse_k * pen
            fx[i] -= f * ux
            fy[i] -= f * uy
            fx[j] += f * ux
            fy[j] += f * uy
            # small torque to reduce overlap via relative rotation
            tq[i] -= 0.05 * f
            tq[j] += 0.05 * f

    # container repulsion (push back if a corner is outside)
    for i in range(n):
        c = corners[i]
        for k in range(6):
            th = Srot + k * math.pi / 3
            nx = math.cos(th)
            ny = math.sin(th)
            support = np.max(c[:, 0] * nx + c[:, 1] * ny)
            limit = S * math.sqrt(3.0) / 2.0
            over = support - limit
            if over > 0.0:
                f = repulse_k * over
                fx[i] += f * nx
                fy[i] += f * ny
                # torque: rotate hexagon to reduce its support in this direction
                tq[i] += 0.05 * f

    # integrate (semi-implicit Euler with damping)
    for i in range(n):
        vel[i, 0] = (vel[i, 0] + fx[i] * dt) * damp
        vel[i, 1] = (vel[i, 1] + fy[i] * dt) * damp
        angvel[i] = (angvel[i] + tq[i] * dt) * damp
        # thermal noise
        vel[i, 0] += temp * (random.gauss(0, 1))
        vel[i, 1] += temp * (random.gauss(0, 1))
        angvel[i] += temp * 0.3 * (random.gauss(0, 1))
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        ang[i] += angvel[i] * dt
        # wrap angle
        ang[i] = ang[i] % (2.0 * math.pi)

    return pos, ang, vel, angvel

def _relax(positions, angles, S0, S1, n_iter, dt, Srot,
           repulse_k=8.0, damp=0.85, temp=0.02):
    """Run the simulation, shrinking container from S0 to S1."""
    pos = positions.copy()
    ang = angles.copy()
    vel = np.zeros((len(pos), 2))
    angvel = np.zeros(len(pos))
    t0 = time.time()
    for it in range(n_iter):
        frac = it / max(1, n_iter - 1)
        S = S0 + (S1 - S0) * frac
        pos, ang, vel, angvel = _step(pos, ang, vel, angvel, dt, S, Srot,
                                      repulse_k, damp, temp)
        if (it + 1) % 200 == 0:
            if time.time() - t0 > 120.0:
                break
    return pos, ang

def _is_valid(positions, angles, S, Srot, tol=1e-7):
    """Check all hexagons disjoint and inside container of side S."""
    n = len(positions)
    corners = [_hex_corners(positions[i, 0], positions[i, 1], angles[i])
               for i in range(n)]
    for i in range(n):
        if not _hex_inside_container(positions[i, 0], positions[i, 1],
                                     angles[i], S, Srot):
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if _penetration(corners[i], corners[j]) > tol:
                return False
    return True

def construct():
    deadline = time.time() + 110.0   # safety margin under 120s
    random.seed(20240517)
    np.random.seed(20240517)

    # ---- initial guess: hexagonal close-packed lattice of unit hexagons ----
    # A regular hexagon tiling: neighbors at distance 2 (edge to edge) in 6 dirs.
    # For 12 hexes, place them in a compact cluster.
    # Use a "flower + ring" arrangement in hex-lattice coordinates.
    # Hex-lattice basis vectors (for edge-to-edge tiling of unit hexagons):
    #   a1 = (2.0, 0.0)
    #   a2 = (1.0, sqrt(3))
    #   a3 = (-1.0, sqrt(3))
    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, math.sqrt(3.0)])
    a3 = np.array([-1.0, math.sqrt(3.0)])
    # 12 positions: center + 6 ring + 5 of second ring (compact)
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),   # center + ring
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),              # 5 more
    ]
    positions = np.zeros((N, 2))
    for i, (u, v) in enumerate(lattice_coords):
        # map (u,v) -> position using a1, a2
        positions[i, 0] = u * a1[0] + v * a2[0]
        positions[i, 1] = u * a1[1] + v * a2[1]
    # center the cluster
    positions -= positions.mean(axis=0)
    angles = np.zeros(N)

    # estimate a starting container size
    S_start = _min_side_for_hexagons(positions, angles) + 0.5
    S_target = S_start * 0.78   # aim to shrink ~22%

    best = (positions.copy(), angles.copy(), S_start, 0.0)
    best_side = S_start

    n_iter = 4000
    dt = 0.02

    # ---- main relaxation loop with multiple container rotations ----
    for Srot_deg in [0.0, 30.0, 15.0, 45.0]:
        if time.time() > deadline - 20.0:
            break
        Srot = math.radians(Srot_deg)
        pos, ang = _relax(positions, angles, S_start, S_target,
                          n_iter, dt, Srot)
        # compute actual minimal side for this configuration
        side = _min_side_for_hexagons(pos, ang)
        # verify validity at the computed side (with a small margin)
        if _is_valid(pos, ang, side * 1.0005, math.radians(Srot_deg)):
            if side < best_side:
                best_side = side
                best = (pos.copy(), ang.copy(), side, Srot_deg)
        # keep the relaxed config for the next rotation as a warm start
        positions, angles = pos, ang

    # ---- local refinement: small random nudges + short re-relax ----
    base_pos, base_ang, base_side, base_rot = best
    for _ in range(40):
        if time.time() > deadline - 5.0:
            break
        trial_pos = base_pos.copy()
        trial_ang = base_ang.copy()
        for i in range(N):
            trial_pos[i, 0] += random.gauss(0, 0.05)
            trial_pos[i, 1] += random.gauss(0, 0.05)
            trial_ang[i] += random.gauss(0, 0.05)
        # re-relax from current best side down a bit more
        pos, ang = _relax(trial_pos, trial_ang, base_side, base_side * 0.97,
                          1500, dt, math.radians(base_rot))
        side = _min_side_for_hexagons(pos, ang)
        if _is_valid(pos, ang, side * 1.0005, math.radians(base_rot)):
            if side < best_side:
                best_side = side
                best = (pos.copy(), ang.copy(), side, base_rot)
                base_pos, base_ang = pos.copy(), ang.copy()

    # ---- final: tighten the container to the exact minimal side ----
    final_pos, final_ang, final_side, final_rot = best
    # binary-search the minimal side for the best configuration (fixed centers)
    lo, hi = final_side * 0.95, final_side * 1.02
    Srot = math.radians(final_rot)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _is_valid(final_pos, final_ang, mid, Srot):
            hi = mid
        else:
            lo = mid
    final_side = hi

    # build the return format
    inner = [[float(final_pos[i, 0]), float(final_pos[i, 1]),
              float(math.degrees(final_ang[i])) % 360.0] for i in range(N)]
    return inner, (0.0, 0.0), float(final_side), float(final_rot)

# EVOLVE-BLOCK-END
