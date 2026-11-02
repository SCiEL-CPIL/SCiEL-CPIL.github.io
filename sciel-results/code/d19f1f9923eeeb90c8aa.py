"""
Strategy: Conformal Map Packing.
Map the regular hexagon to a unit disk via a Möbius-like transformation,
use circle-packing heuristics (Fekete points / energy minimization) in the
disk to place 12 unit circles (approximating hexagons), back-transform to
hexagonal coordinates, then refine with a low-dimensional global search
over cluster rotation and outer hexagon orientation to minimize the
enclosing regular hexagon side length.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0  # apothem of unit hexagon

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    # Conformal map: hexagon -> disk
    # Use a simple radial scaling with angular correction
    def hex_to_disk(x, y, hex_side):
        """Map point in hexagon (centered at origin) to disk of radius 1."""
        r = math.sqrt(x*x + y*y)
        if r < 1e-12:
            return 0.0, 0.0
        theta = math.atan2(y, x)
        # Hexagon boundary radius as function of angle
        # For a regular hexagon with side s, the distance from center to boundary
        # at angle theta is: s * sqrt(3) / (2 * cos(theta - pi/6)) for |theta - pi/6| < pi/3
        # Simplified: use the fact that hexagon has 6-fold symmetry
        # Normalize theta to [0, 60) degrees
        theta_mod = (theta % (math.pi / 3))  # 0 to 60 deg
        # Distance from center to hex boundary at this angle
        # The hexagon with side s has apothem a = s*sqrt(3)/2
        # At angle 0 (along a vertex direction), distance = s
        # At angle 30 deg (along flat side), distance = a = s*sqrt(3)/2
        # Linear interpolation in cos space
        # Actually: r_boundary(theta) = s * cos(30) / cos(theta - 30) for theta in [0,60]
        # = s * (sqrt(3)/2) / cos(theta - pi/6)
        r_boundary = hex_side * (SQRT3 / 2) / math.cos(theta_mod - math.pi / 6)
        # Scale to disk
        if r_boundary < 1e-12:
            return 0.0, 0.0
        scale = 1.0 / r_boundary
        return x * scale, y * scale

    def disk_to_hex(x, y, hex_side):
        """Map point in unit disk back to hexagon coordinates."""
        r = math.sqrt(x*x + y*y)
        if r < 1e-12:
            return 0.0, 0.0
        theta = math.atan2(y, x)
        theta_mod = (theta % (math.pi / 3))
        r_boundary = hex_side * (SQRT3 / 2) / math.cos(theta_mod - math.pi / 6)
        scale = r_boundary
        return x * scale, y * scale

    def feke_initial_placement(n, seed=42):
        """Generate initial positions using a spiral/Fekete-like pattern in the disk."""
        np.random.seed(seed)
        # Start with a hexagonal lattice pattern in the disk
        # For 12 points: 1 center + 6 inner ring + 5 outer ring
        positions = [(0.0, 0.0)]
        # Inner ring: 6 points at radius ~0.5
        for k in range(6):
            angle = 2 * math.pi * k / 6
            r = 0.45
            positions.append((r * math.cos(angle), r * math.sin(angle)))
        # Outer ring: 5 points at radius ~0.85
        for k in range(5):
            angle = 2 * math.pi * k / 6 + math.pi / 6
            r = 0.82
            positions.append((r * math.cos(angle), r * math.sin(angle)))
        return np.array(positions)

    def energy_in_disk(positions, min_dist=0.35):
        """Repulsive energy to keep points apart in the disk."""
        n = len(positions)
        energy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    energy += 100.0 * (min_dist - d)**2
                else:
                    energy += 0.01 / (d + 0.01)**2
        # Also penalize points too close to boundary
        for i in range(n):
            r = math.sqrt(positions[i, 0]**2 + positions[i, 1]**2)
            if r > 0.95:
                energy += 10.0 * (r - 0.95)**2
        return energy

    def min_outer_side(inner_positions, outer_angle_deg, outer_center=(0.0, 0.0)):
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        if not all_verts:
            return 0.0
        all_verts = np.vstack(all_verts)

        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad

        max_proj = 0.0
        for na in normals:
            nx = math.cos(na)
            ny = math.sin(na)
            projs = (all_verts[:, 0] - outer_center[0]) * nx + (all_verts[:, 1] - outer_center[1]) * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p

        return max_proj / (SQRT3 / 2)

    # Step 1: Generate initial placement in disk space
    disk_positions = feke_initial_placement(N, seed=42)

    # Step 2: Optimize in disk space using energy minimization
    def disk_objective(flat_params):
        params = flat_params.reshape(-1, 2)
        return energy_in_disk(params)

    best_disk_positions = disk_positions.copy()
    best_disk_energy = energy_in_disk(disk_positions)

    # Multi-start optimization in disk space
    np.random.seed(123)
    for restart in range(5):
        if time.time() > deadline - 60:
            break
        init = disk_positions.copy()
        if restart > 0:
            init += np.random.normal(0, 0.1, size=init.shape)
            # Keep within disk
            for i in range(len(init)):
                r = np.linalg.norm(init[i])
                if r > 0.95:
                    init[i] *= 0.95 / r
        try:
            res = minimize(disk_objective, init.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_disk_energy:
                best_disk_energy = res.fun
                best_disk_positions = res.x.reshape(-1, 2).copy()
        except Exception:
            pass

    # Step 3: Back-transform to hexagonal space
    # Assume outer hexagon side ~4.0 for initial back-transform
    assumed_side = 4.0
    hex_positions = []
    for dx, dy in best_disk_positions:
        hx, hy = disk_to_hex(dx, dy, assumed_side)
        hex_positions.append((hx, hy))

    # Step 4: Refine in hexagonal space with full 36-DOF search
    # Represent as [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, outer_rot]
    def full_objective(params):
        n_params = len(params)
        outer_rot = params[-1]
        inner_params = params[:-1].reshape(-1, 3)

        # Check disjointness
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = hexagon_vertices(inner_params[i, 0], inner_params[i, 1], 1.0, inner_params[i, 2])
                v2 = hexagon_vertices(inner_params[j, 0], inner_params[j, 1], 1.0, inner_params[j, 2])
                if not polygons_disjoint_sat(v1, v2):
                    penalty += 10.0

        inner_pos = [(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2]) for i in range(N)]
        s = min_outer_side(inner_pos, outer_rot, (0.0, 0.0))
        return s + penalty

    # Initialize full parameter vector
    full_params = np.zeros(N * 3 + 1)
    for i in range(N):
        full_params[i*3] = hex_positions[i][0]
        full_params[i*3+1] = hex_positions[i][1]
        full_params[i*3+2] = 0.0  # angle
    full_params[-1] = 0.0  # outer rotation

    best_full_params = full_params.copy()
    best_full_score = full_objective(full_params)

    # Multi-start Nelder-Mead in full space
    np.random.seed(777)
    for restart in range(8):
        if time.time() > deadline - 30:
            break
        init = full_params.copy()
        if restart > 0:
            # Perturb positions and angles
            for i in range(N):
                init[i*3] += np.random.uniform(-0.3, 0.3)
                init[i*3+1] += np.random.uniform(-0.3, 0.3)
                init[i*3+2] += np.random.uniform(-30, 30)
            init[-1] += np.random.uniform(-30, 30)
        try:
            res = minimize(full_objective, init, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_full_score:
                best_full_score = res.fun
                best_full_params = res.x.copy()
        except Exception:
            pass

    # SA polish
    if time.time() < deadline - 15:
        current = best_full_params.copy()
        current_score = best_full_score
        T = 0.5
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            new = current.copy()
            # Randomly perturb a few hexagons
            for _ in range(3):
                idx = np.random.randint(0, N)
                new[idx*3] += np.random.uniform(-0.1, 0.1)
                new[idx*3+1] += np.random.uniform(-0.1, 0.1)
                new[idx*3+2] += np.random.uniform(-10, 10)
            new[-1] += np.random.uniform(-10, 10)
            new_score = full_objective(new)
            if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
                current = new
                current_score = new_score
                if current_score < best_full_score:
                    best_full_score = current_score
                    best_full_params = current.copy()
            T *= 0.999
            T = max(T, 0.01)

    # Extract results
    inner_params = best_full_params[:-1].reshape(-1, 3)
    inner = [(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2]) for i in range(N)]
    outer_rot = best_full_params[-1]
    s = min_outer_side(inner, outer_rot, (0.0, 0.0))

    center = (0.0, 0.0)
    side = s
    angle = outer_rot

    # Validation
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break

    if not valid:
        side *= 1.02

    return inner, center, side, angle
