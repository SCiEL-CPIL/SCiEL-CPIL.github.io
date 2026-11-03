"""Combined: lattice topology search + full-parameter SLSQP optimization."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def support_unit(n_vec):
        return np.max(unit_verts @ n_vec)

    def compute_outer_side_fast(positions, angles_rad, otheta_rad):
        """Compute minimum outer hexagon side for given configuration.
        positions: (N, 2) array of centers
        angles_rad: (N,) array of inner hex rotations
        otheta_rad: rotation of outer hexagon
        """
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta_rad + math.pi/6 + k * math.pi/3
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            # For each inner hex, compute support in direction n
            # Support of hex i = (P_i - O) . n + h_unit(R_{-alpha_i} n)
            for i in range(N):
                # Rotate n by -alpha_i
                c = math.cos(-angles_rad[i])
                s = math.sin(-angles_rad[i])
                n_rot = np.array([c*n[0] - s*n[1], s*n[0] + c*n[1]])
                h_val = support_unit(n_rot)
                proj = positions[i] @ n + h_val
                if proj > max_proj:
                    max_proj = proj
        return max_proj * 2.0 / sqrt3

    def check_disjoint(positions, min_dist=sqrt3 - 0.001):
        """Check if all hexagons are disjoint."""
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    return False
        return True

    def get_positions_axial(topo, scale=2.0):
        """Convert axial coordinates to Cartesian, centered."""
        pts = []
        for q, r in topo:
            x = scale * (q + r/2.0)
            y = scale * (r * sqrt3/2.0)
            pts.append((x, y))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return np.array([(x-cx, y-cy) for x, y in pts])

    def optimize_full(positions0, angles0_deg, max_iters=200):
        """Full SLSQP optimization: all positions, angles, and outer hex params.
        Returns (positions, angles_rad, otheta_rad, side) or None.
        """
        n = N
        # Variables: x1,y1,a1, ..., xn,yn,an, otheta
        # Total: 3n + 1
        x0 = []
        for i in range(n):
            x0.extend([positions0[i][0], positions0[i][1], math.radians(angles0_deg[i])])
        x0.append(0.0)  # outer hex rotation

        def objective(x):
            if time.time() > deadline - 10:
                return 1e10
            pos = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
            ang = np.array([x[3*i+2] for i in range(n)])
            otheta = x[3*n]
            return compute_outer_side_fast(pos, ang, otheta)

        # Constraints: pairwise distance >= sqrt(3)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                def dist_con(x, i=i, j=j):
                    return (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2 - (sqrt3 - 0.0005)**2
                constraints.append({'type': 'ineq', 'fun': dist_con})

        bounds = [(-6, 6), (-6, 6), (-math.pi, math.pi)] * n + [(-math.pi, math.pi)]

        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints,
                             options={'maxiter': max_iters, 'ftol': 1e-12})
            if result.fun < 1e9:
                pos = np.array([[result.x[3*i], result.x[3*i+1]] for i in range(n)])
                ang = np.array([result.x[3*i+2] for i in range(n)])
                otheta = result.x[3*n]
                side = compute_outer_side_fast(pos, ang, otheta)
                # Verify disjointness
                if check_disjoint(pos):
                    return pos, ang, otheta, side
        except Exception:
            pass
        return None

    def polish_nm(positions, angles_rad, otheta_rad, max_iter=1000):
        """Nelder-Mead polish for high precision."""
        def obj(x):
            pos = np.array([[x[3*i], x[3*i+1]] for i in range(N)])
            ang = np.array([x[3*i+2] for i in range(N)])
            ot = x[3*N]
            return compute_outer_side_fast(pos, ang, ot)

        x0 = []
        for i in range(N):
            x0.extend([positions[i][0], positions[i][1], angles_rad[i]])
        x0.append(otheta_rad)

        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-14})
            if res.fun < 1e9:
                pos = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(N)])
                ang = np.array([res.x[3*i+2] for i in range(N)])
                ot = res.x[3*N]
                if check_disjoint(pos):
                    return pos, ang, ot, res.fun
        except Exception:
            pass
        return None

    # Generate initial configurations
    configs = []

    # Topo 1: 3x4 Parallelogram
    t1 = [(q, r) for q in range(3) for r in range(4)]
    configs.append((get_positions_axial(t1, scale=2.0), [0.0]*12))

    # Topo 2: 4x3 Parallelogram
    t2 = [(q, r) for q in range(4) for r in range(3)]
    configs.append((get_positions_axial(t2, scale=2.0), [0.0]*12))

    # Topo 3: 2x6 Parallelogram
    t3 = [(q, r) for q in range(2) for r in range(6)]
    configs.append((get_positions_axial(t3, scale=2.0), [0.0]*12))

    # Topo 4: 6x2 Parallelogram
    t4 = [(q, r) for q in range(6) for r in range(2)]
    configs.append((get_positions_axial(t4, scale=2.0), [0.0]*12))

    # Topo 5: Hexagonal cluster (center + ring1 + 5 from ring2)
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    if len(set(t5)) == 12:
        configs.append((get_positions_axial(t5, scale=2.0), [0.0]*12))

    # Topo 6: Compact cross
    t6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    ]
    if len(set(t6)) == 12:
        configs.append((get_positions_axial(t6, scale=2.0), [0.0]*12))

    # Topo 7: Linear chain
    t7 = [(q, 0) for q in range(12)]
    configs.append((get_positions_axial(t7, scale=2.0), [0.0]*12))

    # Topo 8: Diamond
    t8 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1), (2,-1)
    ]
    if len(set(t8)) == 12:
        configs.append((get_positions_axial(t8, scale=2.0), [0.0]*12))

    # Topo 9: 3x4 with rotation (30 deg)
    configs.append((get_positions_axial(t1, scale=2.0), [30.0]*12))

    # Topo 10: Hexagonal cluster with 30 deg rotation
    if len(set(t5)) == 12:
        configs.append((get_positions_axial(t5, scale=2.0), [30.0]*12))

    # Topo 11: 2x6 with 30 deg
    configs.append((get_positions_axial(t3, scale=2.0), [30.0]*12))

    # Topo 12: 4x3 with 15 deg
    configs.append((get_positions_axial(t2, scale=2.0), [15.0]*12))

    best_side = float('inf')
    best_positions = None
    best_angles_rad = None
    best_otheta = None

    # Optimize each configuration
    for cfg_idx, (pos0, ang0_deg) in enumerate(configs):
        if time.time() > deadline - 20:
            break

        # Try SLSQP optimization
        result = optimize_full(pos0, ang0_deg, max_iters=150)
        if result is not None:
            pos, ang, ot, side = result
            if side < best_side:
                best_side = side
                best_positions = pos.copy()
                best_angles_rad = ang.copy()
                best_otheta = ot

    # Fallback if all failed
    if best_positions is None:
        pos0, ang0 = configs[0]
        best_positions = pos0
        best_angles_rad = np.array([math.radians(a) for a in ang0])
        best_otheta = 0.0
        best_side = compute_outer_side_fast(best_positions, best_angles_rad, best_otheta)

    # Polish with Nelder-Mead
    if time.time() < deadline - 5 and best_positions is not None:
        polished = polish_nm(best_positions, best_angles_rad, best_otheta, max_iter=2000)
        if polished is not None:
            pos, ang, ot, side = polished
            if side < best_side:
                best_side = side
                best_positions = pos
                best_angles_rad = ang
                best_otheta = ot

    # Final second polish
    if time.time() < deadline - 2 and best_positions is not None:
        polished = polish_nm(best_positions, best_angles_rad, best_otheta, max_iter=3000)
        if polished is not None:
            pos, ang, ot, side = polished
            if side < best_side:
                best_side = side
                best_positions = pos
                best_angles_rad = ang
                best_otheta = ot

    # Construct return values
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], math.degrees(best_angles_rad[i])])

    return inner, (0.0, 0.0), best_side, math.degrees(best_otheta)

# EVOLVE-BLOCK-END
