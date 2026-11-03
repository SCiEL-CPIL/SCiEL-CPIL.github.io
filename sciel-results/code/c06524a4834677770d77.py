import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def compute_min_outer_side_smooth(positions, angles_deg, beta=10.0):
        """
        Compute the minimum outer hexagon side length using a smooth approximation.
        Also returns the optimal rotation angle phi for the outer hexagon.
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # We need to find the rotation phi of the outer hexagon that minimizes the side length.
        # Side length S is determined by the maximum projection of vertices onto the 6 normal directions.
        # For a regular hexagon with side S, the distance from center to edge (apothem) is S * sqrt(3)/2.
        # The projection of a vertex onto a normal direction n must be <= apothem.
        # So S >= 2/sqrt(3) * max(|proj|).
        
        # To make this differentiable and optimizable, we can scan phi or optimize it.
        # Since phi is only 1 parameter, we can use a smooth scan or nested optimization.
        # For L-BFGS on the main variables, we can treat phi as part of the state or optimize it separately.
        # Let's optimize phi separately for a given configuration, as it's low-dimensional.
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            
            # Smooth max using log-sum-exp
            # log(sum(exp(x_i))) approximates max(x_i) for large beta
            # We want max(abs_proj). 
            # logsumexp(beta * abs_proj) / beta approximates max(abs_proj)
            scaled = beta * abs_proj
            # Subtract max for numerical stability
            max_scaled = np.max(scaled, axis=0)
            log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled), axis=0))
            smooth_max = log_sum_exp / beta
            
            side_vals = (2.0 / sqrt3) * smooth_max
            return np.max(side_vals)
        
        # Coarse scan for phi
        best_side = float('inf')
        best_phi = 0.0
        # Vectorized coarse scan
        phis = np.linspace(0, 60, 60)
        for phi_deg in phis:
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune phi with minimize_scalar
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-6})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def smooth_objective(params, beta=10.0):
        """
        Objective function for L-BFGS-B.
        params: [x0, y0, a0, x1, y1, a1, ...]
        Returns: outer_side + collision_penalty
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles[i] = params[3*i+2]
        
        # 1. Compute outer side (smooth)
        outer_side, _, _, _ = compute_min_outer_side_smooth(positions, angles, beta)
        
        # 2. Compute collision penalty (smooth)
        # For each pair, compute the minimum distance.
        # If distance < 0, they overlap.
        # We use a log-sum-exp of the negative distances to create a smooth barrier.
        collision_penalty = 0.0
        angles_rad = np.radians(angles)
        
        # Precompute vertices and normals for all hexagons
        all_verts = get_all_vertices(positions, angles_rad)
        # all_verts shape: (N, 6, 2)
        
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        en1x = _enx[None, :] * c[:, None] - _eny[None, :] * s[:, None]
        en1y = _enx[None, :] * s[:, None] + _eny[None, :] * c[:, None]
        # en1x, en1y shape: (N, 6)
        
        for i in range(N):
            for j in range(i+1, N):
                # Separation axes: normals of hex i and hex j
                # Project vertices of both hexagons onto these axes
                # If projections are disjoint on any axis, no collision.
                
                # Axis from hex i
                ni_x = en1x[i]
                ni_y = en1y[i]
                proj_i_i = all_verts[i, :, 0] * ni_x + all_verts[i, :, 1] * ni_y
                proj_j_i = all_verts[j, :, 0] * ni_x + all_verts[j, :, 1] * ni_y
                
                # Axis from hex j
                nj_x = en1x[j]
                nj_y = en1y[j]
                proj_i_j = all_verts[i, :, 0] * nj_x + all_verts[i, :, 1] * nj_y
                proj_j_j = all_verts[j, :, 0] * nj_x + all_verts[j, :, 1] * nj_y
                
                # Calculate overlap depths (positive if overlapping)
                # Overlap on axis n = min(max_i, max_j) - max(min_i, min_j)
                # If negative, they are separated by -overlap.
                
                o1 = min(proj_i_i.max(), proj_j_i.max()) - max(proj_i_i.min(), proj_j_i.min())
                o2 = min(proj_i_j.max(), proj_j_j.max()) - max(proj_i_j.min(), proj_j_j.min())
                
                # If o1 < 0 or o2 < 0, no collision.
                # We want a smooth penalty for o > 0.
                # Actually, if they are separated, o is negative.
                # We want to penalize when o is positive (overlap).
                # Let d = -o. If d < 0, overlap.
                # We can use log(1 + exp(beta * o)) as a smooth hinge loss for o > 0.
                # But we have two axes. The collision exists if BOTH o1 > 0 AND o2 > 0.
                # This is an AND condition.
                # A smooth approximation for "AND" of positive parts is tricky.
                # Alternative: Use the maximum of the "separation distances".
                # Separation distance on axis 1: s1 = max(min_i, min_j) - min(max_i, max_j). If s1 > 0, separated.
                # We want to penalize when s1 < 0 AND s2 < 0.
                
                # Let's use a simpler heuristic: 
                # Penalty = log(1 + exp(beta * max(o1, o2)))? No, that's not right.
                # If o1 is large positive but o2 is large negative, no collision.
                
                # Correct smooth collision check:
                # The hexagons are disjoint if there exists an axis where projections are disjoint.
                # Let gap1 = min(max_i, max_j) - max(min_i, min_j). If gap1 < 0, disjoint on axis 1.
                # Let gap2 = ... If gap2 < 0, disjoint on axis 2.
                # Collision if gap1 >= 0 AND gap2 >= 0.
                
                # We can approximate the indicator of collision with a smooth function.
                # Let f(g1, g2) = 1 if g1>0, g2>0 else 0.
                # Smooth: 1 / (1 + exp(-beta * min(g1, g2)))? 
                # If min(g1, g2) is large positive, min is positive, exp(-large) is small, result ~1.
                # If min(g1, g2) is large negative, min is negative, exp(-large neg) is large, result ~0.
                # This works if both must be positive.
                
                min_gap = min(o1, o2)
                # Smooth hinge for min_gap > 0
                # log(1 + exp(beta * min_gap)) is ~beta*min_gap for large min_gap, ~0 for large neg min_gap.
                # But we want it to be 0 when min_gap < 0.
                # Actually, log(1+exp(x)) is always positive.
                # Let's use: penalty = (1/beta) * log(1 + exp(beta * min_gap))
                # If min_gap is very negative, exp(beta*min_gap) ~ 0, log(1) = 0. Penalty ~ 0.
                # If min_gap is very positive, exp(beta*min_gap) ~ large, log(large) ~ beta*min_gap. Penalty ~ min_gap.
                # This is a smooth approximation of max(0, min_gap).
                
                collision_penalty += (1.0/beta) * math.log(1.0 + math.exp(beta * min_gap))
        
        # Weight the collision penalty heavily to ensure validity
        # If collision_penalty is large, the total objective is large.
        # We want to minimize outer_side, but avoid collisions.
        # If we add penalty, we might find a local min with small penalty but large side.
        # Better: Constrained optimization? L-BFGS-B supports bounds, but not arbitrary constraints easily.
        # We can use a penalty method.
        
        return outer_side + 10.0 * collision_penalty

    def generate_honeycomb(seed_offset=0):
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        patterns = [
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(2,0),(-2,0),(1,-1),(-1,1),(0,2)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,-1),(-2,-1),(1,-1),(-1,-1),(0,2)],
        ]
        idx = (seed_offset) % len(patterns)
        lattice_coords = patterns[idx]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (li, lj) in enumerate(lattice_coords):
            positions[i] = li * a1 + lj * a2
            angles[i] = 0.0
        return positions, angles

    def generate_random_cluster():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        angles[0] = random.uniform(0, 60)
        
        for i in range(1, N):
            for attempt in range(50):
                ref_idx = random.randint(0, i-1)
                dx = random.uniform(-3.5, 3.5)
                dy = random.uniform(-3.5, 3.5)
                positions[i] = positions[ref_idx] + [dx, dy]
                angles[i] = random.uniform(0, 60)
                
                # Quick check: just ensure distance > 2 (circumradius)
                # This is a loose check, L-BFGS will fix it
                valid = True
                for j in range(i):
                    d = math.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    if d < 1.8:
                        valid = False
                        break
                if valid:
                    break
        return positions, angles

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        en1x = _enx[None, :] * c[:, None] - _eny[None, :] * s[:, None]
        en1y = _enx[None, :] * s[:, None] + _eny[None, :] * c[:, None]
        
        for i in range(N):
            for j in range(i+1, N):
                ni_x = en1x[i]
                ni_y = en1y[i]
                proj_i_i = all_verts[i, :, 0] * ni_x + all_verts[i, :, 1] * ni_y
                proj_j_i = all_verts[j, :, 0] * ni_x + all_verts[j, :, 1] * ni_y
                if proj_i_i.max() < proj_j_i.min() or proj_j_i.max() < proj_i_i.min():
                    continue # Disjoint on this axis
                # Check second axis
                nj_x = en1x[j]
                nj_y = en1y[j]
                proj_i_j = all_verts[i, :, 0] * nj_x + all_verts[i, :, 1] * nj_y
                proj_j_j = all_verts[j, :, 0] * nj_x + all_verts[j, :, 1] * nj_y
                if proj_i_j.max() < proj_j_j.min() or proj_j_j.max() < proj_i_j.min():
                    continue # Disjoint on this axis
                # If neither axis separates them, they are likely colliding
                # For strict validity, we should check all 12 axes? 
                # Actually, for two convex polygons, if they are not separated by any edge normal of either, they intersect.
                # So if we didn't continue, they intersect.
                return False
        return True

    # Portfolio approach
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Generate candidates
    candidates = []
    for offset in range(3):
        pos, ang = generate_honeycomb(offset)
        candidates.append((pos, ang))
    for seed in range(5):
        random.seed(seed * 12345)
        pos, ang = generate_random_cluster()
        candidates.append((pos, ang))
        
    # Optimize each candidate
    for i, (pos, ang) in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        init_params = np.zeros(3*N)
        for j in range(N):
            init_params[3*j] = pos[j, 0]
            init_params[3*j+1] = pos[j, 1]
            init_params[3*j+2] = ang[j]
            
        # Use L-BFGS-B
        # Increase beta over time? Start with smooth, end with sharp.
        # For simplicity, use a fixed high beta for the penalty, but the outer side is already smooth.
        # The outer side function uses beta=10 for log-sum-exp.
        # The collision penalty uses beta=10.
        
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9})
            
            if res.success or res.fun < best_side:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for j in range(N):
                    new_pos[j, 0] = res.x[3*j]
                    new_pos[j, 1] = res.x[3*j+1]
                    new_ang[j] = res.x[3*j+2]
                
                # Verify validity
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side_smooth(new_pos, new_ang, beta=50.0)
                    # Refine with higher beta for more accurate side length
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass

    # If no valid solution found, fallback
    if best_positions is None:
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_smooth(best_positions, best_angles, beta=50.0)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
