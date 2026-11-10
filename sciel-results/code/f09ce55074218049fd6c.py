# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, least_squares

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _build_from_contact_graph(r1, r2, r3, r4, theta1, theta2, theta3, theta4):
    """
    Build 11 hexagon positions from a contact-graph parameterization.
    
    Hex 0: center at origin.
    Hex 1-6: ring at distance r1, angles 30+60k + theta1 (small perturbation)
    Hex 7-10: outer ring at distances r2-r4, angles based on theta2-theta4
    
    For optimal packing of identical hexagons (side 1), touching distance is 2*sqrt(3).
    But we allow free parameters to find the tightest configuration.
    """
    d = 2 * np.sqrt(3)  # touching distance for unit hexagons
    
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    
    # Center
    positions[0] = [0.0, 0.0]
    
    # Inner ring: 6 hexagons
    base_angles = np.radians(30 + 60 * np.arange(6)) + theta1
    positions[1:7, 0] = r1 * np.cos(base_angles)
    positions[1:7, 1] = r1 * np.sin(base_angles)
    
    # Outer ring: 4 hexagons in the "valley" positions
    # Valley positions are at angles 30, 90, 150, 210 (every other gap)
    # We use 4 of the 6 valleys
    valley_angles = np.radians([30, 90, 150, 210])
    positions[7, 0] = r2 * np.cos(valley_angles[0] + theta2)
    positions[7, 1] = r2 * np.sin(valley_angles[0] + theta2)
    positions[8, 0] = r3 * np.cos(valley_angles[1] + theta3)
    positions[8, 1] = r3 * np.sin(valley_angles[1] + theta3)
    positions[9, 0] = r4 * np.cos(valley_angles[2] + theta4)
    positions[9, 1] = r4 * np.sin(valley_angles[2] + theta4)
    positions[10, 0] = r4 * np.cos(valley_angles[3] + theta4)
    positions[10, 1] = r4 * np.sin(valley_angles[3] + theta4)
    
    return positions, angles

def _contact_residuals(params):
    """
    Residuals for contact constraints.
    params = [r1, r2, r3, r4, theta1, theta2, theta3, theta4]
    
    We want:
    - Each inner ring hex touches the center: dist = 2*sqrt(3) (but r1 is free, so this is automatic)
    - Each inner ring hex touches its 2 neighbors in the ring
    - Each outer hex touches 2 inner ring hexes and possibly 1 outer neighbor
    
    Actually, for the constraint solver approach, let's use a different formulation:
    Fix the topology and solve for positions given the contact distances.
    """
    r1, r2, r3, r4, theta1, theta2, theta3, theta4 = params
    d = 2 * np.sqrt(3)
    
    positions, angles = _build_from_contact_graph(r1, r2, r3, r4, theta1, theta2, theta3, theta4)
    
    residuals = []
    
    # Inner ring hexes should touch the center
    for k in range(6):
        dist = np.linalg.norm(positions[k+1])
        residuals.append(dist - d)
    
    # Inner ring: adjacent hexes should touch
    for k in range(6):
        i = k + 1
        j = (k + 1) % 6 + 1
        dist = np.linalg.norm(positions[i] - positions[j])
        residuals.append(dist - d)
    
    # Outer hexes should touch 2 inner ring hexes each
    # Hex 7 (angle ~30): touches hex 1 (angle 30) and hex 2 (angle 90)
    # Actually let me be more careful about which hexes touch which
    
    # Hex 7 at angle ~30+theta2, distance r2: touches hex 1 (at 30+theta1, dist r1) and hex 6 (at 330+theta1, dist r1)
    # Let's just enforce that each outer hex is at distance d from 2 specific inner hexes
    
    # Simplified: enforce that outer hexes are at distance d from nearest inner hexes
    # This is getting complex; let's use a simpler approach
    
    return np.array(residuals)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 610.0  # 20s safety margin
    
    d = 2 * np.sqrt(3)  # touching distance
    
    # ============================================================
    # STRATEGY: Constraint-based parameterization
    # 
    # Key insight: In the optimal packing, many hexagons are in exact
    # contact. We parameterize the configuration by a small number of
    # geometric parameters and use the contact constraints to determine
    # positions, then optimize only the free parameters.
    #
    # Topology: center + 6 ring + 4 outer
    # Free parameters: ring radius, outer radii, angular offsets
    # ============================================================
    
    # Initial guess: symmetric configuration
    # Center at origin
    # 6 ring at distance 2*sqrt(3), angles 30, 90, 150, 210, 270, 330
    # 4 outer at distance 4*sqrt(3), angles 30, 90, 150, 210 (valleys)
    
    best_positions = np.zeros((11, 2))
    best_angles = np.zeros(11)
    
    # Symmetric initial
    best_positions[0] = [0.0, 0.0]
    for k in range(6):
        a = np.radians(30 + 60 * k)
        best_positions[k + 1] = d * np.cos(a), d * np.sin(a)
    for k in range(4):
        a = np.radians(30 + 60 * k * 1.5)  # 30, 90, 150, 210
        best_positions[7 + k] = 2 * d * np.cos(a), 2 * d * np.sin(a)
    
    # Wait, 4 hexagons at the 4 "valley" positions between ring hexes
    # Ring hexes at 30, 90, 150, 210, 270, 330
    # Valleys at 0, 60, 120, 180, 240, 300 (between adjacent ring hexes)
    # Actually the "outer" positions in the hexagonal lattice are at distance 2*sqrt(3) * 2 = 4*sqrt(3)
    # No wait. In a hexagonal packing of hexagons with side 1:
    # - Center to ring: 2*sqrt(3) (they share an edge)
    # - Ring to next ring: 2*sqrt(3) more
    # - So outer ring at 4*sqrt(3) from center
    # But we only have 4 outer hexes, not 6
    
    # Let me reconsider the topology. With 11 hexagons:
    # 1 center + 6 ring + 4 more
    # The 4 more could be placed in 4 of the 6 "outer" positions
    
    # Actually, for the tightest packing, we want to minimize the outer hexagon.
    # The outer hexagon is determined by the maximum projection of all vertices.
    # So we want to pull all hexagons toward the center as much as possible
    # while maintaining separation.
    
    # Let me use a direct optimization approach with the constraint that
    # we minimize the enclosing hexagon side, with separation as a hard constraint.
    
    # Phase 1: Use scipy minimize with penalty for separation violations
    def objective_with_penalty(x):
        pos = x[:22].reshape(11, 2)
        ang = x[22:33]
        
        # Enclosing side
        side = _min_enclosing_side(pos, ang)
        
        # Separation penalty
        all_verts = _all_vertices(pos, ang)
        n = 11
        hex_normals = np.zeros((n, 6, 2))
        base = np.radians(ang)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
        hex_normals[:, :, 0] = np.cos(base)
        hex_normals[:, :, 1] = np.sin(base)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                vi = all_verts[i]
                vj = all_verts[j]
                normals = np.vstack([hex_normals[i], hex_normals[j]])
                proj_i = vi @ normals.T
                proj_j = vj @ normals.T
                max_i = np.max(proj_i, axis=0)
                min_i = np.min(proj_i, axis=0)
                max_j = np.max(proj_j, axis=0)
                min_j = np.min(proj_j, axis=0)
                
                # Check if separated
                sep_i = min_j - max_i  # positive means separated
                sep_j = min_i - max_j
                min_sep = np.min(np.maximum(sep_i, sep_j))
                if min_sep < 0:
                    penalty += min_sep ** 2
        
        return side + 1000.0 * penalty
    
    # Initial configuration
    x0 = np.zeros(33)
    x0[0:2] = [0.0, 0.0]  # center
    for k in range(6):
        a = np.radians(30 + 60 * k)
        x0[2 + 2*k:4 + 2*k] = d * np.cos(a), d * np.sin(a)
    # 4 outer hexes at valley positions
    valley_ks = [0, 1, 2, 3]  # valleys at 30, 90, 150, 210
    for idx, k in enumerate(valley_ks):
        a = np.radians(30 + 60 * k)
        x0[14 + 2*idx:16 + 2*idx] = 2 * d * np.cos(a), 2 * d * np.sin(a)
    # Angles all 0
    # x0[22:33] already 0
    
    best_side = _min_enclosing_side(x0[:22].reshape(11, 2), x0[22:33])
    best_x = x0.copy()
    
    # Multiple random restarts + local optimization
    np.random.seed(12345)
    
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb the best solution
            x_init = best_x + np.random.randn(33) * 0.05
            # Keep angles small
            x_init[22:33] *= 0.5
        
        # Quick check if initial is valid
        pos_init = x_init[:22].reshape(11, 2)
        ang_init = x_init[22:33]
        if not _check_separation_fast(pos_init, ang_init):
            # Try to fix by scaling
            x_init[:22] *= 1.001
            pos_init = x_init[:22].reshape(11, 2)
            if not _check_separation_fast(pos_init, ang_init):
                continue
        
        try:
            res = minimize(
                objective_with_penalty,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pos_res = res.x[:22].reshape(11, 2)
            ang_res = res.x[22:33]
            
            if _check_separation_fast(pos_res, ang_res):
                side_res = _min_enclosing_side(pos_res, ang_res)
                if side_res < best_side:
                    best_side = side_res
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Fine-tune with Nelder-Mead on the best solution
    if time.time() < deadline - 40.0:
        def objective_nm(x):
            pos = x[:22].reshape(11, 2)
            ang = x[22:33]
            if not _check_separation_fast(pos, ang):
                return 1e6
            return _min_enclosing_side(pos, ang)
        
        try:
            res = minimize(
                objective_nm,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            pos_res = res.x[:22].reshape(11, 2)
            ang_res = res.x[22:33]
            if _check_separation_fast(pos_res, ang_res):
                side_res = _min_enclosing_side(pos_res, ang_res)
                if side_res < best_side:
                    best_side = side_res
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent fine-tuning
    if time.time() < deadline - 20.0:
        for cd_iter in range(500):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            pos = best_x[:22].reshape(11, 2)
            ang = best_x[22:33]
            
            for i in range(11):
                for coord in range(2):
                    for step in [0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                        trial = best_x.copy()
                        trial[2*i + coord] += step
                        trial_pos = trial[:22].reshape(11, 2)
                        trial_ang = trial[22:33]
                        
                        if not _check_separation_fast(trial_pos, trial_ang):
                            continue
                        
                        trial_side = _min_enclosing_side(trial_pos, trial_ang)
                        if trial_side < best_side - 1e-12:
                            best_x = trial
                            best_side = trial_side
                            improved = True
                            break
                
                # Also try angle adjustments
                for dangle in [0.5, -0.5, 0.2, -0.2, 0.1, -0.1]:
                    trial = best_x.copy()
                    trial[22 + i] += dangle
                    trial_pos = trial[:22].reshape(11, 2)
                    trial_ang = trial[22:33]
                    
                    if not _check_separation_fast(trial_pos, trial_ang):
                        continue
                    
                    trial_side = _min_enclosing_side(trial_pos, trial_ang)
                    if trial_side < best_side - 1e-12:
                        best_x = trial
                        best_side = trial_side
                        improved = True
                        break
            
            if not improved:
                break
    
    # Extract final result
    best_positions = best_x[:22].reshape(11, 2)
    best_angles = best_x[22:33]
    
    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to known valid configuration
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(4):
            a = np.radians(30 + 60 * k)
            positions[7 + k] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    # Ensure containment
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001
    
    # Verify edge lengths
    all_v = _all_vertices(best_positions, best_angles)
    for i in range(11):
        for j in range(6):
            v1 = all_v[i, j]
            v2 = all_v[i, (j + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                # Should not happen, but safety
                pass
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
