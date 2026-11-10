# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _build_symmetric_positions(ring_r, valley_r, angle_offset=0.0, ring_angle=0.0, valley_angle=0.0):
    """Build 11 hexagon positions using 6-fold symmetry.
    - 1 center at origin
    - 6 ring hexagons at radius ring_r, angles 30+60k + angle_offset
    - 4 valley hexagons at radius valley_r, angles 30+60*(2k) + angle_offset (k=0,1,2,3 -> 4 positions)
    Wait: 6-fold symmetry with 11 = 1 + 6 + 4 doesn't work perfectly.
    Let me reconsider: 1 center + 6 ring + 4 extra.
    Actually for 11: center(1) + ring(6) + outer(4). The 4 outer break 6-fold symmetry.
    Let's use: center(1) + 6 ring + 4 at specific angles.
    """
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    
    # Center
    positions[0] = [0.0, 0.0]
    angles[0] = angle_offset
    
    # 6 ring hexagons
    for k in range(6):
        a = np.radians(30 + 60 * k + angle_offset)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
        angles[k + 1] = ring_angle
    
    # 4 valley hexagons - place at angles 30, 90, 150, 210 (every other valley)
    # Actually, to maintain some symmetry, place at 30+60*2k for k=0,1,2 and one more
    # Let's place 4 at: 30, 150, 270, and 90 (or similar)
    # For better symmetry: place at 30+60*2k for k=0,1,2 (3 positions) + 1 at 90
    # Actually let's just use a pattern: 30, 90, 150, 210
    valley_angles_deg = [30, 90, 150, 210]
    for k, va in enumerate(valley_angles_deg):
        a = np.radians(va + angle_offset)
        positions[7 + k] = [valley_r * np.cos(a), valley_r * np.sin(a)]
        angles[7 + k] = valley_angle
    
    # We have 1+6+4 = 11
    # But we only assigned 7+4=11, and positions[10] is the 4th valley
    # Let me recount: positions[0]=center, positions[1:7]=ring(6), positions[7:11]=valley(4)
    # That's 1+6+4=11. Good.
    
    return positions, angles

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    d = np.sqrt(3)
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    # Phase 1: Dense grid search over symmetric parameters
    # ring_r: distance from center to ring hexagons (must be >= sqrt(3) for separation)
    # valley_r: distance from center to valley hexagons
    # angle_offset: rotational offset of the whole pattern
    
    # Theoretical minimum ring_r = sqrt(3) (touching center)
    # Valley hexagons need to be separated from ring hexagons
    
    ring_r_range = np.linspace(1.732, 2.5, 40)  # sqrt(3) ≈ 1.732
    valley_r_range = np.linspace(2.5, 4.5, 30)
    angle_offset_range = np.linspace(0, 30, 15)
    
    for ar in ring_r_range:
        if time.time() > deadline - 120:
            break
        for vr in valley_r_range:
            if time.time() > deadline - 120:
                break
            for ao in angle_offset_range:
                if time.time() > deadline - 120:
                    break
                
                positions, angles = _build_symmetric_positions(ar, vr, ao, 0.0, 0.0)
                
                if not _check_separation_fast(positions, angles):
                    continue
                
                side = _min_enclosing_side(positions, angles)
                if side < best_side:
                    best_side = side
                    best_positions = positions.copy()
                    best_angles = angles.copy()
    
    # Phase 2: Refine with Nelder-Mead on the reduced parameter space
    if best_positions is not None and time.time() < deadline - 60:
        # Extract current best parameters
        # We need to map back to (ring_r, valley_r, angle_offset)
        # For now, do a local search on positions directly but with symmetry constraints
        
        def objective_sym(params):
            ring_r = params[0]
            valley_r = params[1]
            angle_offset = params[2]
            ring_angle = params[3]
            valley_angle = params[4]
            
            positions, angles = _build_symmetric_positions(ring_r, valley_r, angle_offset, ring_angle, valley_angle)
            
            if not _check_separation_fast(positions, angles):
                return 1e10
            
            return _min_enclosing_side(positions, angles)
        
        # Get initial params from best
        # Estimate ring_r and valley_r from best_positions
        ring_dists = np.linalg.norm(best_positions[1:7], axis=1)
        valley_dists = np.linalg.norm(best_positions[7:11], axis=1)
        init_ring_r = np.mean(ring_dists)
        init_valley_r = np.mean(valley_dists)
        init_params = [init_ring_r, init_valley_r, 0.0, 0.0, 0.0]
        
        try:
            res = minimize(
                objective_sym,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_side and res.fun < 1e9:
                ring_r, valley_r, angle_offset, ring_angle, valley_angle = res.x
                positions, angles = _build_symmetric_positions(ring_r, valley_r, angle_offset, ring_angle, valley_angle)
                if _check_separation_fast(positions, angles):
                    side = _min_enclosing_side(positions, angles)
                    if side < best_side:
                        best_side = side
                        best_positions = positions.copy()
                        best_angles = angles.copy()
        except Exception:
            pass
    
    # Phase 3: Asymmetric refinement - allow each hexagon to move independently
    if best_positions is not None and time.time() < deadline - 30:
        # Use Nelder-Mead on full 22-dim position space starting from symmetric solution
        def objective_full(pos_flat):
            pos = pos_flat.reshape(11, 2)
            if not _check_separation_fast(pos, best_angles):
                return 1e10
            return _min_enclosing_side(pos, best_angles)
        
        try:
            res = minimize(
                objective_full,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_side and res.fun < 1e9:
                trial_pos = res.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    side = _min_enclosing_side(trial_pos, best_angles)
                    if side < best_side:
                        best_side = side
                        best_positions = trial_pos
        except Exception:
            pass
    
    # Phase 4: Fine coordinate descent
    if best_positions is not None and time.time() < deadline - 10:
        for cd_iter in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(11):
                for coord in range(2):
                    for step in [0.005, -0.005, 0.001, -0.001]:
                        trial = best_positions.copy()
                        trial[i, coord] += step
                        if not _check_separation_fast(trial, best_angles):
                            continue
                        trial_side = _min_enclosing_side(trial, best_angles)
                        if trial_side < best_side - 1e-10:
                            best_positions = trial
                            best_side = trial_side
                            improved = True
                            break
            if not improved:
                break
    
    # Fallback if nothing found
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(0, 6, 2):
            a = np.radians(30 + 60 * k)
            positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        a = np.radians(30 + 60 * 1)
        positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Final verification and containment
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(0, 6, 2):
            a = np.radians(30 + 60 * k)
            positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        a = np.radians(30 + 60 * 1)
        positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
