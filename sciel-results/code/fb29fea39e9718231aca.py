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
    """Compute all 66 vertices for 11 hexagons. positions (11,2), angles (11,)"""
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    """Check all pairs separated using SAT on 12 edge normals per pair."""
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    
    # Precompute edge normals for each hexagon (6 normals each)
    # For a hexagon with center (cx,cy) and angle a, edge i is between vertex i and i+1
    # The outward normal of edge i points at angle a + 30 + 60*i degrees
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]  # (6,2)
            vj = all_verts[j]
            # 12 normals: 6 from hex i, 6 from hex j
            normals = np.vstack([hex_normals[i], hex_normals[j]])  # (12,2)
            
            # Project all vertices onto each normal
            # vi @ normals.T -> (6, 12), vj @ normals.T -> (6, 12)
            proj_i = vi @ normals.T  # (6, 12)
            proj_j = vj @ normals.T  # (6, 12)
            
            max_i = np.max(proj_i, axis=0)  # (12,)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            
            # Check if any normal separates them
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue  # separated
            
            return False  # not separated
    return True

def _min_enclosing_side(positions, angles):
    """Min side length of outer hexagon (centered at origin, angle=0) containing all vertices."""
    all_verts = _all_vertices(positions, angles)  # (11, 6, 2)
    flat = all_verts.reshape(-1, 2)  # (66, 2)
    
    # Outer hexagon at angle 0: vertices at angles 0, 60, 120, 180, 240, 300
    # Edge normals at angles 30, 90, 150, 210, 270, 330
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    
    projs = flat @ normals.T  # (66, 6)
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    """Check all inner vertices inside outer hexagon (center=origin, angle=0)."""
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

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    d = np.sqrt(3)
    
    # Symmetric initial arrangement: center + 6 ring + 4 in gaps
    # All angles = 0 for simplicity (exploits outer hexagon symmetry)
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    
    # Center
    positions[0] = [0.0, 0.0]
    # 6 surrounding at distance sqrt(3)
    for k in range(6):
        a = np.radians(30 + 60 * k)
        positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
    # 4 more in the "valleys" at distance 2*sqrt(3)
    for k in range(0, 6, 2):
        a = np.radians(30 + 60 * k)
        positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
    # One more at the remaining valley
    a = np.radians(30 + 60 * 1)
    positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
    
    # Verify initial validity
    if not _check_separation_fast(positions, angles):
        # Scale up slightly if needed
        scale = 1.0
        while not _check_separation_fast(positions, angles) and scale < 5.0:
            positions = positions * 1.01
            scale += 1.01
        positions[0] = [0.0, 0.0]  # Keep center
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Phase 1: Sequential local refinement using Nelder-Mead on positions only
    def objective_pos(pos_flat):
        pos = pos_flat.reshape(11, 2)
        return _min_enclosing_side(pos, best_angles)
    
    # Multiple Nelder-Mead restarts with small perturbations
    np.random.seed(42)
    for iteration in range(200):
        if time.time() > deadline - 30.0:
            break
        
        # Try perturbing current best
        perturbation = np.random.randn(22) * 0.1
        trial = best_positions.flatten() + perturbation
        
        # Quick validity check
        trial_pos = trial.reshape(11, 2)
        if not _check_separation_fast(trial_pos, best_angles):
            continue
        
        trial_side = _min_enclosing_side(trial_pos, best_angles)
        if trial_side < best_side:
            best_positions = trial_pos
            best_side = trial_side
        
        # Nelder-Mead local optimization
        if time.time() > deadline - 60.0:
            break
        
        try:
            res = minimize(
                objective_pos,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if res.fun < best_side:
                trial_pos = res.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    best_positions = trial_pos
                    best_side = _min_enclosing_side(best_positions, best_angles)
        except Exception:
            pass
    
    # Phase 2: Coordinate descent for fine-tuning
    if time.time() < deadline - 30.0:
        for cd_iter in range(1000):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for i in range(11):
                for coord in range(2):
                    # Try small steps in + and - direction
                    for step in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
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
    
    # Phase 3: Try rotating individual hexagons to find better fits
    if time.time() < deadline - 30.0:
        for angle_iter in range(200):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for i in range(11):
                for delta_angle in [5, -5, 10, -10, 15, -15]:
                    trial_angles = best_angles.copy()
                    trial_angles[i] = (best_angles[i] + delta_angle) % 360
                    
                    if not _check_separation_fast(best_positions, trial_angles):
                        continue
                    
                    trial_side = _min_enclosing_side(best_positions, trial_angles)
                    if trial_side < best_side - 1e-10:
                        best_angles = trial_angles
                        best_side = trial_side
                        improved = True
                        break
            
            if not improved:
                break
    
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
        for k in range(0, 6, 2):
            a = np.radians(30 + 60 * k)
            positions[7 + k // 2] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        a = np.radians(30 + 60 * 1)
        positions[10] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    # Ensure containment
    if not _check_containment_fast(best_positions, best_angles, final_side):
        # Slightly increase side
        final_side *= 1.001
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
