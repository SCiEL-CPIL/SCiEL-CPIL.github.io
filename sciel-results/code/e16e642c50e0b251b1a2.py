import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side_from_vertices(vertices):
    """
    Compute the minimum side length s of a regular hexagon centered at origin
    (angle 0) that contains all given vertices.
    """
    # Outward normals of a regular hexagon with angle 0
    # Vertices at 0, 60, 120, 180, 240, 300 degrees.
    # Edge k is between v_k and v_{k+1}.
    # Normal to edge k is at angle (k * 60 + 30) degrees.
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0, tol=1e-8):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    # Get normals for both hexagons
    normals = []
    for v in [v1, v2]:
        for k in range(6):
            edge = v[(k+1)%6] - v[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
                
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If there is a gap > tol, they are separated
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
            
    return False

def _is_valid_config(positions, angles, s, tol=1e-6):
    """
    Full validity check.
    """
    N = len(positions)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(angles)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    # Check containment
    all_verts = []
    for i in range(N):
        all_verts.append(_hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]))
    all_verts = np.vstack(all_verts)
    
    outer_verts = _hex_vertices(0, 0, s, 0.0)
    for v in all_verts:
        inside = True
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k+1)%6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -tol:
                inside = False
                break
        if not inside:
            return False
            
    # Check separation
    for i in range(N):
        for j in range(i+1, N):
            if not _check_separation(positions[i], angles[i], positions[j], angles[j], tol=1e-8):
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_score = -1.0
    best_result = None
    
    N = 12
    # Fixed angles for symmetry: 0 degrees for all
    # This is a valid configuration space. The optimal packing might have rotated hexagons,
    # but for 12 hexagons in a hexagon, the "honeycomb" alignment is often optimal or near-optimal.
    # Allowing rotation adds 12 more dimensions. Let's start with fixed angles=0 to keep it simple and robust.
    angles = np.zeros(N)
    
    # Initial positions: Honeycomb cluster
    # Center: (0,0)
    # Ring 1 (6 hexagons): distance sqrt(3)
    # Ring 2 (5 hexagons): distance 2*sqrt(3)
    
    pos0 = np.zeros((N, 2))
    
    # Index 0: Center
    pos0[0] = [0, 0]
    
    # Indices 1-6: Inner ring
    for i in range(6):
        ang = i * np.pi / 3
        pos0[i+1] = [np.sqrt(3) * np.cos(ang), np.sqrt(3) * np.sin(ang)]
        
    # Indices 7-11: Outer ring (5 hexagons)
    # To make it a "cluster", we place them at the gaps of the outer ring of a full hexagon of side 2.
    # Full outer ring would be at distance 2*sqrt(3) at angles 30, 90, ...
    # We pick 5 of them. To keep it compact, maybe pick the ones closest to the inner ring?
    # Actually, just placing them at the standard honeycomb positions for a side-2 hexagon (minus one) is good.
    # Standard side-2 hexagon vertices for centers:
    # (0,0)
    # (sqrt(3)*cos(i*60), sqrt(3)*sin(i*60)) for i=0..5
    # (2*sqrt(3)*cos(i*60+30), 2*sqrt(3)*sin(i*60+30)) for i=0..5
    # Total 19. We need 12.
    # 1 + 6 + 5.
    # Let's place the 5 outer ones at angles 30, 90, 150, 210, 270.
    outer_angles = np.array([30, 90, 150, 210, 270]) * np.pi / 180.0
    for i in range(5):
        pos0[i+7] = [2 * np.sqrt(3) * np.cos(outer_angles[i]), 2 * np.sqrt(3) * np.sin(outer_angles[i])]
        
    # Objective function
    # Minimize s, subject to:
    # 1. All inner hexagons inside outer hexagon of side s.
    # 2. No overlap between inner hexagons.
    
    # We can use a penalty method or directly minimize s with constraints.
    # L-BFGS-B supports inequality constraints? No, only bounds.
    # We can use SLSQP or trust-constr. SLSQP is often faster for small N.
    # Or we can use a soft penalty:
    # Energy = s + lambda * (sum of overlaps + sum of containment violations)
    
    # Let's use a soft penalty approach with L-BFGS-B or Nelder-Mead on the augmented energy.
    # Actually, minimizing s directly with constraints is cleaner.
    # Let's try minimizing s using SLSQP with constraints.
    
    # Variables: x = [s, pos_0..pos_11] (1 + 24 = 25 variables)
    # Or fix s and minimize energy? No, we want to find min s.
    # Let's parameterize by positions only, and compute s as the min containing side.
    # Then minimize s(positions).
    # This is non-smooth (max of linear functions).
    # But we can smooth it or just use a general optimizer.
    
    # Alternative:
    # Minimize E(x) = s(x) + P * (overlap_penalty(x) + containment_penalty(x, s(x)))
    # This is smooth if we use log-sum-exp or squared penalties.
    
    def objective(x):
        s = x[0]
        pos = x[1:].reshape(N, 2)
        
        # 1. Containment Penalty
        # Compute min s needed for this pos
        all_verts = []
        for i in range(N):
            all_verts.append(_hex_vertices(pos[i,0], pos[i,1], 1.0, angles[i]))
        all_verts = np.vstack(all_verts)
        s_needed = _min_outer_side_from_vertices(all_verts)
        
        # Penalty if s < s_needed
        # We want s to be at least s_needed.
        # If s < s_needed, we are infeasible.
        # Let's define the objective as s + penalty.
        # Actually, if we just minimize s, and s is a variable, we need constraints.
        # Let's use the "expand until valid" idea but optimized.
        
        # Let's try: Minimize s, subject to containment and separation.
        # We'll use scipy.optimize.minimize with method='SLSQP' and constraints.
        
        # But SLSQP is slow for 25 vars? 25 is small.
        
        # Let's define the function for SLSQP
        # x = [s, pos_flat]
        
        # Constraints:
        # 1. Separation: For all pairs (i,j), min distance >= 0 (in terms of SAT)
        #    This is hard to formulate as smooth inequality constraints.
        #    We can use the "penalty" approach with a smooth optimizer like L-BFGS-B.
        
        pass

    # Let's stick to the Penalty Method with L-BFGS-B as it's robust for non-smooth-ish landscapes if smoothed.
    # Or better: Use the "Rigid Body Contact Network" idea from the prompt.
    # Minimize total potential energy.
    # E = sum_{pairs} (penetration_depth)^2 + lambda * (containment_violation)^2
    # And we want to minimize s.
    # We can do a binary search on s?
    # Or we can just minimize s directly, treating the constraints as penalties that go to infinity.
    
    # Let's try a different approach:
    # 1. Start with a valid, slightly expanded configuration.
    # 2. Use gradient descent on the positions to minimize the "max distance to boundary" (which determines s).
    #    This is equivalent to minimizing s.
    #    The function f(pos) = s_needed(pos) is convex (max of linear functions).
    #    We can use subgradient methods or a smooth approximation.
    #    L-BFGS-B can handle this if we provide gradients.
    #    Gradient of s_needed: The subgradient is the normal of the edge that is tight.
    #    If multiple edges are tight, it's the average or any convex combination.
    
    # Let's implement a simple gradient descent on positions to minimize s_needed.
    # This is a convex optimization problem (minimizing a convex function).
    # We can use scipy.optimize.minimize with method='L-BFGS-B' and provide gradients.
    
    def s_and_grad(pos_flat):
        pos = pos_flat.reshape(N, 2)
        all_verts = []
        for i in range(N):
            all_verts.append(_hex_vertices(pos[i,0], pos[i,1], 1.0, angles[i]))
        all_verts = np.vstack(all_verts)
        
        # Compute s_needed
        outward_normals = np.array([
            [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
            for k in range(6)
        ])
        dots = all_verts @ outward_normals.T
        max_dot = np.max(dots, axis=1)
        apothem_min = np.max(max_dot)
        s_min = 2.0 * apothem_min / np.sqrt(3.0)
        
        # Gradient
        # s_min = (2/sqrt(3)) * max_k (max_i dot(v_i, n_k))
        # The gradient is non-zero only for the vertices that achieve the max.
        # Let's find the active constraints.
        # This is tricky to compute exactly.
        # For now, let's use numerical gradient or just use a penalty method which is easier to code correctly.
        
        return s_min

    # Given the complexity of exact gradients for the max function, let's use a penalty method with a smooth optimizer.
    # Objective: E(pos) = s_needed(pos) + M * (sum of overlaps)^2
    # We want to minimize s_needed.
    # If we just minimize s_needed, we might get overlaps.
    # So we must enforce separation.
    
    # Strategy:
    # 1. Use a penalty method to find a valid configuration with minimal s.
    # 2. Start with a valid configuration (expanded).
    # 3. Gradually increase the penalty for overlaps while minimizing s.
    
    # Let's define the energy function.
    def energy(pos_flat, penalty_weight):
        pos = pos_flat.reshape(N, 2)
        
        # 1. s_needed
        all_verts = []
        for i in range(N):
            all_verts.append(_hex_vertices(pos[i,0], pos[i,1], 1.0, angles[i]))
        all_verts = np.vstack(all_verts)
        
        outward_normals = np.array([
            [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
            for k in range(6)
        ])
        dots = all_verts @ outward_normals.T
        max_dot = np.max(dots, axis=1)
        apothem_min = np.max(max_dot)
        s_min = 2.0 * apothem_min / np.sqrt(3.0)
        
        # 2. Overlap Penalty
        # For each pair, compute penetration depth.
        # Penetration depth is not trivially defined for hexagons.
        # We can use the SAT gap. If gap < 0, they overlap.
        # Let's approximate overlap by the sum of squared negative gaps on all axes.
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = _hex_vertices(pos[i,0], pos[i,1], 1.0, angles[i])
                v2 = _hex_vertices(pos[j,0], pos[j,1], 1.0, angles[j])
                
                # Normals
                normals = []
                for v in [v1, v2]:
                    for k in range(6):
                        edge = v[(k+1)%6] - v[k]
                        n = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-12:
                            n = n / norm
                            normals.append(n)
                            
                for n in normals:
                    proj1 = v1 @ n
                    proj2 = v2 @ n
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    gap = min(max1 - min2, max2 - min1)
                    if gap < 0:
                        overlap_pen += gap * gap
                        
        return s_min + penalty_weight * overlap_pen

    # Optimize
    x0 = np.concatenate(([10.0], pos0.flatten())) # Start with large s? No, s is not a variable in this energy.
    # Wait, s is determined by pos. So we just optimize pos.
    x0 = pos0.flatten()
    
    # Use L-BFGS-B
    # We need to be careful with the penalty weight.
    # Start with low weight, increase it.
    
    penalty_weight = 1.0
    current_pos = x0.copy()
    
    for _ in range(10):
        if time.time() > deadline - 20:
            break
            
        # Optimize with current penalty
        try:
            res = minimize(energy, current_pos, args=(penalty_weight,), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            if res.success:
                current_pos = res.x
            else:
                # If it fails, maybe increase penalty
                pass
        except Exception:
            pass
            
        # Check validity
        pos = current_pos.reshape(N, 2)
        all_verts = []
        for i in range(N):
            all_verts.append(_hex_vertices(pos[i,0], pos[i,1], 1.0, angles[i]))
        all_verts = np.vstack(all_verts)
        s_needed = _min_outer_side_from_vertices(all_verts)
        
        # Check separation
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not _check_separation(pos[i], angles[i], pos[j], angles[j]):
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            score = 1.0 / s_needed
            if score > best_score:
                best_score = score
                best_result = (pos, angles, s_needed)
                
        # Increase penalty
        penalty_weight *= 2.0
        
    # Fallback
    if best_result is None:
        # Use the initial configuration
        pos = pos0
        all_verts = []
        for i in range(N):
            all_verts.append(_hex_vertices(pos[i,0], pos[i,1], 1.0, angles[i]))
        all_verts = np.vstack(all_verts)
        s = _min_outer_side_from_vertices(all_verts)
        # Expand if needed
        scale = 1.0
        while not _is_valid_config(pos * scale, angles, s * scale) and scale < 10:
            scale += 0.1
        if _is_valid_config(pos * scale, angles, s * scale):
            best_result = (pos * scale, angles, s * scale)
            best_score = 1.0 / (s * scale)
        else:
            pos = np.random.uniform(-5, 5, (N,2))
            s = 10.0
            while not _is_valid_config(pos, angles, s):
                s += 0.5
            best_result = (pos, angles, s)
            best_score = 1.0/s

    pos_final, ang_final, s_final = best_result
    inner_hex_data = np.column_stack([pos_final, ang_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final
