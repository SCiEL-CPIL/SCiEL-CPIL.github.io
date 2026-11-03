import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Unit hexagon vertices (pointy-top: vertices at 30, 90, 150, 210, 270, 330)
    # This orientation aligns flat sides with horizontal/vertical axes? 
    # Actually, let's use standard vertices at angles 0, 60, 120, 180, 240, 300.
    # This makes the "width" along x-axis = 2, and "height" along y-axis = sqrt(3).
    # Wait, for a regular hexagon with side 1:
    # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This is "pointy" on x-axis.
    
    unit_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] 
        for a in range(0, 360, 60)
    ])
    
    # Precompute rotation matrices for efficient vertex calculation
    def get_rot_matrix(angle_deg):
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[c, -s], [s, c]])
    
    def get_hex_verts(cx, cy, angle_deg):
        R = get_rot_matrix(angle_deg)
        return unit_verts @ R.T + np.array([cx, cy])

    # Minkowski Sum / Difference Body Approach
    # For two unit hexagons H1 (angle a1) and H2 (angle a2), 
    # they are disjoint if dist(center1, center2) >= support_function(H1 - H2, direction)
    # Actually, a more robust way for optimization is to use the support function of the Minkowski Sum.
    # H1 and H2 are disjoint iff for ALL directions u, u . (c1 - c2) >= h_{H1}(u) + h_{H2}(-u) ? No.
    # Disjoint iff c1 - c2 is outside H1 - H2 (Minkowski Difference).
    # Since H1 and H2 are convex, H1 - H2 is convex.
    # We can enforce that the origin is not in (c2 - c1) + H1 - H2 ? 
    # Let's use the condition: c1 - c2 is outside (H1 - H2).
    # This is equivalent to: There exists a separating axis? No, that's SAT.
    # For convex polygons, A and B are disjoint iff max_{u in unit_circle} (u.(c1-c2) - h_A(u) - h_B(u)) > 0 ?
    # Actually, the distance between two convex sets is given by min ||x-y||.
    # A simpler sufficient condition for disjointness of two regular hexagons:
    # If we fix the relative angle delta = a1 - a2, the Minkowski Sum H1 + H2 (or difference) is a fixed convex polygon.
    # Let D = H1 - H2. D is a convex polygon with at most 12 vertices.
    # c1 - c2 must be outside D.
    # This is equivalent to: c1 - c2 must satisfy the linear inequalities defining the exterior of D.
    # However, the "exterior" of a convex polygon is non-convex.
    # But we can assume a specific separation direction? No, they can separate in any direction.
    
    # Alternative: Use the fact that for regular hexagons, the "worst case" overlap occurs when they are aligned or at specific angles.
    # Let's use a differentiable approximation of the non-overlap constraint.
    # The gap between two convex polygons P1 and P2 is min_{u} (u.(c1-c2) - h_P1(u) - h_P2(u)).
    # We want this gap to be >= 0.
    # This is a max-min problem.
    # We can approximate the support function h_P(u) by its maximum over the vertices.
    # h_P(u) = max_{v in vertices(P)} u.v
    # So gap = min_{u} (u.(c1-c2) - max_{v1} u.v1 - max_{v2} u.v2)
    # This is still non-smooth.
    
    # Let's try a different structural change: 
    # Instead of optimizing positions directly, we optimize the **lattice parameters** or **symmetry groups** using a differentiable objective.
    # The current program uses Nelder-Mead on symmetric params. 
    # Let's use **Gradient Descent with analytic gradients** on the symmetric parameters.
    # Objective: Minimize Outer Side Length.
    # Constraint: Min Gap >= 0.
    
    # To make it differentiable, we can use a "soft" gap constraint or simply use the analytic gradient of the outer side length
    # and rely on the fact that if we start from a valid configuration and move slowly, we might stay valid? 
    # No, that's risky.
    
    # Let's stick to the Minkowski Sum idea but implement it correctly for the specific case of regular hexagons.
    # For two regular hexagons of side 1, the Minkowski Sum H1 + H2 is a regular dodecagon if they are aligned?
    # If rotated by 30 deg, it's a regular hexagon?
    # Actually, the support function of a regular hexagon with side 1 and angle alpha is:
    # h(u) = cos(pi/6) * |u|_hex? No.
    
    # Let's use a robust differentiable metric:
    # The "minimum enclosing hexagon" side length is a convex function of the positions?
    # Yes, the support function is convex, and the max of convex functions is convex.
    # So minimizing the outer side length is a Convex Optimization problem!
    # The only non-convex part is the disjointness constraint.
    
    # Strategy:
    # 1. Define a parameterization that guarantees disjointness by construction? 
    #    E.g., place hexagons on a hexagonal lattice with spacing > 2*sqrt(3)/2?
    #    If we place them on a lattice with spacing L, they are disjoint if L >= width across flats = sqrt(3).
    #    If we use a lattice with spacing L > sqrt(3), they are strictly disjoint.
    #    We can then optimize the lattice scaling and rotation to minimize the bounding hexagon.
    #    This reduces the problem to optimizing a few lattice parameters (scale, rotation, shear) + offsets.
    
    # Let's try: **Lattice-Based Packing with Analytic Gradients**
    # Assume the 12 hexagons are arranged in a parallelogram or hexagonal cluster on a lattice.
    # For 12 items, a 3x4 rectangular grid or a hexagonal cluster of radius 2 (1+6+9? No, 1+6=7, 1+6+12=19).
    # 12 items: 
    #   Option A: 2 rows of 6? 
    #   Option B: Hexagonal cluster: Center + 6 neighbors + 5 more?
    #   Option C: 3x4 grid.
    
    # Let's define a generic "Cluster" parameterization.
    # We define a set of relative offsets (dx, dy) in a local coordinate system.
    # We apply a linear transformation (Scale, Rotation, Shear) to these offsets.
    # This guarantees that if the base offsets are disjoint, the transformed ones are disjoint?
    # No, scaling down makes them overlap.
    # So we must ensure the base configuration is "tight" and then we optimize the transformation.
    
    # Better: Use **Free Placement** but with **Analytic Gradients** for the Objective and a **Differentiable Penalty** for Constraints.
    # Penalty: sum(max(0, -gap_ij)^2).
    # Gap_ij can be approximated by the distance between centers minus the sum of inradii? 
    # That's too loose.
    
    # Let's go back to the **Minkowski Sum** but compute it numerically for the current angles, 
    # and then use the linear inequalities of the resulting polygon to define the constraint.
    # This is effectively SAT, but we can make the "gap" function differentiable by using a soft-min over the axes.
    
    # Differentiable Gap Function:
    # Gap(P1, P2) = min_{axis in SAT_axes} (proj_diff(axis))
    # We approximate min with a soft-min: -log(sum(exp(-k * proj_diff))) / k
    # This is smooth and differentiable.
    
    def soft_min(vals, k=100.0):
        # vals: array of projection gaps
        # We want min(vals). 
        # soft_min = -1/k * log(sum(exp(-k * vals)))
        # To avoid overflow, shift by max
        m = np.max(vals)
        return m - (1.0/k) * np.log(np.sum(np.exp(-k * (vals - m))))

    def get_sat_axes(poly1, poly2):
        axes = []
        for i in range(6):
            e1 = poly1[(i+1)%6] - poly1[i]
            axes.append(np.array([-e1[1], e1[0]]))
            e2 = poly2[(i+1)%6] - poly2[i]
            axes.append(np.array([-e2[1], e2[0]]))
        # Normalize
        norms = np.linalg.norm(axes, axis=1)
        norms[norms < 1e-9] = 1.0
        return np.array(axes) / norms[:, np.newaxis]

    def calc_gap_and_grad(params):
        """
        Returns the minimum gap (positive if disjoint) and its gradient.
        params: [x1, y1, a1, ..., x12, y12, a12]
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = []
        for i in range(N):
            polys.append(get_hex_verts(positions[i][0], positions[i][1], angles[i]))
            
        min_gap_val = float('inf')
        grad = np.zeros_like(params)
        
        # We only need to check pairs that are close? For N=12, 66 pairs is small.
        for i in range(N):
            for j in range(i+1, N):
                p1 = polys[i]
                p2 = polys[j]
                axes = get_sat_axes(p1, p2)
                
                # Projection gaps for each axis
                # gap = min(max1 - min2, max2 - min1) ? No, SAT says if any axis has separation, they are disjoint.
                # The "gap" is the minimum separation over all axes.
                # Separation on axis u: 
                # proj1 = p1 @ u, proj2 = p2 @ u
                # sep = max(min(proj2) - max(proj1), min(proj1) - max(proj2))
                # If sep > 0, disjoint.
                
                seps = []
                # We need gradients of seps w.r.t params.
                # This is complex. Let's use numerical gradients for the penalty?
                # Or, we can just use the analytic gradient of the Objective (Outer Side) 
                # and use a **Feasibility Pump**: 
                # 1. Optimize Objective with SLSQP using numerical Jacobian for constraints.
                # 2. The "Jacobian" of the gap is hard, but SLSQP can handle finite differences.
                
                # Let's just return the gap value for the constraint. SLSQP will finite-difference it.
                # To speed up, we can provide an analytic gradient for the Objective.
                
                proj1 = p1 @ axes.T
                proj2 = p2 @ axes.T
                min1, max1 = np.min(proj1, axis=0), np.max(proj1, axis=0)
                min2, max2 = np.min(proj2, axis=0), np.max(proj2, axis=0)
                
                sep1 = min2 - max1
                sep2 = min1 - max2
                # The actual separation is max(sep1, sep2) for a given axis? 
                # No, if they overlap on this axis, both are negative. 
                # The gap on this axis is max(sep1, sep2).
                # The total gap is min(axis_gaps).
                
                axis_gaps = np.maximum(sep1, sep2)
                current_gap = np.min(axis_gaps)
                
                if current_gap < min_gap_val:
                    min_gap_val = current_gap
                    
        return min_gap_val

    def objective_outer_side(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Outer hexagon centered at (0,0), angle 0.
        # Normals for outer hexagon sides (pointy-top outer? Let's assume outer is aligned with inner? No, outer angle is fixed to 0 for simplicity in this step, or optimized?)
        # The task allows outer rotation. Let's optimize outer angle too? 
        # No, let's keep outer angle 0 for now and optimize inner positions. 
        # Actually, optimizing outer angle is a 1D problem. We can do it separately.
        
        theta_rad = 0.0
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i][0], positions[i][1], angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
                
        side = max_proj * 2.0 / math.sqrt(3)
        return side

    # Initial Configuration: Honeycomb cluster
    # 1 center, 6 neighbors, 5 more? 
    # Let's use a 3x4 grid on a hex lattice.
    sqrt3 = math.sqrt(3)
    init_positions = []
    # Generate 12 points in a hex lattice
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            init_positions.append([x, y])
            
    # Sort by distance to origin to pick central 12
    init_positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = init_positions[:12]
    
    # Center them
    init_positions = np.array(init_positions)
    init_positions -= np.mean(init_positions, axis=0)
    
    init_angles = np.zeros(N)
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    # Check initial validity
    gap0 = calc_gap_and_grad(x0)
    if gap0 < 0:
        # Scale up positions to ensure disjointness
        scale = 1.0
        while gap0 < 0 and scale < 10:
            scale += 0.1
            x0[:2*N] = init_positions.flatten() * scale
            gap0 = calc_gap_and_grad(x0)
        if gap0 < 0:
            # Fallback: spread them out more
            x0[:2*N] = init_positions.flatten() * 1.5
            gap0 = calc_gap_and_grad(x0)

    best_x = x0.copy()
    best_score = objective_outer_side(best_x)
    
    # Optimization
    def constraint_gap(params):
        return calc_gap_and_grad(params)
    
    constraints = [{'type': 'ineq', 'fun': constraint_gap}]
    
    try:
        res = minimize(
            objective_outer_side,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.fun < best_score:
            # Verify
            gap_final = calc_gap_and_grad(res.x)
            if gap_final >= -1e-6:
                best_x = res.x
                best_score = res.fun
    except Exception:
        pass
    
    # Polish: Optimize outer angle
    def objective_with_outer_angle(outer_angle_deg):
        positions = best_x[:2*N].reshape(N, 2)
        angles = best_x[2*N:]
        
        theta_rad = math.radians(outer_angle_deg)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i][0], positions[i][1], angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
                
        side = max_proj * 2.0 / math.sqrt(3)
        return side

    # Minimize over outer angle (0 to 60 deg due to symmetry)
    res_outer = minimize(objective_with_outer_angle, 0.0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8})
    best_outer_angle = res_outer.x[0]
    best_score = res_outer.fun
    
    # Final check
    positions = best_x[:2*N].reshape(N, 2)
    angles = best_x[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    # Return interface: inner, center, side, angle
    # Center is (0,0)
    # Side is best_score
    # Angle is best_outer_angle
    
    return inner, (0.0, 0.0), best_score, best_outer_angle

