import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Support function directions for a regular hexagon of side S centered at origin
        # The width of a regular hexagon of side S in the 6 normal directions is S*sqrt(3).
        # The distance from center to side (apothem) is S*sqrt(3)/2.
        # The projection of a point p onto the normal n_k must be <= S*sqrt(3)/2.
        # So S >= 2/sqrt(3) * max_k (p . n_k).
        # The normals for a hexagon with vertices at 0, 60, 120... are at 30, 90, 150...
        
        # Let's verify the linear forms used in the original code:
        # L1 = x + y / sqrt3  -> dot with (1, 1/sqrt3). Norm of (1, 1/sqrt3) is sqrt(1+1/3)=2/sqrt3.
        # Unit normal: (sqrt3/2, 1/2). Correct.
        # L2 = y * 2.0 / sqrt3 -> dot with (0, 2/sqrt3). Norm 2/sqrt3. Unit normal (0, 1). Correct.
        # L3 = -x + y / sqrt3 -> dot with (-1, 1/sqrt3). Unit normal (-sqrt3/2, 1/2). Correct.
        # L4 = -x - y / sqrt3 -> dot with (-1, -1/sqrt3). Unit normal (-sqrt3/2, -1/2). Correct.
        # L5 = -y * 2.0 / sqrt3 -> dot with (0, -2/sqrt3). Unit normal (0, -1). Correct.
        # L6 = x - y / sqrt3 -> dot with (1, -1/sqrt3). Unit normal (sqrt3/2, -1/2). Correct.
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def are_disjoint(v1, v2):
        """
        Check if two convex polygons (hexagons) are disjoint using Separating Axis Theorem.
        Returns True if disjoint, False if overlapping.
        """
        for i in range(6):
            # Edge of v1
            edge = v1[(i + 1) % 6] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True # Separated
        
        for i in range(6):
            # Edge of v2
            edge = v2[(i + 1) % 6] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True # Separated
                
        return False

    def is_valid_configuration(positions):
        """
        Check if all hexagons are disjoint.
        """
        verts = [hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if not are_disjoint(verts[i], verts[j]):
                    return False
        return True

    def penalty_objective(params):
        """
        Objective for optimization: Minimize outer side length + large penalty for overlaps.
        We use a smooth approximation of the overlap indicator to help the optimizer.
        """
        positions = params.reshape(N, 2)
        side = required_outer_side(positions)
        
        # Calculate minimum separation for all pairs
        # If we find an overlap, we need to penalize it heavily.
        # For speed, we just check validity first? No, we need gradient info.
        # Let's use a simpler penalty: sum of squared penetration depths approx.
        # Since SAT doesn't give depth easily, we'll use distance between centers vs min distance.
        # Min distance between two disjoint unit hexagons (aligned) is sqrt(3) * 1.5? 
        # Distance between centers for touching hexagons in lattice is sqrt(3) * 1 = 1.732?
        # Actually, distance between centers of two touching unit hexagons is 2 * apothem = sqrt(3).
        # So if dist < sqrt(3) - epsilon, they overlap.
        
        penalty = 0.0
        min_dist_sq = 3.0 # (sqrt(3))^2
        for i in range(N):
            for j in range(i + 1, N):
                d_sq = np.sum((positions[i] - positions[j])**2)
                if d_sq < min_dist_sq:
                    # Overlap likely. Penalty proportional to violation.
                    # This is a crude proxy but helps push apart.
                    penalty += (min_dist_sq - d_sq) ** 2
        
        return side + 100.0 * penalty

    def gradient(params):
        """
        Numerical gradient via central differences.
        """
        eps = 1e-5
        grad = np.zeros_like(params)
        for i in range(len(params)):
            old = params[i]
            params[i] = old + eps
            f_plus = penalty_objective(params)
            params[i] = old - eps
            f_minus = penalty_objective(params)
            params[i] = old
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def solve_from_init(init_positions, time_budget=5.0):
        """
        Run L-BFGS-B optimization from a given initial position.
        """
        init_params = init_positions.flatten()
        
        # Check if initial is already valid and good
        init_side = required_outer_side(init_positions)
        if not is_valid_configuration(init_positions):
             # If invalid, we still try to optimize, but we must verify at the end.
             pass
             
        try:
            # Use a tighter tolerance and fewer iterations to save time
            res = minimize(
                penalty_objective, 
                init_params, 
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
        except Exception as e:
            return init_positions, init_side
        
        opt_positions = res.x.reshape(N, 2)
        opt_side = required_outer_side(opt_positions)
        
        # Verify validity of the optimized solution
        if is_valid_configuration(opt_positions):
            return opt_positions, opt_side
        else:
            # If optimization resulted in overlap, discard.
            # Return the initial if it was valid, else None
            if is_valid_configuration(init_positions):
                return init_positions, init_side
            return None, float('inf')

    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    # This is the standard densest packing for infinite plane.
    # For N=12, we take a cluster of 12 hexagons.
    # Center at (0,0). Neighbors at distance sqrt(3).
    # Axial coordinates (q, r)
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # We need 12. 1 + 6 + 5 from next ring.
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random seeds
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init3)
    
    np.random.seed(123)
    init4 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init4)
    
    # 4. Perturbed Lattice
    init5 = init1 + np.random.uniform(-0.5, 0.5, (N, 2))
    configs.append(init5)
    
    # 5. Another lattice variant (shifted)
    init6 = init1 + np.array([1.0, 1.0])
    configs.append(init6)

    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        if not is_valid_configuration(init):
            # Try to fix by scaling
            fixed = False
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = init * scale
                if is_valid_configuration(scaled):
                    init = scaled
                    fixed = True
                    break
            if not fixed:
                continue

        # Run optimization
        pos, side = solve_from_init(init)
        if pos is not None and side < best_side:
            best_side = side
            best_positions = pos

    # Fallback
    if best_positions is None:
        # Use the first valid config
        for init in configs:
            if is_valid_configuration(init):
                best_positions = init
                best_side = required_outer_side(init)
                break
        
        if best_positions is None:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
