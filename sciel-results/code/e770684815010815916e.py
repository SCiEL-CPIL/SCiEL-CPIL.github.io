"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # For a regular hexagon with side length s, the distance from center to any vertex is s (circumradius).
    # The distance from center to any edge midpoint is s * sqrt(3)/2 (inradius).
    # We want to place N unit hexagons (side=1) inside an outer hexagon of side S.
    # Each inner hexagon has circumradius 1 and inradius sqrt(3)/2.
    # The inner hexagons can rotate. Let's fix orientations to 0 or 30 degrees for simplicity first,
    # but actually, optimal packings often use specific rotations.
    # 
    # A key insight: the "effective" radius of a hexagon in a given direction depends on its orientation.
    # However, a simpler and robust approach for convex containers like hexagons is to use the
    # "Minkowski sum" or support function approach.
    # 
    # Alternatively, we can formulate this as:
    # Minimize S such that:
    # 1. The outer hexagon of side S (centered at origin, rotated by alpha) contains all inner hexagons.
    # 2. Inner hexagons do not overlap.
    #
    # Let's simplify by fixing the outer hexagon orientation to 0 (flat top/bottom or pointy top/bottom).
    # Standard hexagon with vertices at angles 0, 60, 120, 180, 240, 300.
    # Vertices: (S*cos(k*pi/3), S*sin(k*pi/3)) for k=0..5.
    # The hexagon is the intersection of 6 half-planes:
    # For each edge k, the normal is (cos((k+0.5)*pi/3), sin((k+0.5)*pi/3))? No.
    # Edge between vertex k and k+1. The outward normal for the edge between v_k and v_{k+1} is in the direction of the angle bisector of the exterior.
    # Actually, for a regular hexagon centered at origin with circumradius S, the inradius is S*sqrt(3)/2.
    # The 6 supporting lines are at distance S*sqrt(3)/2 from the center, with normals at angles 30, 90, 150, 210, 270, 330 degrees (if vertices are at 0, 60, ...).
    # Let's define the outer hexagon H_S with vertices at angles 0, 60, ..., 300.
    # The edges are perpendicular to the directions 30, 90, 150, 210, 270, 330.
    # So the constraints for a point p to be inside H_S are:
    # p . n_k <= S * sqrt(3)/2 for k=0..5, where n_k = (cos((k+0.5)*pi/3), sin((k+0.5)*pi/3)).
    
    # Now, for an inner hexagon with center c_i, orientation theta_i, and side length 1.
    # Its vertices are c_i + (cos(theta_i + m*pi/3), sin(theta_i + m*pi/3)) for m=0..5.
    # For the inner hexagon to be inside H_S, ALL its vertices must be inside H_S.
    # This is equivalent to: for all m=0..5, and for all k=0..5:
    # (c_i + v_m(theta_i)) . n_k <= S * sqrt(3)/2
    # where v_m(theta) = (cos(theta + m*pi/3), sin(theta + m*pi/3)).
    
    # Non-overlap constraint:
    # Two convex polygons (hexagons) do not overlap if and only if there exists a separating axis.
    # For two regular hexagons, the separating axes are among the normals of their edges.
    # The edge normals for hexagon i are n_i(k) = (cos(theta_i + (k+0.5)*pi/3), sin(theta_i + (k+0.5)*pi/3)) for k=0..5.
    # The distance between centers projected onto a separating axis n must be >= (inradius_i + inradius_j) = sqrt(3).
    # Wait, this is only true if the separating axis is perpendicular to an edge of both, or if one hexagon is "aligned" with the other in a specific way.
    # Actually, the general condition for two convex polygons P and Q to be disjoint is that there exists a line separating them.
    # The separating line's normal must be perpendicular to an edge of P or Q.
    # So we need to check all edge normals of hexagon i and hexagon j.
    # For a given normal n, the projection of hexagon i onto n is [c_i . n - r_i(n), c_i . n + r_i(n)], where r_i(n) is the support function of the unit hexagon in direction n.
    # For a regular hexagon of side 1, the support function in direction n (unit vector) is:
    # r(n) = max_{m=0..5} v_m . n.
    # The condition for non-overlap is:
    # c_i . n - r_i(n) > c_j . n + r_j(n)  OR  c_j . n - r_j(n) > c_i . n + r_i(n)
    # which simplifies to: |c_i - c_j| . n >= r_i(n) + r_j(n)
    # This must hold for AT LEAST ONE of the separating axes.
    # But checking "exists n in N_i ∪ N_j such that ..." is a disjunctive constraint, which is non-convex.
    
    # This makes the problem hard. Let's try a different approach:
    # Use a grid of orientations and solve for centers.
    # Or, use a penalty method where we penalize overlaps.
    
    # Let's try to use scipy.optimize.minimize with a penalty function.
    # Variables: c_1..c_N (2N coords), theta_1..theta_N (N angles), S (1 scale).
    # Total 3N+1 = 37 variables.
    
    def penalty_function(x):
        # x: [c1x, c1y, theta1, c2x, c2y, theta2, ..., cN, thetaN, S]
        S = x[-1]
        if S < 0:
            return 1e10
        centers = x[:2*N].reshape(N, 2)
        thetas = x[2*N:3*N]
        
        # Check containment: all vertices of all inner hexagons must be in outer hexagon.
        # Outer hexagon constraints: p . n_k <= S * sqrt(3)/2
        n_outer = np.array([[np.cos((k+0.5)*np.pi/3), np.sin((k+0.5)*np.pi/3)] for k in range(6)])
        # Inner hexagon vertices relative to center:
        # v_m(theta) = (cos(theta + m*pi/3), sin(theta + m*pi/3))
        
        contain_violation = 0.0
        for i in range(N):
            c = centers[i]
            theta = thetas[i]
            for m in range(6):
                v = np.array([np.cos(theta + m*np.pi/3), np.sin(theta + m*np.pi/3)])
                p = c + v
                for k in range(6):
                    val = np.dot(p, n_outer[k])
                    limit = S * np.sqrt(3)/2
                    if val > limit:
                        contain_violation += (val - limit)**2
        
        # Check non-overlap: for each pair (i, j), check if they overlap.
        # We use the separating axis theorem. For each pair, we check the 12 potential separating axes (6 from i, 6 from j).
        # If for ALL axes, the projections overlap, then the hexagons overlap.
        # To make this differentiable, we can use a soft penalty:
        # For each pair, compute the minimum "gap" over all separating axes.
        # If the minimum gap is negative, they overlap.
        # gap = min_{n in axes} ( |c_i - c_j| . n - r_i(n) - r_j(n) )
        # We want gap >= 0. Penalty = max(0, -gap)^2.
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                c_i = centers[i]
                c_j = centers[j]
                d = c_i - c_j
                theta_i = thetas[i]
                theta_j = thetas[j]
                
                # Axes from i
                axes_i = np.array([[np.cos(theta_i + (k+0.5)*np.pi/3), np.sin(theta_i + (k+0.5)*np.pi/3)] for k in range(6)])
                # Axes from j
                axes_j = np.array([[np.cos(theta_j + (k+0.5)*np.pi/3), np.sin(theta_j + (k+0.5)*np.pi/3)] for k in range(6)])
                axes = np.vstack([axes_i, axes_j])
                
                # Support functions
                # r(n) for unit hexagon with orientation theta in direction n
                # r(n) = max_m (v_m(theta) . n)
                # v_m(theta) = (cos(theta + m*pi/3), sin(theta + m*pi/3))
                # v_m . n = cos(theta + m*pi/3 - phi) where phi is angle of n.
                # So r(n) = max_m cos(theta + m*pi/3 - phi).
                # This is the support function of a regular hexagon.
                # It can be computed analytically, but let's just compute it numerically for simplicity in the penalty.
                
                def support(theta, n):
                    # n is a unit vector
                    phi = np.arctan2(n[1], n[0])
                    vals = np.cos(theta + np.arange(6)*np.pi/3 - phi)
                    return np.max(vals)
                
                min_gap = np.inf
                for n in axes:
                    r_i = support(theta_i, n)
                    r_j = support(theta_j, n)
                    proj = np.dot(d, n)
                    gap = proj - r_i - r_j
                    # We need |proj| >= r_i + r_j for separation in direction n.
                    # Actually, the condition is: c_i . n - r_i(n) >= c_j . n + r_j(n) OR vice versa.
                    # This is equivalent to |c_i - c_j| . n >= r_i(n) + r_j(n) IF n points from j to i.
                    # But n is a direction. The projection of the segment c_j->c_i onto n is d . n.
                    # The extent of hex i in direction n is r_i(n), and in -n is also r_i(n) (since it's centrally symmetric? No, hexagon is centrally symmetric).
                    # Yes, regular hexagon is centrally symmetric. So r_i(n) = r_i(-n).
                    # So the condition is: |d . n| >= r_i(n) + r_j(n).
                    abs_proj = abs(proj)
                    gap = abs_proj - r_i - r_j
                    if gap < min_gap:
                        min_gap = gap
                
                if min_gap < 0:
                    overlap_penalty += (-min_gap)**2
        
        total_penalty = contain_violation + overlap_penalty
        # We want to minimize S, but also ensure feasibility.
        # Let's return S + lambda * penalty.
        # If penalty is 0, we return S.
        # If penalty > 0, we return a large value to encourage feasibility.
        # Actually, we should minimize S subject to penalty=0.
        # A common trick: return S + M * penalty, where M is large.
        # But M might cause numerical issues.
        # Alternative: return penalty if penalty > 0, else S.
        # This prioritizes feasibility over minimizing S.
        if total_penalty > 1e-6:
            return total_penalty
        else:
            return S
    
    # Initial guess
    # Place hexagons in a hexagonal lattice pattern.
    # For N=12, a 3x4 grid or a hexagonal arrangement.
    # Let's try a hexagonal arrangement of 12 hexagons.
    # Centers at (dx, dy) where dx = 1.5 * (i - j), dy = sqrt(3)/2 * (i + j) ? No, that's for triangular lattice.
    # For hexagonal tiling, the distance between adjacent centers is 2 * inradius = sqrt(3).
    # Let's place them in a 3x4 grid with spacing sqrt(3) in x and 1.5 in y? No.
    # Let's just use a simple grid.
    
    np.random.seed(42)
    x0 = np.zeros(3*N + 1)
    # Place centers in a rough grid
    for i in range(N):
        row = i // 4
        col = i % 4
        x0[2*i] = (col - 1.5) * 2.0  # x
        x0[2*i+1] = (row - 1.0) * 1.8  # y
        x0[2*N+i] = 0.0  # theta
    x0[-1] = 5.0  # initial S
    
    # Use differential evolution for global optimization
    # Bounds: centers in [-10, 10], thetas in [0, pi/3], S in [3, 15]
    bounds = [(-10, 10), (-10, 10)] * N + [(0, np.pi/3)] * N + [(3, 15)]
    
    # Run DE
    result_de = differential_evolution(penalty_function, bounds, maxiter=50, popsize=15, tol=1e-4, seed=42, polish=False)
    
    best_x = result_de.x
    best_val = result_de.fun
    
    # Polish with L-BFGS-B
    if time.time() < deadline:
        result_lbfgs = minimize(penalty_function, best_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        if result_lbfgs.fun < best_val:
            best_x = result_lbfgs.x
            best_val = result_lbfgs.fun
    
    # Extract result
    S = best_x[-1]
    centers = best_x[:2*N].reshape(N, 2)
    thetas = best_x[2*N:3*N]
    
    # Convert thetas to degrees
    thetas_deg = np.degrees(thetas)
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(thetas_deg[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = float(S)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
