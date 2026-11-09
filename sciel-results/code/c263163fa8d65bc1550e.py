"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    sqrt3 = math.sqrt(3.0)
    
    # Precompute unit hexagon vertices (side=1, angle=0, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    unit_hex_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute edge normals for unit hexagon (angle=0)
    # Edges connect v[i] to v[i+1]. Normal is perpendicular to edge vector.
    # For a regular hexagon, normals point at 30, 90, 150, 210, 270, 330 degrees.
    unit_hex_normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 390, 60)])

    def get_vertices_and_normals(params):
        """
        Returns:
        verts: (N, 6, 2) array of vertices for each hexagon
        norms: (N, 6, 2) array of outward normals for each hexagon
        """
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        verts = np.zeros((N, 6, 2))
        norms = np.zeros((N, 6, 2))
        
        for i in range(N):
            angle_rad = math.radians(angles[i])
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            R = np.array([[c, -s], [s, c]])
            verts[i] = (R @ unit_hex_verts.T).T + centers[i]
            norms[i] = (R @ unit_hex_normals.T).T
            
        return verts, norms

    def calc_enclosing_side(verts, outer_angle_deg):
        """
        Calculate the side length of the minimal enclosing regular hexagon.
        """
        outer_angle_rad = math.radians(outer_angle_deg)
        # Axes of the outer hexagon are perpendicular to its sides.
        # If outer hexagon is rotated by phi, its side normals are at phi + 30 + k*60.
        # The projection length along normal n is (max dot - min dot).
        # Side length S satisfies: projection_length = S * sqrt(3).
        # So S = projection_length / sqrt(3).
        
        max_proj_diff = 0.0
        all_verts = verts.reshape(-1, 2)
        
        for k in range(3): # Only 3 unique axes needed (opposite sides give same extent)
            # Normal angle for axis k
            n_angle = outer_angle_rad + math.radians(30 + k * 60)
            n = np.array([math.cos(n_angle), math.sin(n_angle)])
            
            projections = all_verts @ n
            diff = np.max(projections) - np.min(projections)
            if diff > max_proj_diff:
                max_proj_diff = diff
                
        return max_proj_diff / sqrt3

    def check_overlap_penalty(verts, norms):
        """
        Calculate a penalty for overlaps using Separating Axis Theorem.
        Returns 0 if no overlap, positive value if overlap.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Check axes from polygon i
                # Project j onto i's normals
                # If min(proj_j) > max(proj_i) for any axis, they are separated.
                # We check if they are NOT separated on ANY axis -> Overlap.
                
                separated = False
                
                # Axes from i
                for k in range(6):
                    n = norms[i, k]
                    proj_i = verts[i] @ n
                    proj_j = verts[j] @ n
                    
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i < min_j or max_j < min_i:
                        separated = True
                        break
                
                if separated:
                    continue
                
                # Axes from j
                for k in range(6):
                    n = norms[j, k]
                    proj_i = verts[i] @ n
                    proj_j = verts[j] @ n
                    
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i < min_j or max_j < min_i:
                        separated = True
                        break
                
                if not separated:
                    # They overlap. Add penalty based on how deep? 
                    # Simple penalty: count as 1.0 or use a distance metric.
                    # For optimization, a continuous penalty is better.
                    # Approximate penetration depth or just use a large constant.
                    # Let's use a simple constant penalty to push them apart.
                    penalty += 1.0
                    
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        outer_angle = params[-1]
        verts, norms = get_vertices_and_normals(params)
        
        # Check overlap
        overlap_pen = check_overlap_penalty(verts, norms)
        
        # Calculate enclosing side
        S = calc_enclosing_side(verts, outer_angle)
        
        # Total cost: Side length + Penalty
        # We want to minimize S. Penalty should be large enough to prevent overlaps.
        # If S is around 5-6, a penalty of 10 per overlap pair is significant.
        return S + 10.0 * overlap_pen

    # Initial guess: Hexagonal lattice
    # Center at 0,0
    # 6 around it, 5 more? 
    # A 12-hexagon cluster: Center (1), Ring 1 (6), Ring 2 (5) -> 12? 
    # 1 + 6 + 5 = 12. Yes.
    # Or 2 rows of 6?
    # Let's use a lattice start.
    init_params = np.zeros(3*N + 1)
    
    # Coordinates for a compact cluster
    # q, r axial coordinates
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    for i, (q, r) in enumerate(coords):
        # Convert axial to cartesian
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        init_params[2*i] = x
        init_params[2*i+1] = y
        init_params[2*N + i] = 0.0 # angle
    init_params[-1] = 0.0 # outer angle

    best_result = None
    best_score = 1e10

    # Define bounds for DE
    # x, y in [-5, 5], angle in [-60, 60], outer angle in [-30, 30]
    bounds = [(-5, 5)] * (2*N) + [(-60, 60)] * N + [(-30, 30)]
    
    # Run Differential Evolution
    # maxiter and popsize tuned for time
    # Time limit is ~130s. DE can be slow.
    # Let's try a few runs.
    
    n_de_runs = 3
    
    for run_idx in range(n_de_runs):
        if time.time() > deadline - 20.0:
            break
            
        # Vary bounds or seed
        seed = 42 + run_idx
        # Tighten bounds if we have a good result? No, keep wide for global.
        
        try:
            de_res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15,
                tol=0.01,
                seed=seed,
                polish=True, # Let DE do some local polishing
                init='latinhypercube'
            )
        except Exception:
            continue
        
        if de_res.success or de_res.fun < best_score:
            # Verify and polish with Powell
            current_params = de_res.x
            current_score = de_res.fun
            
            if current_score < best_score:
                best_score = current_score
                best_result = current_params
            
            # Local polish with Powell
            # Powell is good for local minima.
            try:
                powell_res = minimize(
                    objective, 
                    current_params, 
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if powell_res.fun < best_score:
                    best_score = powell_res.fun
                    best_result = powell_res.x
            except Exception:
                pass

        # Check time
        if time.time() > deadline - 10.0:
            break

    # If no result found or failed, use initial
    if best_result is None:
        best_result = init_params
        best_score = objective(best_result)

    # Final extraction
    centers = best_result[:2*N].reshape(N, 2)
    angles = best_result[2*N:3*N]
    outer_angle = best_result[-1]
    
    # Calculate final side length (without penalty)
    verts, _ = get_vertices_and_normals(best_result)
    final_side = calc_enclosing_side(verts, outer_angle)
    
    # Construct return values
    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle = float(outer_angle)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
