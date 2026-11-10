import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    # Unit hexagon vertices
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Initial guess: Honeycomb cluster
    d = np.sqrt(3)
    u = np.array([d, 0])
    v = np.array([d * np.cos(np.deg2rad(60)), d * np.sin(np.deg2rad(60))])
    
    core = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (0,2), (-2,0), (0,-2)] # A symmetric subset of ring 2
    
    initial_points = core + ring2
    initial_centers = np.array([i * u + j * v for i, j in initial_points])
    
    # Parameters to optimize:
    # x, y for each of 11 hexagons (22 params)
    # rotation angle of the whole cluster (1 param)
    # Total: 23 params
    
    bounds = [(-5, 5)] * 22 + [(0, np.pi)]
    
    def get_vertices(centers, rot):
        # Rotate centers
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = centers @ R.T
        # Add vertices
        verts = rotated_centers[:, np.newaxis, :] + hex_verts
        return verts
    
    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
            
        centers = params[:22].reshape(11, 2)
        rot = params[22]
        
        verts = get_vertices(centers, rot)
        
        # 1. Check overlaps between inner hexagons
        # For each pair, check distance between centers.
        # If distance < sqrt(3) - epsilon, they overlap.
        # Actually, for regular hexagons of side 1, min center distance is sqrt(3).
        # We need separation > 1e-6.
        # So distance must be > sqrt(3) + 1e-6.
        
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < np.sqrt(3) + 1e-6:
            # Penalty proportional to overlap
            overlap = (np.sqrt(3) + 1e-6 - min_dist)
            return 1e6 + overlap * 1000
            
        # 2. Calculate minimum enclosing hexagon size
        # We want to find the smallest side length S such that all vertices fit in a hexagon of side S.
        # The hexagon can be rotated. But we fixed the cluster rotation `rot`.
        # We can assume the outer hexagon is aligned with the cluster's principal axes?
        # Or we can optimize the outer hexagon rotation too?
        # For simplicity in DE, let's assume outer hexagon is aligned with global axes (angle 0)
        # and we optimize cluster rotation `rot` to minimize the bounding box.
        
        # Vertices are in (11, 6, 2)
        all_pts = verts.reshape(-1, 2)
        
        # Project onto 6 normals of a regular hexagon (angle 0)
        # Normals at 0, 60, 120, 180, 240, 300 degrees?
        # Vertices of outer hexagon (side S, center 0, angle 0) are at S * e^{i * 60k}.
        # The edges are perpendicular to these? No.
        # Let's use the apothem method.
        # A point (x,y) is inside a regular hexagon of side S (angle 0) if:
        # |x| <= S * sqrt(3)/2 AND |y| <= S * sqrt(3)/2 AND |x/2 + y*sqrt(3)/2| <= S * sqrt(3)/2 ...
        # Actually, the support function of a regular hexagon with side S and angle 0:
        # h(theta) = S * cos(theta - 60k) for theta in [60k-30, 60k+30].
        # Max projection of points onto any direction must be <= h(direction).
        
        # Simpler: Compute the maximum projection of all_pts onto the 6 face normals of the hexagon.
        # Face normals of hexagon with vertices at 0, 60... are at 30, 90, 150, 210, 270, 330.
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        normals = np.array([np.cos(angles), np.sin(angles)]).T # (6, 2)
        
        projs = all_pts @ normals.T # (66, 6)
        max_projs = np.max(projs, axis=0) # (6,)
        
        # For a hexagon of side S, the projection on the normal to a face is the apothem: S * sqrt(3) / 2.
        # So we need S * sqrt(3) / 2 >= max(max_projs)
        # S >= 2 / sqrt(3) * max(max_projs)
        
        S = (2.0 / np.sqrt(3.0)) * np.max(max_projs)
        
        return S
    
    # Run Differential Evolution
    # Use a reasonable number of iterations to fit in time budget
    # DE is stochastic, so we might need multiple restarts or a longer run.
    # Given 600s limit, we can afford a few minutes for DE.
    
    best_result = None
    best_side = float('inf')
    
    # Try to run DE
    try:
        res = differential_evolution(objective, bounds, maxiter=500, popsize=15, tol=1e-4, seed=42, polish=True)
        
        if res.success and res.fun < best_side:
            best_side = res.fun
            centers = res.x[:22].reshape(11, 2)
            rot = res.x[22]
            
            # Construct final result
            # Inner hexagons: centers, angle 0 (relative to global frame? No, they are rotated by rot)
            # Wait, the task asks for (x, y, angle) for inner hexagons.
            # In my objective, I rotated the centers by `rot` to check fit.
            # So the actual centers in the global frame are `centers` (unrotated) and the angle is `rot`?
            # No, `get_vertices` rotated the centers.
            # If I return centers as `centers` and angle as `rot`, the evaluator will compute vertices as:
            # cx + s * cos(angle + 60i).
            # My optimization assumed vertices at `rotated_centers + hex_verts`.
            # `rotated_centers` is `centers` rotated by `rot`.
            # `hex_verts` are at angle 0.
            # So the vertices are at `R * centers + hex_verts`.
            # The evaluator computes `centers + R(angle) * hex_verts_unit`.
            # These are NOT the same unless `rot=0` or `centers=0`.
            # Actually, rotating the hexagon by `angle` is equivalent to rotating the vertices.
            # My optimization: `R(rot) * c + v`.
            # Evaluator: `c + R(angle) * v`.
            # If I set `c_eval = R(rot) * c_opt` and `angle_eval = 0`, then `c_eval + v = R(rot)*c_opt + v`. Matches.
            # So I should return `centers_eval = centers @ R.T` and `angle = 0`.
            
            c, s = np.cos(rot), np.sin(rot)
            R = np.array([[c, -s], [s, c]])
            centers_eval = centers @ R.T
            
            inner_data = np.column_stack([centers_eval, np.zeros(11)])
            outer_data = np.array([0, 0, 0.0]) # Outer hexagon angle 0
            best_result = (inner_data, outer_data, best_side)
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    if best_result is None:
        # Fallback to original logic if DE fails or finds nothing better
        # Re-implement the simple scaling logic from the original code as fallback
        d = np.sqrt(3)
        u = np.array([d, 0])
        v = np.array([d * np.cos(np.deg2rad(60)), d * np.sin(np.deg2rad(60))])
        core = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        adds = [(2,0), (0,2), (-2,0), (0,-2)]
        points = core + adds
        centers = np.array([i * u + j * v for i, j in points])
        centers = centers * 1.001
        
        # Calculate S for this configuration
        verts = centers[:, np.newaxis, :] + hex_verts
        all_pts = verts.reshape(-1, 2)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        normals = np.array([np.cos(angles), np.sin(angles)]).T
        projs = all_pts @ normals.T
        max_projs = np.max(projs, axis=0)
        S = (2.0 / np.sqrt(3.0)) * np.max(max_projs)
        
        inner_data = np.column_stack([centers, np.zeros(11)])
        outer_data = np.array([0, 0, 0.0])
        best_result = (inner_data, outer_data, S)
        
    return best_result
