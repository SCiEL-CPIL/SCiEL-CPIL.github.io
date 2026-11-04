"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def get_all_vertices(params):
        """Get vertices for all hexagons given parameters [cx0, cy0, a0, cx1, ...]"""
        n = len(params) // 3
        verts = np.zeros((n, 6, 2))
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts[i] = hex_vertices(cx, cy, angle)
        return verts

    def compute_S_vectorized(verts, oangle):
        """
        Compute minimal side length S of outer hexagon at (0,0) with rotation oangle
        containing all vertices in 'verts' (N x 6 x 2).
        """
        # Transform to container local frame (rotate by -oangle)
        cos_a = np.cos(-oangle)
        sin_a = np.sin(-oangle)
        
        # verts shape: (N, 6, 2)
        # Rotate: x' = x cos - y sin, y' = x sin + y cos
        # Wait, standard rotation of point by -theta:
        # x' = x cos(-t) - y sin(-t) = x cos(t) + y sin(t)
        # y' = x sin(-t) + y cos(-t) = -x sin(t) + y cos(t)
        
        rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
        ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
        
        # Normals for unit hexagon (apothem direction)
        # Angles: pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
        theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(theta_n)
        ny = np.sin(theta_n)
        
        # Projections: (N, 6, 6)
        # proj[i, j, k] = rx[i, j] * nx[k] + ry[i, j] * ny[k]
        proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
               ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
        
        max_proj = np.max(np.abs(proj))
        apothem = np.sqrt(3) / 2
        return max_proj / apothem

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def objective(params):
        """
        params: [cx0, cy0, a0, ..., cxN, cyN, aN, oangle]
        """
        n = N
        oangle = params[3 * n]
        
        verts = get_all_vertices(params[:3*n])
        
        S = compute_S_vectorized(verts, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = verts[i].mean(axis=0)
                c2 = verts[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0: # Only check SAT if potentially overlapping
                    if sat_overlap(verts[i], verts[j]):
                        penalty += 100.0 * (2.0 - dist)
        
        return S + penalty

    # Initial guess: Hexagonal Lattice
    # Axial coords to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate first 11 hexes
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: We need 4 more. 
    # To be symmetric, we can pick 4 from the 12 in Ring 2.
    # Or just use the first 11 in a spiral?
    # Let's use a specific set that is known to be close to optimal or just lattice.
    # Lattice is a valid non-overlapping config.
    
    axial_coords = [(0,0)]
    # Ring 1
    axial_coords += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    # Ring 2 (first 4)
    # (2,0), (2,-1), (2,-2), (1,-2) ?
    # Let's just take the next 4 in a systematic order.
    # Actually, for N=11, a "cross" shape or similar might be better.
    # But lattice is a safe start.
    axial_coords += [(2,0), (2,-1), (2,-2), (1,-2)]
    
    params = np.zeros(3 * N + 1)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cart(q, r)
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = 0.0 # Angle
    
    # Center the configuration
    # The lattice is not centered at 0.
    # Calculate centroid of centers
    centers = params[:3*N].reshape(N, 3)[:, :2]
    centroid = centers.mean(axis=0)
    params[:3*N:3] -= centroid[0]
    params[1:3*N:3] -= centroid[1]
    
    # Initial container rotation 0
    params[3*N] = 0.0
    
    best_params = params
    best_score = objective(best_params)
    
    # Optimize
    # Use Powell for robustness
    # We can try multiple starts if time permits, but 120s is tight.
    # Let's do 1 main optimization with high maxiter.
    
    res = minimize(objective, best_params, method='Powell', 
                  options={'maxiter': 3000, 'maxfev': 20000, 'disp': False})
    
    if res.fun < best_score:
        best_score = res.fun
        best_params = res.x
    
    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    oangle = best_params[3 * N]
    # Container is at (0,0) in our optimization frame.
    # But the task asks for (inner, center, side, angle).
    # The "center" returned should be the center of the outer hexagon.
    # Since we optimized relative to (0,0), the center is (0,0) in the local frame.
    # However, the inner coordinates are absolute.
    # If we return center (0,0), the evaluator will check if inner hexes are inside outer hex at (0,0) with side S and angle oangle.
    # Our optimization ensured this.
    
    # One final polish: refine container rotation only?
    # The joint optimization should have handled it.
    
    return inner, (0.0, 0.0), best_score, np.degrees(oangle)
# EVOLVE-BLOCK-END
