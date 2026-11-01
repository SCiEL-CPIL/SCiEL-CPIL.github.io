import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # Initial configuration: Honeycomb cluster
    # Center hexagon
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Hex 0 is at (0,0)
    # Hex 1-6 are at distance 2, angles 0, 60, ...
    for i in range(1, 7):
        angle = i * np.pi / 3
        centers[i] = [2 * np.cos(angle), 2 * np.sin(angle)]
        
    # Hex 7-10: Place them in the next ring, roughly
    # The next ring in a hexagonal lattice has centers at distance 2*sqrt(3) ~ 3.46
    # Let's place them at angles 30, 150, 210, 330 degrees?
    # Or just extend the "arms".
    # Let's try a 3-3-3-2 arrangement in a grid-like fashion for better initial packing.
    # Row 0 (y=0): x = -2, 0, 2
    # Row 1 (y=sqrt(3)): x = -1, 1, 3 (shifted)
    # Row 2 (y=2*sqrt(3)): x = -2, 0, 2
    # Row 3 (y=3*sqrt(3)): x = -1, 1
    # This is 3+3+3+2 = 11.
    
    # Let's use this grid as initial guess
    sqrt3 = np.sqrt(3)
    init_centers = [
        (-2, 0), (0, 0), (2, 0),
        (-1, sqrt3), (1, sqrt3), (3, sqrt3),
        (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
        (-1, 3*sqrt3), (1, 3*sqrt3)
    ]
    centers = np.array(init_centers)
    # All angles 0 initially
    
    # Symmetry: Fix center of mass at (0,0) and rotation of first hex?
    # For optimization, we optimize all 2N coords and N angles.
    # To avoid translation drift, we can center the cluster in the cost function or after each step.
    # Simpler: Optimize relative to the first hexagon.
    # Let's just optimize all and rely on the optimizer to find a local min.
    # To ensure the outer hexagon is centered, we can compute the center of the bounding box in the cost function?
    # No, the outer hexagon is fixed at (0,0). The inner cluster should center itself.
    # We will optimize the positions such that the cluster is centered.
    # Parameter vector: x1, y1, a1, ..., xN, yN, aN
    # Constraint: mean(x) = 0, mean(y) = 0? 
    # If we don't constrain, the optimizer might shift the whole cluster to reduce L? 
    # No, shifting the cluster relative to the fixed outer hexagon (0,0) changes L.
    # The optimal solution will naturally center the cluster.
    
    def cost(params):
        c = params[:2*N].reshape(N, 2)
        a = params[2*N:]
        
        # Calculate vertices for all hexagons
        # For each hex i, vertices are c[i] + R(a[i]) @ base_vertices
        # Base vertices of unit hexagon (flat top? pointy top?)
        # Let's use pointy top for inner hexes too, doesn't matter as they rotate.
        # Vertices at angles 0, 60, ...
        base_angles = np.arange(6) * np.pi / 3
        base_verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        
        max_L = 0.0
        for i in range(N):
            ci = c[i]
            ai = a[i]
            # Rotate base vertices
            cos_a = np.cos(ai)
            sin_a = np.sin(ai)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            verts = base_verts @ rot.T + ci
            
            # Check each vertex against outer hexagon constraints
            # Outer hexagon: side L, centered at (0,0), flat-top (normals at 0, 60, 120...)
            # Normals: n_k = (cos(k*pi/3), sin(k*pi/3))
            # Constraint: |v . n_k| <= L * sqrt(3) / 2
            # So L >= 2 * |v . n_k| / sqrt(3)
            
            # Vectorized dot products
            # verts: (6, 2)
            # normals: (6, 2)
            normals = base_verts # Same angles for normals in this orientation?
            # If outer hex is flat-top (vertices at 30, 90...), normals are at 0, 60...
            # Yes, base_verts are at 0, 60... which are the normals.
            
            dots = verts @ normals.T # (6, 6)
            abs_dots = np.abs(dots)
            # Max projection for this hexagon
            max_proj = np.max(abs_dots)
            # Required L for this hexagon
            L_i = 2 * max_proj / np.sqrt(3)
            if L_i > max_L:
                max_L = L_i
                
        return max_L

    x0 = np.zeros(3*N)
    x0[:2*N] = centers.flatten()
    x0[2*N:] = angles
    
    # Optimization
    # Nelder-Mead is good for this non-smooth function
    result = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    best_params = result.x
    best_c = best_params[:2*N].reshape(N, 2)
    best_a = best_params[2*N:]
    best_L = result.fun
    
    # Convert to output format
    inner = []
    for i in range(N):
        x, y = best_c[i]
        ang_deg = np.degrees(best_a[i]) % 360
        inner.append([float(x), float(y), float(ang_deg)])
        
    # Outer hexagon center is (0,0), side is best_L, angle 0
    return inner, (0.0, 0.0), float(best_L), 0.0

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import minimize

    N = 11
    sqrt3 = np.sqrt(3)

    # Initial configuration: Grid-like cluster
    init_centers = [
        (-2, 0), (0, 0), (2, 0),
        (-1, sqrt3), (1, sqrt3), (3, sqrt3),
        (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
        (-1, 3*sqrt3), (1, 3*sqrt3)
    ]
    centers = np.array(init_centers)
    angles = np.zeros(N)

    def cost(params):
        c = params[:2*N].reshape(N, 2)
        a = params[2*N:]
        
        # Base vertices of unit hexagon (pointy top, vertices at 0, 60, ...)
        base_angles = np.arange(6) * np.pi / 3
        base_verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        # Normals for outer hexagon (flat top, normals at 0, 60, ...)
        # Note: If outer hex is defined by normals at 0,60..., it is flat-top.
        normals = base_verts

        max_L = 0.0
        for i in range(N):
            ci = c[i]
            ai = a[i]
            cos_a = np.cos(ai)
            sin_a = np.sin(ai)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            verts = base_verts @ rot.T + ci
            
            dots = verts @ normals.T
            abs_dots = np.abs(dots)
            max_proj = np.max(abs_dots)
            L_i = 2 * max_proj / np.sqrt(3)
            if L_i > max_L:
                max_L = L_i
                
        return max_L

    x0 = np.zeros(3*N)
    x0[:2*N] = centers.flatten()
    x0[2*N:] = angles

    try:
        result = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
        best_params = result.x
        best_c = best_params[:2*N].reshape(N, 2)
        best_a = best_params[2*N:]
        best_L = result.fun
    except Exception:
        # Fallback to initial guess if optimization fails
        best_c = centers
        best_a = angles
        best_L = cost(x0)

    inner = []
    for i in range(N):
        x, y = best_c[i]
        ang_deg = np.degrees(best_a[i]) % 360
        inner.append([float(x), float(y), float(ang_deg)])
        
    return inner, (0.0, 0.0), float(best_L), 0.0
# EVOLVE-BLOCK-END
