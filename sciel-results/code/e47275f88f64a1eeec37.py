import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Base vertices of a unit hexagon (flat-top relative to local x-axis? No, let's use pointy-top for local definition, doesn't matter as angle is optimized)
    # Let's use standard math angles: 0, 60, 120...
    # Side length 1. Distance from center to vertex is 1.
    # Vertices: (cos(60*k), sin(60*k))
    base_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    def get_verts(cx, cy, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return (R @ base_verts.T).T + np.array([cx, cy])

    def sat_penetration(v1, v2):
        """
        Returns the penetration depth between two convex polygons.
        Positive if overlapping, negative if separated (gap).
        Specifically, returns min_overlap.
        If min_overlap > 0, they overlap by min_overlap.
        If min_overlap < 0, they are separated by -min_overlap.
        """
        # Axes are normals to edges
        axes = []
        for v in [v1, v2]:
            for i in range(len(v)):
                p1 = v[i]
                p2 = v[(i+1) % len(v)]
                edge = p2 - p1
                # Normal
                n = np.array([-edge[1], edge[0]])
                n = n / (np.linalg.norm(n) + 1e-9)
                axes.append(n)
        
        min_overlap = float('inf')
        for n in axes:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Calculate overlap
            # If max1 < min2 or max2 < min1, they are separated on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def calc_L_from_verts(verts):
        # verts: (N, 6, 2)
        # Outer hexagon: flat-top, centered at (0,0), side L.
        # Inequalities for point (x,y) to be inside flat-top hex of side L:
        # Vertices: (L,0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L,0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2)
        # Edges:
        # 1. (L,0) to (L/2, L*sqrt(3)/2): Line sqrt(3)x + y = L*sqrt(3) -> L >= x + y/sqrt(3)
        # 2. (L/2, L*sqrt(3)/2) to (-L/2, L*sqrt(3)/2): Line y = L*sqrt(3)/2 -> L >= 2y/sqrt(3)
        # 3. (-L/2, L*sqrt(3)/2) to (-L,0): Line -sqrt(3)x + y = L*sqrt(3) -> L >= -x + y/sqrt(3)
        # 4. (-L,0) to (-L/2, -L*sqrt(3)/2): Line -sqrt(3)x - y = L*sqrt(3) -> L >= -x - y/sqrt(3)
        # 5. (-L/2, -L*sqrt(3)/2) to (L/2, -L*sqrt(3)/2): Line y = -L*sqrt(3)/2 -> L >= -2y/sqrt(3)
        # 6. (L/2, -L*sqrt(3)/2) to (L,0): Line sqrt(3)x - y = L*sqrt(3) -> L >= x - y/sqrt(3)
        
        x = verts[:, :, 0]
        y = verts[:, :, 1]
        
        s3 = np.sqrt(3)
        terms = np.stack([
            x + y / s3,
            2 * y / s3,
            -x + y / s3,
            -x - y / s3,
            -2 * y / s3,
            x - y / s3
        ], axis=-1)
        
        L_points = np.max(terms, axis=-1)
        return np.max(L_points)

    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        for i in range(N):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            verts_list.append(get_verts(cx, cy, ang))
        
        verts = np.array(verts_list)
        
        # Calculate L
        L = calc_L_from_verts(verts)
        
        # Calculate overlap penalties
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Quick reject: if center distance > 2.0, no overlap possible (max radius 1)
                ci = np.array([params[i*3], params[i*3+1]])
                cj = np.array([params[j*3], params[j*3+1]])
                dist = np.linalg.norm(ci - cj)
                if dist < 2.0:
                    pen = sat_penetration(verts[i], verts[j])
                    if pen > 0:
                        # Overlap depth is pen
                        penalty += pen**2
                    # If pen <= 0, they are separated, penalty is 0
        
        # We want to minimize L, but penalize overlap heavily
        # Using a large multiplier ensures non-overlap in the optimum
        return L + 50.0 * penalty

    # Initial guess: Hexagonal grid
    # Generate points in a hex grid
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            points.append([x, y])
    
    points = np.array(points)
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    init_centers = points[indices[:N]]
    
    x0 = np.zeros(36)
    for i in range(N):
        x0[i*3] = init_centers[i, 0]
        x0[i*3+1] = init_centers[i, 1]
        x0[i*3+2] = 0.0 # Angle

    # Optimize
    # Nelder-Mead is robust for non-smooth penalties
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000, 'adaptive': True})
    
    best_params = res.x
    best_L = calc_L_from_verts(np.array([get_verts(best_params[i*3], best_params[i*3+1], best_params[i*3+2]) for i in range(N)]))
    
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang_deg = np.rad2deg(best_params[i*3+2])
        inner.append([cx, cy, ang_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0

