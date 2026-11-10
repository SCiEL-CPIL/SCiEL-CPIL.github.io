import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_s_for_hexagon(vertices, center, angle_deg):
    """Min side length of regular hexagon with given center/angle containing all vertices."""
    rel = vertices - center
    a = np.radians(angle_deg)
    # Outward edge normals
    normals = np.column_stack([
        np.cos(a + np.radians(60*i + 30)),
        np.sin(a + np.radians(60*i + 30))
    ] for i in range(6))
    max_proj = np.max(rel @ normals.T)
    return max(0.0, max_proj / (np.sqrt(3)/2))

def hexagon_packing_12():
    t0 = time.time()
    deadline = t0 + 570
    
    # 12 hex positions in axial coords (compact hexagonal cluster)
    axial = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
             (2,0),(1,1),(0,2),(-1,2),(-2,1)]
    
    best_s = np.inf
    best_theta = 0
    best_phi = 0
    best_d = np.sqrt(3) * 1.0001
    
    # Coarse search
    for d_mult in [1.0001, 1.0003, 1.0005]:
        d = np.sqrt(3) * d_mult
        centers = np.array([(d*(q+r/2), d*r*np.sqrt(3)/2) for q,r in axial])
        
        verts = np.vstack([_hex_vertices(cx,cy,1.0,0) for cx,cy in centers])
        centroid = verts.mean(axis=0)
        
        for theta in np.linspace(0, 60, 121):
            if time.time() > deadline:
                break
            th = np.radians(theta)
            R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            rv = verts @ R.T
            rc = centroid @ R.T
            
            for phi in np.linspace(0, 60, 13):
                s = _min_s_for_hexagon(rv, rc, phi)
                if s < best_s:
                    best_s, best_theta, best_phi, best_d = s, theta, phi, d
    
    # Fine search
    for _ in range(4):
        d = best_d
        centers = np.array([(d*(q+r/2), d*r*np.sqrt(3)/2) for q,r in axial])
        verts = np.vstack([_hex_vertices(cx,cy,1.0,0) for cx,cy in centers])
        centroid = verts.mean(axis=0)
        
        improved = False
        for theta in np.linspace(best_theta-1, best_theta+1, 201):
            if time.time() > deadline:
                break
            th = np.radians(theta)
            R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            rv = verts @ R.T
            rc = centroid @ R.T
            for phi in np.linspace(best_phi-1, best_phi+1, 21):
                s = _min_s_for_hexagon(rv, rc, phi)
                if s < best_s - 1e-12:
                    best_s, best_theta, best_phi = s, theta, phi
                    improved = True
        if not improved:
            break
    
    # Final
    d = best_d
    centers = np.array([(d*(q+r/2), d*r*np.sqrt(3)/2) for q,r in axial])
    th = np.radians(best_theta)
    R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    final_centers = centers @ R.T
    centroid = (verts @ R.T).mean(axis=0)
    
    inner = np.array([[final_centers[i][0], final_centers[i][1], best_theta] for i in range(12)])
    outer = np.array([centroid[0], centroid[1], best_phi])
    
    return inner, outer, best_s
