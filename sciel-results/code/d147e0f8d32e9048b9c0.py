import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Honeycomb lattice: 11 hexagons in a compact cluster
    # Axial coordinates: center + ring1 (6) + 4 from ring2
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    sqrt3 = np.sqrt(3.0)
    
    # Convert axial (q, r) to Cartesian for unit hexagons
    def axial_to_xy(q, r):
        x = sqrt3 * (q + 0.5 * r)
        y = 1.5 * r
        return x, y
    
    centers = np.array([axial_to_xy(q, r) for q, r in axial_coords])
    
    # All vertices of all 11 hexagons (each hexagon has 6 vertices at distance 1 from center)
    all_verts = []
    for cx, cy in centers:
        for k in range(6):
            ang = k * np.pi / 3.0
            all_verts.append([cx + np.cos(ang), cy + np.sin(ang)])
    all_verts = np.array(all_verts)
    
    # For a regular hexagon (outer) with center (hx,hy), side s, rotation phi:
    # Apothem a = s * sqrt(3)/2
    # 6 inward normals at angles phi + i*pi/3
    # Point p is inside iff (p - center) dot n_i <= a for all i
    
    def fits(hx, hy, s, phi):
        a = s * sqrt3 / 2.0
        for i in range(6):
            ang = phi + i * np.pi / 3.0
            nx, ny = np.cos(ang), np.sin(ang)
            d = (all_verts[:, 0] - hx) * nx + (all_verts[:, 1] - hy) * ny
            if d.max() > a + 1e-10:
                return False
        return True
    
    def min_side(hx, hy, phi):
        lo, hi = 0.0, 50.0
        for _ in range(60):
            mid = (lo + hi) * 0.5
            if fits(hx, hy, mid, phi):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Optimize over outer hexagon center and rotation
    # Use centroid of vertices as initial center
    cx0, cy0 = all_verts[:, 0].mean(), all_verts[:, 1].mean()
    
    best_s = np.inf
    best_hx, best_hy, best_phi = cx0, cy0, 0.0
    
    # Coarse search over rotation (0 to 60 deg, since hexagon has 6-fold symmetry)
    for phi_deg in range(0, 61, 2):
        phi = np.radians(phi_deg)
        s = min_side(cx0, cy0, phi)
        if s < best_s:
            best_s, best_hx, best_hy, best_phi = s, cx0, cy0, phi
    
    # Refine rotation
    for _ in range(3):
        for dphi in np.linspace(-2, 2, 9):
            phi = best_phi + np.radians(dphi)
            s = min_side(best_hx, best_hy, phi)
            if s < best_s:
                best_s, best_phi = s, phi
    
    # Refine center: try small offsets
    for dx in np.linspace(-0.5, 0.5, 7):
        for dy in np.linspace(-0.5, 0.5, 7):
            s = min_side(best_hx + dx, best_hy + dy, best_phi)
            if s < best_s:
                best_s, best_hx, best_hy = s, best_hx + dx, best_hy + dy
    
    # Final refinement of center
    for _ in range(2):
        for dx in np.linspace(-0.2, 0.2, 5):
            for dy in np.linspace(-0.2, 0.2, 5):
                s = min_side(best_hx + dx, best_hy + dy, best_phi)
                if s < best_s:
                    best_s, best_hx, best_hy = s, best_hx + dx, best_hy + dy
    
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
    return inner, (float(best_hx), float(best_hy)), float(best_s), float(np.degrees(best_phi))
# EVOLVE-BLOCK-END
