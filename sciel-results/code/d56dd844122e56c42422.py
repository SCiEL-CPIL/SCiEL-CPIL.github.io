import numpy as np

N = 11

def construct():
    # 1. Generate lattice points
    # Axial coordinates (q, r)
    # We want 11 points closest to (0,0)
    # Dist^2 = q^2 + q*r + r^2
    
    points = []
    # Manually select 11 points for compactness
    # Ring 0: (0,0)
    points.append((0,0))
    # Ring 1: 6 points
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    points.extend(ring1)
    # Ring 2: 12 points
    ring2 = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), 
             (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)]
    
    # We need 4 more from ring2.
    # To minimize bounding box, we should probably pick points that don't extend the corners too much?
    # Actually, let's try a few candidates for the 4 points.
    # Candidate A: 4 consecutive? No.
    # Candidate B: Symmetric pairs?
    # Since we can rotate the container, the specific choice might matter less if we optimize phi well.
    # But some shapes are inherently "wider".
    # Let's try picking 4 points that are "spread out" to average out the radius.
    # E.g., (2,0), (0,2), (-2,1), (-1,-1)?
    
    # Let's try a simple heuristic: pick the 4 that minimize the max distance from origin?
    # All are at same distance.
    # So it depends on the angles.
    # Let's try: (2,0), (2,-2), (-2,0), (-2,2)? No, those are opposite corners.
    # Let's try: (2,0), (0,2), (-2,1), (-1,-1) ... 
    
    # Actually, let's just pick 4 specific ones and optimize.
    # If we pick (2,0), (2,-1), (2,-2), (1,-2) -> a block of 4.
    # This creates a "tail".
    # A more symmetric distribution is better.
    # Let's pick: (2,0), (0,2), (-2,1), (-2,-1)? No.
    
    # Let's try: (2,0), (-2,0) are opposite. (0,2), (0,-2) are opposite.
    # If we pick these 4, we extend the 4 corners?
    # Ring 2 indices:
    # 0: (2,0) - 0 deg
    # 1: (2,-1) - 30 deg
    # 2: (2,-2) - 60 deg
    # 3: (1,-2) - 90 deg
    # 4: (0,-2) - 120 deg
    # 5: (-1,-1) - 150 deg
    # 6: (-2,0) - 180 deg
    # 7: (-2,1) - 210 deg
    # 8: (-2,2) - 240 deg
    # 9: (-1,2) - 270 deg
    # 10: (0,2) - 300 deg
    # 11: (1,1) - 330 deg
    
    # If we pick 0, 3, 6, 9 (0, 90, 180, 270 degrees), we get a "plus" shape extension.
    # This might be good for a hexagon container?
    # Let's try indices 0, 3, 6, 9.
    
    selected_ring2_indices = [0, 3, 6, 9]
    for idx in selected_ring2_indices:
        points.append(ring2[idx])
        
    # Convert to Cartesian
    # d = sqrt(3)
    d = np.sqrt(3)
    centers = []
    for q, r in points:
        x = d * (q + r/2)
        y = d * (r * np.sqrt(3)/2)
        centers.append((x, y))
        
    # Generate vertices
    # Unit hexagon vertices (flat-top, side 1)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_verts = []
    for cx, cy in centers:
        for v in verts:
            all_verts.append((cx + v[0], cy + v[1]))
            
    all_verts = np.array(all_verts)
    
    # Optimize container rotation phi
    # Minimize S(phi)
    # S(phi) = 2/sqrt(3) * max_{v in all_verts, k in 0..5} | dot(v, n_k) |
    # n_k = (cos(phi + 30 + 60k), sin(phi + 30 + 60k))
    
    def get_side_length(phi):
        max_proj = 0
        for k in range(6):
            theta = phi + np.deg2rad(30 + 60*k)
            nx = np.cos(theta)
            ny = np.sin(theta)
            projs = np.abs(all_verts[:, 0] * nx + all_verts[:, 1] * ny)
            max_proj = max(max_proj, np.max(projs))
        return (2.0 / np.sqrt(3)) * max_proj

    # Search phi in [0, 30]
    # Use golden section or simple sampling
    phi_min = 0
    phi_max = np.deg2rad(30)
    
    # Sampling
    phis = np.linspace(phi_min, phi_max, 1000)
    best_phi = phis[0]
    best_S = float('inf')
    for p in phis:
        S = get_side_length(p)
        if S < best_S:
            best_S = S
            best_phi = p
            
    # Refine with ternary search
    for _ in range(50):
        m1 = phi_min + (phi_max - phi_min) / 3
        m2 = phi_max - (phi_max - phi_min) / 3
        if get_side_length(m1) < get_side_length(m2):
            phi_max = m2
        else:
            phi_min = m1
    best_phi = (phi_min + phi_max) / 2
    best_S = get_side_length(best_phi)
    
    # Prepare output
    inner = []
    for i, (cx, cy) in enumerate(centers):
        # Angle of inner hexagon. We used flat-top for vertices.
        # So angle_degrees is 0.
        inner.append([cx, cy, 0.0])
        
    center = [0.0, 0.0]
    side = best_S
    angle = best_phi * 180.0 / np.pi
    
    return inner, center, side, angle
