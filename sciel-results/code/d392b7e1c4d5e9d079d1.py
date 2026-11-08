"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    sqrt3 = np.sqrt(3.0)
    
    def hex_vertices(cx, cy, angle_rad):
        """Vertices of unit regular hexagon centered at (cx,cy) with rotation angle_rad."""
        verts = []
        for k in range(6):
            a = angle_rad + k * np.pi / 3.0
            verts.append([cx + np.cos(a), cy + np.sin(a)])
        return np.array(verts)
    
    def enclosing_hex_side(vertices, ox, oy, phi_rad):
        """Compute minimum side length of regular hexagon centered at (ox,oy) with rotation phi_rad
        that contains all points in vertices."""
        max_proj = 0.0
        for k in range(6):
            angle = phi_rad + k * np.pi / 3.0
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # Project relative to center
            dx = vertices[:, 0] - ox
            dy = vertices[:, 1] - oy
            projections = dx * cos_a + dy * sin_a
            proj_max = np.max(projections)
            if proj_max > max_proj:
                max_proj = proj_max
        # inradius = s * sqrt(3)/2, so s = max_proj / (sqrt(3)/2) = max_proj * 2/sqrt3
        return max_proj * 2.0 / sqrt3
    
    # Generate multiple candidate arrangements
    arrangements = []
    
    # Arrangement 1: Hexagonal cluster radius 1 (7 hex) + 4 extra
    base = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if max(abs(q), abs(r), abs(q + r)) <= 1:
                base.append((q, r))
    
    # Try different sets of 4 extra positions
    extra_options = [
        [(2,0), (0,2), (-2,0), (0,-2)],
        [(2,-1), (1,-2), (-2,1), (-1,2)],
        [(2,0), (0,2), (-1,2), (-1,-2)],
        [(2,0), (1,1), (-2,0), (0,-2)],
        [(2,-1), (1,-2), (-1,2), (0,-2)],
        [(2,0), (0,2), (2,-1), (1,-2)],
        [(2,0), (0,2), (-2,1), (-1,2)],
        [(2,-1), (1,-2), (-2,0), (0,-2)],
        [(2,0), (-2,0), (0,2), (0,-2)],
        [(2,0), (0,2), (-2,0), (-1,2)],
    ]
    
    for extras in extra_options:
        pos = list(base) + list(extras)
        if len(pos) == N:
            arrangements.append(pos)
    
    # Arrangement 2: Linear-ish but compact
    # 3-4-4 pattern or similar
    linear_arrangements = [
        # Row-based: 3,4,4
        [(-1,0),(0,0),(1,0), (-2,1),(-1,1),(0,1),(1,1), (-2,-1),(-1,-1),(0,-1),(1,-1)],
        # 2,3,3,3
        [(-1,0),(0,0), (-2,1),(-1,1),(0,1), (-2,-1),(-1,-1),(0,-1), (-3,0),(-4,0),(-5,0)],
        # 4,3,4
        [(-2,0),(-1,0),(0,0),(1,0), (-1,1),(0,1),(1,1), (-2,-1),(-1,-1),(0,-1),(1,-1)],
        # 5,3,3
        [(-2,0),(-1,0),(0,0),(1,0),(2,0), (-1,1),(0,1),(1,1), (-1,-1),(0,-1),(1,-1)],
        # 3,5,3
        [(-1,0),(0,0),(1,0), (-2,1),(-1,1),(0,1),(1,1),(2,1), (-1,-1),(0,-1),(1,-1)],
    ]
    for arr in linear_arrangements:
        if len(arr) == N:
            arrangements.append(arr)
    
    best_result = None
    best_s = float('inf')
    
    for arr in arrangements:
        # Convert axial to Cartesian
        centers = np.array([[sqrt3 * (q + r/2.0), 1.5 * r] for q, r in arr])
        
        # Center at origin
        center_offset = centers.mean(axis=0)
        centers = centers - center_offset
        
        # Get all vertices with angle 0
        all_vertices = []
        for cx, cy in centers:
            verts = hex_vertices(cx, cy, 0.0)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # Try different outer hexagon rotations and centers
        # First try center at origin
        ox, oy = 0.0, 0.0
        
        best_local_s = float('inf')
        best_local_phi = 0.0
        
        # Coarse search
        for phi_deg in range(0, 360, 3):
            phi = np.radians(phi_deg)
            s = enclosing_hex_side(all_vertices, ox, oy, phi)
            if s < best_local_s:
                best_local_s = s
                best_local_phi = phi
        
        # Fine search
        for phi_deg in np.arange(best_local_phi - 3, best_local_phi + 3, 0.5):
            phi = np.radians(phi_deg)
            s = enclosing_hex_side(all_vertices, ox, oy, phi)
            if s < best_local_s:
                best_local_s = s
                best_local_phi = phi
        
        for phi_deg in np.arange(best_local_phi - 0.5, best_local_phi + 0.5, 0.1):
            phi = np.radians(phi_deg)
            s = enclosing_hex_side(all_vertices, ox, oy, phi)
            if s < best_local_s:
                best_local_s = s
                best_local_phi = phi
        
        if best_local_s < best_s:
            best_s = best_local_s
            best_result = (centers, best_local_phi)
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback: simple linear arrangement
        centers = np.array([[sqrt3 * k, 0.0] for k in range(-(N//2), N - N//2)])
        center_offset = centers.mean(axis=0)
        centers = centers - center_offset
        all_vertices = np.vstack([hex_vertices(cx, cy, 0.0) for cx, cy in centers])
        best_s = 0.0
        best_phi = 0.0
        for phi_deg in range(0, 360, 5):
            phi = np.radians(phi_deg)
            s = enclosing_hex_side(all_vertices, 0.0, 0.0, phi)
            if s < best_s or best_s == 0.0:
                best_s = s
                best_phi = phi
        best_result = (centers, best_phi)
    
    centers, best_phi = best_result
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
    
    return inner, (0.0, 0.0), float(best_s), float(best_phi)
# EVOLVE-BLOCK-END
