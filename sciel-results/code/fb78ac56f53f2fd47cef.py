import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _all_inner_vertices(inner):
    """Get all vertices of all inner hexagons."""
    all_v = []
    for (x, y, ang) in inner:
        all_v.append(_hex_vertices(x, y, 1.0, ang))
    return np.vstack(all_v)

def _min_enclosing_hex_side(verts, cx, cy, angle_deg):
    """
    Given all inner vertices and a candidate outer hex center/rotation,
    compute the minimum side length needed.
    The outer hex has 6 half-planes. For each direction d_k (angle = angle_deg + k*60),
    the constraint is: dot(p - c, d_k) <= s * sqrt(3)/2 for all p.
    So s >= max_k max_p dot(p-c, d_k) / (sqrt(3)/2)
    """
    a = np.radians(angle_deg)
    # 6 outward normal directions of the outer hexagon sides
    normals = np.array([[np.cos(a + k * np.pi / 3), np.sin(a + k * np.pi / 3)] for k in range(6)])
    # Actually for a regular hexagon, the apothem direction is perpendicular to each side.
    # The 6 side normals are at angles: angle_deg + k*60 for k=0..5? Let me reconsider.
    # A regular hexagon with "pointy-top" orientation has vertices at 0,60,120,...
    # The sides are between consecutive vertices. The outward normal to side k is at angle (k+0.5)*60 + rotation.
    # Let's use: the 6 support directions are at angles: angle_deg + 30 + k*60 for k=0..5
    # This gives the apothem directions.
    support_angles = np.radians(angle_deg + 30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(support_angles), np.sin(support_angles)])
    
    # For each vertex p, and each normal n: dot(p - c, n)
    # The max over all p and n gives the required apothem
    diff = verts - np.array([cx, cy])
    # shape: (num_verts, 2) x (6, 2) -> (num_verts, 6)
    projections = diff @ normals.T  # (num_verts, 6)
    max_proj = projections.max()
    apothem_needed = max_proj
    # side = apothem / (sqrt(3)/2)
    side = apothem_needed / (np.sqrt(3) / 2)
    return max(0.0, side)

def _compute_outer_hex(verts):
    """
    Find the optimal (cx, cy, angle) and minimal side for enclosing hex.
    We'll do a small grid search over angle and center, then refine.
    """
    # Initial guess: center of mass of all vertices
    cx0, cy0 = verts[:, 0].mean(), verts[:, 1].mean()
    
    best_side = float('inf')
    best_params = (cx0, cy0, 0.0)
    
    # Try several rotation angles
    for angle_deg in range(0, 360, 15):
        # For a given rotation, the optimal center minimizes the max projection
        # This is a linear programming problem, but let's do a simple approach:
        # The center that minimizes max projection is related to the Chebyshev center
        # For now, use the centroid and check
        side = _min_enclosing_hex_side(verts, cx0, cy0, angle_deg)
        if side < best_side:
            best_side = side
            best_params = (cx0, cy0, float(angle_deg))
    
    # Refine: try shifting center
    for dx in np.linspace(-2, 2, 17):
        for dy in np.linspace(-2, 2, 17):
            for angle_deg in np.linspace(0, 180, 13):
                side = _min_enclosing_hex_side(verts, cx0 + dx, cy0 + dy, angle_deg)
                if side < best_side:
                    best_side = side
                    best_params = (cx0 + dx, cy0 + dy, float(angle_deg))
    
    return best_params, best_side

def _check_overlap(inner, eps=1e-6):
    """Check if any two inner hexagons overlap. Returns True if valid (no overlap)."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            # Check if any vertex of i is inside j or vice versa, or edges cross
            # Simple check: if min distance between any vertex pair < some threshold, likely overlap
            # More robust: check if centroids are close enough
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            # Two unit hexagons can overlap if centers are within ~2*sqrt(3)/2 + 1 or so
            # Actually the maximum distance between centers for non-overlapping is at least the sum of inradii in worst case
            # For same orientation, min center distance for no overlap is sqrt(3) (flat-to-flat)
            # For different orientations, it can be less
            # Let's use a conservative check: if dist < 1.5, check more carefully
            if dist < 2.5:
                # Check if any vertex of hex i is inside hex j
                # A point is inside a convex hexagon if it's on the same side of each edge
                def point_in_hex(px, py, hex_verts):
                    n_v = len(hex_verts)
                    for k in range(n_v):
                        x1, y1 = hex_verts[k]
                        x2, y2 = hex_verts[(k + 1) % n_v]
                        # Cross product to check side
                        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                        if cross < -eps:
                            return False
                    return True
                
                # Check vertices of i against j
                for vx, vy in vi:
                    if point_in_hex(vx, vy, vj):
                        return False
                # Check vertices of j against i
                for vx, vy in vj:
                    if point_in_hex(vx, vy, vi):
                        return False
    return True

def construct():
    start_time = time.time()
    
    # Initial placement: honeycomb-like pattern
    # Place hexagons in a hexagonal arrangement
    # For 12 hexagons, try a 3-4-3-2 or similar pattern
    # Honeycomb: each hex has 6 neighbors
    # Let's place them in rows
    
    # A good initial guess: place in a roughly circular/hexagonal cluster
    # Using honeycomb coordinates
    # For 12 hexes, one arrangement:
    # Row 0: 4 hexes
    # Row 1: 3 hexes (offset)
    # Row 2: 3 hexes (offset other way)  
    # Row 3: 2 hexes
    # That's 12
    
    # Actually, let me use a more systematic approach
    # Place hexagons at honeycomb lattice positions
    
    # Honeycomb lattice: positions are at (sqrt(3)*i + sqrt(3)/2*j, 1.5*j) for integer i,j
    # with appropriate offsets for the two sublattices
    
    # Let's just place them in a grid-like pattern initially and let SA fix it
    dx = np.sqrt(3)  # horizontal spacing for flat-top hexagons
    dy = 1.5  # vertical spacing between rows
    
    # Place in a 4x3 grid with offset
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y, 0.0])
    
    inner = positions[:N]
    
    # Center the configuration
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    inner = [[p[0] - cx, p[1] - cy, p[2]] for p in inner]
    
    # Simulated Annealing
    best_inner = [list(p) for p in inner]
    best_score = float('inf')
    
    # Compute initial score
    verts = _all_inner_vertices(inner)
    _, init_side = _compute_outer_hex(verts)
    if _check_overlap(inner):
        best_score = init_side
    else:
        best_score = init_side + 100  # penalty for overlap
    
    T = 0.5  # initial temperature
    cooling = 0.999
    min_T = 0.001
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
        if T < min_T:
            break
        
        # Propose a move
        move_type = np.random.random()
        new_inner = [list(p) for p in inner]
        
        if move_type < 0.6:
            # Move one hexagon
            idx = np.random.randint(N)
            new_inner[idx][0] += np.random.normal(0, 0.3)
            new_inner[idx][1] += np.random.normal(0, 0.3)
        elif move_type < 0.8:
            # Rotate one hexagon
            idx = np.random.randint(N)
            new_inner[idx][2] += np.random.uniform(-30, 30)
            new_inner[idx][2] = new_inner[idx][2] % 360
        elif move_type < 0.9:
            # Move two hexagons
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            new_inner[idx1][0] += np.random.normal(0, 0.3)
            new_inner[idx1][1] += np.random.normal(0, 0.3)
            new_inner[idx2][0] += np.random.normal(0, 0.3)
            new_inner[idx2][1] += np.random.normal(0, 0.3)
        else:
            # Shift all
            dx_all = np.random.normal(0, 0.1)
            dy_all = np.random.normal(0, 0.1)
            for i in range(N):
                new_inner[i][0] += dx_all
                new_inner[i][1] += dy_all
        
        # Check validity
        if not _check_overlap(new_inner):
            new_score = best_score + 50
        else:
            verts_new = _all_inner_vertices(new_inner)
            _, new_side = _compute_outer_hex(verts_new)
            new_score = new_side
        
        # Accept or reject
        if new_score < best_score or np.random.random() < np.exp(-(new_score - best_score) / T):
            inner = new_inner
            if new_score < best_score:
                best_score = new_score
                best_inner = [list(p) for p in inner]
        
        T *= cooling
    
    # Final: use best_inner
    inner_final = best_inner
    verts = _all_inner_vertices(inner_final)
    (cx, cy, angle_deg), side = _compute_outer_hex(verts)
    
    # Final overlap check
    if not _check_overlap(inner_final):
        # Fallback to a valid configuration
        # Use the simple row arrangement
        inner_final = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        verts = _all_inner_vertices(inner_final)
        (cx, cy, angle_deg), side = _compute_outer_hex(verts)
    
    return inner_final, (cx, cy), side, angle_deg
