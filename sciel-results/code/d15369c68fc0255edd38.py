import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0, 60, ..., 300) for a hexagon with flat top?
# Let's use the standard "pointy-top" or "flat-top"?
# The task allows rotation. Let's define a base hexagon.
# Vertices of a unit hexagon (side 1) centered at origin, "flat-top" (vertex at 0 deg? No, flat top means edge is horizontal).
# Let's use "pointy-top" vertices: (1,0), (0.5, S3/2), ...
# Normals are perpendicular to edges.
# For pointy-top hex:
# Edges are at angles 30, 90, 150, 210, 270, 330?
# Let's stick to the previous definition which worked well for calculations.
# NORM[k] are unit vectors perpendicular to the 6 edges.
# If vertices are at 0, 60, 120... (pointy right), edges are at 30, 90...
# Normals are at 30, 90, 150...
NORM = np.array([
    [np.cos(np.pi/6), np.sin(np.pi/6)],      # 30 deg
    [np.cos(np.pi/2), np.sin(np.pi/2)],      # 90 deg
    [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],  # 150 deg
    [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],  # 210 deg
    [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],  # 270 deg
    [np.cos(11*np.pi/6), np.sin(11*np.pi/6)],# 330 deg
])

# Vertices of a unit hexagon (side 1) centered at origin, "pointy-top" (vertex at 0 deg)
UNIT_VERTS = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

def get_cluster_vertices(theta, tx, ty):
    """
    Returns the 66 vertices (11 hexagons * 6 vertices) of the rigid flower cluster
    rotated by theta and translated by (tx, ty).
    """
    # Base centers for the 11-hexagon flower
    # 1 center at (0,0)
    # 6 neighbors at distance S3
    # 4 outer at distance 2*S3? 
    # Let's verify the "flower" geometry.
    # Center (0,0).
    # Ring 1: 6 hexes. Centers at distance S3. Angles 0, 60, 120, 180, 240, 300.
    # Ring 2: 4 hexes. Where do they go?
    # In a perfect hexagonal lattice, the "gaps" between ring 1 hexes are filled by ring 2?
    # No, the standard 11-hexagon cluster is:
    # Row 1: 1
    # Row 2: 3
    # Row 3: 3
    # Row 4: 3
    # Row 5: 1
    # Total 11.
    # This is a rhombus shape? No, it's a hexagon of side 2?
    # A hexagon of side 2 has 1 + 6 + 12 = 19 hexes.
    # A "flower" of 11 is typically 1 center + 6 ring + 4 in the "corners" of the ring?
    # Let's look at the inspiration's analytic solution:
    # "1 center, 6 ring-1, 4 ring-2".
    # Ring 1 centers: dist S3, angles 0, 60, 120, 180, 240, 300.
    # Ring 2 centers: dist ? 
    # The inspiration said: "4 more at dist 3 + 2*eps. Angles 30, 90, 150, 210".
    # Wait, distance 3? 
    # Distance between center and ring 1 is S3 ~ 1.732.
    # Distance between ring 1 and ring 2?
    # If ring 2 is at 30 deg, and ring 1 is at 0 and 60 deg.
    # Midpoint of 0 and 60 is 30 deg.
    # Distance from center to midpoint of ring 1 edge?
    # Let's calculate the lattice positions.
    # Basis vectors: u = (S3, 0), v = (S3/2, 1.5).
    # Center: (0,0)
    # Ring 1:
    # u, v, v-u, -u, -v, u-v
    # (S3, 0), (S3/2, 1.5), (-S3/2, 1.5), (-S3, 0), (-S3/2, -1.5), (S3/2, -1.5)
    # Ring 2 (the 4 ones):
    # The inspiration suggests angles 30, 90, 150, 210.
    # Let's check the position (S3/2, 1.5) + v = (S3, 3.0)? No.
    # Let's use the lattice coordinates.
    # 11 hexes in a "hexagon" shape of side 2 is 19 hexes.
    # 11 hexes is a "hexagon" of side 2 with 4 corners removed?
    # Or a cross?
    # Let's trust the inspiration's coordinates:
    # Centers:
    # (0,0)
    # 6 at S3 * (cos(k*60), sin(k*60))
    # 4 at 3.0 * (cos(30+k*60), sin(30+k*60))? 
    # Let's check distance.
    # If ring 2 is at dist 3, angle 30.
    # Nearest ring 1 is at dist S3, angle 0.
    # Dist^2 = 9 + 3 - 2*3*S3*cos(30) = 12 - 6*S3*S3/2 = 12 - 9 = 3.
    # Dist = S3. So they touch!
    # So the 4 outer hexes are at distance 3.0, angles 30, 90, 150, 210, 270, 330?
    # Inspiration said 4 more. Angles 30, 90, 150, 210.
    # Why not 270, 330? 
    # Maybe the shape is not symmetric?
    # "4 ring-2 hexagons".
    # If we put them at 30, 90, 150, 210, 270, 330, that's 6.
    # 1 + 6 + 6 = 13.
    # We need 11.
    # So 1 + 6 + 4.
    # The inspiration code:
    # for k in range(4):
    #   angle = 30 + k * 60
    #   x = (3.0 + 2*eps) * cos(angle)
    #   y = (3.0 + 2*eps) * sin(angle)
    # This creates a shape with 4-fold symmetry? No, 60 deg spacing.
    # 30, 90, 150, 210.
    # This leaves gaps at 270 and 330.
    # This shape is a "kite" or asymmetric?
    # Actually, a regular hexagon packing of 11 items usually looks like a hexagon of side 2 (19 items) minus the 4 "side" items?
    # Let's generate the lattice points for a hexagon of side 2 (radius 2 in hex grid).
    # And pick the 11 closest to center?
    # Or just use the inspiration's explicit coordinates.
    
    centers = np.array([
        [0.0, 0.0],
    ])
    
    # Ring 1
    for k in range(6):
        a = k * np.pi / 3.0
        centers = np.vstack([centers, [S3 * np.cos(a), S3 * np.sin(a)]])
        
    # Ring 2 (4 items)
    # The inspiration used angles 30, 90, 150, 210.
    # Let's verify if this is the optimal 11-packing.
    # A 3-4-3-1 lattice?
    # Row 0: 1
    # Row 1: 3
    # Row 2: 4? No, staggered.
    # Let's stick to the inspiration's "flower" which scored 10.0.
    for k in range(4):
        a = np.radians(30 + k * 60)
        centers = np.vstack([centers, [3.0 * np.cos(a), 3.0 * np.sin(a)]])
        
    # Rotate and translate centers
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    C_rot = centers @ R.T
    C_trans = C_rot + np.array([tx, ty])
    
    # Get all vertices
    # Each center has 6 vertices.
    # Vertices of unit hex at origin: UNIT_VERTS
    # Rotated vertices: UNIT_VERTS @ R.T
    # Translated: + C_trans[i]
    
    V_rot = UNIT_VERTS @ R.T
    # V_rot shape (6, 2)
    # C_trans shape (11, 2)
    
    # We need max projection of ALL vertices.
    # Instead of generating 66 points, we can compute max projection analytically?
    # Max_{i} (C_trans[i] + V_rot[j]) . n
    # = Max_{i} (C_trans[i] . n) + Max_{j} (V_rot[j] . n)
    # Since the hexagons are all rotated by the same theta, V_rot is the same for all.
    # So Max_{all vertices} (v . n) = Max_{i} (C_trans[i] . n) + Max_{j} (V_rot[j] . n)
    
    # Max_{j} (V_rot[j] . n) is the support function of the unit hexagon rotated by theta.
    # Let h(theta, n) be the support function of the unit hexagon rotated by theta in direction n.
    # Then we just need Max_{i} (C_trans[i] . n) + h(theta, n).
    
    # This is much faster!
    
    # Pre-compute C_trans . n for all 6 normals?
    # n is fixed (outer hex normals).
    # C_trans . n = (C_rot + t) . n = C_rot . n + t . n
    # C_rot . n = (C @ R.T) . n = C @ (R.T n) = C @ R^T n.
    # Let n' = R.T n.
    # Then we need Max_i (C . n') + t . n + h(theta, n).
    
    # h(theta, n) for a unit hexagon rotated by theta.
    # The support function of a regular hexagon is periodic with period 60 deg.
    # h(theta, n) = 1.0 if n is aligned with a vertex normal?
    # Actually, for a unit hexagon (side 1), the max projection onto a unit vector is:
    # If the vector is along a vertex (0 deg), proj = 1.
    # If the vector is along an edge normal (30 deg), proj = S3/2 ~ 0.866? No.
    # Vertices at 0, 60...
    # Edge normals at 30, 90...
    # Support function s(u) = max v . u.
    # For hexagon, s(u) = 1 / cos(delta) where delta is angle to nearest vertex? No.
    # s(u) = 1 if u is vertex direction.
    # s(u) = S3/2 if u is edge normal direction?
    # Let's check.
    # Vertices: (1,0). Edge normal at 30 deg: (cos30, sin30).
    # Dot product (1,0).(cos30, sin30) = cos30 = S3/2.
    # Next vertex (0.5, S3/2). Dot = 0.5*S3/2 + S3/2*0.5 = S3/2.
    # So yes, support is S3/2 at edge normals, 1 at vertices.
    # It varies as 1/cos(angle_to_vertex) for small angles?
    # Actually, we can just compute it numerically for the 6 vertices.
    # h(theta, n) = max( (UNIT_VERTS @ R.T) . n )
    
    return C_trans, V_rot

def objective(params):
    """
    params: [theta, tx, ty]
    Returns the required outer side length.
    """
    theta, tx, ty = params
    
    # Calculate max projection for each of the 6 outer normals
    max_proj = 0.0
    
    # Pre-compute rotated unit hex vertices support?
    # V_rot = UNIT_VERTS @ R.T
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    V_rot = UNIT_VERTS @ R.T
    
    # Centers
    centers = np.array([
        [0.0, 0.0],
    ])
    for k in range(6):
        a = k * np.pi / 3.0
        centers = np.vstack([centers, [S3 * np.cos(a), S3 * np.sin(a)]])
    for k in range(4):
        a = np.radians(30 + k * 60)
        centers = np.vstack([centers, [3.0 * np.cos(a), 3.0 * np.sin(a)]])
        
    C_rot = centers @ R.T
    C_trans = C_rot + np.array([tx, ty])
    
    for k in range(6):
        n = NORM[k]
        
        # Max projection of centers onto n
        proj_centers = C_trans @ n
        max_center_proj = np.max(proj_centers)
        
        # Max projection of unit hex vertices (rotated) onto n
        proj_verts = V_rot @ n
        max_vert_proj = np.max(proj_verts)
        
        total_max_proj = max_center_proj + max_vert_proj
        
        if total_max_proj > max_proj:
            max_proj = total_max_proj
            
    # Side length S such that apothem = max_proj
    # Apothem = S * S3 / 2
    # S = 2 * max_proj / S3
    return 2.0 * max_proj / S3

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin
    
    # Initial guess
    # The optimal rotation is likely 0 or 30 degrees relative to the lattice.
    # Let's try a few starts.
    best_side = float('inf')
    best_params = [0.0, 0.0, 0.0]
    
    starts = [
        [0.0, 0.0, 0.0],
        [np.pi/6, 0.0, 0.0],  # 30 deg
        [np.pi/12, 0.0, 0.0], # 15 deg
        [0.0, 0.1, 0.1],
        [np.pi/6, 0.1, 0.1],
    ]
    
    for p0 in starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # High precision polish
    if time.time() < deadline - 2:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
            
    theta, tx, ty = best_params
    
    # Construct the inner hexagons
    # We need to return (x, y, angle_degrees) for each of the 11 hexagons.
    # And the outer hexagon center, side, angle.
    # Outer hexagon is centered at (0,0) with angle 0?
    # The problem asks for: inner rows (x, y, angle_degrees), outer (x, y, side, angle).
    # We optimized the cluster relative to a fixed outer hex (angle 0).
    # So outer center is (0,0), angle is 0.
    # Inner hexagons are rotated by theta.
    
    inner = []
    centers = np.array([
        [0.0, 0.0],
    ])
    for k in range(6):
        a = k * np.pi / 3.0
        centers = np.vstack([centers, [S3 * np.cos(a), S3 * np.sin(a)]])
    for k in range(4):
        a = np.radians(30 + k * 60)
        centers = np.vstack([centers, [3.0 * np.cos(a), 3.0 * np.sin(a)]])
        
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    C_rot = centers @ R.T
    C_trans = C_rot + np.array([tx, ty])
    
    angle_deg = np.degrees(theta)
    
    for i in range(N):
        inner.append([float(C_trans[i, 0]), float(C_trans[i, 1]), float(angle_deg)])
        
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# Fallback if optimization fails or returns inf
def construct_safe():
    try:
        return construct()
    except Exception:
        # Fallback to a simple valid configuration
        # 11 hexes in a line
        inner = []
        for i in range(N):
            inner.append([i * S3, 0.0, 0.0])
        # Outer side length for this line
        # Length is 10 * S3 + 2 (width of hex)
        # Approx side length
        return inner, (0.0, 0.0), 20.0, 0.0

# Wrap the main construct to ensure safety
_original_construct = construct
def construct():
    try:
        return _original_construct()
    except Exception:
        return construct_safe()

# Keep the name 'construct' as the entry point
# The evaluator calls construct()
