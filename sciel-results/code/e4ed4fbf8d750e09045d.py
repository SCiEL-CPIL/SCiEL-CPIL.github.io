"""Rigid hex-grid cluster + smooth log-sum-exp surrogate for outer side length,
with Nelder-Mead refinement over (phi, tx, ty) and final exact polish."""
N = 11
import numpy as np
import math
import time
from scipy.optimize import minimize

# ---- geometry helpers ----------------------------------------------------

def _hex_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    angs = a + np.pi/6 + np.arange(6)*np.pi/3
    return np.column_stack([cx + s*np.cos(angs), cy + s*np.sin(angs)])

def _all_verts(positions, angles, s=1.0):
    n = len(positions)
    v = np.empty((n*6, 2))
    for i in range(n):
        v[i*6:(i+1)*6] = _hex_verts(positions[i][0], positions[i][1], s, angles[i])
    return v

def _exact_side(positions, angles, phi_rad, center=(0.0, 0.0)):
    """Exact required side length of outer hex with rotation phi centered at center."""
    verts = _all_verts(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    max_proj = 0.0
    for t in thetas:
        proj = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (math.sqrt(3)/2)

def _smooth_side(positions, angles, x, beta=30.0):
    """Smooth surrogate for outer side length.
    x = [phi, tx, ty]: outer rotation, and cluster translation.
    The outer hexagon is centered at origin; inner positions are translated by (tx,ty).
    """
    phi, tx, ty = x[0], x[1], x[2]
    # Translate inner positions
    verts = _all_verts(positions, angles, 1.0)
    verts = verts - np.array([tx, ty])
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    # Stack all projections: (6 directions, n*6 vertices)
    projs = np.zeros((6, len(verts)))
    for k, t in enumerate(thetas):
        projs[k] = verts[:, 0]*math.cos(t) + verts[:, 1]*math.sin(t)
    # Smooth max over vertices for each direction
    m = np.max(projs, axis=1)  # (6,)
    lse_per_dir = m + (1.0/beta) * np.log(np.sum(np.exp(projs - m[:, None]), axis=1))
    # Smooth max over directions
    m2 = np.max(lse_per_dir)
    lse_final = m2 + (1.0/beta) * math.log(np.sum(np.exp((lse_per_dir - m2)*beta)))
    return lse_final / (math.sqrt(3)/2)

# 36 candidate support directions (every 5 deg) for the outer hexagon's 6 normals.
# The outer hexagon has 6 normals at angles phi + (2k+1)*pi/6 for k=0..5.
# We optimize over phi (outer rotation) and over the cluster's translation.
# The side length is determined by the max over all vertices of the projection onto
# each of the 6 normals, divided by (sqrt(3)/2).



# ---- SAT overlap check ---------------------------------------------------

def _sat_gap(p1, a1, p2, a2, s=1.0):
    v1 = _hex_verts(p1[0], p1[1], s, a1)
    v2 = _hex_verts(p2[0], p2[1], s, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        nv = np.array([-e[1], e[0]])
        n = np.linalg.norm(nv)
        if n < 1e-12:
            continue
        nv /= n
        proj1 = v1 @ nv
        proj2 = v2 @ nv
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-9):
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -tol:
                return False
    return True

# ---- Build the rigid 3-4-3 cluster ---------------------------------------

def _build_343_cluster():
    """
    3-4-3 arrangement of unit hexagons on a hex lattice.
    Using axial coordinates (q, r) with pointy-top hexagons.
    Row 0 (top): 3 hexagons
    Row 1 (middle): 4 hexagons  
    Row 2 (bottom): 3 hexagons
    
    For pointy-top hexagons with side 1:
    - Horizontal spacing between adjacent hexes in same row: sqrt(3)
    - Vertical spacing between rows: 1.5
    - Offset between rows: sqrt(3)/2
    
    Actually, let me use the standard hex grid:
    For flat-top hexagons (angle 0):
    - x spacing: 1.5 * side
    - y spacing: sqrt(3) * side
    - Offset for odd columns: sqrt(3)/2 * side
    
    Let me just place them manually for a compact 3-4-3:
    """
    s3 = math.sqrt(3)
    # Flat-top hexagons, side 1
    # Hex centers in a compact triangular arrangement:
    # Row 0: 3 hexes at y = 0, x = -s3, 0, s3
    # Row 1: 4 hexes at y = -1.5, x = -1.5*s3/2 - s3/2, ... 
    # Actually let me use axial coordinates properly.
    
    # Axial coords (q, r) for flat-top:
    # x = 1.5*q, y = sqrt(3)*(r + q/2)
    # Adjacent: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    # 3-4-3 cluster:
    # Center at (0,0)
    # Ring 1: 6 hexes
    # We need 11 total. 3+4+3 = 10, plus 1 center = 11.
    
    # Let me define specific axial coords:
    # Row structure (by y-level):
    # Using (q, r):
    # y = sqrt(3)*(r + q/2)
    
    # Let me just enumerate a known compact 11-hex packing:
    # Center: (0,0)
    # 6 around it: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 more from ring 2: (2,0), (-2,1), (0,2), (1,-2)  -- need to check compactness
    
    # Actually, the 3-4-3 pattern:
    # Top row (3): (-1, 1), (0, 1), (1, 1)  -> but (1,1) isn't adjacent to center
    # Let me think differently.
    
    # Compact 11-hex cluster (like a hexagon of "radius" ~1.5):
    # Axial coords:
    coords = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (0, 2), (-1, 2),  # part of ring 2 (4 more)
    ]
    # That's 1+6+4 = 11. Let me verify they're all distinct.
    # (2,0), (1,1), (0,2), (-1,2) - yes, all distinct from above.
    
    def axial_to_xy(q, r):
        return (1.5 * q, s3 * (r + q / 2.0))
    
    positions = [axial_to_xy(q, r) for q, r in coords]
    angles = [0.0] * 11
    return positions, angles

def _build_alternative_clusters():
    """Generate several candidate rigid clusters."""
    s3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        return (1.5 * q, s3 * (r + q / 2.0))
    
    clusters = []
    
    # Cluster 1: center + ring1 + 4 from ring2 (as above)
    c1 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2),
    ]
    clusters.append([axial_to_xy(q, r) for q, r in c1])
    
    # Cluster 2: center + ring1 + different 4 from ring2
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    for subset in [
        [(2,0), (0,2), (-2,2), (0,-2)],       # symmetric
        [(2,0), (1,1), (-2,0), (-1,-1)],       # another symmetric
        [(2,0), (1,1), (0,2), (1,-2)],         # asymmetric
        [(2,0), (0,2), (-2,1), (1,-2)],        # another
    ]:
        c = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + subset
        clusters.append([axial_to_xy(q, r) for q, r in c])
    
    # Cluster 3: 5-1-5 (two rows of 5 with 1 in middle) - might be wider
    c3 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,2),  # 5 from ring 2
    ]
    # That's 12, too many. Let me do 5+1+5 differently.
    # Actually 3-4-3 is the standard compact shape.
    
    return clusters

# ---- Main construct -------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    s3 = math.sqrt(3)

    # Generate candidate clusters
    clusters = _build_alternative_clusters()

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0

    for cluster in clusters:
        if time.time() > deadline - 30.0:
            break

        angles = [0.0] * N
        if not _check_disjoint(cluster, angles):
            continue

        # Centroid of cluster
        cx0 = np.mean([p[0] for p in cluster])
        cy0 = np.mean([p[1] for p in cluster])

        # Coarse grid search to find a good starting point
        best_cluster_side = float('inf')
        best_x0 = None

        for phi_deg in np.arange(0, 60, 3.0):
            if time.time() > deadline - 25.0:
                break
            phi = math.radians(phi_deg)
            for dx in np.arange(-1.5, 1.51, 0.5):
                for dy in np.arange(-1.5, 1.51, 0.5):
                    # x = [phi, tx, ty] where tx,ty is the translation applied to cluster
                    # The outer hex is at origin. Cluster centers are at cluster[i] + (tx,ty).
                    # We want the cluster centered near origin, so tx ≈ -cx0, ty ≈ -cy0.
                    tx = -cx0 + dx
                    ty = -cy0 + dy
                    x = [phi, tx, ty]
                    side = _exact_side(cluster, angles, phi, (-tx, -ty))
                    # Wait, _exact_side takes center as the outer hex center.
                    # Let me reframe: positions are in absolute coords, outer hex at center.
                    # So if cluster is at (cx0,cy0) and we want outer at (cx0+dx, cy0+dy),
                    # the relative positions are cluster - (cx0+dx, cy0+dy).
                    # But in _smooth_side, x=[phi,tx,ty] means positions are translated by (tx,ty).
                    # So the effective positions are cluster[i] + (tx,ty), outer at origin.
                    # For the cluster to be centered at origin: tx = -cx0, ty = -cy0.
                    # Let me just use _exact_side with center=(cx0+dx, cy0+dy) on the raw cluster.
                    center = (cx0 + dx, cy0 + dy)
                    side = _exact_side(cluster, angles, phi, center)
                    if side < best_cluster_side:
                        best_cluster_side = side
                        best_x0 = [phi, -cx0 + dx, -cy0 + dy]

        if best_x0 is None:
            continue

        # Nelder-Mead refinement using smooth surrogate
        # Multiple restarts from nearby points
        for restart in range(4):
            if time.time() > deadline - 15.0:
                break

            if restart == 0:
                x_start = np.array(best_x0)
            else:
                x_start = np.array(best_x0) + np.array([
                    np.random.uniform(-0.1, 0.1),
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.3, 0.3)
                ])

            def obj(x):
                return _smooth_side(cluster, angles, x, beta=30.0)

            try:
                res = minimize(obj, x_start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < best_cluster_side:
                    # Evaluate exact side at the solution
                    phi_sol, tx_sol, ty_sol = res.x
                    center = (-tx_sol, -ty_sol)
                    exact = _exact_side(cluster, angles, phi_sol, center)
                    if exact < best_cluster_side:
                        best_cluster_side = exact
                        best_x0 = [phi_sol, tx_sol, ty_sol]
            except Exception:
                pass

        if best_cluster_side < best_side:
            # Convert: positions relative to outer hex center (origin)
            phi_sol, tx_sol, ty_sol = best_x0
            best_positions = [(p[0] + tx_sol, p[1] + ty_sol) for p in cluster]
            best_angles = angles
            best_phi = phi_sol
            best_side = best_cluster_side

    # Final polish: fine grid on phi and center
    if best_positions is not None:
        # Fine-tune phi
        for phi_deg in np.arange(best_phi*180/math.pi - 2, best_phi*180/math.pi + 2.01, 0.1):
            if time.time() > deadline - 5.0:
                break
            phi = math.radians(phi_deg)
            side = _exact_side(best_positions, best_angles, phi, (0.0, 0.0))
            if side < best_side:
                best_side = side
                best_phi = phi

        # Fine-tune center (shift all positions)
        for dx in np.arange(-0.3, 0.31, 0.02):
            for dy in np.arange(-0.3, 0.31, 0.02):
                if time.time() > deadline - 3.0:
                    break
                shifted = [(p[0]+dx, p[1]+dy) for p in best_positions]
                side = _exact_side(shifted, best_angles, best_phi, (0.0, 0.0))
                if side < best_side:
                    best_side = side
                    best_positions = shifted

    if best_positions is None:
        # Fallback: linear arrangement
        best_positions = [(4.0*k, 0.0) for k in range(-(N//2), N-N//2)]
        best_angles = [0.0]*N
        best_side = _exact_side(best_positions, best_angles, 0.0, (0.0, 0.0))
        best_phi = 0.0

    # Final verification
    if not _check_disjoint(best_positions, best_angles):
        best_positions = [(4.0*k, 0.0) for k in range(-(N//2), N-N//2)]
        best_angles = [0.0]*N
        best_side = _exact_side(best_positions, best_angles, 0.0, (0.0, 0.0))
        best_phi = 0.0

    best_side *= (1.0 + 1e-9)

    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, math.degrees(best_phi)
