import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0
SQRT3 = np.sqrt(3)

def _min_side_for_orientation(verts, outer_angle):
    """
    Compute the minimum outer side length required to contain the vertices
    of the inner hexagons, given the outer hexagon's orientation and center (0,0).
    
    The outer hexagon is centered at the origin. Its sides are defined by normals.
    For a regular hexagon with side S and orientation theta (rotation of the hexagon itself),
    the distance from center to each side (apothem) is a = S * sqrt(3) / 2.
    The normals to the sides are at angles theta + k*pi/3 for k=0..5.
    
    A point (x,y) is inside if its projection onto any normal n is <= a.
    So for each normal n_k, we need max_i (dot(v_i, n_k)) <= a.
    Thus a >= max_i (dot(v_i, n_k)) for all k.
    So a >= max_k (max_i dot(v_i, n_k)).
    Then S = 2 * a / sqrt(3).
    """
    a_outer = np.radians(outer_angle)
    # Normals to the 6 sides
    # If the hexagon is rotated by theta, the sides are at angles theta + 30, theta + 90, ...?
    # Let's check the seed program's definition.
    # Seed: verts = (S,0), (S/2, S*sqrt3/2)... for angle 0.
    # Vertex 0: (S, 0). Vertex 1: (S/2, S*sqrt3/2).
    # Edge 0-1: from (S,0) to (S/2, S*sqrt3/2).
    # Midpoint: (3S/4, S*sqrt3/4). Normal is perpendicular to edge.
    # Edge vector: (-S/2, S*sqrt3/2). Normal: (S*sqrt3/2, S/2) -> angle 30 deg.
    # So for angle 0, normals are at 30, 90, 150, 210, 270, 330.
    # Generally, normals are at outer_angle + 30 + k*60.
    
    base_angles = np.pi/6 + np.arange(6) * np.pi/3 + a_outer
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # projections = verts @ normals.T  # (N*6, 6)
    # max_proj_per_normal = np.max(projections, axis=0)
    # required_apothem = np.max(max_proj_per_normal)
    
    # Vectorized
    proj = verts @ normals.T
    max_proj = np.max(proj, axis=0)
    required_apothem = np.max(max_proj)
    
    return 2.0 * required_apothem / SQRT3

def _compute_min_side(verts):
    """
    Find the optimal orientation and translation to minimize the bounding hexagon side.
    Since the outer hexagon is centered at (0,0) in the return interface, we can shift
    the inner cluster. But the interface asks for (inner, center, side, angle).
    We can return the inner positions relative to the outer center.
    
    Actually, the objective function in the optimizer will compute the side length
    for a given configuration. We can optimize the orientation of the *cluster* 
    relative to the *outer hexagon*.
    
    Let the outer hexagon be fixed at angle 0 (for calculation simplicity, we can rotate
    the result back or just return the angle).
    Let the inner cluster have a global rotation phi.
    The vertices are R(phi) * v_i.
    We want to minimize S(phi) = min_side_for_orientation(R(phi) * v_i, 0).
    
    This is a 1D optimization problem for phi!
    """
    # We can just scan phi from 0 to 60 degrees (symmetry of hexagon)
    # Or use scipy minimize on phi.
    
    # To be fast, we can just evaluate at a few points or use a simple minimization.
    # Given 120s limit, we can afford a small optimization.
    
    def obj(phi):
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        rotated_verts = verts @ R.T
        return _min_side_for_orientation(rotated_verts, 0.0)
    
    # Initial guess 0
    res = minimize(obj, 0.0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
    best_phi = res.x[0]
    best_side = res.fun
    
    # Also check a few other points just in case
    for phi in [np.pi/6, np.pi/12, np.pi/3]:
        s = obj(phi)
        if s < best_side:
            best_side = s
            best_phi = phi
            
    return best_side, best_phi

def _objective(params):
    """
    Objective for internal configuration optimization.
    params: [x_0...x_10, y_0...y_10, ang_0...ang_10]
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    
    # 1. Compute all vertices
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    verts = np.column_stack([verts_x.ravel(), verts_y.ravel()])
    
    # 2. Compute minimum outer side length (optimizing global rotation)
    side, _ = _compute_min_side(verts)
    
    # 3. Penalty for overlaps
    # We need to check overlaps between hexagons.
    # For unit hexagons, if center distance < 2*sqrt(3) - epsilon, they might overlap.
    # But rotation matters.
    # A cheap proxy: distance between centers.
    # If d < 2*sqrt(3), they *might* overlap.
    # We use a smooth penalty for d < 2*sqrt(3).
    # Actually, the minimum distance between two unit hexagons that don't overlap is 2*sqrt(3) 
    # only if they are aligned in a specific way? No, the minimum distance between centers
    # for non-overlapping unit hexagons is 2*sqrt(3) regardless of rotation?
    # Let's check. The inradius is sqrt(3)/2. The circumradius is 1.
    # If two hexagons touch at a vertex, distance is 2.
    # If they touch at an edge, distance is 2*sqrt(3) ~ 3.46.
    # Wait, the "touching" in the seed program was at 2*sqrt(3).
    # Let's re-verify the geometry.
    # Regular hexagon side 1.
    # Distance between parallel sides is sqrt(3).
    # If two hexagons are aligned and touch edge-to-edge, center distance is sqrt(3) + sqrt(3) = 2*sqrt(3).
    # If they touch vertex-to-vertex, center distance is 1 + 1 = 2.
    # So 2*sqrt(3) is the distance for edge-to-edge contact.
    # Vertex-to-vertex is 2.
    # So the minimum center distance for non-overlap is 2.
    # But 2*sqrt(3) is a "stable" contact for edge alignment.
    
    # To be safe, we should penalize if distance < 2 (absolute minimum).
    # But we want a dense packing, so we want them to be close.
    # Let's use a soft constraint that pushes them apart if d < 2.0.
    # And maybe a weak attraction to d = 2*sqrt(3) to maintain the topology?
    # No, just minimize side length. The side length minimization will naturally pack them.
    # We just need to prevent overlaps.
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            # Hard constraint: d >= 2.0
            if d < 2.0:
                penalty += 100.0 * (2.0 - d)**2
            # Soft constraint: encourage d >= 2*sqrt(3) to avoid "vertex overlap" issues
            # that might be missed by simple distance check?
            # Actually, if d=2, they touch at vertices. That's valid.
            # If d < 2, they overlap.
            # So d >= 2 is the correct constraint for non-overlap of unit hexagons?
            # Yes, the minimum distance between any two points on the boundary of a unit hexagon
            # is 0 (if touching). The minimum distance between centers is 2 (vertex-vertex).
            
    return side + penalty

def _initial_configuration():
    """
    Generate a specific initial configuration for 11 hexagons.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2 (4 hexagons)
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        # Convert ang to radians for calculation
        a = np.radians(ang)
        base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
        angles = a + base_angles
        vx = x + np.cos(angles)
        vy = y + np.sin(angles)
        verts = np.column_stack([vx, vy])
        verts_list.append(verts)
    
    # Check overlaps using SAT (Simplified: check distance between centers? No, SAT is better)
    # But SAT is slow. Let's use a simpler check:
    # For each pair, check if any vertex of one is inside the other.
    # This is O(N^2 * 36) point-in-polygon checks.
    # N=11, so 55 pairs * 6 vertices = 330 checks. Fast enough.
    
    def point_in_hexagon(pt, verts):
        # pt is (x,y), verts is (6,2)
        # Use cross product sign check
        x, y = pt
        inside = True
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i+1)%6]
            # Vector (x1-x, y1-y) to (x2-x2, y2-y2)?
            # Cross product of (v_i - p) and (v_{i+1} - p)
            # If all cross products have the same sign, p is inside.
            cross = (x2 - x) * (y1 - y) - (x1 - x) * (y2 - y)
            # Wait, standard cross product for 2D: (a,b) x (c,d) = ad - bc
            # Vector 1: v_i - p = (x1-x, y1-y)
            # Vector 2: v_{i+1} - p = (x2-x, y2-y)
            # Cross = (x1-x)(y2-y) - (y1-y)(x2-x)
            cross = (x1-x)*(y2-y) - (y1-y)*(x2-x)
            if i == 0:
                sign = np.sign(cross)
            else:
                if sign * np.sign(cross) < 0:
                    return False
        return True

    for i in range(N):
        for j in range(i+1, N):
            # Check if any vertex of i is inside j
            for v in verts_list[i]:
                if point_in_hexagon(v, verts_list[j]):
                    return False
            # Check if any vertex of j is inside i
            for v in verts_list[j]:
                if point_in_hexagon(v, verts_list[i]):
                    return False

    # Check containment
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    for verts in verts_list:
        for v in verts:
            if not point_in_hexagon(v, outer_verts):
                return False
    return True

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def construct():
    global START_TIME, DEADLINE
    
    pos_init, ang_init = _initial_configuration()
    best_pos = pos_init.copy()
    best_ang = ang_init.copy()
    
    x0 = np.concatenate([pos_init.ravel(), ang_init])
    
    # Initial evaluation
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang_init[:, None] + base_angles[None, :]
    verts_x = pos_init[:, 0:1] + np.cos(total_angles)
    verts_y = pos_init[:, 1:2] + np.sin(total_angles)
    verts_init = np.column_stack([verts_x.ravel(), verts_y.ravel()])
    best_side, best_global_rot = _compute_min_side(verts_init)
    
    # Optimization
    if time.time() < DEADLINE - 20.0:
        try:
            # Use L-BFGS-B or Nelder-Mead?
            # L-BFGS-B is faster for smooth problems.
            # Our objective is smooth (max of linear functions is convex, sum of squares is smooth).
            # So L-BFGS-B should work well.
            res = minimize(_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.fun < _objective(x0):
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                ang_opt = x_opt[2*N:]
                
                # Compute side for this config
                total_angles = ang_opt[:, None] + base_angles[None, :]
                verts_x = pos_opt[:, 0:1] + np.cos(total_angles)
                verts_y = pos_opt[:, 1:2] + np.sin(total_angles)
                verts_opt = np.column_stack([verts_x.ravel(), verts_y.ravel()])
                side_opt, rot_opt = _compute_min_side(verts_opt)
                
                # Check validity
                # We need to apply the global rotation to the vertices to check validity?
                # No, the validity check uses the positions and angles as given.
                # The "global rotation" found by _compute_min_side is just for determining the side length.
                # The inner positions are in the local frame.
                # The outer hexagon is rotated by rot_opt relative to the local frame?
                # Or we rotate the inner frame by -rot_opt to align with outer?
                
                # Let's stick to the interface:
                # inner: (x, y, angle_deg)
                # outer: (center, side, angle)
                
                # If we rotate the inner cluster by rot_opt, the side length is minimized.
                # So we should return the rotated positions.
                
                c, s = np.cos(rot_opt), np.sin(rot_opt)
                R = np.array([[c, -s], [s, c]])
                
                # Rotate positions
                pos_rot = pos_opt @ R.T
                # Rotate angles
                ang_rot = ang_opt + np.degrees(rot_opt)
                
                inner_list = [[float(pos_rot[i,0]), float(pos_rot[i,1]), float(ang_rot[i])] for i in range(N)]
                
                # The outer hexagon is now at angle 0 (since we rotated the cluster to align with optimal orientation)
                # Wait, _compute_min_side found the optimal rotation of the cluster *relative to* the outer hexagon.
                # If we fix the outer hexagon at angle 0, the cluster is rotated by rot_opt.
                # So the inner positions should be rotated by rot_opt.
                # And the outer angle is 0.
                
                if _check_validity(inner_list, (0.0, 0.0), side_opt, 0.0):
                    if side_opt < best_side:
                        best_pos = pos_rot
                        best_ang = ang_rot
                        best_side = side_opt
        except Exception:
            pass

    # Fallback: if optimization failed or didn't improve, use initial
    if time.time() < DEADLINE - 5.0:
        # Try a few random restarts if time permits
        for _ in range(5):
            if time.time() > DEADLINE - 5.0:
                break
            pos_jit = pos_init + np.random.normal(0, 0.1, (N, 2))
            ang_jit = ang_init + np.random.normal(0, 0.1, N)
            x_jit = np.concatenate([pos_jit.ravel(), ang_jit])
            try:
                res = minimize(_objective, x_jit, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-6})
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                ang_opt = x_opt[2*N:]
                
                total_angles = ang_opt[:, None] + base_angles[None, :]
                verts_x = pos_opt[:, 0:1] + np.cos(total_angles)
                verts_y = pos_opt[:, 1:2] + np.sin(total_angles)
                verts_opt = np.column_stack([verts_x.ravel(), verts_y.ravel()])
                side_opt, rot_opt = _compute_min_side(verts_opt)
                
                c, s = np.cos(rot_opt), np.sin(rot_opt)
                R = np.array([[c, -s], [s, c]])
                pos_rot = pos_opt @ R.T
                ang_rot = ang_opt + np.degrees(rot_opt)
                
                inner_list = [[float(pos_rot[i,0]), float(pos_rot[i,1]), float(ang_rot[i])] for i in range(N)]
                
                if _check_validity(inner_list, (0.0, 0.0), side_opt, 0.0):
                    if side_opt < best_side:
                        best_pos = pos_rot
                        best_ang = ang_rot
                        best_side = side_opt
            except Exception:
                pass

    # Finalize
    # The best_pos and best_ang are already rotated to align with the optimal outer hexagon orientation (0 degrees)
    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])] for i in range(N)]
    
    # Final validity check
    if not _check_validity(inner_list, (0.0, 0.0), best_side, 0.0):
        # Fallback to initial
        pos_f, ang_f = _initial_configuration()
        inner_list = [[float(pos_f[i,0]), float(pos_f[i,1]), float(ang_f[i])] for i in range(N)]
        # Recompute side for initial
        total_angles = ang_f[:, None] + base_angles[None, :]
        verts_x = pos_f[:, 0:1] + np.cos(total_angles)
        verts_y = pos_f[:, 1:2] + np.sin(total_angles)
        verts_f = np.column_stack([verts_x.ravel(), verts_y.ravel()])
        best_side, _ = _compute_min_side(verts_f)
        
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
