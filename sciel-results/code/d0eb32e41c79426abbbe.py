"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 145.0  # Safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta):
    """
    Returns vertices of a regular hexagon with side length 1, circumradius 1,
    centered at origin, rotated by theta.
    """
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _sat_overlap(A, B):
    """
    Fast SAT check for two convex polygons A and B (arrays of (x,y)).
    Returns True if they overlap.
    """
    # Check axes from A
    for i in range(6):
        p1 = A[i]
        p2 = A[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        projA = A @ normal
        projB = B @ normal
        if projA.max() < projB.min() or projB.max() < projA.min():
            return False
    # Check axes from B
    for i in range(6):
        p1 = B[i]
        p2 = B[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        projA = A @ normal
        projB = B @ normal
        if projA.max() < projB.min() or projB.max() < projA.min():
            return False
    return True

def _bounding_side_fast(centers, angles, outer_theta):
    """
    Calculates the minimum side length of a regular hexagon (orientation outer_theta)
    that contains all inner hexagons defined by centers and angles.
    """
    # Normals for the outer hexagon
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    
    min_proj = np.inf
    max_proj = -np.inf
    
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        # Project vertices onto the 6 normal directions
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
        
    widths = max_proj - min_proj
    # Side length s relates to width w by w = s * sqrt(3)
    return float(np.max(widths) / math.sqrt(3.0))

def _is_valid(centers, angles, tol=1e-9):
    """
    Checks if the configuration has no overlaps.
    """
    verts = [_hex_vertices(angles[i]) + centers[i] for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return False
    return True

def _generate_topologies():
    """
    Generates relative coordinates for different honeycomb topologies of 12 hexagons.
    Returns a list of arrays of shape (N, 2).
    """
    topologies = []
    
    # Topology 1: 3x4 rectangular block on hex grid
    # Hex grid coords (u, v)
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # Spacing: dx = 2 * sqrt(3) for horizontal neighbors? 
    # Standard hex tiling:
    # Center distance is 2 * sqrt(3) if side is 1? No.
    # Side length 1. Distance between centers of touching hexagons is 2 * sqrt(3) / 2 * 2?
    # Apothem = sqrt(3)/2. Distance = 2 * apothem = sqrt(3).
    # Wait. If side is 1, circumradius is 1.
    # Distance between centers of two touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Let's verify.
    # Hexagon side 1. Height = 2 * apothem = sqrt(3). Width = 2 * circumradius = 2.
    # If aligned flat-top, distance between centers in x direction is 1.5? No.
    # Flat-top:
    # x spacing: 1.5 * side? No.
    # Let's use the standard hex grid basis.
    # a1 = (sqrt(3), 1) ? No.
    # Let's just use the fact that centers form a triangular lattice with spacing sqrt(3).
    # Basis vectors:
    # v1 = (sqrt(3), 1) * (2/3)? 
    # Let's calculate directly.
    # Center at (0,0).
    # Neighbor at 60 deg: (sqrt(3), 1)? Distance = sqrt(3+1) = 2. Too far.
    # Distance should be sqrt(3).
    # (sqrt(3)/2 * 2, 1/2 * 2) = (sqrt(3), 1). Distance 2.
    # Ah, the distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # So if one is at (0,0), the neighbor at 0 deg is at (sqrt(3), 0).
    # Neighbor at 60 deg is at (sqrt(3)/2, 3/2)? Distance sqrt(3/4 + 9/4) = sqrt(3). Correct.
    # So basis vectors:
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 1.5) -> Wait, 1.5 is 3/2.
    # Let's check e2 length: sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
    # Angle of e2: atan(1.5 / sqrt(3)) = atan(sqrt(3)) = 60 deg. Correct.
    
    e1 = np.array([math.sqrt(3), 0])
    e2 = np.array([math.sqrt(3)/2, 1.5])
    
    # 3x4 grid
    pts = []
    for i in range(4):
        for j in range(3):
            p = i * e1 + j * e2
            pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    topologies.append(pts)
    
    # Topology 2: 2x6 grid
    pts = []
    for i in range(6):
        for j in range(2):
            p = i * e1 + j * e2
            pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    topologies.append(pts)
    
    # Topology 3: Compact 7+5
    # Center 7: (0,0) and 6 neighbors
    # 5 more attached to the perimeter
    pts = []
    # Center
    pts.append(np.array([0,0]))
    # 6 neighbors
    for k in range(6):
        angle = k * math.pi / 3
        p = math.sqrt(3) * np.array([math.cos(angle), math.sin(angle)])
        pts.append(p)
    # 5 more
    # Attach to neighbors 0, 1, 2, 3, 4
    for k in range(5):
        angle = k * math.pi / 3
        base = math.sqrt(3) * np.array([math.cos(angle), math.sin(angle)])
        # Extend further in the same direction
        p = base + math.sqrt(3) * np.array([math.cos(angle), math.sin(angle)])
        pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    topologies.append(pts)
    
    return topologies

def _rigid_optimize(topology, num_rotations=60):
    """
    Optimizes global rotation and translation for a fixed topology.
    Returns best centers, angles, side length.
    """
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_outer_theta = 0.0
    
    # We assume all inner hexagons have the same orientation relative to the cluster.
    # Since the cluster is rigid, if we rotate the cluster by theta, all hexagons rotate by theta.
    # We can fix the "inner" angle to 0 relative to the cluster, so absolute angle is theta.
    
    for rot_idx in range(num_rotations):
        theta = rot_idx * (math.pi / 3.0) / num_rotations
        
        # Rotate topology points
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_pts = topology @ R.T
        
        # The absolute angle of each hexagon is theta
        angles = np.full(N, theta)
        
        # We need to find the optimal translation (tx, ty) and outer hexagon orientation (outer_theta)
        # To minimize the bounding side length.
        # The bounding side length depends on the relative orientation between the cluster and the outer hexagon.
        # Let's iterate over outer_theta as well, or assume outer_theta = 0 and rotate the cluster?
        # Due to symmetry, we can fix outer_theta = 0 and rotate the cluster.
        # So we are effectively searching over relative angle.
        
        # For a fixed relative angle, the optimal translation centers the convex hull of all vertices.
        # Let's compute the vertices of all hexagons in the rotated cluster.
        # This is expensive if done naively, but N=12 is small.
        
        # Actually, we can just compute the bounding box in the 6 normal directions for outer_theta=0.
        # The width in each direction is fixed by the cluster shape.
        # The side length is max(widths) / sqrt(3).
        # The translation doesn't affect the side length, only the position.
        # So we just need to find the minimum side length over all relative orientations.
        
        # Compute widths for this rotation
        # Normals for outer hexagon (orientation 0)
        norms = np.array([[math.cos(math.pi / 6.0 + k * math.pi / 3.0),
                           math.sin(math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
        
        min_proj = np.inf
        max_proj = -np.inf
        
        for i in range(N):
            # Vertices of hexagon i
            # Center is rotated_pts[i]
            # Angle is theta
            verts = _hex_vertices(theta) + rotated_pts[i]
            projs = verts @ norms.T
            min_proj = np.minimum(min_proj, projs.min(axis=0))
            max_proj = np.maximum(max_proj, projs.max(axis=0))
            
        widths = max_proj - min_proj
        side = float(np.max(widths) / math.sqrt(3.0))
        
        if side < best_side:
            best_side = side
            best_theta = theta
            # Calculate optimal translation to center the pack
            # The center of the bounding box in normal space is (min+max)/2
            # We want to map this back to Cartesian.
            # For a regular hexagon, the center is the intersection of the bisectors.
            # A good approximation is the centroid of all vertices.
            all_verts = []
            for i in range(N):
                verts = _hex_vertices(theta) + rotated_pts[i]
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            centroid = all_verts.mean(axis=0)
            
            best_centers = rotated_pts + centroid # Wait, rotated_pts is relative to origin.
            # We want the centers to be in absolute coordinates.
            # If we shift the cluster by -centroid, the new centroid is 0.
            # So centers = rotated_pts - centroid.
            best_centers = rotated_pts - centroid
            best_angles = np.full(N, theta)
            best_outer_theta = 0.0 # Since we fixed outer to 0
            
    return best_centers, best_angles, best_side, best_outer_theta

def _smooth_collision_penalty(centers, angles, alpha=1.0):
    """
    Calculates a smooth approximation of the collision penalty.
    Uses the minimum distance between edges? Or just center distance?
    Center distance is not sufficient for rotated hexagons.
    We can use a simple potential: if centers are close, penalty is high.
    A more robust smooth penalty: 
    For each pair, compute the minimum translation vector (MTV) magnitude.
    If MTV > 0, no overlap.
    If MTV < 0, overlap.
    Smooth penalty = max(0, -MTV)^2.
    Calculating MTV is expensive.
    Alternative: Use a "soft" SAT check.
    For each axis, compute the separation. If separation < 0, add penalty.
    Penalty = sum(max(0, -separation)^2) over all axes and pairs.
    """
    penalty = 0.0
    verts = [_hex_vertices(angles[i]) + centers[i] for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            A = verts[i]
            B = verts[j]
            # Check axes from A
            for k in range(6):
                p1 = A[k]
                p2 = A[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                nl = math.hypot(normal[0], normal[1])
                if nl < 1e-9: continue
                normal = normal / nl
                projA = A @ normal
                projB = B @ normal
                # Separation: min distance between intervals
                # If intervals overlap, separation is negative.
                sep = min(projA.max() - projB.min(), projB.max() - projA.min())
                if sep < 0:
                    penalty += sep * sep
            # Check axes from B
            for k in range(6):
                p1 = B[k]
                p2 = B[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                nl = math.hypot(normal[0], normal[1])
                if nl < 1e-9: continue
                normal = normal / nl
                projA = A @ normal
                projB = B @ normal
                sep = min(projA.max() - projB.min(), projB.max() - projA.min())
                if sep < 0:
                    penalty += sep * sep
                    
    return penalty

def _smooth_objective(x, outer_theta):
    """
    Objective function for L-BFGS-B.
    x: array of shape (3*N,) = [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = np.array(x[0:2*N]).reshape(N, 2)
    angles = np.array(x[2*N:3*N])
    
    # Bounding side
    side = _bounding_side_fast(centers, angles, outer_theta)
    
    # Collision penalty
    penalty = _smooth_collision_penalty(centers, angles)
    
    # We want to minimize side, but we MUST avoid collisions.
    # If penalty is large, the configuration is invalid.
    # We can use a large weight for penalty.
    weight = 100.0
    return side + weight * penalty

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_inner = None
    best_outer = (0.0, 0.0)
    best_theta = 0.0

    topologies = _generate_topologies()
    
    # Phase 1: Rigid Cluster Optimization
    for topo in topologies:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        c_rigid, a_rigid, side_rigid, outer_theta_rigid = _rigid_optimize(topo, num_rotations=120)
        
        if side_rigid < best_side:
            best_side = side_rigid
            best_centers = c_rigid
            best_angles = a_rigid
            best_outer_theta = outer_theta_rigid
            
            # Calculate optimal center for this rigid configuration
            # The rigid optimizer returns centers centered at 0.
            # We can just use (0,0) as the outer center.
            best_inner = [[float(best_centers[i][0]), float(best_centers[i][1]), float(math.degrees(best_angles[i]))] for i in range(N)]
            best_outer = (0.0, 0.0)
            best_theta = outer_theta_rigid

    # Phase 2: Local Optimization with L-BFGS-B
    # Start from the best rigid configuration
    if best_inner is not None:
        # Flatten the state
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[2*i] = best_centers[i][0]
            x0[2*i+1] = best_centers[i][1]
            x0[2*N+i] = best_angles[i]
            
        # Try to optimize
        # We need to be careful with time.
        if time.time() - start_time < DEADLINE - 10:
            try:
                # Use L-BFGS-B
                res = minimize(_smooth_objective, x0, args=(best_outer_theta,), method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 1000})
                
                if res.success or res.fun < best_side:
                    x_opt = res.x
                    centers_opt = x_opt[0:2*N].reshape(N, 2)
                    angles_opt = x_opt[2*N:3*N]
                    
                    # Check validity
                    if _is_valid(centers_opt, angles_opt):
                        side_opt = _bounding_side_fast(centers_opt, angles_opt, best_outer_theta)
                        if side_opt < best_side:
                            best_side = side_opt
                            best_inner = [[float(centers_opt[i][0]), float(centers_opt[i][1]), float(math.degrees(angles_opt[i]))] for i in range(N)]
                            # Recalculate outer center
                            all_verts = []
                            for i in range(N):
                                all_verts.append(_hex_vertices(angles_opt[i]) + centers_opt[i])
                            all_verts = np.vstack(all_verts)
                            centroid = all_verts.mean(axis=0)
                            best_outer = tuple(float(x) for x in centroid)
                            best_theta = best_outer_theta
            except Exception as e:
                pass # Ignore optimization errors

    if best_inner is None:
        # Fallback
        centers, angles = _generate_topologies()[0], np.zeros(N)
        best_inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
        best_side = _bounding_side_fast(np.array(centers), np.array(angles), 0.0)
        best_outer = (0.0, 0.0)
        best_theta = 0.0

    return best_inner, best_outer, best_side, math.degrees(best_theta)

# EVOLVE-BLOCK-END
