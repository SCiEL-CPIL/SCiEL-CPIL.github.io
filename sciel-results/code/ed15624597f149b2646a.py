import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    Vertices are at distance 1 from center.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Base vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon (centered at 0,0, rotated by outer_angle_deg)
    that contains all inner hexagons.
    """
    # Outer hexagon constraints for side S, angle theta:
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with vertices at angles theta, theta+60...
    # The normal vectors to the sides are at angles theta+30, theta+90...
    # Distance from center to side is apothem A = S * sqrt(3) / 2.
    # So for any point (x,y) in the hexagon, the projection onto any normal vector n must be <= A.
    # n_k = (cos(alpha_k), sin(alpha_k)) where alpha_k = theta + 30 + k*60.
    # proj = x*n_x + y*n_y <= A.
    # S >= 2 * max(proj) / sqrt(3).
    
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + np.deg2rad(30 + k * 60)
        normals.append((np.cos(alpha), np.sin(alpha)))
    
    max_proj = 0.0
    for (cx, cy), ang in zip(inner_centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def construct():
    start_time = time.time()
    deadline = 150.0 # Safety margin for 120s limit? Task says 120s. Use 110s.
    
    # Lattice basis vectors
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5]) # Wait, 3/2 = 1.5. Yes.
    
    # Generate candidate topologies (list of lattice indices (i,j))
    # Topology 1: 7 central + 4 at (2,0), (0,2), (-2,0), (0,-2)
    t1 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(2,0), (0,2), (-2,0), (0,-2)]
    # Topology 2: 7 central + 4 at (1,1), (1,-1), (-1,1), (-1,-1)
    t2 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(1,1), (1,-1), (-1,1), (-1,-1)]
    # Topology 3: 7 central + 4 consecutive: (2,0), (1,1), (0,2), (-1,1)
    t3 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(2,0), (1,1), (0,2), (-1,1)]
    # Topology 4: 7 central + 4 at (2,0), (1,1), (-1,1), (-2,0) ? Symmetric top/bottom
    t4 = [(0,0)] + [(i,j) for i,j in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]] + [(2,0), (1,1), (-1,1), (-2,0)]
    
    topologies = [t1, t2, t3, t4]
    
    best_S = float('inf')
    best_config = None
    
    for topo in topologies:
        # Convert indices to coordinates
        centers = []
        for i, j in topo:
            c = i * A + j * B
            centers.append(tuple(c))
        
        # Try rotations of the inner set: 0, 30, 60...
        # Since the lattice is symmetric, 0 and 30 are the distinct orientations relative to the outer hexagon.
        # Actually, rotating the inner set by 30 degrees moves the centers to new positions.
        # We just need to try a few rotations of the entire inner cluster.
        for rot_deg in [0, 30, 60, 90, 120, 150]:
            # Rotate centers around origin
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            rotated_centers = []
            for (x, y) in centers:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated_centers.append((rx, ry))
            
            # Inner hexagons can also be rotated. 
            # For optimal packing, inner hexagons should align with the outer hexagon or be rotated 30 deg.
            # Let's try inner angle 0 and 30.
            for inner_ang in [0, 30]:
                if time.time() > deadline - 10:
                    break
                S = _min_outer_side(rotated_centers, [inner_ang]*N, outer_angle_deg=0.0)
                if S < best_S:
                    best_S = S
                    best_config = (rotated_centers, [inner_ang]*N, 0.0)
    
    if best_config is None:
        # Fallback to seed-like behavior if something went wrong
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    centers, angles, outer_ang = best_config
    # Format output
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
