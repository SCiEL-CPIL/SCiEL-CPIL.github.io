"""
Strategy: Fixed-Topology Rigid Body Placement with Differential Evolution.

This approach fundamentally changes the problem framing:
1. Rigid Cluster: The 11 unit hexagons are treated as a single rigid body. 
   Their relative positions and orientations are fixed to a high-quality 
   "seed" packing derived from the hexagonal lattice.
2. Global Optimization: We optimize only the global rotation and translation 
   of this rigid cluster relative to the outer hexagon.
3. Differential Evolution: We use DE (a population-based stochastic method) 
   instead of local gradient methods (Nelder-Mead/Powell) to avoid local 
   minima in the global placement landscape.
4. Objective: Minimize the maximum support function of the cluster's 
   vertices against the six facets of the outer hexagon.

This reduces the search space from ~30+ variables (positions/angles of 11 
hexagons) to 3 variables (tx, ty, theta), allowing the optimizer to explore 
the global landscape thoroughly within the time budget.
"""

import time
import numpy as np
import math
from scipy.optimize import differential_evolution

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

def _get_seed_configuration():
    """
    Generate a high-quality initial rigid configuration of 11 hexagons.
    This is a compact cluster: 1 center + 6 ring + 4 second ring.
    The second ring hexagons are placed in the "valleys" of the first ring
    to create a convex, compact shape.
    
    Coordinates are in the cluster's local frame (centered at 0,0).
    Angles are in degrees.
    """
    sqrt3 = math.sqrt(3)
    # Distance between centers of touching hexagons is 2 * sqrt(3) * cos(30) = 3? 
    # Wait. Side length 1. Distance between parallel sides is sqrt(3).
    # If two hexagons touch side-to-side, center distance is sqrt(3) * 2 / sqrt(3) * 1? 
    # Let's verify.
    # Hexagon vertices at distance 1 from center.
    # Flat-to-flat distance is sqrt(3).
    # If two hexagons share an edge, their centers are distance sqrt(3) apart? No.
    # Let's calculate.
    # Hexagon 1 centered at (0,0). Vertex at (1,0).
    # Hexagon 2 centered at (d, 0). Vertex at (d-1, 0).
    # If they touch at (0.5, 0)? No, side length is 1.
    # The distance from center to vertex is 1.
    # The distance from center to edge midpoint is sqrt(3)/2.
    # If two hexagons touch edge-to-edge, the distance between centers is 
    # 2 * (sqrt(3)/2) = sqrt(3).
    # Let's check the lattice.
    # Standard hex lattice basis vectors: (sqrt(3), 0) and (sqrt(3)/2, 1.5).
    # Length of basis vector 1 is sqrt(3).
    # Length of basis vector 2 is sqrt(3/4 + 2.25) = sqrt(0.75 + 2.25) = sqrt(3).
    # So yes, center distance for touching is sqrt(3).
    
    # However, the current seed program used 2*sqrt(3) for ring 1. That implies
    # they were NOT touching? Or I am misremembering the geometry.
    # Let's re-read the seed program.
    # init_pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
    # Distance is 2*sqrt(3) ≈ 3.46.
    # If they are not touching, there is a gap.
    # For a *compact* packing, we want them touching.
    # Let's use distance = sqrt(3).
    
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0.0, 0.0]
    
    # 1-6: Ring 1 (touching center)
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [sqrt3 * np.cos(a), sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2
    # To make it compact, we place them in the gaps between ring 1 hexagons.
    # The gaps are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Distance for ring 2 to touch ring 1?
    # If hex 1 is at angle 0, and hex 7 is at angle 30.
    # Hex 1 center: (sqrt3, 0).
    # Hex 7 center: (d * cos(30), d * sin(30)).
    # Distance between centers should be sqrt(3).
    # (sqrt3 - d*cos(30))^2 + (0 - d*sin(30))^2 = 3
    # 3 - 2*sqrt3*d*cos(30) + d^2 = 3
    # d^2 - 2*sqrt3*d*(sqrt3/2) = 0
    # d^2 - 3d = 0 => d = 3.
    # So ring 2 distance is 3.
    
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [3.0 * np.cos(a), 3.0 * np.sin(a)]
        
    return pos, ang

def _cluster_vertices(pos, ang):
    """
    Compute all vertices of the rigid cluster in the local frame.
    Returns an array of shape (N*6, 2).
    """
    verts_list = []
    for i in range(N):
        cx, cy = pos[i]
        a_rad = np.radians(ang[i])
        for k in range(6):
            theta = a_rad + k * np.pi / 3
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            verts_list.append([vx, vy])
    return np.array(verts_list)

def _min_outer_side_global(local_verts, tx, ty, rot_deg):
    """
    Calculate the minimum outer hexagon side length required to contain the
    cluster defined by local_verts, translated by (tx,ty) and rotated by rot_deg.
    The outer hexagon is centered at (0,0) and has angle 0.
    """
    a = np.radians(rot_deg)
    ca, sa = np.cos(a), np.sin(a)
    
    # Rotate local vertices
    rx = ca * local_verts[:, 0] - sa * local_verts[:, 1]
    ry = sa * local_verts[:, 0] + ca * local_verts[:, 1]
    
    # Translate
    rx += tx
    ry += ty
    
    # Outer hexagon constraints (centered at 0,0, angle 0)
    # The outer hexagon is defined by |x| <= s, |x + y/sqrt(3)| <= s, |x - y/sqrt(3)| <= s
    # Wait, let's derive the support function.
    # Vertices of outer hexagon with side s at angle 0:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # Facets are:
    # 1. x = s (Right)
    # 2. x/2 + y*sqrt(3)/2 = s (Top Right) -> x + y*sqrt(3) = 2s
    # 3. -x/2 + y*sqrt(3)/2 = s (Top Left) -> -x + y*sqrt(3) = 2s
    # 4. x = -s (Left)
    # 5. -x/2 - y*sqrt(3)/2 = s (Bottom Left) -> -x - y*sqrt(3) = 2s
    # 6. x/2 - y*sqrt(3)/2 = s (Bottom Right) -> x - y*sqrt(3) = 2s
    
    # So for a point (x,y) to be inside, we need:
    # s >= x
    # s >= (x + y*sqrt(3)) / 2
    # s >= (-x + y*sqrt(3)) / 2
    # s >= -x
    # s >= (-x - y*sqrt(3)) / 2
    # s >= (x - y*sqrt(3)) / 2
    
    # Thus, s = max( x, (x + y*sqrt(3))/2, (-x + y*sqrt(3))/2, -x, (-x - y*sqrt(3))/2, (x - y*sqrt(3))/2 )
    # Note: max(x, -x) = |x|.
    # Let's just compute the max of these 6 expressions for all vertices.
    
    sqrt3 = math.sqrt(3)
    v1 = rx
    v2 = (rx + ry * sqrt3) / 2.0
    v3 = (-rx + ry * sqrt3) / 2.0
    v4 = -rx
    v5 = (-rx - ry * sqrt3) / 2.0
    v6 = (rx - ry * sqrt3) / 2.0
    
    # The required side length is the maximum of these values across all vertices.
    # Also, side length must be positive.
    max_req = np.max(np.maximum(np.maximum(np.maximum(v1, v2), np.maximum(v3, v4)), np.maximum(v5, v6)))
    
    return float(max_req)

def _objective(params):
    """
    Objective function for Differential Evolution.
    params: [tx, ty, rot_deg]
    """
    if time.time() > DEADLINE:
        return 1e10
        
    tx, ty, rot_deg = params
    
    # We use the pre-computed local vertices for speed
    # But local_verts depends on the fixed configuration.
    # We compute it once outside.
    global _LOCAL_VERTS
    return _min_outer_side_global(_LOCAL_VERTS, tx, ty, rot_deg)

def construct():
    global START_TIME, DEADLINE
    global _LOCAL_VERTS
    
    # 1. Generate seed configuration
    seed_pos, seed_ang = _get_seed_configuration()
    
    # 2. Compute local vertices for the rigid cluster
    _LOCAL_VERTS = _cluster_vertices(seed_pos, seed_ang)
    
    # 3. Define bounds for Differential Evolution
    # The cluster has a diameter of roughly 6-7 units.
    # tx, ty should be within +/- 5.
    # rot_deg in [0, 60) due to hexagonal symmetry.
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 60.0)]
    
    # 4. Run Differential Evolution
    # We use a small population and max iterations to fit the time budget.
    # DE is stochastic, so we might run it multiple times if time permits,
    # but one run with reasonable popsize/iter is usually sufficient for 3 vars.
    
    best_result = None
    best_side = float('inf')
    
    # We can try a few different DE configurations or just one robust one.
    # popsize=20, maxiter=200 should be fast for 3 variables.
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=1000, 
            popsize=30, 
            tol=1e-6, 
            seed=42,
            polish=True  # Use L-BFGS-B to polish the result
        )
        
        if res.fun < best_side:
            best_side = res.fun
            best_result = res.x
            
    except Exception as e:
        pass
    
    # If DE fails or takes too long, fall back to a simple grid search or the initial guess
    if best_result is None:
        # Simple grid search fallback
        best_side = float('inf')
        best_result = [0.0, 0.0, 0.0]
        for tx in np.linspace(-2, 2, 10):
            for ty in np.linspace(-2, 2, 10):
                for rot in np.linspace(0, 60, 10):
                    side = _min_outer_side_global(_LOCAL_VERTS, tx, ty, rot)
                    if side < best_side:
                        best_side = side
                        best_result = [tx, ty, rot]
    
    tx, ty, rot_deg = best_result
    
    # 5. Transform the seed positions and angles to the global frame
    # The inner hexagons are rigid, so their relative angles don't change,
    # but their global positions and angles do.
    
    a_rad = np.radians(rot_deg)
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    
    final_pos = np.zeros((N, 2))
    final_ang = np.zeros(N)
    
    for i in range(N):
        # Rotate position
        lx, ly = seed_pos[i]
        gx = ca * lx - sa * ly + tx
        gy = sa * lx + ca * ly + ty
        final_pos[i] = [gx, gy]
        
        # Rotate angle
        final_ang[i] = seed_ang[i] + rot_deg
    
    # 6. Construct output
    # The outer hexagon is centered at (0,0) and has angle 0.
    # The side length is best_side.
    # Wait, the evaluator expects:
    # inner rows (x, y, angle_degrees)
    # outer center (x, y), side length, angle
    # If we place the outer hexagon at (0,0) with angle 0, the inner hexagons
    # are at final_pos.
    
    inner_list = [[float(final_pos[i,0]), float(final_pos[i,1]), float(final_ang[i])] for i in range(N)]
    
    # It is often better to center the output at (0,0) for the outer hexagon.
    # Our calculation assumed outer center (0,0).
    # So we return outer center (0,0), side best_side, angle 0.
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
