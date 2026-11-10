# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = 550.0  # Safety margin
    
    # Initial guess: Hexagonal grid arrangement
    # Side length 1 hexagons
    # dx = 1.5, dy = sqrt(3)/2
    dx = 1.5
    dy = np.sqrt(3) / 2.0
    
    # 11 hexagons in a compact cluster
    # Center (0,0)
    # 6 neighbors
    # 4 more to make 11? 
    # Let's use a 3-4-3-1 or similar? 
    # Actually, a standard hexagonal number is 1, 7, 19. 11 is 7 + 4.
    # Let's place 7 in a flower, and 4 around.
    
    centers = []
    centers.append([0, 0])
    for i in range(6):
        angle = np.pi/3 * i
        centers.append([dx * np.cos(angle), dx * np.sin(angle)])
    
    # Add 4 more. Maybe extending the ring?
    # Or just a 3x3 grid minus corners?
    # Let's try a 3x3 grid of hexagons (9) + 2?
    # 3x3 grid:
    # Row 0: 0, 1, 2
    # Row 1: 0.5, 1.5, 2.5 (shifted)
    # Row 2: 0, 1, 2
    # This is 9. Add 2 more?
    
    # Let's stick to the 7+4 idea or just optimize a random-ish valid start.
    # A safe start: 11 hexagons in a line? No, too long.
    # 5x2 grid?
    
    # Let's use the 7 flower + 4 on the outer ring, spaced out.
    # Or simply 11 in a hexagonal-ish blob.
    
    # Let's define initial positions manually to ensure they are disjoint.
    # 1: (0,0)
    # 2-7: Neighbors
    # 8-11: Further out
    # (1.5, 2.6), (-1.5, 2.6), (1.5, -2.6), (-1.5, -2.6) -> These are at distance sqrt(1.5^2 + 2.6^2) ~ 3.
    # Distance from center (0,0) to (1.5, 2.6) is > 2*1 (diameter 2). So they don't overlap with center.
    # Do they overlap with neighbors?
    # Neighbor at (1.5, 0). Distance to (1.5, 2.6) is 2.6 > 2. OK.
    # Neighbor at (0.75, 1.3). Distance to (1.5, 2.6) is sqrt(0.75^2 + 1.3^2) = sqrt(0.56 + 1.69) = sqrt(2.25) = 1.5.
    # Wait, distance 1.5 < 2. So they overlap!
    # The "flower" neighbors are at distance 1.5 from center? No, distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732.
    # My dx=1.5 was for vertex-to-vertex? No.
    # For unit side hexagons:
    # Width (flat-to-flat) = sqrt(3).
    # Distance between centers of touching hexagons = sqrt(3).
    # So dx = 1.5 is wrong for center distance.
    # Center distance = sqrt(3).
    
    dist_c = np.sqrt(3)
    
    centers = []
    centers.append([0, 0])
    for i in range(6):
        angle = np.pi/3 * i
        centers.append([dist_c * np.cos(angle), dist_c * np.sin(angle)])
        
    # Add 4 more.
    # Place them at the "corners" of the hexagon formed by the 7.
    # The 7 form a hexagon of side 2*dist_c? No, the outer vertices of the neighbors are at distance 2*dist_c?
    # Neighbor center is at dist_c. Its far vertex is at dist_c + 1 (circumradius).
    # So radius of cluster is ~ 1.732 + 1 = 2.732.
    # We need to place 4 more such that they don't overlap.
    # Let's try placing them at angles 30, 150, 210, 330 deg, at a larger radius.
    # Radius R = dist_c * 2 = 3.464?
    # Distance between (0,0) and (R, 30) is R. R > 2*1.732? No.
    # Distance between center (0,0) and new hex center must be > 2*1.732? No, > sqrt(3) is enough to not touch?
    # Wait, non-overlap condition: distance between centers > 2 * apothem? No.
    # Two hexagons of side 1 overlap if distance between centers < 2 * apothem?
    # Apothem = sqrt(3)/2 ~ 0.866.
    # If aligned, they touch when center distance = 2 * apothem = sqrt(3).
    # So if dist > sqrt(3), they are disjoint (assuming optimal orientation).
    
    # So my initial 7 are at dist sqrt(3). They are touching.
    # To make them disjoint, I need dist > sqrt(3).
    # Let's scale the initial configuration by 1.1 to ensure disjointness.
    
    scale = 1.1
    for i in range(len(centers)):
        centers[i] = [centers[i][0]*scale, centers[i][1]*scale]
        
    # Add 4 more at larger radius
    R = dist_c * 2 * scale
    for i in range(4):
        angle = np.pi/6 + np.pi/2 * i  # 30, 150, 270, 30? No, 30, 150, 270, 330?
        # Let's use 45, 135, 225, 315
        angle = np.pi/4 + np.pi/2 * i
        centers.append([R * np.cos(angle), R * np.sin(angle)])
        
    # Total 11 centers.
    
    # Initial angles: all 0
    angles = np.zeros(11)
    
    # Variable vector: x1, y1, a1, x2, y2, a2, ...
    x0 = []
    for i in range(11):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(angles[i])
    x0 = np.array(x0)
    
    def compute_side(params):
        # Extract positions and angles
        xs = params[0::3]
        ys = params[1::3]
        as_ = params[2::3] # angles in radians
        
        # Outer hexagon is centered at (0,0) with angle 0 for simplicity?
        # The problem allows the outer hexagon to have an angle.
        # But by symmetry, we can fix the outer hexagon's center to (0,0).
        # Should we optimize the outer hexagon's angle too?
        # The task says outer is (x,y,angle).
        # If we fix outer center to (0,0), we save 2 vars.
        # Let's optimize outer angle too? 
        # Actually, the "side" is the constraint.
        # The outer hexagon must contain all inner vertices.
        # For a given set of inner hexagons, the minimum side length of an enclosing regular hexagon (centered at 0,0, variable angle) is determined by the maximum projection of any vertex onto the 6 edge normals.
        
        # Let's assume outer hexagon is centered at (0,0).
        # We can optimize its angle 'alpha'.
        # But for now, let's fix alpha=0 and just compute the required side.
        # If we fix alpha=0, the side length S must be >= max over all vertices v and all 6 normals n_k of (v dot n_k).
        # Wait, the distance from center to edge is S * apothem_factor?
        # For regular hexagon of side S, apothem = S * sqrt(3)/2.
        # So S * sqrt(3)/2 >= max_projection.
        # S >= max_projection / (sqrt(3)/2).
        
        # Let's compute vertices of all inner hexagons.
        # Vertices of hex i:
        # (x_i + cos(a_i + 60j), y_i + sin(a_i + 60j))
        
        all_vertices = []
        for i in range(11):
            cx, cy, a = xs[i], ys[i], as_[i]
            for j in range(6):
                vx = cx + np.cos(a + j * np.pi/3)
                vy = cy + np.sin(a + j * np.pi/3)
                all_vertices.append([vx, vy])
                
        all_vertices = np.array(all_vertices)
        
        # Outer hexagon normals for angle 0:
        # Vertices at 0, 60, ...
        # Edge normals are at 30, 90, ...
        # n_k = (cos(30 + 60k), sin(30 + 60k))
        
        max_proj = 0
        for k in range(6):
            angle_n = np.pi/6 + k * np.pi/3
            nx = np.cos(angle_n)
            ny = np.sin(angle_n)
            projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_proj = max(max_proj, np.max(projs))
            
        side = max_proj / (np.sqrt(3) / 2)
        return side
        
    def objective(params):
        # Compute side length
        side = compute_side(params)
        
        # Compute overlap penalty
        # For each pair, check separation.
        # If overlapping, add penalty.
        # We need to check if they are disjoint.
        # Disjoint if for all 12 edge normals (6 of A, 6 of B), the projected intervals are apart.
        # Actually, the condition is: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6".
        # This is the Separating Axis Theorem (SAT).
        # If there is NO separating axis, they overlap.
        
        xs = params[0::3]
        ys = params[1::3]
        as_ = params[2::3]
        
        penalty = 0
        for i in range(11):
            for j in range(i+1, 11):
                # Get vertices
                vi = []
                for k in range(6):
                    vi.append([xs[i] + np.cos(as_[i] + k*np.pi/3), ys[i] + np.sin(as_[i] + k*np.pi/3)])
                vj = []
                for k in range(6):
                    vj.append([xs[j] + np.cos(as_[j] + k*np.pi/3), ys[j] + np.sin(as_[j] + k*np.pi/3)])
                
                vi = np.array(vi)
                vj = np.array(vj)
                
                # Check SAT
                # Axes are edge normals of both hexagons
                # Hexagon edge normals are perpendicular to edges.
                # Edge k connects vertex k and k+1.
                # Normal to edge k is in direction of vertex k (for regular hexagon centered at 0)?
                # No, normal to edge between v_k and v_{k+1} is in direction of (v_k + v_{k+1})/2?
                # For unit hexagon, v_k = (cos(a+60k), sin(a+60k)).
                # Midpoint of edge k is (cos(a+60k+30)*cos30, ...)?
                # Actually, the normal to the edge opposite to vertex k is in direction of vertex k?
                # Let's check.
                # Vertex 0 is (1,0). Edge 5-0 is vertical? No.
                # Vertices: 0:(1,0), 1:(0.5, 0.866), 2:(-0.5, 0.866)...
                # Edge 0-1: vector (-0.5, 0.866). Normal (0.866, 0.5) -> direction 30 deg.
                # Vertex 0 is 0 deg.
                # So normal to edge 0 is 30 deg.
                # Normal to edge k is angle a + 60k + 30.
                
                axes = []
                for k in range(6):
                    axes.append(as_[i] + k*np.pi/3 + np.pi/6)
                    axes.append(as_[j] + k*np.pi/3 + np.pi/6)
                    
                is_separated = False
                for angle in axes:
                    nx = np.cos(angle)
                    ny = np.sin(angle)
                    pi_proj = vi[:, 0]*nx + vi[:, 1]*ny
                    pj_proj = vj[:, 0]*nx + vj[:, 1]*ny
                    
                    min_i, max_i = np.min(pi_proj), np.max(pi_proj)
                    min_j, max_j = np.min(pj_proj), np.max(pj_proj)
                    
                    # Check if intervals [min_i, max_i] and [min_j, max_j] are apart
                    # Gap > 1e-6
                    if max_i + 1e-6 < min_j or max_j + 1e-6 < min_i:
                        is_separated = True
                        break
                
                if not is_separated:
                    # Overlapping. Add penalty.
                    # How much?
                    # We can compute the minimum translation distance (MTD) or just a fixed penalty.
                    # A fixed penalty might be too weak or too strong.
                    # Let's add penalty proportional to the overlap depth if we can compute it easily.
                    # Or just a large constant.
                    penalty += 10.0
                    
        return side + penalty
        
    # Optimization
    # L-BFGS-B
    # Bounds:
    # x, y in [-10, 10]
    # angle in [0, 2*pi]
    bounds = []
    for i in range(11):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 2*np.pi))
        
    # Run optimization
    best_side = compute_side(x0)
    best_params = x0.copy()
    
    # Check time
    if time.time() > deadline:
        # Return initial
        return _format_output(best_params, best_side)
        
    # Use multiple restarts?
    # For now, one run with high maxiter.
    # But we need to be careful with time.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if time.time() > deadline:
             # Interrupt? Can't easily.
             # But if it finishes, check time.
             pass
             
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
            
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # If time is still available, do a second run with different start?
    # Or just return.
    
    return _format_output(best_params, best_side)

def _format_output(params, side):
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = params[i*3]
        inner[i, 1] = params[i*3+1]
        inner[i, 2] = np.degrees(params[i*3+2])
        
    outer = np.array([0, 0, 0])
    return inner, outer, float(side)

# EVOLVE-BLOCK-END
