"""Strategy: Structured lattice initialization + smooth SAT-based penalty minimization with scipy L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    def remaining_time():
        return max(0.0, deadline - time.time())

    # Helper: vertices of a unit regular hexagon centered at (cx, cy) with rotation angle (radians)
    def hex_vertices(cx, cy, angle):
        base_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
        final_angles = base_angles + angle
        xs = cx + np.cos(final_angles)
        ys = cy + np.sin(final_angles)
        return np.column_stack((xs, ys))

    # Helper: edge normals of a unit hexagon (outward)
    def hex_edge_normals(angle):
        # Edges are perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees + rotation
        normals = np.array([np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6]) + angle
        return np.column_stack((np.cos(normals), np.sin(normals)))

    # Minimum enclosing side length for a set of points (vertices)
    # For a regular hexagon with side S and rotation 0 (flat top, vertices at 0, 60, ...),
    # the inradius is S * sqrt(3) / 2.
    # A point (x, y) is inside if for all 6 edge normals n_k, n_k . (x, y) <= R_out.
    # So R_out >= max_k(n_k . (x, y)).
    # S = R_out / (sqrt(3) / 2) = 2 * R_out / sqrt(3).
    def min_enclosing_side(points):
        # points: (M, 2) array
        if len(points) == 0:
            return 0.0
        # Edge normals for outer hexagon (rotation 0)
        normals = np.array([[np.cos(a), np.sin(a)] for a in [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6]])
        # Project all points onto normals
        projections = points @ normals.T  # (M, 6)
        # For each normal, the max projection gives the required R_out in that direction
        max_projections = np.max(projections, axis=0)  # (6,)
        # The required R_out is the max of these
        R_out = np.max(max_projections)
        S = 2.0 * R_out / np.sqrt(3.0)
        return S

    # SAT-based overlap check between two hexagons
    # Returns True if they overlap (intersect)
    def hex_overlap(verts1, verts2, normals1, normals2):
        # Check all normals from both hexagons
        all_normals = np.vstack([normals1, normals2])
        for n in all_normals:
            # Project all vertices onto normal n
            proj1 = verts1 @ n
            proj2 = verts2 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # If there's a gap, they don't overlap
            if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
                return False
        return True

    # Compute smooth penalty for a given configuration
    def compute_penalty(params):
        # params: (3N,) array: [x1, y1, a1, x2, y2, a2, ...]
        positions = params[0::3]
        ys = params[1::3]
        angles = params[2::3]
        
        # Compute vertices and normals for all hexagons
        verts_list = []
        normals_list = []
        for i in range(N):
            v = hex_vertices(positions[i], ys[i], angles[i])
            n = hex_edge_normals(angles[i])
            verts_list.append(v)
            normals_list.append(n)
        
        penalty = 0.0
        
        # Containment penalty: all vertices must be inside outer hexagon
        # We'll compute the required S and penalize if it exceeds a target
        # But for the optimizer, we want a smooth function.
        # Let's define the objective as: S_required + lambda * overlap_penalty
        # But we want to minimize S. So we can just return S_required + penalty.
        
        all_verts = np.vstack(verts_list)
        S_req = min_enclosing_side(all_verts)
        
        # Overlap penalty: for each pair, if they overlap, add penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if hex_overlap(verts_list[i], verts_list[j], normals_list[i], normals_list[j]):
                    # Compute minimum distance between the two hexagons (approximate)
                    # Use center distance as a proxy for penalty magnitude
                    dx = positions[j] - positions[i]
                    dy = ys[j] - ys[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # If they overlap, dist is small. Penalty should be large when dist is small.
                    # Use a smooth penalty: 1 / (dist + 0.1) or similar
                    overlap_pen += 10.0 / (dist + 0.1)
        
        return S_req + overlap_pen

    # Initial configuration: honeycomb lattice
    def initial_lattice():
        # 11 hexagons in a honeycomb pattern
        # Center at (0,0)
        # 6 around at distance sqrt(3)
        # 4 more at distance 2*sqrt(3) in specific directions
        pos = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Center
        pos[0] = [0.0, 0.0]
        
        # 6 around
        for i in range(6):
            angle = i * np.pi / 3
            pos[i+1] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
        
        # 4 more: at distance 2*sqrt(3) at angles 0, pi/2, pi, 3pi/2? 
        # Or better: extend the honeycomb.
        # Let's place them at:
        # (2*sqrt(3), 0), (0, 2*sqrt(3)), (-2*sqrt(3), 0), (0, -2*sqrt(3))
        for i in range(4):
            angle = i * np.pi / 2
            pos[i+7] = [2*np.sqrt(3) * np.cos(angle), 2*np.sqrt(3) * np.sin(angle)]
        
        # Pack into params
        params = np.zeros(3*N)
        for i in range(N):
            params[3*i] = pos[i, 0]
            params[3*i+1] = pos[i, 1]
            params[3*i+2] = angles[i]
        
        return params

    # Run optimization
    best_params = None
    best_score = np.inf
    
    # Try multiple initializations
    inits = [initial_lattice()]
    
    # Add a few random initializations
    for _ in range(3):
        pos = np.random.randn(N, 2) * 2.0
        angles = np.random.uniform(0, np.pi/3, N)
        params = np.zeros(3*N)
        for i in range(N):
            params[3*i] = pos[i, 0]
            params[3*i+1] = pos[i, 1]
            params[3*i+2] = angles[i]
        inits.append(params)
    
    for init in inits:
        if remaining_time() < 10.0:
            break
        
        try:
            res = minimize(
                compute_penalty,
                init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue
    
    # If no valid result, use the best we have
    if best_params is None:
        best_params = inits[0]
    
    # Extract final configuration
    positions = best_params[0::3]
    ys = best_params[1::3]
    angles = best_params[2::3]
    
    # Compute the actual required S
    verts_list = [hex_vertices(positions[i], ys[i], angles[i]) for i in range(N)]
    all_verts = np.vstack(verts_list)
    S = min_enclosing_side(all_verts)
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([positions[i], ys[i], np.degrees(angles[i])])
    
    return inner, (0.0, 0.0), S, 0.0
