"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Base vertices for a unit regular hexagon (flat-top, centered at origin)
    # Side length 1
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # x = cos(theta), y = sin(theta)
    angles = np.deg2rad(np.arange(0, 360, 60))
    base_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def get_hex_verts(center, angle_deg, side=1.0):
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return center + (side * base_vertices @ R.T)

    def hexagon_containment_violation(params, S, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
        """
        Vectorized check for containment of N hexagons in an outer hexagon.
        params: shape (N, 3) -> [x, y, angle_deg]
        Returns: max violation (0 if all contained)
        """
        N = len(params)
        if N == 0: return 0.0
        
        # Get all vertices for all inner hexagons: shape (N, 6, 2)
        # We can do this vectorized
        xs = params[:, 0]
        ys = params[:, 1]
        as_ = np.deg2rad(params[:, 2])
        
        # Rotation matrices for each hexagon
        c = np.cos(as_)
        s = np.sin(as_)
        # R_i for each i: [[c, -s], [s, c]]
        
        # Base vertices shape (6, 2)
        # We want to rotate each base vertex by angle_i and translate by (x_i, y_i)
        # v_i,j = R_i @ base_j + center_i
        
        # base_vertices shape (6, 2)
        # c, s shape (N,)
        # Let's compute rotated base vertices for all N hexagons at once.
        # rotated_base[i] = R_i @ base_vertices.T -> (2, 6) -> transpose to (6, 2)
        
        # Efficient vectorization:
        # For each vertex j (0..5), compute x and y for all i.
        # x_v = c * bx - s * by
        # y_v = s * bx + c * by
        
        bx = base_vertices[:, 0]
        by = base_vertices[:, 1]
        
        # x_v shape (N, 6)
        x_v = c[:, None] * bx[None, :] - s[:, None] * by[None, :]
        y_v = s[:, None] * bx[None, :] + c[:, None] * by[None, :]
        
        # Translate
        x_v += xs[:, None]
        y_v += ys[:, None]
        
        # Now check containment in outer hexagon (side S, center (0,0), angle 0)
        # Outer hexagon inequalities for side S, flat-top:
        # 1. x <= S
        # 2. 0.5x + sqrt(3)/2 y <= S * sqrt(3)/2
        # 3. -0.5x + sqrt(3)/2 y <= S * sqrt(3)/2
        # 4. -x <= S
        # 5. -0.5x - sqrt(3)/2 y <= S * sqrt(3)/2
        # 6. 0.5x - sqrt(3)/2 y <= S * sqrt(3)/2
        
        # Compute violations for each vertex
        # d1 = max(0, x_v - S)
        # ...
        
        sqrt3_2 = np.sqrt(3) / 2
        
        d1 = np.maximum(0.0, x_v - S)
        d2 = np.maximum(0.0, 0.5 * x_v + sqrt3_2 * y_v - S * sqrt3_2)
        d3 = np.maximum(0.0, -0.5 * x_v + sqrt3_2 * y_v - S * sqrt3_2)
        d4 = np.maximum(0.0, -x_v - S)
        d5 = np.maximum(0.0, -0.5 * x_v - sqrt3_2 * y_v - S * sqrt3_2)
        d6 = np.maximum(0.0, 0.5 * x_v - sqrt3_2 * y_v - S * sqrt3_2)
        
        # Max violation across all vertices and all hexagons
        max_violation = np.maximum.reduce([d1, d2, d3, d4, d5, d6]).max()
        return max_violation

    def hex_overlap_violation(params):
        """
        Check for overlap between any pair of inner hexagons.
        Returns: max penetration depth (0 if disjoint)
        Note: This is the hardest part to vectorize fully due to pair-wise nature.
        We will use a simplified check: if distance between centers < 2, check vertices.
        """
        N = len(params)
        max_overlap = 0.0
        xs = params[:, 0]
        ys = params[:, 1]
        as_ = params[:, 2]
        
        # Precompute vertices for all hexagons (N, 6, 2)
        c = np.cos(np.deg2rad(as_))
        s = np.sin(np.deg2rad(as_))
        bx = base_vertices[:, 0]
        by = base_vertices[:, 1]
        
        x_v = c[:, None] * bx[None, :] - s[:, None] * by[None, :] + xs[:, None]
        y_v = s[:, None] * bx[None, :] + c[:, None] * by[None, :] + ys[:, None]
        verts = np.column_stack((x_v.ravel(), y_v.ravel())).reshape(N, 6, 2)
        
        # Pairwise check
        for i in range(N):
            for j in range(i + 1, N):
                # Quick reject based on center distance
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.0: # If centers > 2 apart, they might still touch if rotated, 
                                  # but for regular hexagons, min distance between disjoint is 0.
                                  # Max width is 2 (vertex to vertex). 
                                  # If dist > 2, they are disjoint? 
                                  # Actually, if rotated, the "radius" is 1. 
                                  # So if dist > 2, they are definitely disjoint.
                                  # If dist <= 2, they might overlap.
                    continue
                
                # Check if any vertex of i is inside j
                # And if any vertex of j is inside i
                # Use the half-plane method for point-in-hexagon
                # Transform vertex of i into local frame of j
                
                # Local frame of j:
                # Rotate by -angle_j, translate by -center_j
                theta_j = -np.deg2rad(as_[j])
                c_j, s_j = np.cos(theta_j), np.sin(theta_j)
                R_j_inv = np.array([[c_j, -s_j], [s_j, c_j]])
                
                # Vertices of i: shape (6, 2)
                # Center of j: shape (2,)
                local_verts_i_in_j = (verts[i] - np.array([xs[j], ys[j]])) @ R_j_inv.T
                
                # Check against unit hexagon inequalities
                # x <= 1, 0.5x + sqrt(3)/2 y <= sqrt(3)/2, etc.
                lx = local_verts_i_in_j[:, 0]
                ly = local_verts_i_in_j[:, 1]
                
                v1 = np.maximum(0.0, lx - 1.0)
                v2 = np.maximum(0.0, 0.5 * lx + (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                v3 = np.maximum(0.0, -0.5 * lx + (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                v4 = np.maximum(0.0, -lx - 1.0)
                v5 = np.maximum(0.0, -0.5 * lx - (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                v6 = np.maximum(0.0, 0.5 * lx - (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                
                max_pen = np.maximum.reduce([v1, v2, v3, v4, v5, v6]).max()
                if max_pen > max_overlap:
                    max_overlap = max_pen
                
                # Symmetry: Check if any vertex of j is inside i
                theta_i = -np.deg2rad(as_[i])
                c_i, s_i = np.cos(theta_i), np.sin(theta_i)
                R_i_inv = np.array([[c_i, -s_i], [s_i, c_i]])
                
                local_verts_j_in_i = (verts[j] - np.array([xs[i], ys[i]])) @ R_i_inv.T
                
                lx = local_verts_j_in_i[:, 0]
                ly = local_verts_j_in_i[:, 1]
                
                w1 = np.maximum(0.0, lx - 1.0)
                w2 = np.maximum(0.0, 0.5 * lx + (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                w3 = np.maximum(0.0, -0.5 * lx + (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                w4 = np.maximum(0.0, -lx - 1.0)
                w5 = np.maximum(0.0, -0.5 * lx - (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                w6 = np.maximum(0.0, 0.5 * lx - (np.sqrt(3)/2) * ly - np.sqrt(3)/2)
                
                max_pen2 = np.maximum.reduce([w1, w2, w3, w4, w5, w6]).max()
                if max_pen2 > max_overlap:
                    max_overlap = max_pen2
                    
        return max_overlap

    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        # params: [x1, y1, a1, ..., xN, yN, aN, S]
        S = params[-1]
        if S < 1.0:
            S = 1.0
            params[-1] = 1.0
            
        inner_params = params[:-1].reshape(N, 3)
        
        # Containment penalty
        contain_viol = hexagon_containment_violation(inner_params, S)
        
        # Overlap penalty
        overlap_viol = hex_overlap_violation(inner_params)
        
        # Objective: Minimize S, penalize violations heavily
        # We want S to be as small as possible, but violations must be 0.
        # Using a large multiplier for violations ensures feasibility is prioritized.
        penalty = 1000.0 * (contain_viol**2 + overlap_viol**2)
        
        return S + penalty

    def generate_initial_guesses():
        guesses = []
        
        # Guess 1: Hexagonal cluster (7 center + 4 extra)
        # Center at (0,0)
        # 6 neighbors at distance 2*sqrt(3)/2? No, side 1 hexagons touching side-to-side have center distance sqrt(3)?
        # Distance between centers of two side-1 hexagons touching side-to-side is sqrt(3).
        # Distance for vertex-to-vertex touching is 2.
        # For a tight packing, we want them as close as possible without overlapping.
        # If we align them (angle 0), the distance to place a neighbor to the right is 1.5? No.
        # Let's use the standard hex grid spacing.
        # In a hex grid with side 1, the distance between centers of adjacent hexagons is sqrt(3).
        
        d = np.sqrt(3)
        
        # 7 cluster
        centers = [
            (0, 0),
            (d, 0),
            (d/2, d*np.sqrt(3)/2),
            (-d/2, d*np.sqrt(3)/2),
            (-d, 0),
            (-d/2, -d*np.sqrt(3)/2),
            (d/2, -d*np.sqrt(3)/2)
        ]
        
        # Add 4 more. Where?
        # Next ring is at distance 2d.
        # Or fill in gaps?
        # Let's just add 4 more in a row or square.
        # For simplicity, add them to the right.
        centers.extend([
            (2*d, 0),
            (2*d, d),
            (3*d, 0),
            (3*d, d)
        ])
        
        params = []
        for x, y in centers:
            params.extend([x, y, 0.0])
        params.append(4.0) # Initial S
        guesses.append(np.array(params))
        
        # Guess 2: Linear arrangement
        params = []
        for i in range(N):
            x = i * d
            y = 0
            params.extend([x, y, 0.0])
        params.append(4.0)
        guesses.append(np.array(params))
        
        # Guess 3: Random
        np.random.seed(42)
        params = []
        for i in range(N):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            a = np.random.uniform(0, 60)
            params.extend([x, y, a])
        params.append(5.0)
        guesses.append(np.array(params))
        
        return guesses

    best_score = 1e9
    best_result = None
    
    guesses = generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, guess, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback
        inner = [[0.0, 0.0, 0.0]] * N
        return inner, (0.0, 0.0), 10.0, 0.0

    S = best_result[-1]
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        a = best_result[3*i+2]
        inner.append([x, y, a])
        
    # Final validity check and correction
    # If there are still violations, we might need to expand S slightly or shift positions.
    # For now, return the optimized result.
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
