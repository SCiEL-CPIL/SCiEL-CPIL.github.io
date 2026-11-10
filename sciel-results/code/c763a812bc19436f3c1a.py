import numpy as np
import time
import random
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons in a regular hexagon to minimize the outer side length s.
    Uses gradient-based optimization (L-BFGS-B) on a smooth surrogate objective.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    N = 12
    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices (angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    unit_verts = np.array([
        [math.cos(math.radians(i * 60)), math.sin(math.radians(i * 60))] 
        for i in range(6)
    ])
    
    # Precompute edge normals for SAT
    # Edge i connects vertex i and i+1. Normal is perpendicular to edge.
    # For a CCW polygon, the outward normal of edge (v_i, v_{i+1}) is 
    # (v_{i+1} - v_i) rotated -90 deg? 
    # Let's check.
    # v0=(1,0), v1=(0.5, sqrt3/2). Edge = (-0.5, sqrt3/2).
    # Normal pointing outward (away from origin): 
    # Midpoint of edge is at angle 30. Normal should be at angle 30.
    # cos(30)=sqrt3/2, sin(30)=0.5.
    # Edge vector e = v1 - v0.
    # Normal n = (e_y, -e_x) or (-e_y, e_x)?
    # e = (-0.5, 0.866).
    # n1 = (0.866, 0.5). This is angle 30. Correct.
    # So n = (e_y, -e_x).
    
    edges = np.roll(unit_verts, -1, axis=0) - unit_verts
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    # Normalize normals (they should be unit length for a regular hex side 1? 
    # Edge length is 1. Normal length is 1. Yes.)
    
    def compute_s(params):
        """
        params: (24,) array -> 12 centers (x, y). Angles fixed to 0.
        Returns s (side length of outer hex) and the outer center (which is centroid of inner centers? 
        Actually, to minimize s, the outer hex center should be chosen optimally.
        However, for a regular hexagon, the optimal center for a set of points is the center of the minimum enclosing regular hexagon.
        Approximation: The centroid of the inner hexagons' centers is a good starting point, but we can also optimize the outer center.
        Let's fix the outer center to (0,0) by centering the inner hexagons first?
        No, the problem allows translating the whole system. 
        We can assume WLOG that the outer hexagon is centered at (0,0).
        Then we optimize the positions of the 12 inner hexagons.
        The objective is s = max over all vertices v of inner hexes of (max over 6 normals n_i of dot(v, n_i)) / (sqrt3/2).
        Wait, the support function h(n) = max dot(v, n).
        For a regular hexagon centered at origin with side s, the distance to edge is s*sqrt3/2.
        So we need h(n_i) <= s * sqrt3/2 for all 6 normals n_i (and their negatives? No, the 6 normals cover all 6 directions).
        Actually, the 6 outward normals of the outer hexagon define the constraints.
        s >= max_i ( h(n_i) / (sqrt3/2) ) where h(n_i) is the max projection of any inner vertex onto n_i.
        """
        centers = params.reshape(12, 2)
        
        # Generate all vertices of all inner hexagons
        # V is (72, 2)
        V = centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
        V = V.reshape(-1, 2) # (72, 2)
        
        # Compute projections onto the 6 outer hexagon normals
        # Outer normals are at angles 30, 90, 150, 210, 270, 330 (for angle 0 outer hex)
        # Wait, the outer hexagon angle is also a variable? 
        # The prompt says outer is (x, y, angle). 
        # We can fix outer angle to 0 WLOG? 
        # No, rotating the outer hexagon relative to inner might help.
        # However, since inner hexagons are also rotatable, we can fix inner angle to 0 and optimize outer angle.
        # Or fix outer angle to 0 and optimize inner angles.
        # Due to symmetry, fixing one to 0 is usually optimal for hexagonal packings.
        # Let's fix outer angle to 0.
        
        outer_normals = np.array([
            [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] 
            for i in range(6)
        ])
        
        # Projections: (72, 6)
        projs = V @ outer_normals.T
        
        # Max projection for each normal
        max_projs = np.max(projs, axis=0)
        
        # s is determined by the largest constraint
        # s * sqrt3/2 >= max_projs
        s = np.max(max_projs) * 2.0 / sqrt3
        
        return s

    def objective(params):
        return compute_s(params)

    def validate_and_finalize(centers):
        """
        Takes centers (12,2), computes s, shifts centers so centroid is (0,0),
        validates collisions and containment, returns (inner, outer, s) or None.
        """
        # Center the configuration
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        
        s = compute_s(centers.flatten())
        
        # Check collisions using SAT
        # Get all vertices again for the centered config
        V = centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
        
        for i in range(12):
            verts_i = V[i*6:(i+1)*6]
            for j in range(i+1, 12):
                verts_j = V[j*6:(j+1)*6]
                
                # SAT check
                # Normals from i
                axes = normals
                # Normals from j (same set since angle 0)
                
                separated = False
                for n in axes:
                    proj_i = np.dot(verts_i, n)
                    proj_j = np.dot(verts_j, n)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        separated = True
                        break
                
                if not separated:
                    return None

        # Check containment
        # Outer hexagon centered at (0,0), side s, angle 0.
        # Vertices of outer hex:
        outer_verts = s * unit_verts # Wait, unit_verts are for side 1.
        # Vertices are at distance s from center.
        # My unit_verts are at distance 1. So s * unit_verts gives vertices at distance s.
        
        # Check all inner vertices against outer hex
        # For each inner vertex v, check if it's inside outer hex.
        # Cross product test.
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1)%6]
            edge = p2 - p1
            # For all inner vertices
            vecs = V - p1
            crosses = vecs[:, 0] * edge[1] - vecs[:, 1] * edge[0]
            if np.any(crosses < -1e-6):
                return None

        inner_arr = np.zeros((12, 3))
        inner_arr[:, 0] = centers[:, 0]
        inner_arr[:, 1] = centers[:, 1]
        inner_arr[:, 2] = 0.0
        
        outer_arr = np.array([0.0, 0.0, 0.0])
        
        return inner_arr, outer_arr, s

    # Initial configurations
    def get_hex_grid_config():
        """
        Standard hexagonal close packing.
        Rows of 3, 4, 3, 2? Or 2,3,4,3?
        Let's try a 3-4-3-2 arrangement or similar.
        Distance between centers in same row: 2.5?
        Distance between rows: sqrt(3) * 1.5?
        For unit hexes (side 1):
        Width = 2. Height = sqrt(3).
        Horizontal spacing in same row: 2.5 (touching at vertices? No, touching at edges requires specific offset).
        If flat-top:
        Row y = 0. x = -2.5, 0, 2.5. (Spacing 2.5)
        Row y = sqrt(3)*1.5? No.
        Let's use the known good seed from the prompt's current program which worked.
        """
        return np.array([
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.17],
            [1.25, 2.17],
            [-1.25, -2.17],
            [1.25, -2.17],
            [-3.75, 2.17],
            [3.75, 2.17],
            [-3.75, -2.17],
            [3.75, -2.17],
            [0, -4.0]
        ])

    def get_random_config():
        return np.random.uniform(-5, 5, (12, 2))

    best_result = None
    best_s = float('inf')
    
    # Try multiple starts
    starts = [get_hex_grid_config()]
    # Add some random perturbations of the grid
    for _ in range(5):
        pert = get_hex_grid_config() + np.random.normal(0, 0.5, (12, 2))
        starts.append(pert)
    
    # Add pure randoms
    for _ in range(5):
        starts.append(get_random_config())

    for init in starts:
        if time.time() > deadline:
            break
        
        # Optimization
        # Use L-BFGS-B
        # The objective is non-smooth (max), but L-BFGS-B can handle it reasonably well.
        # We can also use a smooth approximation if needed, but let's try raw first.
        
        res = minimize(objective, init.flatten(), method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success:
            centers = res.x.reshape(12, 2)
            result = validate_and_finalize(centers)
            if result:
                inner, outer, s = result
                if s < best_s:
                    best_s = s
                    best_result = (inner, outer, s)
                    
                    # Polish: Coordinate descent
                    # Try small moves in x and y for each hex
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        improved = False
                        for i in range(12):
                            for delta in [0.01, -0.01, 0.005, -0.005]:
                                new_centers = centers.copy()
                                new_centers[i, 0] += delta
                                new_s = compute_s(new_centers.flatten())
                                if new_s < best_s - 1e-8:
                                    # Validate quickly? Or assume it's valid if s decreased?
                                    # Risky. Let's validate.
                                    temp_res = validate_and_finalize(new_centers)
                                    if temp_res:
                                        inner, outer, s = temp_res
                                        if s < best_s:
                                            best_s = s
                                            best_result = (inner, outer, s)
                                            centers = new_centers
                                            improved = True
                                            break
                            if improved:
                                break
                        if not improved:
                            break

    if best_result is None:
        # Fallback to the initial grid if optimization failed
        init = get_hex_grid_config()
        result = validate_and_finalize(init)
        if result:
            best_result = result
        else:
            # Should not happen, but just in case
            raise RuntimeError("Failed to find valid packing")

    return best_result

# EVOLVE-BLOCK-END
