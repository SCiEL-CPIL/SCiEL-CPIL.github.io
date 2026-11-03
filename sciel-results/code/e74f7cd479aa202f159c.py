"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def compute_gaps_and_side(params):
        """
        Efficiently computes the minimum gap between all pairs and the outer side length.
        Returns: (min_gap, side_length)
        min_gap < 0 means overlap.
        """
        # Unpack
        centers = params[0:36].reshape(N, 2)
        angles_deg = params[36:48]
        outer_cx = params[48]
        outer_cy = params[49]
        outer_angle_deg = params[50]
        
        # Precompute vertices for all inner hexagons
        # verts[i] is (6, 2)
        verts_list = []
        for i in range(N):
            verts_list.append(hex_vertices(centers[i, 0], centers[i, 1], angles_deg[i]))
        
        # 1. Compute Overlap (Minimum Gap)
        # We only need to check 3 unique edge normals per hexagon pair if we consider relative orientation,
        # but for simplicity and robustness, we check all 6 normals from both hexagons (12 checks total per pair).
        # Vectorized approach:
        min_gap = float('inf')
        
        for i in range(N):
            v1 = verts_list[i]
            for j in range(i+1, N):
                v2 = verts_list[j]
                
                # Get edge normals for v1 and v2
                # Edge i is from v[i] to v[(i+1)%6]
                # Normal is perpendicular to edge.
                
                # We can compute projections for all normals of v1 and v2 against both v1 and v2.
                # To save time, we can just check the 6 normals of v1. If they are disjoint on any of these,
                # the gap is determined by that normal. If not, we must check v2's normals.
                # Actually, for convex polygons, it's sufficient to check the normals of both.
                
                # Let's compute the gap for each of the 12 normals.
                # To be fast, we'll do it in a loop but with minimal overhead.
                # A fully vectorized pair check is complex due to the min/max logic.
                
                # Optimized loop:
                pair_min_gap = float('inf')
                
                # Normals for v1
                for k in range(6):
                    e = v1[(k+1)%6] - v1[k]
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-9: continue
                    n = n / norm
                    
                    proj1 = np.dot(v1, n)
                    proj2 = np.dot(v2, n)
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 < min2:
                        gap = min2 - max1
                    elif max2 < min1:
                        gap = min1 - max2
                    else:
                        # Overlap or touching
                        # Gap is negative overlap depth
                        overlap = min(max1, max2) - max(min1, min2)
                        gap = -overlap
                    
                    if gap < pair_min_gap:
                        pair_min_gap = gap
                
                # Normals for v2 (only needed if we didn't find a separating axis in v1's normals? 
                # No, we need to check all to find the *minimum* gap (most negative)).
                # However, if pair_min_gap is already < 0, we know they overlap. 
                # To find the *deepest* overlap, we must check all.
                # But for the penalty, we usually just need to know *how much* they overlap.
                # The "maximum penetration depth" is the minimum gap over all axes.
                
                for k in range(6):
                    e = v2[(k+1)%6] - v2[k]
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-9: continue
                    n = n / n
                    
                    proj1 = np.dot(v1, n)
                    proj2 = np.dot(v2, n)
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 < min2:
                        gap = min2 - max1
                    elif max2 < min1:
                        gap = min1 - max2
                    else:
                        overlap = min(max1, max2) - max(min1, min2)
                        gap = -overlap
                    
                    if gap < pair_min_gap:
                        pair_min_gap = gap
                
                if pair_min_gap < min_gap:
                    min_gap = pair_min_gap
        
        # 2. Compute Outer Side Length
        # Outer hexagon centered at (outer_cx, outer_cy) with angle outer_angle_deg
        # The side length S is determined by the maximum projection of any inner vertex onto the outer normals.
        # For a regular hexagon of side S, the distance from center to vertex is S, and to edge midpoint is S*sqrt(3)/2.
        # Wait, the "side length" in the problem usually refers to the edge length of the outer hexagon.
        # Let R be the circumradius (distance to vertex). R = S.
        # The apothem (distance to edge) is S * sqrt(3) / 2.
        # The condition for containment is: for all outer normals n_k, max_{v in inner} dot(v - c_outer, n_k) <= apothem.
        # So apothem >= max_proj.
        # S = 2 * apothem / sqrt(3).
        
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        # We can vectorize this part significantly.
        # All inner vertices: shape (N*6, 2)
        all_verts = np.vstack(verts_list)
        rel_verts = all_verts - outer_center
        
        # Outer normals: 6 vectors
        outer_normals = np.array([
            [math.cos(outer_angle_rad + k * math.pi / 3), math.sin(outer_angle_rad + k * math.pi / 3)]
            for k in range(6)
        ])
        
        # Projections: (N*6, 6)
        projs = np.dot(rel_verts, outer_normals.T)
        max_proj = np.max(projs)
        
        side_length = 2.0 * max_proj / math.sqrt(3)
        
        return min_gap, side_length

    def objective(params):
        min_gap, side_length = compute_gaps_and_side(params)
        # Penalty for overlap
        # If min_gap < 0, there is overlap.
        # We use a quadratic penalty to make the objective smooth.
        penalty = 0.0
        if min_gap < 0:
            penalty = (-min_gap) ** 2
        
        return side_length + 100.0 * penalty

    # Initial Configuration: Hexagonal Grid
    # A 12-hexagon cluster can be formed by a central hexagon and its 6 neighbors, plus 5 more?
    # Or a 3x4 grid?
    # Let's use a compact cluster: 1 center, 6 around it, then 5 more attached to the outside.
    # Actually, a 3x4 staggered grid is a good start.
    
    s = 1.0
    dx = 2.0 * s
    dy = math.sqrt(3) * s
    
    centers = []
    # 4 rows, 3 cols
    for r in range(4):
        for c in range(3):
            x = c * dx
            y = r * dy
            if r % 2 == 1:
                x += s # offset
            centers.append((x, y))
            
    # Center the cluster
    cx_mean = sum(c[0] for c in centers) / N
    cy_mean = sum(c[1] for c in centers) / N
    adjusted_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # Initialize parameters
    # Format: [cx1, cy1, ang1, ..., cx12, cy12, ang12, outer_cx, outer_cy, outer_ang]
    x0 = np.zeros(51)
    for i in range(N):
        x0[3*i] = adjusted_centers[i][0]
        x0[3*i+1] = adjusted_centers[i][1]
        x0[3*i+2] = 0.0
        
    x0[48] = 0.0
    x0[49] = 0.0
    x0[50] = 0.0
    
    best_side = float('inf')
    best_params = x0.copy()
    best_valid = False
    
    def simulated_annealing(x_start, max_iter=30000, T0=0.5, T_min=0.0001, alpha=0.9995):
        current_x = x_start.copy()
        current_obj = objective(current_x)
        best_x = current_x.copy()
        best_obj = current_obj
        
        # Check initial validity
        min_gap, _ = compute_gaps_and_side(current_x)
        current_valid = min_gap >= -1e-5
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Perturb
            new_x = current_x.copy()
            # Perturb centers and angles
            # Step size decays with temperature
            step_size = T0 * (alpha ** it) * 2.0
            
            # Randomly select a few hexagons to move
            num_move = np.random.randint(1, 4)
            indices = np.random.choice(N, num_move, replace=False)
            for i in indices:
                new_x[3*i] += np.random.normal(0, step_size)
                new_x[3*i+1] += np.random.normal(0, step_size)
                new_x[3*i+2] += np.random.normal(0, step_size * 10) # Angles can change faster
                
            # Also perturb outer hexagon slightly
            new_x[48] += np.random.normal(0, step_size * 0.5)
            new_x[49] += np.random.normal(0, step_size * 0.5)
            new_x[50] += np.random.normal(0, step_size * 5)
            
            new_obj = objective(new_x)
            
            # Check validity of new state for tracking
            new_gap, _ = compute_gaps_and_side(new_x)
            new_valid = new_gap >= -1e-5
            
            # Acceptance
            T = T0 * (alpha ** it)
            delta = new_obj - current_obj
            if delta < 0 or (T > T_min and np.random.random() < math.exp(-delta / T)):
                current_x = new_x
                current_obj = new_obj
                
                if new_valid and new_obj < best_obj:
                    best_x = new_x.copy()
                    best_obj = new_obj
                    # Update best side if valid
                    _, side = compute_gaps_and_side(new_x)
                    if side < best_side:
                        best_side = side
                        best_params = new_x.copy()
                        best_valid = True
                        
        return best_x, best_obj, best_valid

    # Run multiple restarts
    for restart in range(8):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb initial grid
            x_start = x0.copy()
            x_start[:36] += np.random.normal(0, 0.2, size=36)
            x_start[36:48] += np.random.normal(0, 0.1, size=12) # Angles
            
        try:
            res_x, res_obj, res_valid = simulated_annealing(x_start)
            
            if res_valid and res_obj < best_obj:
                # Re-verify side length
                _, side = compute_gaps_and_side(res_x)
                if side < best_side:
                    best_side = side
                    best_params = res_x
                    best_valid = True
        except Exception:
            pass

    # Polish Phase: High precision local search on the best candidate
    if best_valid:
        # Use a smaller step size and more iterations
        # We can also try to fix the outer hexagon center to the centroid of inner hexagons
        # and just optimize the outer angle and inner positions.
        
        # Simple polish: SA with very small steps
        x_polish = best_params.copy()
        T0_polish = 0.01
        alpha_polish = 0.999
        for it in range(5000):
            if time.time() > deadline:
                break
            new_x = x_polish.copy()
            step = T0_polish * (alpha_polish ** it)
            
            # Perturb all slightly
            new_x[:48] += np.random.normal(0, step, size=48)
            new_x[48:51] += np.random.normal(0, step*0.5, size=3)
            
            new_obj = objective(new_x)
            current_obj = objective(x_polish)
            
            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < math.exp(-delta / T0_polish * (alpha_polish ** it)):
                x_polish = new_x
                _, side = compute_gaps_and_side(x_polish)
                min_gap, _ = compute_gaps_and_side(x_polish)
                if min_gap >= -1e-5 and side < best_side:
                    best_side = side
                    best_params = x_polish
                    best_valid = True

    # Final Construction
    if best_valid:
        inner = []
        for i in range(N):
            cx = best_params[3*i]
            cy = best_params[3*i+1]
            ang = best_params[3*i+2]
            inner.append([float(cx), float(cy), float(ang)])
            
        outer_cx = best_params[48]
        outer_cy = best_params[49]
        outer_angle_deg = best_params[50]
        
        # Add safety margin
        final_side = best_side * 1.001 + 1e-4
        
        return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle_deg)
    else:
        # Fallback
        inner_list = []
        for c, a in zip(adjusted_centers, [0.0]*N):
            inner_list.append([float(c[0]), float(c[1]), float(a)])
        return inner_list, (0.0, 0.0), 10.0, 0.0

# EVOLVE-BLOCK-END
