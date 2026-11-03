import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at angles 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts_batch(positions, angles):
        """
        Vectorized computation of hexagon vertices.
        positions: (N, 2)
        angles: (N,)
        Returns: (N, 6, 2)
        """
        N_local = positions.shape[0]
        angles_rad = np.radians(angles)
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
        
        # Rotation matrices: (N, 2, 2)
        rot = np.zeros((N_local, 2, 2))
        rot[:, 0, 0] = cos_a
        rot[:, 0, 1] = -sin_a
        rot[:, 1, 0] = sin_a
        rot[:, 1, 1] = cos_a
        
        # Rotate base vertices: (N, 6, 2)
        # unit_verts_0 is (6, 2)
        # We want v_rot = R @ v0
        # Using einsum: 'nij,nkj->nik' where i,j are matrix indices, k is vertex index? 
        # Actually: v_rot[k] = sum_j R[i,j] v0[j,k] ? No, R is 2x2, v0 is 2x1 per vertex.
        # v_rot = R @ v0.T -> (N, 2, 6) -> transpose to (N, 6, 2)
        
        # Let's do it simply:
        # v0 is (6,2). R is (N,2,2).
        # result[n, k, i] = sum_j R[n, i, j] * v0[k, j]
        verts_rot = np.einsum('nij,kj->nki', rot, unit_verts_0)
        
        # Add positions
        verts = verts_rot + positions[:, np.newaxis, :]
        return verts

    def calc_min_outer_side_vectorized(positions, angles, outer_angle=0.0):
        """
        Calculate minimum side length of regular hexagon centered at (0,0) with rotation outer_angle
        containing all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the 6 sides of the outer hexagon
        # Side normals are perpendicular to edges. For a regular hexagon, normals point to vertices?
        # No, for a regular hexagon with flat top/bottom (or pointy top), the support function is max projection on 6 axes.
        # The axes are the normals to the 6 sides.
        # For a hexagon with side length S, distance from center to side (apothem) is S * sqrt(3) / 2.
        # If max projection on normal n is P, then P = S * sqrt(3) / 2.
        # So S = P * 2 / sqrt(3).
        
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        verts = get_hex_verts_batch(positions, angles) # (N, 6, 2)
        
        # Project all vertices onto all 6 normals
        # projs[n, k, i] = dot(verts[n, k], normals[i])
        projs = np.einsum('nki,ji->nkj', verts, normals) # (N, 6, 6)
        
        # Max projection per hexagon per normal
        max_proj_per_hex_normal = np.max(projs, axis=1) # (N, 6)
        
        # Global max projection
        global_max_proj = np.max(max_proj_per_hex_normal)
        
        return global_max_proj * 2.0 / SQRT3

    def check_disjoint_fast_vectorized(positions, angles):
        """
        Check if all hexagons are disjoint.
        Returns True if valid, False otherwise.
        Uses SAT.
        """
        verts = get_hex_verts_batch(positions, angles) # (N, 6, 2)
        
        # Precompute edges and normals for each hexagon
        # edges[n, k] = verts[n, (k+1)%6] - verts[n, k]
        verts_shifted = np.roll(verts, -1, axis=1)
        edges = verts_shifted - verts # (N, 6, 2)
        
        # Normals to edges: rotate edge by 90 deg
        # n = [-dy, dx]
        edge_normals = np.empty_like(edges)
        edge_normals[:, :, 0] = -edges[:, :, 1]
        edge_normals[:, :, 1] = edges[:, :, 0]
        
        # Normalize normals (they should be unit length for unit hexagons, but just in case)
        norms = np.linalg.norm(edge_normals, axis=2, keepdims=True)
        norms[norms == 0] = 1.0
        edge_normals = edge_normals / norms
        
        # For each pair (i, j), check SAT
        # To speed up, we can check all pairs in a loop but vectorize the projection part for a specific pair?
        # Or just loop pairs, it's only 66 pairs.
        
        for i in range(N):
            for j in range(i+1, N):
                v1 = verts[i] # (6, 2)
                v2 = verts[j] # (6, 2)
                
                # Axes: normals from i and normals from j
                axes = np.vstack([edge_normals[i], edge_normals[j]]) # (12, 2)
                
                # Project v1 and v2 onto axes
                proj1 = v1 @ axes.T # (6, 12)
                proj2 = v2 @ axes.T # (6, 12)
                
                min1 = np.min(proj1, axis=0)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                
                # Check for separation
                # If max1 < min2 or max2 < min1 for any axis, they are separated.
                separated = np.any((max1 < min2) | (max2 < min1))
                
                if not separated:
                    return False
        return True

    def objective(params):
        """
        Objective for differential evolution.
        Minimizes outer side length + penalty for overlaps.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side = calc_min_outer_side_vectorized(positions, angles, 0.0)
        
        # Check validity
        if not check_disjoint_fast_vectorized(positions, angles):
            # Calculate overlap penalty
            # We can use a simple heuristic: sum of squared overlaps?
            # Or just return a large value.
            # To help DE, let's provide a gradient-like signal via penalty magnitude.
            # Approximate penalty by checking how much they overlap on SAT axes.
            verts = get_hex_verts_batch(positions, angles)
            edges = np.roll(verts, -1, axis=1) - verts
            edge_normals = np.empty_like(edges)
            edge_normals[:, :, 0] = -edges[:, :, 1]
            edge_normals[:, :, 1] = edges[:, :, 0]
            norms = np.linalg.norm(edge_normals, axis=2, keepdims=True)
            norms[norms == 0] = 1.0
            edge_normals = edge_normals / norms
            
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    v1 = verts[i]
                    v2 = verts[j]
                    axes = np.vstack([edge_normals[i], edge_normals[j]])
                    proj1 = v1 @ axes.T
                    proj2 = v2 @ axes.T
                    min1 = np.min(proj1, axis=0)
                    max1 = np.max(proj1, axis=0)
                    min2 = np.min(proj2, axis=0)
                    max2 = np.max(proj2, axis=0)
                    
                    overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
                    overlaps = np.where(overlaps > 0, overlaps, 0)
                    penalty += np.sum(overlaps**2)
            
            return side + 100.0 * penalty + 10.0 * side # Heuristic weighting
            
        return side

    # Bounds
    # Positions: within a reasonable box, say [-10, 10]
    # Angles: [0, 360]
    bounds = [(-10, 10)] * (2*N) + [(0, 360)] * N
    
    # Initial population strategy: 
    # DE will generate random, but we can seed with a few good guesses if time permits?
    # DE doesn't take x0, it generates.
    # We can run DE with a small population size and maxiter to fit in time.
    
    best_result = None
    best_score = float('inf')
    
    # Run Differential Evolution
    # maxiter and popsize control the budget.
    # 150s limit. 
    # Let's estimate time per iteration. 
    # Objective is vectorized but SAT check is a loop over 66 pairs.
    # 66 pairs * 12 axes * vector ops. Should be fast.
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=20,
            tol=1e-4,
            seed=42,
            polish=False, # We will polish manually
            callback=None, # Could add time check here
            update_interval=10 # Print progress? No, silent.
        )
        
        if de_result.success or de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
            
    except Exception:
        pass
    
    # If DE didn't find a valid solution or score is high, try a few random restarts with local optimization
    # Or just use DE result.
    
    if best_result is None:
        # Fallback: Honeycomb
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * SQRT3 / 2.0 + j * SQRT3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        best_result = np.concatenate([init_positions.flatten(), init_angles])
        best_score = objective(best_result)
        
    # Local refinement using Nelder-Mead or Powell on the best found
    # We need a smooth objective for local opt.
    # The penalty method objective is piecewise smooth.
    
    def local_obj(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        side = calc_min_outer_side_vectorized(positions, angles, 0.0)
        
        if not check_disjoint_fast_vectorized(positions, angles):
            # Same penalty as in objective
            verts = get_hex_verts_batch(positions, angles)
            edges = np.roll(verts, -1, axis=1) - verts
            edge_normals = np.empty_like(edges)
            edge_normals[:, :, 0] = -edges[:, :, 1]
            edge_normals[:, :, 1] = edges[:, :, 0]
            norms = np.linalg.norm(edge_normals, axis=2, keepdims=True)
            norms[norms == 0] = 1.0
            edge_normals = edge_normals / norms
            
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    v1 = verts[i]
                    v2 = verts[j]
                    axes = np.vstack([edge_normals[i], edge_normals[j]])
                    proj1 = v1 @ axes.T
                    proj2 = v2 @ axes.T
                    min1 = np.min(proj1, axis=0)
                    max1 = np.max(proj1, axis=0)
                    min2 = np.min(proj2, axis=0)
                    max2 = np.max(proj2, axis=0)
                    overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
                    overlaps = np.where(overlaps > 0, overlaps, 0)
                    penalty += np.sum(overlaps**2)
            return side + 100.0 * penalty + 10.0 * side
        return side

    # Run local minimization from the DE best
    if time.time() < deadline - 20:
        try:
            local_res = minimize(
                local_obj,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if local_res.fun < best_score:
                # Check if valid
                pos = local_res.x[:2*N].reshape(N, 2)
                ang = local_res.x[2*N:]
                if check_disjoint_fast_vectorized(pos, ang):
                    best_score = local_res.fun
                    best_result = local_res.x
        except Exception:
            pass
            
    # Final validation and extraction
    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:]
    
    # Ensure validity
    if not check_disjoint_fast_vectorized(final_positions, final_angles):
        # If invalid, try to fix by expanding slightly? Or fallback?
        # For now, if DE failed to find valid, we might return an invalid solution which scores 0.
        # Let's try a few random local searches if invalid.
        
        found_valid = False
        for _ in range(10):
            if time.time() > deadline - 5: break
            # Perturb
            pos = final_positions + np.random.normal(0, 0.1, (N, 2))
            ang = final_angles + np.random.normal(0, 10, N)
            ang = ang % 360
            if check_disjoint_fast_vectorized(pos, ang):
                s = calc_min_outer_side_vectorized(pos, ang, 0.0)
                if s < best_score:
                    best_score = s
                    final_positions = pos
                    final_angles = ang
                    found_valid = True
                break # Take first valid found in this quick loop
                
        if not found_valid:
            # Fallback to honeycomb
            lattice_points = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    x = i * 1.5
                    y = i * SQRT3 / 2.0 + j * SQRT3
                    lattice_points.append((x, y))
            lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
            final_positions = np.array(lattice_points[:12])
            final_positions -= np.mean(final_positions, axis=0)
            final_angles = np.zeros(N)
            best_score = calc_min_outer_side_vectorized(final_positions, final_angles, 0.0)

    final_side = calc_min_outer_side_vectorized(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
