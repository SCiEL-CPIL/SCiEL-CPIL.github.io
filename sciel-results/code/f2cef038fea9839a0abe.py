"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    n_centers = len(all_centers)  # 19
    
    # Precompute vertex offsets for unit hexagon (flat-top)
    vertex_angles = np.array([math.radians(60 * k) for k in range(6)])
    vertex_offsets = np.column_stack([np.cos(vertex_angles), np.sin(vertex_angles)])
    
    # Precompute normal directions for outer hexagon containment check
    normal_angles = np.array([math.radians(60 * k + 30) for k in range(6)])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Precompute center positions as numpy array
    center_positions = np.array([(c[0], c[1]) for c in all_centers])
    
    # For each center and each vertex, precompute the max projection onto each normal
    # This gives us the L needed for that specific vertex-center pair
    # L_needed = max_proj / (sqrt(3)/2)
    
    # Compute all vertex positions for all centers: shape (19, 6, 2)
    all_vertices = center_positions[:, np.newaxis, :] + vertex_offsets[np.newaxis, :, :]
    
    # Compute projections onto all normals: shape (19, 6, 6)
    # proj[i, j, k] = dot(vertex[i,j], normal[k])
    all_projections = np.einsum('ijk,kl->ijl', all_vertices, normals)
    
    # For each center i, the L needed is max over all vertices j and normals k of proj / (sqrt3/2)
    # L_i = max(all_projections[i, :, :]) / (sqrt3/2)
    max_projections_per_center = np.max(all_projections, axis=(1, 2))
    L_per_center = max_projections_per_center / (sqrt3 / 2.0)
    
    # For a subset of centers, the required L is the max of L_per_center over the subset
    # So we want to choose 12 centers that minimize max(L_per_center[i] for i in subset)
    
    # Sort centers by their individual L requirement
    sorted_indices = np.argsort(L_per_center)
    sorted_L = L_per_center[sorted_indices]
    
    # The optimal subset of size 12 should include the 12 centers with smallest individual L
    # But wait - this is only valid if the L for a subset is just the max of individual Ls.
    # Let's verify: for a subset S, L(S) = max over centers in S of (max over vertices and normals of proj)
    # = max over centers in S of L_per_center[i]
    # Yes! So the optimal subset is simply the 12 centers with smallest L_per_center.
    
    # However, we need to check if there's a better arrangement with rotation.
    # Let's first get the baseline from the grid subset.
    
    # Select the 12 centers with smallest individual L
    best_subset_indices = sorted_indices[:12]
    best_L_ideal = np.max(L_per_center[best_subset_indices])
    
    # But we should also try other subsets to be safe, in case my reasoning is wrong
    # Let's do a quick check with a few random subsets
    from itertools import combinations
    
    # Actually, let's verify with brute force on a smaller set first
    # The key insight: L(subset) = max(L_per_center[i] for i in subset)
    # So we want min over all subsets of size 12 of max(L_per_center[i] for i in subset)
    # This is achieved by picking the 12 smallest L_per_center values.
    
    # Let's double-check by trying a few random subsets
    np.random.seed(42)
    for trial in range(100):
        random_subset = np.random.choice(n_centers, 12, replace=False)
        L_trial = np.max(L_per_center[random_subset])
        if L_trial < best_L_ideal:
            best_L_ideal = L_trial
            best_subset_indices = random_subset
    
    # Now, let's try rotating the inner hexagons to potentially reduce L
    # If we rotate inner hexagons by some angle, the vertex positions change
    # This could allow a smaller outer hexagon
    
    def compute_L_with_rotation(centers, rotation_angle_deg):
        """Compute required outer L for given centers with inner hexagons rotated."""
        rot_rad = math.radians(rotation_angle_deg)
        rot_matrix = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                               [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated_vertices = vertex_offsets @ rot_matrix.T  # shape (6, 2)
        
        all_vertices_rot = centers[:, np.newaxis, :] + rotated_vertices[np.newaxis, :, :]
        all_projs_rot = np.einsum('ijk,kl->ijl', all_vertices_rot, normals)
        max_projs = np.max(all_projs_rot, axis=(1, 2))
        return np.max(max_projs) / (sqrt3 / 2.0)
    
    # Try different rotation angles for the inner hexagons
    best_rotation = 0.0
    best_L_rotated = best_L_ideal
    
    selected_centers = center_positions[best_subset_indices]
    
    # Try rotations from 0 to 60 degrees in small steps
    for rot_deg in np.arange(0, 60.0, 1.0):
        L_rot = compute_L_with_rotation(selected_centers, rot_deg)
        if L_rot < best_L_rotated:
            best_L_rotated = L_rot
            best_rotation = rot_deg
    
    # Refine rotation with finer steps around the best
    if best_rotation > 0:
        for rot_deg in np.arange(max(0, best_rotation - 2), min(60, best_rotation + 2), 0.1):
            L_rot = compute_L_with_rotation(selected_centers, rot_deg)
            if L_rot < best_L_rotated:
                best_L_rotated = L_rot
                best_rotation = rot_deg
    
    # Now try different subsets with the best rotation
    # We need to find which subset of 12 centers, when inner hexagons are rotated by best_rotation,
    # gives the smallest outer L
    
    def compute_L_subset_with_rotation(subset_indices, rotation_angle_deg):
        centers_subset = center_positions[subset_indices]
        return compute_L_with_rotation(centers_subset, rotation_angle_deg)
    
    best_L_final = best_L_rotated
    best_final_subset = best_subset_indices.copy()
    best_final_rotation = best_rotation
    
    # Try all subsets if time permits
    # C(19,12) = C(19,7) = 50388
    # With numpy, each evaluation is fast
    
    if time.time() < deadline - 30:
        # Try to find better subsets with rotation
        # First, precompute L for each center with the best rotation
        rot_rad = math.radians(best_final_rotation)
        rot_matrix = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                               [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated_vertices = vertex_offsets @ rot_matrix.T
        
        all_vertices_rot = center_positions[:, np.newaxis, :] + rotated_vertices[np.newaxis, :, :]
        all_projs_rot = np.einsum('ijk,kl->ijl', all_vertices_rot, normals)
        max_projs_rot = np.max(all_projs_rot, axis=(1, 2))
        L_per_center_rot = max_projs_rot / (sqrt3 / 2.0)
        
        # The optimal subset is still the 12 with smallest L_per_center_rot
        sorted_indices_rot = np.argsort(L_per_center_rot)
        candidate_subset = sorted_indices_rot[:12]
        L_candidate = np.max(L_per_center_rot[candidate_subset])
        
        if L_candidate < best_L_final:
            best_L_final = L_candidate
            best_final_subset = candidate_subset
    
    # Scale to ensure strict disjointness
    # Centers on the hex grid are at distance sqrt(3) apart
    # We need gap > 0, so scale by factor slightly > 1
    SCALE = 1.001
    
    inner = []
    final_centers = []
    for i in best_final_subset:
        x, y = center_positions[i]
        sx, sy = x * SCALE, y * SCALE
        inner.append([sx, sy, best_final_rotation])
        final_centers.append((sx, sy))
    
    final_centers_arr = np.array(final_centers)
    
    # Compute final L with scaled centers and rotation
    final_L = compute_L_with_rotation(final_centers_arr, best_final_rotation)
    
    # Add a small buffer to ensure containment
    final_L *= 1.0001
    
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
