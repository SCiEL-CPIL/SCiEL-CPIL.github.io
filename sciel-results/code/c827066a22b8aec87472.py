"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
from itertools import combinations

def _hex_vertices_array(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_vertices(positions, angles, side=1.0):
    """Get all vertices of all hexagons as (N*6, 2) array."""
    n = len(positions)
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices_array(positions[i][0], positions[i][1], side, angles[i])
    return verts

def _min_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation_deg=0.0):
    """Compute minimum side length of outer hexagon centered at outer_center with given rotation
    to contain all inner hexagons."""
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array(outer_center)
    
    phi = np.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    
    return max_proj / (np.sqrt(3) / 2)

def _check_overlap_pair(p1, a1, p2, a2, side=1.0):
    """Check if two unit hexagons overlap using SAT (Separating Axis Theorem)."""
    v1 = _hex_vertices_array(p1[0], p1[1], side, a1)
    v2 = _hex_vertices_array(p2[0], p2[1], side, a2)
    
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    axes = []
    for e in list(edges1) + list(edges2):
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            axes.append(n / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
            return False
    
    return True

def _check_all_disjoint(positions, angles):
    """Check all pairs are disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _check_overlap_pair(positions[i], angles[i], positions[j], angles[j]):
                return False
    return True

def _overlap_penalty(positions, angles, beta=50.0):
    """Compute smooth overlap penalty using SAT gaps."""
    n = len(positions)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v1 = _hex_vertices_array(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = _hex_vertices_array(positions[j][0], positions[j][1], 1.0, angles[j])
            
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            
            min_gap = float('inf')
            for e in list(edges1) + list(edges2):
                n_vec = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n_vec)
                if norm < 1e-12:
                    continue
                n_vec = n_vec / norm
                
                proj1 = np.dot(v1, n_vec)
                proj2 = np.dot(v2, n_vec)
                
                gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                if gap < min_gap:
                    min_gap = gap
            
            if min_gap < 0:
                penalty += math.exp(-beta * min_gap)
    
    return penalty

def _objective(x, angles, outer_rotation_deg):
    """Objective: outer side length + overlap penalty."""
    positions = [(x[2*i], x[2*i+1]) for i in range(N)]
    
    if not _check_all_disjoint(positions, angles):
        pen = _overlap_penalty(positions, angles, beta=50.0)
        if pen > 1.0:
            return 1e6 + pen
    
    side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), outer_rotation_deg)
    return side

def _refine_positions(initial_positions, angles, outer_rotation_deg, deadline):
    """Use scipy.optimize to refine positions for tighter packing."""
    x0 = np.array([initial_positions[i][0] for i in range(N)] + \
                  [initial_positions[i][1] for i in range(N)])
    
    try:
        result = minimize(
            _objective, x0, args=(angles, outer_rotation_deg),
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success and result.fun < 1e5:
            refined_positions = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            if _check_all_disjoint(refined_positions, angles):
                return refined_positions
    except Exception:
        pass
    
    return initial_positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Generate candidate arrangements
    arrangements = []
    
    # Option 1: 4 consecutive from ring2
    for start in range(12):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Option 2: Symmetric selections (opposite pairs)
    for i in range(6):
        for j in range(i+1, 6):
            subset = [ring2[i], ring2[i+6], ring2[j], ring2[j+6]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy(q, r) for q, r in axial_coords]
            arrangements.append(positions)
    
    # Option 3: All C(12,4) combinations (495 total, but many are equivalent)
    for combo in combinations(range(12), 4):
        subset = [ring2[i] for i in combo]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Deduplicate
    unique_arrangements = []
    seen = set()
    for arr in arrangements:
        key = tuple((round(p[0], 4), round(p[1], 4)) for p in arr)
        if key not in seen:
            seen.add(key)
            unique_arrangements.append(arr)
    
    arrangements = unique_arrangements
    
    # Phase 1: Quick evaluation of all arrangements to find top candidates
    candidates = []
    for arr in arrangements:
        if time.time() > deadline - 60.0:
            break
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            continue
        
        best_side_for_arr = float('inf')
        best_rot = 0.0
        for outer_rot in [0.0, 30.0]:
            side = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), outer_rot)
            if side < best_side_for_arr:
                best_side_for_arr = side
                best_rot = outer_rot
        
        candidates.append((best_side_for_arr, arr, best_rot))
    
    # Sort by side length and keep top 10 for refinement
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:10]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_rotation = 0.0
    
    # Phase 2: Refine top candidates with optimization
    for side_est, arr, outer_rot in top_candidates:
        if time.time() > deadline - 10.0:
            break
        
        angles = [0.0] * N
        
        # Try refinement
        refined = _refine_positions(arr, angles, outer_rot, deadline)
        
        # Evaluate refined
        for pos in [arr, refined]:
            if not _check_all_disjoint(pos, angles):
                continue
            for rot in [outer_rot, (outer_rot + 30.0) % 60.0]:
                side = _min_enclosing_hex_side(pos, angles, (0.0, 0.0), rot)
                if side < best_side:
                    best_side = side
                    best_positions = pos[:]
                    best_angles = angles[:]
                    best_outer_rotation = rot
    
    # If no valid arrangement found, fallback
    if best_positions is None:
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_side = _min_enclosing_hex_side(best_positions, best_angles)
        best_outer_rotation = 0.0
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, best_outer_rotation
# EVOLVE-BLOCK-END
