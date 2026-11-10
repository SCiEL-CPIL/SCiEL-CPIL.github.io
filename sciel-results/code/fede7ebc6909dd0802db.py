# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _compute_min_side(inner, outer_center=None):
    """Compute the minimum side length of outer hexagon needed to contain all inner hexagons."""
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])
    
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(11):
        verts = _hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], 1.0, inner_shifted[i, 2])
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # The 6 edge normals of a regular hexagon centered at origin (angle 0)
    normals = np.column_stack([np.cos(np.radians(30 + 60 * np.arange(6))), 
                                 np.sin(np.radians(30 + 60 * np.arange(6)))])
    
    # For each normal direction, compute max projection
    projections = all_verts @ normals.T  # shape (66, 6)
    max_projs = projections.max(axis=0)  # shape (6,)
    
    # The side length s must satisfy: s * sqrt(3)/2 >= max_projs
    # So s >= max_projs * 2 / sqrt(3)
    min_side = np.max(max_projs) * 2.0 / np.sqrt(3)
    
    return min_side


def _check_separation(inner, tol=1e-6):
    """Check if all pairs of inner hexagons are separated."""
    for i in range(11):
        for j in range(i+1, 11):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            
            # Get edge normals for both hexagons
            def get_edge_normals(verts):
                normals = []
                for k in range(6):
                    e = verts[(k+1) % 6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                return normals
            
            ni = get_edge_normals(vi)
            nj = get_edge_normals(vj)
            
            # Check on at least one of 12 edge normals the projected intervals are apart by > tol
            separated = False
            for n in ni + nj:
                proj_i = vi @ n
                proj_j = vj @ n
                max_i = proj_i.max()
                min_i = proj_i.min()
                max_j = proj_j.max()
                min_j = proj_j.min()
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Flower pattern: 1 center + 6 surrounding at distance 2
    # For flat-top hexagons (angle=0), neighbors at 0,60,120,180,240,300 degrees
    # Distance between touching flat-top hex centers = 2
    
    sqrt3 = np.sqrt(3)
    initial = np.array([
        [0.0, 0.0, 0.0],       # center
        [2.0, 0.0, 0.0],       # right (0 deg)
        [1.0, sqrt3, 0.0],     # top-right (60 deg)
        [-1.0, sqrt3, 0.0],    # top-left (120 deg)
        [-2.0, 0.0, 0.0],      # left (180 deg)
        [-1.0, -sqrt3, 0.0],   # bottom-left (240 deg)
        [1.0, -sqrt3, 0.0],    # bottom-right (300 deg)
        [3.0, sqrt3, 0.0],     # far top-right
        [-3.0, sqrt3, 0.0],    # far top-left
        [3.0, -sqrt3, 0.0],    # far bottom-right
        [-3.0, -sqrt3, 0.0],   # far bottom-left
    ])
    
    best_inner = initial.copy()
    best_side = _compute_min_side(initial)
    
    # Verify initial configuration is valid
    if _check_separation(best_inner):
        pass
    else:
        # Adjust if needed - add small gap
        best_inner[1:7, 0] *= 1.001
        best_inner[1:7, 1] *= 1.001
        best_inner[7:, 0] *= 1.002
        best_inner[7:, 1] *= 1.002
        best_side = _compute_min_side(best_inner)
    
    # Optimize positions to minimize outer hexagon side length
    # Use iterative local search with random restarts
    
    rng = np.random.RandomState(42)
    x0 = best_inner[:, 0].copy()
    y0 = best_inner[:, 1].copy()
    a0 = np.radians(best_inner[:, 2]).copy()
    
    current_side = best_side
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 20 and no_improve_count < max_no_improve:
        improved = False
        
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random perturbation with decreasing step size
            step = 0.05 * (1.0 + no_improve_count / 100.0)
            perturbation = rng.normal(0, step, size=33)
            
            # Keep center fixed for symmetry
            perturbation[0] = 0
            perturbation[1] = 0
            perturbation[2] = 0
            
            params_trial = np.concatenate([x0, y0, a0]) + perturbation
            
            # Normalize angles to [-pi, pi]
            params_trial[22:33] = np.arctan2(np.sin(params_trial[22:33]), np.cos(params_trial[22:33]))
            
            # Build trial inner hexagons
            inner_trial = np.column_stack([
                params_trial[:11],
                params_trial[11:22],
                np.degrees(params_trial[22:33])
            ])
            
            # Quick separation check
            if not _check_separation(inner_trial):
                continue
            
            side_trial = _compute_min_side(inner_trial)
            
            if side_trial < current_side - 1e-10:
                x0 = params_trial[:11].copy()
                y0 = params_trial[11:22].copy()
                a0 = params_trial[22:33].copy()
                current_side = side_trial
                best_inner = inner_trial.copy()
                best_side = side_trial
                improved = True
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    # Final verification
    if not _check_separation(best_inner):
        # Fall back to valid configuration
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        best_side = 8.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return best_inner, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
