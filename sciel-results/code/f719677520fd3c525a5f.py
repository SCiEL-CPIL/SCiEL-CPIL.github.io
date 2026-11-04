import numpy as np
import time
from scipy.optimize import minimize

N = 11
# Minimum center-to-center distance for non-overlapping unit hexagons
# For two unit hexagons, the minimum distance between centers is 2 (diameter)
MIN_CENTER_DIST = 2.0

def get_hex_vertices(center, side_length=1.0, angle_deg=0.0):
    """Get the 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle_rad
    vertices = center + side_length * np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

def min_enclosing_hexagon_side(vertices):
    """
    Compute the minimum side length of a regular hexagon (any orientation) 
    that encloses the given vertices.
    
    Returns: (min_side_length, best_orientation_angle_rad)
    """
    # For a regular hexagon with side L, the distance from center to each side is L*sqrt(3)/2
    # We need to find the orientation and center that minimizes L
    # This is equivalent to finding the minimum width in all 6 directions
    
    # Try different orientations
    best_L = float('inf')
    best_angle = 0.0
    
    # Sample orientations
    for orient_deg in np.linspace(0, 60, 13):
        orient_rad = np.radians(orient_deg)
        normals = np.array([[np.cos(orient_rad + np.pi/3 * k), np.sin(orient_rad + np.pi/3 * k)] for k in range(6)])
        
        # For each orientation, find the minimum enclosing hexagon centered optimally
        # The constraint is: for all vertices v, |v . n_k| <= L * sqrt(3)/2 for all k
        # But we can also shift the center. Let's center at the centroid first as a heuristic.
        
        centroid = np.mean(vertices, axis=0)
        centered_verts = vertices - centroid
        
        projections = np.abs(centered_verts @ normals.T)
        max_proj = np.max(projections)
        L = max_proj / (np.sqrt(3) / 2)
        
        if L < best_L:
            best_L = L
            best_angle = orient_rad
    
    return best_L, best_angle

def compute_container_side(inner_positions, inner_angles):
    """
    Compute the minimum outer hexagon side length given inner hexagon positions and angles.
    """
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_positions)):
        pos = np.array(inner_positions[i])
        ang = inner_angles[i]
        verts = get_hex_vertices(pos, 1.0, ang)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Find minimum enclosing regular hexagon
    L, _ = min_enclosing_hexagon_side(all_vertices)
    return L

def cost_func(params):
    """
    Cost function for optimization.
    params: 22-element array [x1, y1, x2, y2, ..., x11, y11]
    Assumes all inner hexagons have angle 0 (flat-top).
    """
    positions = params.reshape(-1, 2)
    
    # Center the configuration
    centroid = np.mean(positions, axis=0)
    positions_centered = positions - centroid
    
    # Check non-overlap: minimum distance between centers must be >= 2
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions_centered[i] - positions_centered[j])
            if d < MIN_CENTER_DIST:
                penalty += (MIN_CENTER_DIST - d) ** 2
    
    if penalty > 0.01:
        # Heavy penalty for overlaps
        return 1000.0 + penalty
    
    # Compute the enclosing hexagon side length
    inner_angles = [0.0] * N
    L = compute_container_side(positions_centered, inner_angles)
    
    return L

def construct():
    best_L = float('inf')
    best_positions = None
    
    # Multiple restarts with different initializations
    for restart in range(30):
        if restart == 0:
            # Hexagonal lattice initialization: 1 center + 6 ring + 4 more
            # Standard hex packing: positions at distance 2 from each other
            init_pts = np.array([
                [0, 0],
                [2, 0],
                [1, np.sqrt(3)],
                [-1, np.sqrt(3)],
                [-2, 0],
                [-1, -np.sqrt(3)],
                [1, -np.sqrt(3)],
                # 4 more in the next ring
                [0, 2*np.sqrt(3)],
                [2, np.sqrt(3)],
                [2, -np.sqrt(3)],
                [0, -2*np.sqrt(3)]
            ])
        elif restart == 1:
            # Tighter hexagonal arrangement
            init_pts = np.array([
                [0, 0],
                [2, 0],
                [1, np.sqrt(3)],
                [-1, np.sqrt(3)],
                [-2, 0],
                [-1, -np.sqrt(3)],
                [1, -np.sqrt(3)],
                [3, np.sqrt(3)],
                [-3, np.sqrt(3)],
                [3, -np.sqrt(3)],
                [-3, -np.sqrt(3)]
            ])
        else:
            # Random initialization in a disk
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.random.uniform(0, 3, N)
            init_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        initial_params = init_pts.flatten()
        
        # Optimize
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if res.fun < 1000:  # No heavy penalty
            final_pts = res.x.reshape(-1, 2)
            centroid = np.mean(final_pts, axis=0)
            final_pts_centered = final_pts - centroid
            
            # Verify no overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(final_pts_centered[i] - final_pts_centered[j])
                    if d < MIN_CENTER_DIST - 1e-4:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                L = compute_container_side(final_pts_centered, [0.0] * N)
                if L < best_L:
                    best_L = L
                    best_positions = final_pts_centered.copy()
        
        # Time check
        if time.time() > 100:
            break
    
    # If no valid solution found, use a safe fallback
    if best_positions is None:
        # Linear arrangement with spacing 2
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    # Construct output
    inner = [[float(x), float(y), 0.0] for x, y in best_positions]
    outer_side = best_L + 1e-6  # Small buffer for numerical safety
    
    return inner, (0.0, 0.0), outer_side, 0.0
