import numpy as np
import time
import itertools

N = 11

# Hexagonal lattice basis vectors
# Axial coordinates (q, r) to Cartesian (x, y)
# x = q + r/2
# y = r * sqrt(3)/2
# Side length of unit hexagon is 1.
# Distance between adjacent lattice points is sqrt(3).
# Wait, the standard hexagonal tiling of unit hexagons has centers at distance sqrt(3).
# Let's verify.
# Vertices of unit hex at (0,0): (1,0), (0.5, sqrt(3)/2), ...
# Neighbor in x direction: (sqrt(3), 0). Distance is sqrt(3).
# Neighbor in 60 deg direction: (sqrt(3)/2, 1.5). Distance is sqrt(3/4 + 2.25) = sqrt(3).
# Yes, lattice spacing is sqrt(3).

def axial_to_cartesian(q, r):
    x = (q + r / 2.0) * np.sqrt(3)
    y = r * 1.5
    return x, y

def cartesian_to_axial(x, y):
    r = y / 1.5
    q = x / np.sqrt(3) - r / 2.0
    return q, r

# Precompute unit hexagon vertices for aligned orientation
unit_hex_verts = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2.0],
    [-0.5, np.sqrt(3)/2.0],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2.0],
    [0.5, -np.sqrt(3)/2.0]
])

# Normals for outer hexagon (aligned)
# The outer hexagon contains the union of inner hexagons.
# The boundary of the union is determined by the max projection of any vertex onto the 6 normal directions.
# Normals are at 30, 90, 150, 210, 270, 330 degrees.
outer_normals = np.array([
    [np.cos(np.pi/6), np.sin(np.pi/6)],
    [np.cos(np.pi/2), np.sin(np.pi/2)],
    [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
    [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
    [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
    [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
])

def calc_min_side_aligned(centers_cart):
    """
    centers_cart: (N, 2) array of centers of inner hexagons (aligned).
    Returns the minimum side length S of an aligned outer hexagon centered at origin 
    that contains all inner hexagons.
    Note: We can also optimize the outer center. 
    For aligned hexagons, the minimal enclosing hexagon is centered at the "median" of the projections?
    Actually, for a set of points, the minimal enclosing regular hexagon (aligned) side length S
    is determined by the width of the point set in the 6 normal directions.
    Let W_k be the width in direction k. S_k = W_k / sqrt(3) * 2? 
    No.
    For a regular hexagon with side S, the width (distance between parallel sides) is S * sqrt(3).
    So if the projection of the inner set onto normal k has range [min_k, max_k], the width is max_k - min_k.
    We need S * sqrt(3) >= max_k - min_k for all k.
    So S >= max_k (max_k - min_k) / sqrt(3).
    
    However, the outer hexagon can be translated.
    If we translate the outer hexagon by vector t, the projections shift by t . n_k.
    We want to minimize S such that there exists t where for all k:
    min_k + t.n_k <= -S*sqrt(3)/2
    max_k + t.n_k >= S*sqrt(3)/2
    This implies max_k - min_k <= S*sqrt(3).
    The translation t doesn't change the width (max-min).
    So the minimal S is simply max_k (width_k) / sqrt(3).
    Wait, is the outer hexagon allowed to be centered anywhere? Yes.
    Does the center of the outer hexagon need to be at the centroid? No.
    But the width of the set of points is invariant to translation.
    So, for aligned hexagons, the minimal side length is determined solely by the widths of the inner configuration in the 6 normal directions.
    """
    # Get all vertices of all inner hexagons
    # centers_cart is (N, 2)
    # Vertices for each hexagon: center + unit_hex_verts
    # All vertices: (N, 6, 2)
    all_verts = centers_cart[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    # Project onto normals
    projs = all_verts_flat @ outer_normals.T # (N*6, 6)
    widths = np.max(projs, axis=0) - np.min(projs, axis=0) # (6,)
    max_width = np.max(widths)
    S = max_width / np.sqrt(3)
    return S

def get_initial_config():
    """
    Generate a compact initial configuration on the lattice.
    """
    # Axial coordinates for a compact cluster of N=11
    # Center (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 6 more? 
    # Total 1 + 6 + 4 = 11.
    
    # List of axial coords
    coords = [(0,0)]
    # Ring 1
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    coords.extend(ring1)
    # Ring 2 (pick 4 to make it compact)
    # Neighbors of ring 1 that are not ring 0 or 1
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Let's pick (2,0), (1,1), (0,2), (-1,2) -> this makes a triangular shape?
    # Or pick symmetric ones: (2,0), (0,2), (-2,1), (-1,-1)? 
    # Let's just pick the first 4 from the ring 2 list.
    ring2_candidates = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    coords.extend(ring2_candidates[:4])
    
    return coords

def local_search(coords, time_limit):
    """
    Perform local search to minimize calc_min_side_aligned.
    coords: list of (q, r) tuples.
    """
    start_time = time.time()
    best_coords = list(coords)
    best_centers = np.array([axial_to_cartesian(q, r) for q, r in best_coords])
    best_S = calc_min_side_aligned(best_centers)
    
    # Directions to move (6 neighbors)
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Set of occupied coords for O(1) lookup
    occupied = set(best_coords)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try moving each hexagon
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
                
            q, r = best_coords[i]
            
            # Try all 6 neighbors
            for dq, dr in neighbors:
                nq, nr = q + dq, r + dr
                
                # Check if valid (not occupied)
                if (nq, nr) in occupied:
                    continue
                
                # Tentatively move
                # Remove old, add new
                occupied.remove((q, r))
                occupied.add((nq, nr))
                
                new_coords = list(best_coords)
                new_coords[i] = (nq, nr)
                
                new_centers = np.array([axial_to_cartesian(cq, cr) for cq, cr in new_coords])
                new_S = calc_min_side_aligned(new_centers)
                
                if new_S < best_S - 1e-9:
                    # Accept move
                    best_coords = new_coords
                    best_centers = new_centers
                    best_S = new_S
                    improved = True
                else:
                    # Revert
                    occupied.remove((nq, nr))
                    occupied.add((q, r))
                    
        if not improved:
            # Try random restart or larger jump?
            # For now, break if no improvement in a full pass
            break
            
    return best_coords, best_S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_coords = None
    best_S = np.inf
    
    # Run multiple random restarts of local search
    # We can randomize the initial set of 4 ring-2 hexagons
    ring2_candidates = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    
    # Generate a few different initial configurations
    initial_configs = []
    # Config 1: Default
    initial_configs.append(get_initial_config())
    
    # Config 2: Different ring 2 selection
    # Pick 4 random from ring 2
    np.random.seed(123)
    for _ in range(10):
        base = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        extra = np.random.choice(len(ring2_candidates), 4, replace=False)
        coords = base + [ring2_candidates[i] for i in extra]
        initial_configs.append(coords)
        
    for i, init_coords in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # Time limit per restart
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Allocate time based on remaining
        t_limit = min(remaining / 2.0, 10.0) # Max 10s per restart
        
        coords, S = local_search(init_coords, t_limit)
        
        if S < best_S:
            best_S = S
            best_coords = coords
            
    if best_coords is None:
        # Fallback
        best_coords = get_initial_config()
        best_S = calc_min_side_aligned(np.array([axial_to_cartesian(q, r) for q, r in best_coords]))
        
    # Prepare output
    inner = []
    for q, r in best_coords:
        x, y = axial_to_cartesian(q, r)
        inner.append([float(x), float(y), 0.0])
        
    # The outer hexagon center should be chosen to minimize the required side?
    # Our calc_min_side_aligned assumed the outer hexagon is centered such that the widths fit.
    # Actually, the width calculation `max - min` gives the required width.
    # The center of the outer hexagon should be the midpoint of the min and max projections?
    # For each normal k, the outer hexagon side is at distance S*sqrt(3)/2 from the center.
    # So center projection c_k should be (min_k + max_k) / 2.
    # We need a single center (x_c, y_c) that satisfies this for all 6 normals.
    # This is a system of linear equations:
    # x_c * n_k[0] + y_c * n_k[1] = (min_k + max_k) / 2
    # We can solve this least squares.
    
    # Recalculate projections for the best configuration
    centers_cart = np.array([axial_to_cartesian(q, r) for q, r in best_coords])
    all_verts = centers_cart[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    projs = all_verts_flat @ outer_normals.T
    mins = np.min(projs, axis=0)
    maxs = np.max(projs, axis=0)
    targets = (mins + maxs) / 2.0
    
    # Solve A * c = b
    # A is 6x2 (outer_normals)
    # b is targets (6,)
    try:
        center_x, center_y = np.linalg.lstsq(outer_normals, targets, rcond=None)[0]
    except:
        center_x, center_y = 0.0, 0.0
        
    # The side length S is max_width / sqrt(3)
    widths = maxs - mins
    S = np.max(widths) / np.sqrt(3)
    
    # The outer hexagon rotation is 0 (aligned)
    outer_rotation = 0.0
    
    return inner, (float(center_x), float(center_y)), float(S), float(outer_rotation)
