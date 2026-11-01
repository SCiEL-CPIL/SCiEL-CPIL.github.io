"""
Circle packing in perimeter-4 rectangle: Voronoi Inradius Maximization.
Strategy:
1. Initialize with a hexagonal lattice pattern.
2. Use SLSQP to minimize the negative sum of Voronoi inradii.
3. The rectangle aspect ratio (W) is also an optimization variable (H = 2 - W).
4. Polishing phase with high-precision local search.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError
from scipy.spatial.distance import cdist

N = 21

def _compute_inradii(cx, cy, W, H):
    """
    Compute the inradius for each circle in the packing.
    Inradius of circle i = min(distance to walls, distance to nearest other circle / 2).
    Note: For Voronoi cells, the inradius is the distance to the nearest boundary of the cell.
    The boundaries are formed by walls and perpendicular bisectors to neighbors.
    Distance to bisector between i and j is dist(i,j)/2.
    So inradius_i = min( wall_dist_i, min_{j!=i} dist(i,j)/2 ).
    """
    n = len(cx)
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(cx, W - cx),
        np.minimum(cy, H - cy)
    )
    
    # Pairwise distances
    # Using cdist is efficient for N=21
    dists = cdist(np.column_stack((cx, cy)), np.column_stack((cx, cy)))
    np.fill_diagonal(dists, np.inf)
    
    # Distance to nearest neighbor
    min_neighbor_dists = np.min(dists, axis=1)
    
    # Inradius is the minimum of wall distance and half the min neighbor distance
    inradii = np.minimum(wall_dists, min_neighbor_dists / 2.0)
    
    # Safety: if any inradius is negative or nan, return 0
    if np.any(np.isnan(inradii)) or np.any(inradii < 0):
        return np.zeros(n)
        
    return inradii

def _objective(z):
    """
    Objective function for SLSQP.
    Minimize negative sum of inradii.
    z = [x_1, ..., x_N, y_1, ..., y_N, W]
    H is derived as 2 - W.
    """
    cx = z[:N]
    cy = z[N:2*N]
    W = z[2*N]
    H = 2.0 - W
    
    # Basic validity checks to avoid crashing or huge penalties
    if W < 0.1 or H < 0.1:
        return 1e6
    if np.any(cx < 0) or np.any(cx > W) or np.any(cy < 0) or np.any(cy > H):
        # Penalty for going out of bounds (soft)
        # Actually, SLSQP handles bounds, but let's ensure we don't evaluate invalid states
        # that might cause issues in distance calcs, though cdist is safe.
        # The inradii calculation will naturally yield small/negative values if overlapping/out of bounds,
        # but we want to maximize sum, so negative inradii reduce the sum.
        # However, if a circle is outside, its "inradius" relative to the box is effectively negative?
        # No, the definition of inradius inside the box assumes it's inside.
        # If it's outside, the packing is invalid.
        # We rely on bounds in SLSQP to keep cx in [0, W] etc? 
        # Problem: W is variable, so bounds on cx depend on W.
        # SLSQP bounds are static. 
        # Strategy: Use a penalty method for bounds that depend on variables, 
        # OR restrict W to a range and use dynamic bounds? 
        # SLSQP does not support dynamic bounds easily in the standard interface.
        # Alternative: Reformulate.
        # Let's stick to standard SLSQP with static bounds for cx, cy in [0, 2] (max possible W/H is 2).
        # And add a penalty if cx > W or cx < 0 etc.
        
        pass
    
    inradii = _compute_inradii(cx, cy, W, H)
    
    # Penalty for circles outside the rectangle
    # If cx < 0 or cx > W, the circle is outside.
    # We want to maximize sum(inradii).
    # If a circle is outside, its contribution should be heavily penalized.
    # Let's define effective_inradius.
    # If circle is outside, effective_inradius = -large_penalty.
    
    out_of_bounds_penalty = 0.0
    for i in range(N):
        if cx[i] < 0 or cx[i] > W or cy[i] < 0 or cy[i] > H:
            # Distance outside
            d_out = 0
            if cx[i] < 0: d_out = min(d_out, -cx[i])
            if cx[i] > W: d_out = min(d_out, cx[i] - W)
            if cy[i] < 0: d_out = min(d_out, -cy[i])
            if cy[i] > H: d_out = min(d_out, cy[i] - H)
            out_of_bounds_penalty += 100.0 * d_out + 10.0
            
    # Also penalty for overlaps? 
    # The inradius calculation already handles overlaps by making inradius small/negative?
    # If dist(i,j) < 0, it's impossible. dist is always positive.
    # If circles overlap, dist(i,j) < r_i + r_j.
    # In our metric, r_i = min(wall, dist/2).
    # If they overlap, dist(i,j) is small, so r_i and r_j are small.
    # This naturally discourages overlap because it reduces the sum.
    # However, if two circles are on top of each other, dist=0, r=0.
    # This is a valid local minimum (bad one). We need to avoid this.
    # The initialization and multiple restarts should help.
    
    total_radius = np.sum(inradii)
    return -(total_radius - out_of_bounds_penalty)

def _get_initial_guesses(W_init, H_init, n_cols=None):
    """
    Generate initial positions based on a grid or hexagonal pattern.
    """
    if n_cols is None:
        # Estimate optimal cols
        # For N=21, 5x4 or 6x4 with one missing?
        # Let's try a few patterns
        n_cols = 5
    
    xs = []
    ys = []
    
    # Hexagonal packing initialization
    # Rows alternate offset
    n_rows = int(np.ceil(N / float(n_cols)))
    
    for row in range(n_rows):
        for col in range(n_cols):
            if len(xs) >= N:
                break
            # Center of cell
            x = (col + 0.5) * (W_init / n_cols)
            y = (row + 0.5) * (H_init / n_rows)
            
            # Add hexagonal offset
            if row % 2 == 1:
                x += (W_init / n_cols) * 0.5
            
            # Keep within bounds
            x = np.clip(x, 0.1, W_init - 0.1)
            y = np.clip(y, 0.1, H_init - 0.1)
            
            xs.append(x)
            ys.append(y)
            
    # If we didn't fill enough, add random points?
    # Or just repeat the last point? No, that causes overlap.
    # Let's ensure we have exactly N points.
    while len(xs) < N:
        xs.append(np.random.uniform(0.1, W_init - 0.1))
        ys.append(np.random.uniform(0.1, H_init - 0.1))
        
    return np.array(xs[:N]), np.array(ys[:N])

def _optimize(W_init, H_init, time_limit=10):
    """
    Run SLSQP optimization starting from a given aspect ratio.
    """
    start_time = time.time()
    
    # Try multiple initial patterns
    best_val = -1e9
    best_z = None
    
    patterns = [
        (5, 4), # 5 cols, 4 rows approx
        (4, 5),
        (6, 4),
        (3, 7),
    ]
    
    for n_cols, n_rows_hint in patterns:
        if time.time() - start_time > time_limit:
            break
            
        cx0, cy0 = _get_initial_guesses(W_init, H_init, n_cols)
        z0 = np.concatenate([cx0, cy0, [W_init]])
        
        # Bounds
        # cx, cy are in [0, 2] since W+H=2, max dim is 2.
        # W is in [0.5, 1.5] to keep aspect ratio reasonable (avoid extreme slivers)
        bounds = []
        for i in range(N):
            bounds.append((0, 2)) # x
        for i in range(N):
            bounds.append((0, 2)) # y
        bounds.append((0.5, 1.5)) # W
        
        try:
            res = minimize(
                _objective,
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_val:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_z = res.x
                    
        except Exception:
            continue
            
    return best_z, best_val

def _polish(z, time_limit=10):
    """
    High precision polishing using smaller steps and tighter tolerances.
    """
    start_time = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 2))
    for i in range(N):
        bounds.append((0, 2))
    bounds.append((0.5, 1.5))
    
    try:
        res = minimize(
            _objective,
            z,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        return res.x
    except Exception:
        return z

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_val = -1e9
    best_z = None
    
    # Try different initial aspect ratios
    # Square is 1x1 (W=1, H=1)
    # Wide is 1.5x0.5
    # Tall is 0.5x1.5
    initial_ws = [1.0, 1.2, 0.8, 1.4, 0.6]
    
    for W_init in initial_ws:
        if time.time() > deadline - 20:
            break
            
        H_init = 2.0 - W_init
        z, val = _optimize(W_init, H_init, time_limit=10)
        
        if val > best_val:
            best_val = val
            best_z = z
            
    if best_z is None:
        # Fallback: simple grid
        cx = np.linspace(0.2, 1.8, N)
        cy = np.linspace(0.2, 1.8, N)
        best_z = np.concatenate([cx, cy, [1.0]])
        best_val = _objective(best_z)
        
    # Polish
    if time.time() < deadline - 5:
        best_z = _polish(best_z, time_limit=10)
        best_val = -_objective(best_z)
        
    # Extract result
    cx = best_z[:N]
    cy = best_z[N:2*N]
    W = best_z[2*N]
    H = 2.0 - W
    
    # Compute final radii
    radii = _compute_inradii(cx, cy, W, H)
    
    # Safety check: ensure radii are positive and circles are inside
    # If any radius is <= 0, shrink them
    min_r = np.min(radii)
    if min_r <= 0:
        # This shouldn't happen if optimization worked, but just in case
        # Find max valid radius
        # For simplicity, just return a valid small packing
        # 5x4 grid of radius 0.1
        cx = []
        cy = []
        for i in range(N):
            row, col = divmod(i, 5)
            cx.append(0.2 + col * 0.4)
            cy.append(0.2 + row * 0.4)
        cx = np.array(cx)
        cy = np.array(cy)
        radii = np.full(N, 0.19)
        W = 2.0
        H = 0.0 # Wait, perimeter 4 means W+H=2. If W=2, H=0. Invalid.
        # Let's use a valid square
        W = 1.0
        H = 1.0
        # 5x4 grid in 1x1 square
        # Spacing 0.2. Radius 0.1.
        # Centers at 0.1, 0.3, 0.5, 0.7, 0.9
        cx = []
        cy = []
        for i in range(N):
            row, col = divmod(i, 5)
            cx.append(0.1 + col * 0.2)
            cy.append(0.1 + row * 0.2)
        cx = np.array(cx)
        cy = np.array(cy)
        radii = np.full(N, 0.09) # Slightly smaller to be safe
        
    # Final validation and shrink if necessary
    # Check if circles overlap or go out of bounds
    # We already computed radii as inradii, so they should be valid by definition
    # unless numerical errors occurred.
    
    # One final check
    # If any circle center is outside [r, W-r] or [r, H-r], shrink r
    for i in range(N):
        r_i = radii[i]
        if cx[i] < r_i or cx[i] > W - r_i or cy[i] < r_i or cy[i] > H - r_i:
            # Shrink this circle
            # New r = min(cx[i], W-cx[i], cy[i], H-cy[i])
            new_r = np.min([cx[i], W-cx[i], cy[i], H-cy[i]])
            radii[i] = max(0, new_r)
            
    # Check overlaps again with updated radii
    # If overlap, shrink all radii by a small factor
    # Simple global shrink
    for _ in range(10):
        valid = True
        dists = cdist(np.column_stack((cx, cy)), np.column_stack((cx, cy)))
        np.fill_diagonal(dists, np.inf)
        # Check if dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                if dists[i,j] < radii[i] + radii[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        radii *= 0.95
        
    return np.column_stack((cx, cy, radii))

if __name__ == "__main__":
    res = construct()
    print(res)
