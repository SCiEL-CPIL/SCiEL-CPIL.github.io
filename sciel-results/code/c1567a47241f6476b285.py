# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _union_volume_exact(x_positions, y_positions, N=8, n_samples=20000):
    """
    Compute the exact union volume of the 3D sheared tubes.
    Since the tubes are axis-aligned boxes at any fixed height c, the cross-section
    area A(c) is the area of the union of N^2 axis-aligned squares in the xy-plane.
    The volume is integral_0^1 A(c) dc.
    
    We estimate this integral using deterministic quadrature (Simpson's rule or similar)
    on a grid of c values, and for each c, compute the 2D union area exactly via 
    coordinate compression (sweep line).
    """
    n = N * N
    i_grid = np.arange(n) // N
    j_grid = np.arange(n) % N
    
    # We will sample c values
    c_samples = np.linspace(0, 1, n_samples)
    
    total_area = 0.0
    # Use trapezoidal rule for integration
    # Actually, since the area function is piecewise linear in c (vertices move linearly),
    # we can just use a fine grid and trapezoidal rule.
    
    # To speed up 2D union area calculation:
    # For a given c, the x-interval for tube k is [x_k + c*i_k/N, x_k + 1/N + c*i_k/N]
    # The y-interval for tube k is [y_k + c*j_k/N, y_k + 1/N + c*j_k/N]
    
    # We collect all unique x-coordinates and y-coordinates to build a grid.
    # However, doing this for every c is slow.
    # Alternative: Monte Carlo for the integral? No, we want exactness for the objective.
    # But the objective function in DE is called many times.
    # Let's optimize the 2D union area calculation.
    
    # For a set of axis-aligned rectangles, the union area can be computed by 
    # sorting x-coords and y-coords and using a sweep line or simple grid summation.
    # With 64 rectangles, there are at most 128 unique x coords and 128 unique y coords.
    # Grid size 128x128 = 16384 cells. This is fast enough in NumPy if vectorized.
    
    areas = np.zeros(len(c_samples))
    
    for idx, c in enumerate(c_samples):
        # Compute x and y intervals
        x_starts = x_positions + c * i_grid / N
        x_ends = x_starts + 1.0 / N
        y_starts = y_positions + c * j_grid / N
        y_ends = y_starts + 1.0 / N
        
        # Get unique coordinates
        x_coords = np.unique(np.concatenate([x_starts, x_ends]))
        y_coords = np.unique(np.concatenate([y_starts, y_ends]))
        
        # If the number of unique coords is small, we can build a grid
        # But np.unique is somewhat slow. 
        # Alternative: Just use a bounding box and a fixed fine grid for the 2D area?
        # No, that introduces error.
        
        # Let's try a vectorized approach for the 2D union area using the "marking" method on the unique coords.
        # This is still O(N^3) roughly per c.
        
        # Faster approach for 2D union area of 64 rects:
        # Use the formula: Area = sum over cells in the grid formed by all x and y coords.
        # Cell (ix, iy) is covered if any rect covers it.
        
        # To make this faster, we can pre-compute the structure? No, positions change.
        
        # Let's try to be clever. 
        # The 2D union area of axis-aligned rectangles can be computed by:
        # 1. Sort x coords.
        # 2. For each strip between x[i] and x[i+1], determine which rectangles cover this strip.
        # 3. The union of y-intervals of those rectangles gives the height.
        # 4. Area += width * height.
        
        # This is O(N^2 log N) per c. With 20000 c samples, that's 20000 * 64^2 ~ 80M ops.
        # This might be too slow for DE which calls objective ~1000 times. 80B ops. Too slow.
        
        # We need a faster objective.
        # Option 1: Reduce c_samples to ~100. Trapezoidal rule on 100 points for a piecewise linear function is exact?
        # The area A(c) is piecewise linear?
        # The vertices of the squares move linearly with c.
        # The union area of moving rectangles is a continuous function. Is it piecewise linear?
        # The area of a single rectangle is constant (1/64).
        # The area of the union is sum of areas - sum of intersections + ...
        # Intersections of two rectangles moving linearly:
        # Width of intersection = max(0, min(x1_end, x2_end) - max(x1_start, x2_start)).
        # min/max of linear functions is piecewise linear.
        # So the intersection area is piecewise linear.
        # Thus A(c) is piecewise linear.
        # The breakpoints occur when x1_start = x2_end or x1_end = x2_start etc.
        # There are O(N^2) such breakpoints in c.
        # So if we sample at all breakpoints, we can integrate exactly.
        # Number of breakpoints: For each pair of rectangles, there are 4 conditions in x and 4 in y.
        # Total O(N^4) breakpoints? No, O(N^2) pairs, each contributes a few breakpoints.
        # So ~ 64^2 * constant = ~20000 breakpoints.
        # Sampling 20000 points is feasible for ONE evaluation, but not for 1000 DE iterations.
        
        # Option 2: Use a coarse Monte Carlo for the objective, but refine at the end?
        # DE is stochastic, so a noisy objective might be okay if the noise is low.
        # Or use a very small deterministic grid, e.g., 20 points.
        # If A(c) is piecewise linear with ~20000 segments, 20 points is very inaccurate.
        
        # Option 3: Analytical 1D optimization first, then local 2D?
        # The prompt says "DIFFERENT strategy... not refining".
        # The inspirations used 1D optimization.
        # My strategy is 2D DE.
        
        # Let's optimize the 2D union area calculation using NumPy vectorization over the rectangles.
        # For a fixed c:
        # We have 64 rects.
        # We can compute the union area by checking coverage on a grid of the unique coords.
        # But we need to do this fast.
        
        # Let's try a different approach: 
        # Approximate the volume using a fixed set of c values (e.g., 10) and a fixed grid for xy?
        # No, the box bounds change.
        
        # Let's go with the sweep line for 2D area, but implemented efficiently.
        # Or, better yet:
        # The volume is the measure of the set of points (x,y,c).
        # We can just do a 3D Monte Carlo for the objective?
        # 5000 points in 3D for objective? 
        # DE calls objective ~1000 times. 5M points total. Very fast.
        # But noisy.
        # DE can handle noisy objectives if we use a large population or careful settings.
        # Let's try 3D Monte Carlo with a fixed seed for consistency within an objective call?
        # No, random points should be fixed for the whole optimization to be consistent?
        # If we use fixed points, it's a deterministic objective (fixed sample).
        # Let's generate 10,000 random points in a large bounding box?
        # But the bounding box depends on the positions.
        # We can normalize the coordinates?
        # The score is 1/V. V depends on the box size.
        # If we scale the positions, V scales with area*height.
        # The problem is translationally invariant? No, the box is defined by min/max.
        # But relative positions matter.
        
        # Let's stick to the exact 2D sweep line but optimize it.
        # Actually, 20000 c samples * 64 rects is too slow in Python loops.
        # We need to vectorize the c dimension.
        
        # For each pair of rectangles (k1, k2), the intersection area is non-zero only in a range of c.
        # This is getting complicated.
        
        # Let's try a hybrid: 
        # 1. Use 1D Keich as initial guess.
        # 2. Use `scipy.optimize.minimize` with L-BFGS-B on the 128 variables, using a finite difference gradient of the 3D Monte Carlo volume (fixed points).
        # 3. Or use DE with a fast approximate objective.
        
        # Let's use a fixed set of 3D points for the Monte Carlo objective.
        # We need to handle the changing bounding box.
        # We can map the points to [0,1]^3 and then scale to the current box?
        # No, the points are uniform in the box.
        # If we fix the points in [0,1]^3, and the box is [X0,X1]x[Y0,Y1]x[0,1],
        # then a point p in [0,1]^3 maps to X = X0 + p[0]*(X1-X0), etc.
        # The condition "inside tube" is linear in X, Y, C.
        # So we can pre-compute the tube parameters and check the fixed points.
        
        # This is a great strategy!
        # 1. Generate M fixed random points in [0,1]^3 once.
        # 2. For a given set of positions, compute X0, X1, Y0, Y1.
        # 3. Map the fixed points to the current box.
        # 4. Check if each point is inside any tube.
        # 5. V = (count/M) * (X1-X0)*(Y1-Y0)*1.
        
        # This is O(M * N^2) per objective call.
        # M=10000, N^2=64. 640k ops per call.
        # 1000 calls = 640M ops. In NumPy, this is fast (<1s).
        
        pass # Placeholder for logic below

def _make_objective(fixed_points, N=8):
    """
    Create an objective function that uses a fixed set of 3D points in [0,1]^3
    to estimate the volume of the union of tubes.
    """
    M = len(fixed_points)
    
    def objective(params):
        # params: shape (128,) -> 64 x positions, 64 y positions
        x_pos = params[:64]
        y_pos = params[64:]
        
        # Compute bounding box
        i_grid = np.arange(64) // N
        j_grid = np.arange(64) % N
        
        # X range for tube k: [x_k + c*i_k/N, x_k + 1/N + c*i_k/N] for c in [0,1]
        # Min X for tube k is x_k (at c=0) if i_k >= 0. Max is x_k + 1/N + i_k/N (at c=1).
        # Actually, min is min(x_k, x_k + i_k/N) and max is max(x_k + 1/N, x_k + 1/N + i_k/N).
        # Since i_k >= 0, min is x_k, max is x_k + (1+i_k)/N.
        x_min = np.min(x_pos)
        x_max = np.max(x_pos + (1 + i_grid) / N)
        
        y_min = np.min(y_pos)
        y_max = np.max(y_pos + (1 + j_grid) / N)
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        if box_w < 1e-9 or box_h < 1e-9:
            return 1e10 # Invalid
            
        # Map fixed points [0,1]^3 to the box
        # P = (a, b, c) in [0,1]^3
        # X = x_min + a * box_w
        # Y = y_min + b * box_h
        # C = c
        
        a = fixed_points[:, 0]
        b = fixed_points[:, 1]
        c = fixed_points[:, 2]
        
        X = x_min + a * box_w
        Y = y_min + b * box_h
        C = c
        
        # Check if (X, Y, C) is inside any tube
        # Tube k is defined by:
        # X in [x_k + C*i_k/N, x_k + 1/N + C*i_k/N]
        # Y in [y_k + C*j_k/N, y_k + 1/N + C*j_k/N]
        
        # Vectorized check
        # We have M points and 64 tubes.
        # X: (M,), x_pos: (64,)
        # We want to check for each point if there exists a tube k such that...
        
        # This creates a M x 64 matrix. 10000 x 64 = 640k floats. OK.
        
        # X in [x_k + C*i_k/N, x_k + 1/N + C*i_k/N]
        # X - x_k - C*i_k/N >= 0
        # X - x_k - C*i_k/N <= 1/N
        
        # Let's expand:
        # X[:, None] - x_pos[None, :] - C[:, None] * (i_grid[None, :] / N)
        
        x_diff = X[:, None] - x_pos[None, :] - C[:, None] * (i_grid[None, :] / N)
        x_ok = (x_diff >= 0) & (x_diff <= 1.0 / N)
        
        y_diff = Y[:, None] - y_pos[None, :] - C[:, None] * (j_grid[None, :] / N)
        y_ok = (y_diff >= 0) & (y_diff <= 1.0 / N)
        
        inside = x_ok & y_ok
        
        # If any tube contains the point
        hit = np.any(inside, axis=1)
        
        fraction = np.sum(hit) / M
        
        V = fraction * box_w * box_h
        
        return V

    return objective

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    timeout = 600 # seconds
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    init_params = np.concatenate([base_x, base_y])
    
    # Generate fixed points for objective
    rng = np.random.default_rng(42)
    M = 20000
    fixed_points = rng.uniform(0, 1, (M, 3))
    
    objective = _make_objective(fixed_points, N)
    
    # Define bounds for DE
    # The positions can be anywhere, but relative positions matter.
    # We can constrain them to a reasonable range, e.g., [-1, 2]
    bounds = [(-1.0, 2.0)] * 128
    
    # Run DE
    # popsize=10, maxiter=100, tol=0.01
    # We need to be careful with time.
    
    best_params = init_params
    best_val = objective(init_params)
    
    # Use a callback to check time
    def callback(xk, convergence=True):
        nonlocal best_params, best_val
        val = objective(xk)
        if val < best_val:
            best_val = val
            best_params = xk
        if time.time() - start_time > timeout - 30:
            raise KeyboardInterrupt
        
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            init=init_params,
            maxiter=100, 
            popsize=15, 
            tol=0.01,
            callback=callback,
            polish=False, # Polish is slow and might not help much
            seed=42
        )
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except KeyboardInterrupt:
        pass
    
    # If we have time, try a local refinement with L-BFGS-B
    if time.time() - start_time < timeout - 60:
        try:
            from scipy.optimize import minimize
            # Use the best DE result as start
            local_res = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if local_res.fun < best_val:
                best_params = local_res.x
                best_val = local_res.fun
        except Exception:
            pass
            
    # Extract positions
    x_pos = best_params[:64]
    y_pos = best_params[64:]
    positions = np.stack([x_pos, y_pos], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
