"""Pack 50 equal circles in a unit square using Delaunay-based sparse optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _get_delaunay_edges(centers):
    """
    Get edges of the Delaunay triangulation.
    Returns a list of unique pairs (i, j).
    """
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to complete graph if Delaunay fails (e.g. collinear points)
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _init_hexagonal():
    """
    Create a high-quality hexagonal packing initialization.
    We want to fit 50 circles.
    Hexagonal rows: 6, 5, 6, 5, 6, 5, 6, 5, 6 -> 50? 
    6+5+6+5+6+5+6+5+6 = 50.
    Rows: 9 rows.
    Let's try to fit this into the square.
    """
    centers = []
    # Parameters
    # We want to maximize r.
    # Let's start with a normalized grid and let the optimizer scale/move it.
    
    # Standard hex grid:
    # Row 0: 6 points
    # Row 1: 5 points (offset)
    # ...
    
    rows = [6, 5, 6, 5, 6, 5, 6, 5, 6]
    total = sum(rows)
    if total != N:
        # Adjust to make 50
        # 7, 6, 7, 6, 7, 6, 7, 4 -> 50?
        # 7+6+7+6+7+6+7+4 = 50
        # Or just use 5x10 grid as fallback if hex fails?
        # Let's stick to the 6-5 pattern, it's denser.
        # If sum != 50, we pad or trim.
        # For now, let's force the generation to exactly N.
        pass

    # Let's generate a hex grid that spans [-0.5, 0.5]
    # We'll define the grid in a local coordinate system and then scale/center it.
    
    # Let's try a specific configuration known to be good:
    # 8 rows of roughly 6 or 7.
    # 6, 7, 6, 7, 6, 7, 6, 5 -> 40? No.
    
    # Let's just generate a hex grid with enough points and take the first N.
    # Spacing 1.0
    pts = []
    dy = np.sqrt(3) / 2
    for row in range(10):
        for col in range(10):
            x = col + (0.5 if row % 2 else 0)
            y = row * dy
            pts.append([x, y])
    pts = np.array(pts)
    
    # Center and scale to fit in [-0.45, 0.45] roughly
    # We want to fill the square.
    # The hex grid aspect ratio is sqrt(3) ~ 1.73.
    # Square is 1x1.
    # So we should scale x more than y? Or just fit bounding box.
    
    # Let's try to fit the 50 points into the square optimally.
    # Select 50 points from a larger hex grid that best approximates a square shape.
    # Actually, a rectangular grid 7x8 or 8x7 is a safer bet for initialization,
    # but hex is better for local search.
    
    # Let's generate a hex grid, then keep only points inside a square region
    # that maximizes density.
    
    # Simpler: Just use a 7x8 grid, but perturb it slightly into a hex pattern?
    # No, let's use the 6-5-6... pattern explicitly.
    
    centers = []
    # Width of square is 1.
    # If we have 6 points in a row, spacing is 1/6? No, we have margins.
    # Let's assume r is roughly 1/12 = 0.083.
    # Diameter 0.166.
    # 6 points * 0.166 = 1.0. Fits perfectly if touching walls.
    # So x coords: -0.5 + r, -0.5 + 3r, ...
    
    # Let's just create a grid of 50 points in a hex arrangement.
    # Row 0: 6 pts
    # Row 1: 5 pts
    # Row 2: 6 pts
    # Row 3: 5 pts
    # Row 4: 6 pts
    # Row 5: 5 pts
    # Row 6: 6 pts
    # Row 7: 5 pts
    # Row 8: 6 pts
    # Total 50.
    
    # We need to determine the spacing dx and dy.
    # For hex packing, dx = 2r, dy = r*sqrt(3).
    # Total width = 5*dx + 2r (for 6 points) = 12r? 
    # If 6 points: x_0 ... x_5. Distance x_5 - x_0 = 5 * 2r = 10r.
    # Plus margins r on each side: 10r + 2r = 12r = 1 => r = 1/12 = 0.0833.
    # This is the theoretical limit for 6 columns.
    # Height: 9 rows.
    # y_0 ... y_8. Distance y_8 - y_0 = 8 * r*sqrt(3).
    # Plus margins: 8*r*sqrt(3) + 2r = 2r(4*sqrt(3) + 1) = 2*(1/12)*(6.928+1) = 1/6 * 7.928 = 1.32.
    # This is too tall for a unit square (height 1).
    # So we cannot have 9 rows of hex packing with r=1/12.
    # We must compress vertically or use fewer rows with more columns?
    # If we use 5 rows:
    # 10, 10, 10, 10, 10 -> 50.
    # Width: 10 cols. 9*2r + 2r = 20r = 1 => r = 0.05. Too small.
    
    # So a mix is needed.
    # The optimal packing for 50 is likely irregular.
    # Let's initialize with a rectangular grid 7x8 (56 pts, take 50) or 8x7.
    # 7x8 grid:
    # dx = 1/7, dy = 1/8.
    # r_wall_x = 0.5 - 0.5/7 = 3/7 ~ 0.42? No.
    # Center at 0.
    # x: -3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7.
    # Max |x| = 3/7. Wall dist = 0.5 - 3/7 = (3.5-3)/7 = 0.5/7 = 1/14 ~ 0.0714.
    # Pair dist x = 1/7 ~ 0.1428. r_pair = 1/14 ~ 0.0714.
    # y: -3/8 ... 3/8.
    # Max |y| = 3/8. Wall dist = 0.5 - 3/8 = 1/8 = 0.125.
    # Pair dist y = 1/8 = 0.125. r_pair = 1/16 = 0.0625.
    # So limiting factor is y. r = 0.0625.
    
    # 8x7 grid:
    # x: 8 cols. Spacing 1/8. Max |x| = 3.5/8 = 7/16. Wall = 0.5 - 7/16 = 1/16 = 0.0625.
    # y: 7 rows. Spacing 1/7. Max |y| = 3/7. Wall = 0.5 - 3/7 = 1/14 ~ 0.0714.
    # Pair x: 1/8 -> r=1/16=0.0625.
    # Pair y: 1/7 -> r=1/14=0.0714.
    # Limiting factor x. r = 0.0625.
    
    # So grid is ~0.0625. Best known is 0.0713.
    # We need a better init.
    # Let's use the 6-5 hex pattern but compress it to fit height 1.
    # Rows: 6, 5, 6, 5, 6, 5, 6, 5, 6.
    # Let's set the y-coordinates to be evenly spaced to fit height 1.
    # 9 rows. 8 gaps.
    # Let dy = 1/8 = 0.125.
    # y_0 = -0.5 + r, y_8 = 0.5 - r.
    # If r is small, dy is approx 1/8.
    # For hex packing, dy should be dx * sqrt(3)/2.
    # If dy = 0.125, then dx = 2 * 0.125 / sqrt(3) = 0.25 / 1.732 = 0.1443.
    # Check width for 6 points:
    # 5 gaps of dx. Total width = 5 * 0.1443 = 0.7215.
    # Plus margins 2r.
    # If r = dx/2 = 0.07215.
    # Total width = 0.7215 + 2*0.07215 = 0.8658.
    # This fits in width 1 with room to spare.
    # So we can expand dx to use the full width?
    # If we expand dx, we increase r.
    # Max width is 1.
    # 5*dx + 2r = 1. Since r=dx/2, 5dx + dx = 6dx = 1 => dx = 1/6 = 0.1666.
    # Then r = 1/12 = 0.0833.
    # But we are constrained by height.
    # Height = 8*dy + 2r = 1.
    # Hex constraint: dy = dx * sqrt(3)/2.
    # 8 * (dx * sqrt(3)/2) + 2 * (dx/2) = 1
    # 4*sqrt(3)*dx + dx = 1
    # dx * (4*sqrt(3) + 1) = 1
    # dx = 1 / (4*1.732 + 1) = 1 / (6.928 + 1) = 1 / 7.928 = 0.1261.
    # r = dx/2 = 0.06305.
    # This is worse than the grid (0.0625)? No, 0.063 > 0.0625.
    # Wait, grid was 0.0625. This hex is 0.0630.
    # But this assumes perfect hex geometry.
    # The 6-5 pattern has different widths for even/odd rows.
    # Row 6: width 5dx.
    # Row 5: width 4dx.
    # If we center them, the "envelope" width is 5dx + 2r.
    # So the calculation holds for the max width.
    
    # Let's construct this specific hex init.
    dx = 1.0 / (4*np.sqrt(3) + 1)
    r = dx / 2.0
    dy = dx * np.sqrt(3) / 2.0
    
    # Verify height
    h = 8*dy + 2*r
    # h should be 1.
    
    centers = []
    # Rows: 6, 5, 6, 5, 6, 5, 6, 5, 6
    counts = [6, 5, 6, 5, 6, 5, 6, 5, 6]
    for i_row, count in enumerate(counts):
        y = -0.5 + r + i_row * dy
        if i_row % 2 == 0:
            # 6 points
            x_start = -0.5 + r
            for i_col in range(count):
                x = x_start + i_col * dx
                centers.append([x, y])
        else:
            # 5 points, offset by dx/2
            x_start = -0.5 + r + dx/2.0
            for i_col in range(count):
                x = x_start + i_col * dx
                centers.append([x, y])
                
    return np.array(centers)

def _init_grid():
    """7x8 grid, take first 50."""
    # 7 cols, 8 rows = 56.
    # We want 50.
    # Let's just make a 7x8 grid and remove 6 corners?
    # Or just use 8x7 and remove 6?
    # Let's use a standard grid generation and take the center 50.
    xs = (np.arange(7) + 0.5) / 7.0 - 0.5
    ys = (np.arange(8) + 0.5) / 8.0 - 0.5
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:50]

def _slsqp_polish(centers, time_limit=10.0):
    """
    Use SLSQP to maximize the minimum radius.
    We formulate this as:
    Maximize r
    Subject to:
    |c_i - c_j| >= 2r  for all i<j
    0.5 - |c_ik| >= r  for all i,k
    
    This is a non-linear program. SLSQP can handle it.
    To make it sparse, we only enforce constraints for Delaunay neighbors
    and walls. Non-Delaunay pairs are likely not active constraints
    (distance > 2r) in a good packing.
    """
    x0 = centers.ravel()
    
    def objective(x):
        c = x.reshape(N, 2)
        # We want to maximize r.
        # SLSQP minimizes. So minimize -r.
        # But r is not in x.
        # We can treat r as a variable?
        # x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
        # Size N*2 + 1.
        
        c = x[:N*2].reshape(N, 2)
        r = x[N*2]
        
        # Check validity
        if not np.all(np.isfinite(c)):
            return 1e6
        if r < 0:
            return 1e6
            
        # Calculate actual max radius for this configuration
        # If r is larger than the feasible max, constraints will be violated.
        # We want to push r as high as possible.
        
        # For the objective, we can just return -r.
        # The constraints will ensure r is feasible.
        return -r

    # Constraints
    # We need to define constraints dynamically based on the current Delaunay graph?
    # SLSQP requires a fixed number of constraints.
    # We can pre-allocate a large number of constraints and disable inactive ones?
    # Or, we can use a "soft" constraint approach?
    # No, SLSQP is active-set.
    
    # Strategy:
    # 1. Compute Delaunay graph for current centers.
    # 2. Define constraints for these edges + walls.
    # 3. Run SLSQP.
    # 4. If SLSQP converges, check if new Delaunay graph is different.
    # 5. If yes, re-run with new graph.
    
    # Since SLSQP setup is expensive, we do this in a loop.
    
    current_centers = centers.copy()
    current_r = _radius(current_centers)
    
    start_time = time.time()
    
    # Max iterations of the outer loop (graph updates)
    max_outer = 20
    
    for outer_iter in range(max_outer):
        if time.time() - start_time > time_limit:
            break
            
        # Get active edges
        edges = _get_delaunay_edges(current_centers)
        
        # Number of constraints:
        # Pairwise: len(edges)
        # Wall: 4 * N
        n_pairs = len(edges)
        n_walls = 4 * N
        n_cons = n_pairs + n_walls
        
        # We need to map indices.
        # x vector: [c_1x, c_1y, ..., c_Nx, c_Ny, r]
        
        # Define constraint functions
        # SLSQP expects a list of dicts: {'type': 'ineq', 'fun': lambda x: ...}
        # fun(x) >= 0
        
        cons = []
        
        # Pairwise constraints: |c_i - c_j| - 2r >= 0
        # To be differentiable, we can use squared distance?
        # (c_i - c_j)^2 - (2r)^2 >= 0 is not correct for sign if dist < 2r.
        # dist - 2r >= 0 is non-smooth at dist=2r? No, dist is smooth except at c_i=c_j.
        # SLSQP can handle non-smoothness somewhat, but it's better to use
        # (dist^2 - (2r)^2) / (dist + 2r) >= 0 ?
        # Actually, just using dist - 2r is fine for SLSQP usually.
        
        for idx, (i, j) in enumerate(edges):
            def cons_pair(x, i=i, j=j):
                c = x[:N*2].reshape(N, 2)
                r = x[N*2]
                d = np.linalg.norm(c[i] - c[j])
                return d - 2*r
            cons.append({'type': 'ineq', 'fun': cons_pair})
            
        # Wall constraints:
        # 0.5 - |c_ix| - r >= 0
        # 0.5 - |c_iy| - r >= 0
        
        for i in range(N):
            # x > 0
            def cons_wall_x_pos(x, i=i):
                c = x[:N*2].reshape(N, 2)
                r = x[N*2]
                return 0.5 - c[i, 0] - r
            cons.append({'type': 'ineq', 'fun': cons_wall_x_pos})
            
            # x < 0
            def cons_wall_x_neg(x, i=i):
                c = x[:N*2].reshape(N, 2)
                r = x[N*2]
                return 0.5 + c[i, 0] - r
            cons.append({'type': 'ineq', 'fun': cons_wall_x_neg})
            
            # y > 0
            def cons_wall_y_pos(x, i=i):
                c = x[:N*2].reshape(N, 2)
                r = x[N*2]
                return 0.5 - c[i, 1] - r
            cons.append({'type': 'ineq', 'fun': cons_wall_y_pos})
            
            # y < 0
            def cons_wall_y_neg(x, i=i):
                c = x[:N*2].reshape(N, 2)
                r = x[N*2]
                return 0.5 + c[i, 1] - r
            cons.append({'type': 'ineq', 'fun': cons_wall_y_neg})
            
        # Initial guess for x
        x_init = np.zeros(N*2 + 1)
        x_init[:N*2] = current_centers.ravel()
        x_init[N*2] = current_r * 0.99 # Start slightly below current r to be feasible?
        # Or start at current_r.
        x_init[N*2] = current_r
        
        # Bounds
        # Centers in [-0.5, 0.5]
        # r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((-0.5, 0.5))
            bounds.append((-0.5, 0.5))
        bounds.append((0.0, 0.5))
        
        try:
            res = minimize(
                objective,
                x_init,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success:
                new_centers = res.x[:N*2].reshape(N, 2)
                new_r = res.x[N*2]
                
                # Verify new_r is actually feasible
                actual_r = _radius(new_centers)
                if actual_r >= current_r - 1e-8:
                    current_centers = new_centers
                    current_r = actual_r
                else:
                    # SLSQP might have returned an infeasible point or local min
                    # Keep old
                    pass
            else:
                # If SLSQP fails, try to use the last iterated point if it's better
                # res.x might still be useful
                if res.x is not None:
                    new_centers = res.x[:N*2].reshape(N, 2)
                    actual_r = _radius(new_centers)
                    if actual_r > current_r:
                        current_centers = new_centers
                        current_r = actual_r
                        
        except Exception:
            pass
            
        # Check if graph changed
        # If graph is stable, we are likely at a local optimum for this topology.
        # We could break, but let's allow a few more iterations in case of numerical noise.
        # Actually, if r didn't improve, break.
        # We don't track old_r easily in the loop without a variable.
        # Let's assume if SLSQP ran, it tried to improve.
        
    return current_centers, current_r

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # 1. Hexagonal Init
    c_hex = _init_hexagonal()
    r_hex = _radius(c_hex)
    
    if r_hex > best_r:
        best_r = r_hex
        best_centers = c_hex.copy()
    
    # Polish hex
    if time.time() - start_time < DEADLINE - 10:
        c_pol, r_pol = _slsqp_polish(c_hex, time_limit=min(20.0, DEADLINE - 10 - (time.time() - start_time)))
        if r_pol > best_r:
            best_r = r_pol
            best_centers = c_pol.copy()
            
    # 2. Grid Init
    c_grid = _init_grid()
    r_grid = _radius(c_grid)
    
    if r_grid > best_r:
        best_r = r_grid
        best_centers = c_grid.copy()
        
    # Polish grid
    if time.time() - start_time < DEADLINE - 10:
        c_pol, r_pol = _slsqp_polish(c_grid, time_limit=min(20.0, DEADLINE - 10 - (time.time() - start_time)))
        if r_pol > best_r:
            best_r = r_pol
            best_centers = c_pol.copy()
            
    # 3. Random Inits + Polish
    # Use remaining time for random restarts
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        
        # Generate a random perturbation of the best known or a random start?
        # Random start is better for diversity.
        # But random uniform is usually bad (r ~ 0.02).
        # Let's perturb the best current solution.
        if best_centers is not None:
            c_rand = best_centers + np.random.normal(0, 0.01, size=(N, 2))
            # Clip to square
            c_rand = np.clip(c_rand, -0.49, 0.49)
        else:
            c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
            
        r_rand = _radius(c_rand)
        if r_rand > best_r:
            best_r = r_rand
            best_centers = c_rand.copy()
            
        # Short polish
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left > 2:
            c_pol, r_pol = _slsqp_polish(c_rand, time_limit=min(5.0, time_left))
            if r_pol > best_r:
                best_r = r_pol
                best_centers = c_pol.copy()
                
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(f"Radius: {_radius(centers)}")
