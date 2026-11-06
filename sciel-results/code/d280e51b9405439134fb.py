"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _neg_radius(centers):
    """Negative radius for minimization."""
    return -_radius(centers)


def _bcc_initialization(n_points):
    """
    Generate initial positions on a BCC-like lattice.
    We generate points in a grid and add the body-center points,
    then select the first n_points.
    """
    # We want to pack roughly sqrt(50) ~ 7 per side if cubic, but BCC is denser.
    # Let's try a 4x4x4 BCC superlattice which has 4^3 * 2 = 128 points.
    # Or a 3x3x3 BCC which has 3^3 * 2 = 54 points. This is very close to 50.
    
    m = 3
    # Generate corner points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    corners = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Generate body center points
    # The body center of a cell [x, x+dx] x [y, y+dy] x [z, z+dz] is at x+dx/2, etc.
    # Our ticks are centers of cells of size 1/m.
    # So the cell for tick index i is [ticks[i]-0.5/m, ticks[i]+0.5/m].
    # The body center is at ticks[i] + 0.5/m? No.
    # The grid points are at ticks. The cell boundaries are ticks +/- 0.5/m.
    # The center of the cell containing ticks[i] is ticks[i].
    # Wait, the BCC lattice points are at corners and centers.
    # If we have a simple cubic grid with spacing s, the BCC points are at (i,j,k)s and (i+0.5, j+0.5, k+0.5)s.
    # Our ticks are centered at 0. Let's just generate a dense grid and pick.
    
    # Simpler approach: Generate a grid of points that looks like BCC.
    # Spacing s = 1/m.
    # Points at (i, j, k) * s for i,j,k in 0..m-1 (shifted to center)
    # And points at (i+0.5, j+0.5, k+0.5) * s
    
    s = 1.0 / m
    points = []
    for k in range(m):
        for j in range(m):
            for i in range(m):
                # Corner
                points.append(((i + 0.5) * s - 0.5, (j + 0.5) * s - 0.5, (k + 0.5) * s - 0.5))
                # Body Center
                if i < m-1 and j < m-1 and k < m-1:
                     # This logic is tricky. Let's just use a dense random grid or specific BCC coords.
                     pass
    
    # Let's use a standard BCC generation
    # Unit cell size a = 1/m.
    # Points at (0,0,0) + n*a and (0.5, 0.5, 0.5)*a + n*a
    # We need to fit them in [-0.5, 0.5].
    
    pts = []
    for k in range(m):
        for j in range(m):
            for i in range(m):
                # Corner point
                x = (i + 0.5) * s - 0.5
                y = (j + 0.5) * s - 0.5
                z = (k + 0.5) * s - 0.5
                pts.append([x, y, z])
                
                # Body center point
                # The body center of the cell [x-s/2, x+s/2] ... is at x,y,z? No.
                # The cell is centered at x,y,z. The corners of the cell are at x +/- s/2.
                # The BCC point is at the center of the cell? No, that's simple cubic.
                # BCC has points at corners of the cube and one in the center.
                # If we tile the cube with m^3 cells of size s, the "corners" of these small cells are the grid points.
                # The "center" of each small cell is an additional point.
                # So yes, the point (x, y, z) is the center of the cell.
                # Wait, if I put a point at the center of every cell, that's just a simple cubic grid shifted?
                # No.
                # Simple cubic: points at (i,j,k)*s.
                # BCC: points at (i,j,k)*s AND (i+0.5, j+0.5, k+0.5)*s.
                
                # My grid ticks are (i+0.5)*s - 0.5.
                # So the "simple cubic" points are at ticks.
                # The "body center" points should be at the midpoints between adjacent ticks?
                # No, the BCC point is at the center of the unit cell defined by 8 simple cubic points.
                # If simple cubic points are at x_i, the unit cell spans [x_i, x_{i+1}].
                # The center is (x_i + x_{i+1}) / 2.
                
                if i < m-1 and j < m-1 and k < m-1:
                    # Midpoint in x, y, z
                    mx = (x + ((i + 1.5) * s - 0.5)) / 2.0
                    my = (y + ((j + 1.5) * s - 0.5)) / 2.0
                    mz = (z + ((k + 1.5) * s - 0.5)) / 2.0
                    pts.append([mx, my, mz])
                    
    pts = np.array(pts)
    # Add some random jitter to break symmetry
    pts = pts + np.random.normal(0, 0.01, pts.shape)
    pts = np.clip(pts, -0.5, 0.5)
    
    # Select n_points
    # To ensure good spread, we can just take the first n_points
    # But the order is deterministic. Let's shuffle to be safe?
    # No, for reproducibility in this context, just taking a subset is fine.
    # However, taking just the first 50 from a 54-point BCC might bias towards one corner.
    # Let's pick 50 points uniformly at random from the generated set.
    if len(pts) > n_points:
        indices = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[indices]
        
    return pts


def _optimize_single(centers, time_budget):
    """Optimize a single configuration using L-BFGS-B."""
    start = time.time()
    
    # Flatten centers
    x0 = centers.flatten()
    
    # Define objective and gradient
    def objective(x):
        c = x.reshape(-1, 3)
        # Check bounds
        if np.any(np.abs(c) > 0.5):
            return 1e6
        return _neg_radius(c)
        
    # Gradient approximation or analytical?
    # L-BFGS-B can use numerical gradient, but it's slow.
    # Let's try to compute a numerical gradient efficiently or just use BFGS with finite differences.
    # Given time constraints, let's use 'L-BFGS-B' with default finite difference gradient.
    
    # Constraints:
    # We want to maximize min(dist/2, wall_dist).
    # This is non-smooth.
    # A better approach: Maximize r such that all constraints are satisfied.
    # But L-BFGS-B minimizes a scalar function.
    # Using -min(...) is non-differentiable.
    
    # Alternative: Use a smooth approximation or just rely on the optimizer to handle it somewhat.
    # Or, use a penalty method.
    # Let's stick to the current simple approach but maybe use a better initial guess.
    
    # Actually, the current SA approach is quite robust.
    # The insight suggests BCC initialization.
    # Let's combine BCC init with SA or L-BFGS-B.
    
    # Let's try L-BFGS-B on the -radius function.
    # It might get stuck at non-differentiable points.
    # A common trick is to maximize the sum of log(distances) or similar, but that changes the problem.
    
    # Let's stick to the robust SA but start from BCC.
    # And add a final L-BFGS-B polish if time permits.
    
    # For now, just return the input if no specific optimizer is implemented here.
    # We will call SA in the main construct.
    
    return centers


def _sa_optimize(centers, time_budget):
    """Simulated Annealing optimization."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Adaptive step size
    T = 0.005
    T_min = 1e-6
    step_size = 0.02
    decay = 0.9998
    
    while time.time() - start < time_budget:
        # Pick a random point
        i = np.random.randint(n)
        
        # Perturb
        delta = np.random.normal(0, step_size, 3)
        new_center = centers[i].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        new_centers = centers.copy()
        new_centers[i] = new_center
        new_r = _radius(new_centers)
        
        # Acceptance criterion
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            if np.random.rand() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
        
        T *= decay
        step_size *= decay
        if T < T_min:
            T = T_min
            step_size = max(step_size, 0.001)
            
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Phase 1: BCC Initialization
    centers_bcc = _bcc_initialization(N)
    
    # Also try a simple grid for comparison
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers_grid = grid[:N].copy()
    
    # Also try random initialization
    centers_rand = np.random.uniform(-0.5, 0.5, (N, 3))
    
    best_centers = centers_bcc.copy()
    best_r = _radius(best_centers)
    
    # Optimize BCC
    if time.time() - start_time < deadline - 20:
        budget = min(40.0, deadline - 20 - (time.time() - start_time))
        sa_centers, sa_r = _sa_optimize(centers_bcc, budget)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # Optimize Grid
    if time.time() - start_time < deadline - 20:
        budget = min(40.0, deadline - 20 - (time.time() - start_time))
        sa_centers, sa_r = _sa_optimize(centers_grid, budget)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # Optimize Random
    if time.time() - start_time < deadline - 10:
        budget = min(30.0, deadline - 10 - (time.time() - start_time))
        sa_centers, sa_r = _sa_optimize(centers_rand, budget)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # Final polish with a tighter SA or L-BFGS-B?
    # Let's do a final short SA with smaller steps to polish.
    if time.time() - start_time < deadline - 5:
        budget = min(10.0, deadline - 5 - (time.time() - start_time))
        # Clone best and run a short SA with lower T
        polish_centers = best_centers.copy()
        # Temporarily modify SA parameters? Or just run SA again.
        # SA is stochastic, so running it again might improve slightly.
        sa_centers, sa_r = _sa_optimize(polish_centers, budget)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        best_centers = centers_grid.copy()

    return best_centers
# EVOLVE-BLOCK-END
