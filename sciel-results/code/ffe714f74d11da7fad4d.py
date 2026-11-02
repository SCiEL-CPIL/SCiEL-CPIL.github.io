# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    """
    
    def solve_lp(centers, W, H):
        """Solve LP for optimal radii given fixed centers and box size."""
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
        
        # Boundary constraints: r_i <= dist_to_boundary
        for i in range(n):
            # Left
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][0])
            # Right
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(W - centers[i][0])
            # Bottom
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][1])
            # Top
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(H - centers[i][1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x, res
        except Exception:
            pass
        return None, None

    def optimize_packing(centers, W, H, max_iters=1000, tol=1e-8):
        """
        Optimize centers using force-directed method with Delaunay filtering and line search.
        """
        centers = centers.copy()
        n = len(centers)
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        # Pre-compute all pairs for LP, but we will filter for forces
        # We need Delaunay for force filtering
        
        step_size = 0.1
        momentum = np.zeros_like(centers)
        stall_count = 0
        
        for it in range(max_iters):
            radii, res = solve_lp(centers, W, H)
            if radii is None:
                break
            
            val = np.sum(radii)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_radii = radii.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            if stall_count > 100:
                # Shake to escape local minima
                noise = np.random.normal(0, 0.05, size=centers.shape)
                centers += noise
                centers[:, 0] = np.clip(centers[:, 0], 0, W)
                centers[:, 1] = np.clip(centers[:, 1], 0, H)
                stall_count = 0
                step_size = max(0.01, step_size * 0.5)
                momentum = np.zeros_like(centers)
                continue

            # Compute Delaunay triangulation to find neighbors
            try:
                tri = Delaunay(centers)
                # Get unique edges
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            a, b = simplex[i], simplex[j]
                            if a > b: a, b = b, a
                            edges.add((a, b))
            except Exception:
                edges = [(i, j) for i in range(n) for j in range(i+1, n)] # Fallback to all pairs

            # Compute forces based on active constraints
            forces = np.zeros_like(centers)
            
            # Pair forces
            # We check only Delaunay edges for potential contact
            for i, j in edges:
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                # Check if constraint is tight
                # Tolerance needs to be small enough to identify active sets
                if abs(radii[i] + radii[j] - d) < 1e-6:
                    # Push apart
                    direction = d_vec / d
                    forces[i] += direction
                    forces[j] -= direction
            
            # Boundary forces
            for i in range(n):
                if abs(radii[i] - centers[i][0]) < 1e-6:
                    forces[i][0] += 1.0
                if abs(radii[i] - (W - centers[i][0])) < 1e-6:
                    forces[i][0] -= 1.0
                if abs(radii[i] - centers[i][1]) < 1e-6:
                    forces[i][1] += 1.0
                if abs(radii[i] - (H - centers[i][1])) < 1e-6:
                    forces[i][1] -= 1.0
            
            # Normalize forces
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                forces = forces / max_norm
                
                # Add momentum
                momentum = 0.9 * momentum + forces
                update = momentum * step_size
                
                # Line search
                alpha = step_size
                improved = False
                for _ in range(10):
                    new_centers = centers + update * alpha
                    new_centers[:, 0] = np.clip(new_centers[:, 0], 0, W)
                    new_centers[:, 1] = np.clip(new_centers[:, 1], 0, H)
                    
                    new_radii, _ = solve_lp(new_centers, W, H)
                    if new_radii is not None:
                        new_val = np.sum(new_radii)
                        if new_val > val:
                            centers = new_centers
                            improved = True
                            break
                    alpha *= 0.5
                
                if not improved:
                    # If no improvement, reduce step size
                    step_size *= 0.8
                    momentum = np.zeros_like(centers)
                else:
                    step_size *= 1.05 # Slightly increase if successful, capped later
                    step_size = min(step_size, 0.5)
            else:
                # No forces, likely converged or stuck
                step_size *= 0.9

            if abs(val - best_val) < tol and it > 50:
                break

        if best_radii is not None:
            return best_centers, best_radii
        return None, None

    def run_optimization(W, H, initial_centers):
        """Run local optimization for a specific box size and initial config."""
        centers, radii = optimize_packing(initial_centers, W, H, max_iters=500)
        if radii is not None:
            return np.sum(radii), centers, radii
        return -np.inf, None, None

    # Aspect ratios to try
    # Perimeter 2(W+H) = 4 => W+H=2
    aspects = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.75, 1.5, 0.6, 1.6, 0.5, 1.8, 0.4]
    
    best_val_global = -np.inf
    best_result_global = None
    
    # Initial configurations
    configs = []
    
    # 5x5 grid
    gx, gy = 5, 5
    cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
    cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
    grid_pts = np.array([[x, y] for y in cy for x in cx])
    configs.append(grid_pts[:21])
    
    # 4x6 grid
    gx, gy = 4, 6
    cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
    cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
    grid_pts = np.array([[x, y] for y in cy for x in cx])
    configs.append(grid_pts[:21])
    
    # 7x3 grid
    gx, gy = 7, 3
    cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
    cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
    grid_pts = np.array([[x, y] for y in cy for x in cx])
    configs.append(grid_pts[:21])
    
    # Random
    np.random.seed(42)
    configs.append(np.random.rand(21, 2))
    
    # Add a few more random inits
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        configs.append(np.random.rand(21, 2))

    for aspect in aspects:
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        for config in configs:
            # Scale config to box
            centers_init = config * np.array([W, H])
            
            val, centers, radii = run_optimization(W, H, centers_init)
            
            if val > best_val_global:
                best_val_global = val
                result = np.zeros((21, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_result_global = result

    if best_result_global is not None:
        return best_result_global
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
