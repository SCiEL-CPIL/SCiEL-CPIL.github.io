"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

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
                return res.x
        except Exception:
            pass
        return None

    def optimize_packing(centers, W, H, max_iters=500, tol=1e-6):
        """
        Iteratively optimize centers to maximize sum of radii.
        Uses force-based method derived from tight constraints.
        """
        centers = centers.copy()
        n = len(centers)
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        # Pre-compute pairs for efficiency (N=21 is small, but let's be neat)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        step_size = 0.1
        stall_count = 0
        
        for it in range(max_iters):
            radii = solve_lp(centers, W, H)
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
            
            if stall_count > 50:
                # Shake to escape local minima
                centers += np.random.normal(0, 0.01, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 0, W)
                centers[:, 1] = np.clip(centers[:, 1], 0, H)
                stall_count = 0
                step_size = max(0.01, step_size * 0.5)
                continue
                
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Pair forces
            for i, j in pairs:
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                # Check if constraint is tight
                if abs(radii[i] + radii[j] - d) < 1e-5:
                    # Push apart
                    direction = d_vec / d
                    forces[i] += direction
                    forces[j] -= direction
            
            # Boundary forces
            for i in range(n):
                # Left
                if abs(radii[i] - centers[i][0]) < 1e-5:
                    forces[i][0] += 1.0
                # Right
                if abs(radii[i] - (W - centers[i][0])) < 1e-5:
                    forces[i][0] -= 1.0
                # Bottom
                if abs(radii[i] - centers[i][1]) < 1e-5:
                    forces[i][1] += 1.0
                # Top
                if abs(radii[i] - (H - centers[i][1])) < 1e-5:
                    forces[i][1] -= 1.0
            
            # Normalize and apply step
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                forces = forces / max_norm * step_size
                centers += forces
                
                # Clamp to box
                centers[:, 0] = np.clip(centers[:, 0], 0, W)
                centers[:, 1] = np.clip(centers[:, 1], 0, H)
                
            # Decay step size
            step_size *= 0.995
            
            if abs(val - best_val) < tol and it > 100:
                break

        if best_radii is not None:
            return best_centers, best_radii
        return None, None

    def run_optimization(W, H, initial_centers):
        """Run local optimization for a specific box size and initial config."""
        centers, radii = optimize_packing(initial_centers, W, H, max_iters=300)
        if radii is not None:
            return np.sum(radii), centers, radii
        return -np.inf, None, None

    # Aspect ratios to try
    # Perimeter 2(W+H) = 4 => W+H=2
    aspects = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.75, 1.5, 0.6, 1.6, 0.5, 1.8, 0.4]
    
    best_val_global = -np.inf
    best_result_global = None
    
    # Initial configurations
    # 1. Grid 5x5 (take 21)
    # 2. Grid 4x6 (take 21)
    # 3. Random
    
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
