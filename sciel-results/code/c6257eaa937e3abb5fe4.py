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
        
        # Pair constraints
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
        
        # Boundary constraints
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

    def local_optimize(initial_centers, W, H, max_iters=100):
        """Iteratively optimize centers and radii."""
        centers = initial_centers.copy()
        n = len(centers)
        best_r = None
        best_val = -np.inf
        
        for it in range(max_iters):
            radii = solve_lp(centers, W, H)
            if radii is None:
                # Infeasible, shrink radii or move centers? 
                # For now, just break or try to fix by reducing step size in previous step.
                # If we are in the loop, we assume previous step was valid.
                # If initial is invalid, return None.
                break
            
            val = np.sum(radii)
            if val > best_val:
                best_val = val
                best_r = radii.copy()
                best_centers = centers.copy()
            
            # Compute forces to move centers
            forces = np.zeros_like(centers)
            
            # Get Delaunay neighbors to limit constraint checking
            try:
                tri = Delaunay(centers)
                edges = set()
                for simplex in tri.simplices:
                    edges.add(tuple(sorted(simplex[:2])))
                    edges.add(tuple(sorted(simplex[1:3])))
                    edges.add(tuple(sorted(simplex[2:1]))) # This is wrong, simplices are triangles (3 points)
                # Correct edge extraction from Delaunay
                edges = set()
                for simplex in tri.simplices:
                    edges.add(tuple(sorted(simplex[0:2])))
                    edges.add(tuple(sorted(simplex[1:3])))
                    edges.add(tuple(sorted(simplex[2:1]))) # Still wrong logic, simplex is [i,j,k]
                # Let's just do all pairs for N=21, it's cheap.
                edges = set()
                for i in range(n):
                    for j in range(i+1, n):
                        edges.add((i,j))
            except Exception:
                edges = set()
                for i in range(n):
                    for j in range(i+1, n):
                        edges.add((i,j))

            # Apply forces from tight constraints
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 1e-10: continue
                    # Check if constraint is tight
                    if abs(radii[i] + radii[j] - d) < 1e-6:
                        # Push apart
                        direction = (centers[i] - centers[j]) / d
                        forces[i] += direction
                        forces[j] -= direction
                
                # Boundary forces
                if abs(radii[i] - centers[i][0]) < 1e-6:
                    forces[i][0] += 1.0
                if abs(radii[i] - (W - centers[i][0])) < 1e-6:
                    forces[i][0] -= 1.0
                if abs(radii[i] - centers[i][1]) < 1e-6:
                    forces[i][1] += 1.0
                if abs(radii[i] - (H - centers[i][1])) < 1e-6:
                    forces[i][1] -= 1.0
            
            # Update centers
            step = 0.05
            # Normalize forces to avoid huge jumps
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 0:
                forces = forces / max_norm * step
            centers += forces
            
            # Ensure centers stay inside box (clamp)
            centers[:, 0] = np.clip(centers[:, 0], 0, W)
            centers[:, 1] = np.clip(centers[:, 1], 0, H)
            
            # Check for no improvement
            if it > 10 and abs(val - best_val) < 1e-8:
                break
                
        if best_r is not None:
            return best_centers, best_r
        return None, None

    # Try different aspect ratios
    best_result = None
    best_value = -np.inf
    
    # Grid initialization
    # 21 circles. 5x5 grid is 25, so we drop 4.
    # Or 4x6 grid is 24, drop 3.
    # Let's use a 5x5 grid and remove corners or edge points to make it more square-like?
    # Actually, a 5x4 grid (20) + 1 is awkward.
    # Let's stick to 5x5 grid, take first 21.
    
    grid_rows, grid_cols = 5, 5
    base_centers = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            base_centers.append([j, i])
    base_centers = np.array(base_centers[:N], dtype=float)
    
    # Normalize base centers to [0,1]x[0,1]
    cmin = base_centers.min(axis=0)
    cmax = base_centers.max(axis=0)
    base_centers = (base_centers - cmin) / (cmax - cmin)
    
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.33]
    
    for aspect in aspects:
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        # Scale base centers to W, H
        centers = base_centers * np.array([W, H])
        
        # Add some jitter to break symmetry
        np.random.seed(42)
        centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
        
        opt_centers, radii = local_optimize(centers, W, H, max_iters=50)
        
        if radii is not None:
            val = np.sum(radii)
            if val > best_value:
                best_value = val
                result = np.zeros((N, 3))
                result[:, :2] = opt_centers
                result[:, 2] = radii
                best_result = result
                
    if best_result is not None:
        return best_result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
