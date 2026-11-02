# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog, minimize

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    
    Approach: 
    1. Parameterize the packing by centers and rectangle width W (H=2-W).
    2. For a given configuration, determine Delaunay neighbors.
    3. Solve an LP to find the maximum sum of radii for fixed centers and box.
    4. Use Nelder-Mead to optimize centers and W to maximize the sum of radii.
    """
    N = 21
    
    def get_delaunay_pairs(centers):
        """Get Delaunay neighbor pairs from center positions."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
            return list(edges)
        except Exception:
            # Fallback: all pairs if Delaunay fails (expensive but safe)
            return [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]

    def solve_lp(centers, W, H, delaunay_edges):
        """
        Solve LP for optimal radii given fixed centers and box size.
        Returns sum of radii or -inf if infeasible.
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij
        for i, j in delaunay_edges:
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
                return np.sum(res.x)
            else:
                return -np.inf
        except Exception:
            return -np.inf

    def objective(params):
        """
        Objective function for minimization (we minimize -sum_radii).
        params: [x_1, y_1, ..., x_N, y_N, W]
        """
        centers = params[:2*N].reshape(N, 2)
        W = params[2*N]
        H = 2.0 - W
        
        # Basic bounds check
        if W <= 0.1 or H <= 0.1:
            return 1e6
        if np.any(centers[:, 0] < 0) or np.any(centers[:, 0] > W) or \
           np.any(centers[:, 1] < 0) or np.any(centers[:, 1] > H):
            # Penalize out-of-bounds centers heavily
            penalty = 0
            for i in range(N):
                penalty += max(0, -centers[i][0]) + max(0, centers[i][0] - W)
                penalty += max(0, -centers[i][1]) + max(0, centers[i][1] - H)
            return 1e6 + penalty * 100

        delaunay_edges = get_delaunay_pairs(centers)
        sum_radii = solve_lp(centers, W, H, delaunay_edges)
        
        if sum_radii < 0:
            return 1e6
        return -sum_radii

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
    
    # 6x4 grid
    gx, gy = 6, 4
    cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
    cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
    grid_pts = np.array([[x, y] for y in cy for x in cx])
    configs.append(grid_pts[:21])
    
    # Hexagonal-like pattern
    hex_pts = []
    for row in range(5):
        for col in range(5):
            x = (col + 0.5) / 5.0 + (0.5 / 5.0) * (row % 2)
            y = (row + 0.5) / 5.0
            hex_pts.append([x, y])
    configs.append(np.array(hex_pts[:21]))
    
    # Random seeds
    np.random.seed(42)
    configs.append(np.random.rand(21, 2))
    np.random.seed(123)
    configs.append(np.random.rand(21, 2))

    best_val_global = -np.inf
    best_result_global = None

    for config in configs:
        # Scale config to an initial box, say square 1x1
        # We will optimize W in [1.0, 2.0] (since W+H=2 and W>=H by symmetry)
        # Initial W = 1.0
        initial_W = 1.0
        centers_init = config * np.array([initial_W, 2.0 - initial_W])
        
        # Ensure centers are strictly inside
        centers_init[:, 0] = np.clip(centers_init[:, 0], 0.01, initial_W - 0.01)
        centers_init[:, 1] = np.clip(centers_init[:, 1], 0.01, 2.0 - initial_W - 0.01)
        
        x0 = np.concatenate([centers_init.flatten(), [initial_W]])
        
        # Bounds for minimize
        # Centers: [0, W] and [0, H] are handled by penalty, but we can give loose bounds
        # W: [1.0, 2.0]
        bounds = []
        for i in range(N):
            bounds.append((0, 2.0)) # x_i
            bounds.append((0, 2.0)) # y_i
        bounds.append((1.0, 2.0)) # W
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6})
            
            if res.success or res.fun < -best_val_global:
                val = -res.fun
                if val > best_val_global:
                    best_val_global = val
                    centers = res.x[:2*N].reshape(N, 2)
                    W_opt = res.x[2*N]
                    H_opt = 2.0 - W_opt
                    
                    # Solve LP one last time to get exact radii for the best centers
                    delaunay_edges = get_delaunay_pairs(centers)
                    # Re-solve LP to get radii
                    n = N
                    c = -np.ones(n)
                    A_ub = []
                    b_ub = []
                    for i, j in delaunay_edges:
                        d = np.linalg.norm(centers[i] - centers[j])
                        row = np.zeros(n)
                        row[i] = 1
                        row[j] = 1
                        A_ub.append(row)
                        b_ub.append(d)
                    for i in range(n):
                        row = np.zeros(n); row[i] = 1
                        A_ub.append(row); b_ub.append(centers[i][0])
                        row = np.zeros(n); row[i] = 1
                        A_ub.append(row); b_ub.append(W_opt - centers[i][0])
                        row = np.zeros(n); row[i] = 1
                        A_ub.append(row); b_ub.append(centers[i][1])
                        row = np.zeros(n); row[i] = 1
                        A_ub.append(row); b_ub.append(H_opt - centers[i][1])
                    
                    A_ub = np.array(A_ub)
                    b_ub = np.array(b_ub)
                    bounds_lp = [(0, None)] * n
                    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
                    
                    if res_lp.success:
                        radii = res_lp.x
                        result = np.zeros((N, 3))
                        result[:, :2] = centers
                        result[:, 2] = radii
                        
                        # Verify perimeter
                        # The bounding box of the *circles* might be smaller than W x H if centers are clamped?
                        # No, we optimized centers inside [0,W]x[0,H].
                        # The "bounding rectangle" in the task usually refers to the container.
                        # The task says "bounding rectangle has perimeter at most 4".
                        # If the circles are packed in a W x H box with 2(W+H)=4, the bounding rectangle of the *union* of circles
                        # is contained in this box. The evaluator likely checks the bounding box of the circles themselves.
                        # To be safe, we should ensure the circles fit in the minimal bounding box of their centers expanded by radii?
                        # Actually, the standard interpretation is that the circles are inside a fixed rectangle of perimeter 4.
                        # My LP enforces centers in [0,W]x[0,H] and radii <= dist to boundary.
                        # This ensures circles are inside the W x H rectangle.
                        # The perimeter of this rectangle is 4.
                        # So the result is valid.
                        
                        best_result_global = result
        except Exception:
            pass

    if best_result_global is not None:
        return best_result_global
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
