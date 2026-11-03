import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)

# Vertices of the equilateral triangle
# (0,0), (1,0), (0.5, sqrt(3)/2)
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])

# Barycentric coordinates conversion matrices
# Point P = a*V1 + b*V2 + c*V3, where a+b+c=1
# In Cartesian:
# x = b + 0.5*c
# y = (sqrt(3)/2)*c
# Inverse:
# c = 2*y / sqrt(3)
# b = x - 0.5*c
# a = 1 - b - c

def cart_to_bary(pts_cart):
    """Convert (N, 2) Cartesian points to (N, 3) Barycentric coordinates."""
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    c = 2.0 * y / SQRT3
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack((a, b, c))

def bary_to_cart(pts_bary):
    """Convert (N, 3) Barycentric coordinates to (N, 2) Cartesian points."""
    a = pts_bary[:, 0]
    b = pts_bary[:, 1]
    c = pts_bary[:, 2]
    x = b + 0.5 * c
    y = (SQRT3 / 2.0) * c
    return np.column_stack((x, y))

def project_bary(pts_bary):
    """
    Project barycentric coordinates to the valid simplex domain (a,b,c >= 0, sum=1).
    This ensures points stay inside the triangle.
    """
    # Clip negative values
    # A simple projection onto the simplex for 3D is:
    # 1. Clip to 0
    # 2. Normalize to sum 1
    # This is a valid projection for the constraint a,b,c >= 0, a+b+c=1
    
    p = pts_bary.copy()
    p[p < 0] = 0.0
    sums = np.sum(p, axis=1, keepdims=True)
    # Avoid division by zero (should not happen if we start valid and perturb small)
    sums[sums == 0] = 1.0
    p = p / sums
    return p

def min_triangle_area(pts_cart):
    """
    Calculate the minimum area of any triangle formed by N points.
    Vectorized for speed.
    """
    n = len(pts_cart)
    # Generate indices for all unique triples i < j < k
    # C(11, 3) = 165 triples. Very fast.
    idx = np.indices((n, n, n))
    i_idx, j_idx, k_idx = idx[0], idx[1], idx[2]
    
    # Mask for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    p_i = pts_cart[i_vals]
    p_j = pts_cart[j_vals]
    p_k = pts_cart[k_vals]
    
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    # Using cross product in 2D
    v_ij = p_j - p_i
    v_ik = p_k - p_i
    
    # Cross product z-component
    cross = v_ij[:, 0] * v_ik[:, 1] - v_ij[:, 1] * v_ik[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_points_bary = None
    best_points_cart = None
    
    # Strategy: Multi-restart jitter search in Barycentric coordinates
    # 1. Initialize with several diverse starting configurations
    # 2. For each config, perform many small random perturbations in barycentric space
    # 3. Project back to simplex (valid domain)
    # 4. Convert to Cartesian, compute min area
    # 5. If improved, accept and continue from there (greedy local search)
    # 6. If no improvement for many steps, try larger steps or restart
    
    num_restarts = 50  # We have time for many restarts
    
    # Pre-compute indices for min_triangle_area to save time inside loop?
    # No, min_triangle_area creates indices every time. 
    # Let's optimize min_triangle_area to use pre-allocated arrays if possible, 
    # or just rely on the fact that 165 ops is tiny.
    # Actually, creating indices every time is overhead. Let's create them once.
    
    n = N
    idx = np.indices((n, n, n))
    i_idx, j_idx, k_idx = idx[0], idx[1], idx[2]
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    def fast_min_area(pts):
        p_i = pts[i_vals]
        p_j = pts[j_vals]
        p_k = pts[k_vals]
        v_ij = p_j - p_i
        v_ik = p_k - p_i
        cross = v_ij[:, 0] * v_ik[:, 1] - v_ij[:, 1] * v_ik[:, 0]
        return np.min(0.5 * np.abs(cross))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialization
        if restart == 0:
            # Structured init: Hexagonal-ish packing
            # Center
            c_bary = np.array([1/3, 1/3, 1/3])
            pts_bary = np.tile(c_bary, (N, 1))
            # Perturb slightly to break symmetry
            pts_bary += rng.normal(0, 0.05, (N, 3))
            pts_bary = project_bary(pts_bary)
        elif restart == 1:
            # Grid-like init in barycentric
            # Try to spread out
            # 1 center, 6 around, 4 corners/edges
            pts_bary = np.array([
                [1/3, 1/3, 1/3],
                [0.1, 0.45, 0.45],
                [0.45, 0.1, 0.45],
                [0.45, 0.45, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.8, 0.1],
                [0.8, 0.1, 0.1],
                [0.25, 0.25, 0.5],
                [0.25, 0.5, 0.25],
                [0.5, 0.25, 0.25],
                [0.5, 0.5, 0.0]
            ])
        else:
            # Random init
            pts_bary = rng.dirichlet(np.ones(N), size=N)
            
        pts_bary = project_bary(pts_bary)
        pts_cart = bary_to_cart(pts_bary)
        current_score = fast_min_area(pts_cart)
        
        if current_score > best_score:
            best_score = current_score
            best_points_bary = pts_bary.copy()
            best_points_cart = pts_cart.copy()
            
        # Local Search Loop
        # We will perform a sequence of jitter steps
        # Step size starts moderate, decreases if stuck, or increases if stuck?
        # Standard jitter: try small random moves.
        
        current_bary = pts_bary.copy()
        current_cart = pts_cart.copy()
        current_val = current_score
        
        no_improve_count = 0
        max_no_improve = 200
        base_step = 0.02
        
        for step in range(2000):  # Max steps per restart
            if time.time() > deadline - 2.0:
                break
                
            # Dynamic step size: decrease if improving, increase if stuck?
            # Let's just use a fixed small step for simplicity and robustness, 
            # or a slightly decaying step.
            step_size = base_step * (1.0 - 0.5 * (step / 2000.0))
            
            # Generate random perturbation in barycentric space
            # Perturb each point's barycentric coords
            perturb = rng.normal(0, step_size, (N, 3))
            
            new_bary = current_bary + perturb
            new_bary = project_bary(new_bary)
            
            new_cart = bary_to_cart(new_bary)
            new_val = fast_min_area(new_cart)
            
            if new_val > current_val:
                current_bary = new_bary
                current_cart = new_cart
                current_val = new_val
                no_improve_count = 0
                
                if current_val > best_score:
                    best_score = current_val
                    best_points_bary = current_bary.copy()
                    best_points_cart = current_cart.copy()
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Stuck, break and restart
                    break
    
    if best_points_cart is None:
        # Fallback
        pts_bary = np.tile([1/3, 1/3, 1/3], (N, 1))
        pts_bary += rng.normal(0, 0.01, (N, 3))
        pts_bary = project_bary(pts_bary)
        best_points_cart = bary_to_cart(pts_bary)
        
    return best_points_cart

