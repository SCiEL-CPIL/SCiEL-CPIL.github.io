import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 145.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
M = len(tri_indices)
tri_idx = np.array(tri_indices)  # (M, 3)

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    # u corresponds to V0 weight, v to V1 weight
    # x = v + 0.5*w, y = w*H
    # w = y / H
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def calc_min_area_cartesian(cart):
    """Calculate the minimum area of triangles formed by N points in Cartesian coords."""
    x = cart[:, 0]
    y = cart[:, 1]
    
    xi = x[tri_idx[:, 0]]
    yi = y[tri_idx[:, 0]]
    xj = x[tri_idx[:, 1]]
    yj = y[tri_idx[:, 1]]
    xk = x[tri_idx[:, 2]]
    yk = y[tri_idx[:, 2]]
    
    # Area = 0.5 * |det|
    det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(det)
    return np.min(areas)

def get_areas_cartesian(cart):
    """Calculate all triangle areas."""
    x = cart[:, 0]
    y = cart[:, 1]
    
    xi = x[tri_idx[:, 0]]
    yi = y[tri_idx[:, 0]]
    xj = x[tri_idx[:, 1]]
    yj = y[tri_idx[:, 1]]
    xk = x[tri_idx[:, 2]]
    yk = y[tri_idx[:, 2]]
    
    det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(det)
    return areas

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def cart_in_triangle(cart, eps=1e-9):
    """Check if Cartesian points are inside the triangle."""
    x = cart[:, 0]
    y = cart[:, 1]
    # y >= 0
    # y <= H * (1 - x) for x in [0, 0.5]... actually the triangle is:
    # 0 <= y <= H
    # 0 <= x <= 1
    # y <= H * (1 - x) for x >= 0.5? No.
    # The triangle vertices are (0,0), (1,0), (0.5, H).
    # Line from (0,0) to (0.5, H): y = H * x / 0.5 = 2H * x => x = y / (2H)
    # Line from (1,0) to (0.5, H): y = -2H * (x - 1) = 2H * (1 - x) => x = 1 - y/(2H)
    # So: x >= y/(2H) and x <= 1 - y/(2H) and y >= 0 and y <= H
    
    cond1 = y >= -eps
    cond2 = y <= H + eps
    cond3 = x >= y / (2 * H) - eps
    cond4 = x <= 1 - y / (2 * H) + eps
    
    return np.all(cond1 & cond2 & cond3 & cond4)

def clamp_cart(cart, eps=1e-9):
    """Clamp Cartesian points to be inside the triangle."""
    cart = cart.copy()
    x = cart[:, 0]
    y = cart[:, 1]
    
    y = np.clip(y, eps, H - eps)
    
    x_min = y / (2 * H)
    x_max = 1 - y / (2 * H)
    
    x = np.clip(x, x_min + eps, x_max - eps)
    
    return np.column_stack([x, y])

def soft_min_obj_cartesian(cart, alpha=10.0):
    """Smooth surrogate: -log-sum-exp of negative areas scaled by alpha."""
    areas = get_areas_cartesian(cart)
    
    neg_alpha_areas = -alpha * areas
    max_neg_alpha = np.max(neg_alpha_areas)
    if max_neg_alpha > 700:
        return -np.min(areas)
    
    lse = max_neg_alpha + np.log(np.sum(np.exp(neg_alpha_areas - max_neg_alpha)))
    return -lse / alpha

def active_triangles(cart, k=10):
    """Find the k triangles with the smallest areas."""
    areas = get_areas_cartesian(cart)
    indices = np.argsort(areas)[:k]
    return indices, areas[indices]

def coordinated_ascent(cart, step_size=0.01, max_iter=1000, deadline=None):
    """
    Coordinated Ascent: 
    1. Identify the 'critical' triangles (smallest areas).
    2. For each point involved in a critical triangle, try to move it 
       to increase the area of that specific triangle, while checking 
       that it doesn't decrease the global minimum area too much.
    3. This is a form of local search / hill climbing.
    """
    current_cart = cart.copy()
    current_min_area = calc_min_area_cartesian(current_cart)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Find critical triangles
        crit_idx, crit_areas = active_triangles(current_cart, k=5)
        
        improved = False
        for idx in crit_idx:
            # Get the 3 points of this triangle
            p_idx, q_idx, r_idx = tri_idx[idx]
            
            # Try to move each of the 3 points to increase this triangle's area
            # while maintaining the global minimum area
            for point_idx in [p_idx, q_idx, r_idx]:
                orig_point = current_cart[point_idx].copy()
                
                # Define a local objective: maximize the area of the critical triangle
                # subject to the constraint that the global min area doesn't drop below current_min_area - tolerance
                
                def local_obj(x):
                    new_cart = current_cart.copy()
                    new_cart[point_idx] = x
                    new_cart = clamp_cart(new_cart)
                    
                    # Calculate area of the critical triangle
                    p = new_cart[p_idx]
                    q = new_cart[q_idx]
                    r = new_cart[r_idx]
                    area_crit = 0.5 * np.abs(np.cross(q - p, r - p))
                    
                    # Calculate global min area
                    global_min = calc_min_area_cartesian(new_cart)
                    
                    # If global min drops too much, penalize heavily
                    if global_min < current_min_area - 1e-6:
                        return -area_crit - 100 * (current_min_area - global_min)
                    
                    return -area_crit
                
                # Optimize this single point
                try:
                    res = minimize(
                        local_obj,
                        orig_point,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    
                    new_point = res.x
                    new_cart = current_cart.copy()
                    new_cart[point_idx] = new_point
                    new_cart = clamp_cart(new_cart)
                    
                    new_global_min = calc_min_area_cartesian(new_cart)
                    
                    if new_global_min > current_min_area + 1e-8:
                        current_cart = new_cart
                        current_min_area = new_global_min
                        improved = True
                except Exception:
                    pass
        
        if not improved:
            # Try random perturbations if no improvement
            rng = np.random.default_rng()
            for _ in range(10):
                perturbed = current_cart + rng.normal(0, step_size, current_cart.shape)
                perturbed = clamp_cart(perturbed)
                new_global_min = calc_min_area_cartesian(perturbed)
                if new_global_min > current_min_area + 1e-8:
                    current_cart = perturbed
                    current_min_area = new_global_min
                    improved = True
                    break
            
            if not improved:
                # Reduce step size
                step_size *= 0.9
                if step_size < 1e-8:
                    break
    
    return current_cart, current_min_area

def generate_lattice_init():
    """Generate a lattice-based initialization."""
    # Use a triangular lattice pattern
    bary = []
    
    # 4 rows in barycentric space
    # Row 0: (0,0)
    # Row 1: (0, 0.25), (0.25, 0)
    # Row 2: (0, 0.5), (0.5, 0), (0.25, 0.25)
    # Row 3: (0, 0.75), (0.75, 0), (0.5, 0.25), (0.25, 0.5)
    # Row 4: (0, 1), (1, 0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75)
    
    # We need 11 points. Let's pick a subset.
    candidates = [
        [0, 0], [0, 1], [1, 0],  # Vertices
        [0, 0.5], [0.5, 0], [0.5, 0.5],  # Midpoints
        [1/3, 1/3],  # Center
        [0, 0.25], [0.25, 0], [0.25, 0.25], [0.5, 0.25]  # Additional
    ]
    
    # Select 11 from candidates
    selected = candidates[:11]
    bary = np.array(selected)
    
    # Add small perturbation
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.005, bary.shape)
    bary += noise
    bary = clamp_bary(bary)
    
    return bary_to_cart(bary)

def generate_hexagonal_init(seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # Center
    bary[0] = [1/3, 1/3]
    
    # First ring (6 points)
    # In barycentric, a hexagon around the center.
    # Approximate positions:
    # (1/3 + r*cos(theta), 1/3 + r*sin(theta)) in some metric
    # Simpler: use a fixed set of "good" points.
    
    # Let's use a perturbed grid.
    grid_pts = []
    for i in range(5):
        for j in range(5):
            u = i * 0.25
            v = j * 0.25
            if u + v <= 1.0 + 1e-9:
                grid_pts.append([u, v])
    
    # grid_pts has 15 entries.
    # Select 11.
    selected_indices = [0, 4, 14, 2, 9, 11, 6, 7, 10, 3, 8]
    bary = np.array([grid_pts[i] for i in selected_indices])
    
    # Add small perturbation
    noise = rng.normal(0, 0.01, bary.shape)
    bary += noise
    bary = clamp_bary(bary)
    
    return bary_to_cart(bary)

def generate_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    bary = np.column_stack([u, v])
    return bary_to_cart(bary)

def refine_local(cart, deadline):
    """
    Multi-stage local refinement:
    1. Coordinated Ascent on critical triangles.
    2. L-BFGS-B with soft objective.
    3. Final SLSQP polish.
    """
    current_cart = cart.copy()
    current_min = calc_min_area_cartesian(current_cart)
    
    # Stage 1: Coordinated Ascent
    if time.time() < deadline - 10:
        current_cart, current_min = coordinated_ascent(
            current_cart, step_size=0.05, max_iter=500, deadline=deadline - 10
        )
    
    # Stage 2: L-BFGS-B with soft objective
    if time.time() < deadline - 5:
        def soft_obj(x):
            c = x.reshape(N, 2)
            c = clamp_cart(c)
            return soft_min_obj_cartesian(c, alpha=10.0)
        
        try:
            res = minimize(
                soft_obj,
                current_cart.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            new_cart = clamp_cart(res.x.reshape(N, 2))
            new_min = calc_min_area_cartesian(new_cart)
            
            if new_min > current_min:
                current_cart = new_cart
                current_min = new_min
        except Exception:
            pass
    
    # Stage 3: Final SLSQP polish with epigraph
    if time.time() < deadline - 2:
        def epigraph_obj(x):
            c = x[:N*2].reshape(N, 2)
            t = x[N*2]
            return -t
        
        def epigraph_grad(x):
            grad = np.zeros_like(x)
            grad[N*2] = -1.0
            return grad
        
        def epigraph_con_fun(x):
            c = x[:N*2].reshape(N, 2)
            t = x[N*2]
            c = clamp_cart(c)
            areas = get_areas_cartesian(c)
            cons = areas - t
            return cons
        
        constraints = {'type': 'ineq', 'fun': epigraph_con_fun}
        x0 = np.concatenate([current_cart.flatten(), [current_min]])
        bounds = [(0, 1)] * (N * 2) + [(0, 1)]
        
        try:
            res = minimize(
                epigraph_obj,
                x0,
                method='SLSQP',
                jac=epigraph_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < 0:
                new_cart = clamp_cart(res.x[:N*2].reshape(N, 2))
                new_min = calc_min_area_cartesian(new_cart)
                
                if new_min > current_min:
                    current_cart = new_cart
                    current_min = new_min
        except Exception:
            pass
    
    return current_cart, current_min

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # Safety margin
    
    best_cart = None
    best_val = -np.inf
    
    def update_best(cart):
        nonlocal best_cart, best_val
        if cart is None:
            return
        cart = clamp_cart(cart)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_cart = cart.copy()
    
    # Phase 1: Generate diverse initializations
    inits = []
    
    # Lattice init
    inits.append(generate_lattice_init())
    
    # Hexagonal init
    inits.append(generate_hexagonal_init(0))
    
    # Random inits
    for i in range(5):
        inits.append(generate_random_init(100 + i))
    
    # Phase 2: Refine each initialization
    for init_cart in inits:
        if time.time() > deadline - 15:
            break
        
        # Quick evaluation
        init_val = calc_min_area_cartesian(init_cart)
        
        # Refine
        refined_cart, refined_val = refine_local(init_cart, deadline)
        
        update_best(refined_cart)
        
        # Also try a few more random perturbations from the refined solution
        if time.time() < deadline - 10:
            rng = np.random.default_rng()
            for _ in range(3):
                perturbed = refined_cart + rng.normal(0, 0.01, refined_cart.shape)
                perturbed = clamp_cart(perturbed)
                pert_val = calc_min_area_cartesian(perturbed)
                
                if pert_val > best_val:
                    # Refine this perturbation too
                    refined2, val2 = refine_local(perturbed, deadline)
                    update_best(refined2)
    
    # Fallback
    if best_cart is None:
        best_cart = generate_random_init(0)
    
    return best_cart
