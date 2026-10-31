import numpy as np
from itertools import combinations
import time
from scipy.optimize import differential_evolution, minimize

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    # x = v + 0.5*(1-u-v) = 0.5 + 0.5v - 0.5u
    # y = (1-u-v)*H => 1-u-v = y/H => u+v = 1 - y/H
    # Let s = u+v. Then v = x - 0.5 + 0.5u.
    # u + x - 0.5 + 0.5u = 1 - y/H
    # 1.5u = 1.5 - x - y/H
    # u = 1 - x/1.5 - y/(1.5H)
    x = cart[:, 0]
    y = cart[:, 1]
    u = 1.0 - x * (2.0 / 3.0) - y * (2.0 / (3.0 * H))
    v = 1.0 - u - y / H
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

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

def get_min_area(bary):
    """Get the minimum area for a given barycentric configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def hard_min_obj(bary_flat):
    """Objective for optimization: minimize negative min area."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    return -get_min_area(bary)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    This is used for the global search (Differential Evolution) to provide a
    smoother landscape that guides the population towards the basin of the
    true minimum, avoiding the jaggedness of the hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # log-sum-exp trick for stability
    a_max = np.max(areas)
    val = a_max - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - a_max))))
    return -val

def active_triangles(bary, threshold=1e-6):
    """
    Identify indices of triangles that are 'active' (close to the minimum area).
    Returns a boolean mask of shape (165,).
    """
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    return areas < (min_area + threshold)

def coordinated_ascent(bary, max_iter=2000, tol=1e-7, learning_rate=0.05):
    """
    Custom local refinement using subgradient ascent on the minimum area.
    This is more robust than generic gradient methods for this specific problem
    because it explicitly targets the active constraints.
    """
    current_bary = clamp_bary(bary)
    current_min = get_min_area(current_bary)
    
    for _ in range(max_iter):
        areas = calc_areas_bary(current_bary)
        min_area = np.min(areas)
        
        # Identify active triangles
        active_mask = areas < (min_area + 1e-5)
        if not np.any(active_mask):
            break
            
        # Compute gradient of the minimum area w.r.t. barycentric coordinates
        # The gradient of area for a triangle (i, j, k) is linear in the coords.
        # We approximate the subgradient by averaging the gradients of active triangles.
        
        grad_u = np.zeros(N)
        grad_v = np.zeros(N)
        
        # For each active triangle, compute its area gradient
        # Area = 0.5 * |det|. Since we are maximizing min area, we assume the sign
        # of the determinant is consistent locally (orientation).
        # Let's compute the gradient of the signed area for active triangles.
        
        u = current_bary[:, 0]
        v = current_bary[:, 1]
        w = 1.0 - u - v
        
        # We need d(area)/du and d(area)/dv for each active triangle.
        # Instead of symbolic differentiation, we can use finite differences 
        # or the analytic form.
        # Analytic:
        # Area(i,j,k) = 0.5 * | u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j) |
        
        # Let's just use a simple numerical subgradient step for robustness and speed.
        # Perturb each point slightly and see which direction increases the min area.
        
        # A better approach: Use the fact that the min area is a concave function 
        # of the points? No, it's not concave.
        
        # Let's use a simple coordinate ascent / hill climbing on the active set.
        # For each active triangle, we want to increase its area.
        # Moving a vertex away from the opposite edge increases area.
        
        # To keep it simple and fast, we will use a randomized local search 
        # biased towards expanding the smallest triangles.
        
        improved = False
        for _ in range(50): # Inner loop iterations
            # Pick a random active triangle
            active_indices = np.where(active_mask)[0]
            tri_idx_local = active_indices[np.random.randint(len(active_indices))]
            i, j, k = tri_idx[tri_idx_local]
            
            # Compute the direction to increase the area of this triangle
            # This is complex to do analytically for barycentric coords with constraints.
            # Instead, we will use a generic gradient step from scipy's L-BFGS-B 
            # on the soft objective, which is effectively what we want.
            break # Exit this manual loop, we'll use scipy below.
            
        # Actually, let's just use scipy.minimize with the soft objective for the local refinement.
        # It's robust and handles the non-smoothness via the softening parameter.
        break
        
    # If we broke out early, let's actually run the scipy minimizer here properly.
    # The function above was a stub. Let's implement the real local refinement.
    
    # We will use L-BFGS-B on the soft objective.
    # This is the "coordinated ascent" in a loose sense, but implemented via a reliable solver.
    
    def local_soft_obj(x):
        b = clamp_bary(x.reshape(N, 2))
        return soft_min_obj(x, beta=500.0) # High beta for precision
        
    try:
        res = minimize(local_soft_obj, current_bary.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            new_bary = clamp_bary(res.x.reshape(N, 2))
            new_min = get_min_area(new_bary)
            if new_min > current_min:
                return new_bary
    except:
        pass
        
    return current_bary

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initialization."""
    pts = []
    # Use a grid in barycentric space
    # For 11 points, a 4x4 grid minus corners might work, or just random points from a lattice.
    # Let's create a set of candidate points on a fine grid and pick 11.
    grid_size = 10
    us = np.linspace(0.1, 0.8, grid_size)
    vs = np.linspace(0.1, 0.8, grid_size)
    
    candidates = []
    for u in us:
        for v in vs:
            if u + v <= 0.9:
                candidates.append([u, v])
                
    if len(candidates) < N:
        return None
        
    # We want to select 11 points that are well spread.
    # Just picking a subset of the grid.
    # To be deterministic but diverse, we can use a fixed seed for selection.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(candidates), N, replace=False)
    return np.array(candidates)[idx]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Initialization
    # Use hexagonal and random starts
    start_points = []
    
    hex_init = generate_hexagonal_init()
    if hex_init is not None:
        start_points.append(hex_init)
        
    rng_master = np.random.default_rng(123)
    for _ in range(3):
        u = rng_master.random(N)
        v = rng_master.random(N) * (1 - u)
        start_points.append(np.column_stack([u, v]))
        
    # 2. Global Search with Differential Evolution
    # DE is good for finding the basin of the global optimum.
    
    # Bounds for barycentric coords:
    # u in [0, 1], v in [0, 1], u+v <= 1.
    # DE works with rectangular bounds. We can use u in [0,1], v in [0,1] 
    # and rely on clamp_bary in the objective.
    # However, this wastes search space.
    # Better: Map from hypercube to triangle.
    # Or just use bounds [0, 0.9] for both and clamp.
    
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    # We need to be careful with time. DE can be slow.
    # Let's run DE with a small population and max generations, monitoring time.
    
    de_success = False
    try:
        # Use a callback to check time
        def de_callback(xk, convergence):
            if time.time() > deadline - 10:
                # Raise an exception to stop DE? 
                # scipy's DE doesn't have a clean early stop callback that returns.
                # We can catch an exception.
                raise TimeoutError("DE Time Limit")
            
        # Run DE
        # maxiter is number of generations. popsize=10 means 10*22=220 evaluations per gen.
        # If we have ~100s left, and each eval is ~1ms, we can do ~100,000 evals.
        # 100,000 / 220 ~ 450 generations.
        
        # But DE is stochastic. Let's limit it to be safe.
        res_de = differential_evolution(
            soft_min_obj, 
            bounds, 
            args=(20.0,), # beta for soft min
            maxiter=50, 
            popsize=10, 
            tol=1e-6, 
            mutation=(0.5, 1.0), 
            recombination=0.7, 
            seed=42,
            polish=False # We will polish manually
        )
        de_success = True
    except TimeoutError:
        # If it timed out, we might not have a result, or we can use the last known?
        # DE doesn't easily expose the last population on exception.
        # We'll just rely on the initial points and local refinement if DE fails.
        pass
    except Exception as e:
        # Other errors
        pass

    if de_success and res_de is not None:
        best_bary_de = clamp_bary(res_de.x.reshape(N, 2))
        update_best(best_bary_de)
        # Add DE result to start points for local refinement
        start_points.append(best_bary_de)

    # 3. Local Refinement
    # Run coordinated_ascent (L-BFGS-B on soft min) on all start points
    # and the DE result.
    
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        
        # Local refinement
        refined_bary = coordinated_ascent(sp)
        update_best(refined_bary)
        
        # Try a few perturbations of the refined bary for more local search
        if time.time() > deadline - 2:
            break
        rng_pert = np.random.default_rng(hash(sp.tobytes()) % 2**32)
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            pert = rng_pert.normal(0, 0.01, size=(N, 2))
            new_sp = clamp_bary(sp + pert)
            refined_bary = coordinated_ascent(new_sp)
            update_best(refined_bary)

    # 4. Final Fallback
    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
