from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0

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
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

def calc_min_area_cartesian(points):
    """
    Calculate the minimum area of all triangles formed by N points.
    points: (N, 2) array of Cartesian coordinates.
    """
    p_i = points[tri_idx[:, 0]]
    p_j = points[tri_idx[:, 1]]
    p_k = points[tri_idx[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

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

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def generate_lattice_init():
    """Generate a triangular lattice initialization with 10 points + 1 center."""
    pts = []
    for i in range(5):
        for j in range(5 - i):
            pts.append([i / 4.0, j / 4.0])
    pts = np.array(pts)
    
    center = np.array([1/3, 1/3])
    pts = np.vstack([pts, center])
    
    return pts

def generate_hexagonal_init():
    """Generate a hexagonal packing-inspired initialization."""
    s = 0.25
    hex_pts = []
    for k in range(6):
        angle = k * np.pi / 3
        hex_pts.append([s * np.cos(angle), s * np.sin(angle)])
    
    inner_pts = []
    inner_pts.append([0.0, 0.0])
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4
        inner_pts.append([0.15 * np.cos(angle), 0.15 * np.sin(angle)])
    
    all_pts = np.array(hex_pts + inner_pts)
    bary = cart_to_bary(all_pts)
    bary = clamp_bary(bary)
    return bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, scale=0.05, seed=None):
    """Generate perturbed version of base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, scale, base.shape)
    perturbed = base + noise
    return clamp_bary(perturbed)

def refine_local(bary_init, time_limit=5.0, use_soft_min=False):
    """
    Refine a configuration using coordinated ascent on active constraints.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    def get_active_triangles(bary, tol=1e-6):
        """Return indices of triangles achieving (near) the minimum area."""
        cart = bary_to_cart(bary)
        p_i = cart[tri_idx[:, 0]]
        p_j = cart[tri_idx[:, 1]]
        p_k = cart[tri_idx[:, 2]]
        cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
                (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        active = np.where(areas <= min_area + tol)[0]
        return active, min_area
    
    def point_gradient(bary, active_tris, point_idx):
        """
        Compute a gradient-like direction for point_idx that increases
        the minimum area among active triangles.
        """
        cart = bary_to_cart(bary)
        grad = np.zeros(2)
        
        for tri in active_tris:
            i, j, k = tri_idx[tri]
            pi = cart[i]
            pj = cart[j]
            pk = cart[k]
            
            # Area = 0.5 * |cross(pj-pi, pk-pi)|
            # For a fixed triangle, the gradient w.r.t. each vertex:
            # d(area)/d(pi) = -0.5 * cross_sign * perp(pj-pk)
            # where perp(a) = (-a[1], a[0])
            
            cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1])
            sign = 1.0 if cross_val >= 0 else -1.0
            
            # Gradient w.r.t. pi: -0.5 * sign * perp(pj - pk)
            # perp(pj-pk) = (-(pj[1]-pk[1]), pj[0]-pk[0])
            grad_pi = -0.5 * sign * np.array([-(pj[1] - pk[1]), pj[0] - pk[0]])
            # Gradient w.r.t. pj: 0.5 * sign * perp(pk - pi)
            grad_pj = 0.5 * sign * np.array([-(pk[1] - pi[1]), pk[0] - pi[0]])
            # Gradient w.r.t. pk: -0.5 * sign * perp(pj - pi)
            grad_pk = -0.5 * sign * np.array([-(pj[1] - pi[1]), pj[0] - pi[0]])
            
            if point_idx == i:
                grad += grad_pi
            elif point_idx == j:
                grad += grad_pj
            elif point_idx == k:
                grad += grad_pk
        
        # Convert Cartesian gradient to barycentric gradient
        # x = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
        # y = H*(1-u-v) = H - H*u - H*v
        # dx/du = -0.5, dx/dv = 0.5
        # dy/du = -H, dy/dv = -H
        # So grad_bary_u = grad_x * (-0.5) + grad_y * (-H)
        #    grad_bary_v = grad_x * (0.5) + grad_y * (-H)
        grad_bary_u = grad[0] * (-0.5) + grad[1] * (-H)
        grad_bary_v = grad[0] * (0.5) + grad[1] * (-H)
        
        return np.array([grad_bary_u, grad_bary_v])
    
    # Coordinated ascent: iterate over points, take small steps in gradient direction
    learning_rate = 0.01
    max_iter = 200
    
    for iteration in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        
        active_tris, min_area = get_active_triangles(bary_current)
        
        if len(active_tris) == 0:
            break
        
        # Pick a point to move (round-robin or random)
        for pt_idx in range(N):
            if time.time() > start_time + time_limit:
                break
            
            grad = point_gradient(bary_current, active_tris, pt_idx)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                continue
            
            # Take a step in the gradient direction
            step = learning_rate * grad / grad_norm
            new_bary = bary_current.copy()
            new_bary[pt_idx] += step
            new_bary = clamp_bary(new_bary)
            
            # Check if improvement
            new_val = calc_min_area_cartesian(bary_to_cart(new_bary))
            if new_val > min_area + 1e-10:
                bary_current = new_bary
    
    # Final Nelder-Mead polish
    if time.time() < start_time + time_limit:
        try:
            result = minimize(
                hard_min_obj_bary,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            bary_current = clamp_bary(result.x.reshape(N, 2))
        except Exception:
            pass
    
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    good_candidates = []
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def track_good(bary):
        nonlocal best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        if len(good_candidates) < 5 or val > best_val * 0.98:
            good_candidates.append((val, bary.copy()))
            good_candidates.sort(key=lambda x: -x[0])
            del good_candidates[5:]

    # 1. Lattice Init
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=8.0)
    update_best(refined_lat)
    track_good(refined_lat)
    
    # 1b. Hexagonal Init
    if time.time() < deadline - 60:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=6.0)
        update_best(refined_hex)
        track_good(refined_hex)
    
    # 2. Global Search with Genetic Algorithm
    if time.time() < deadline - 40:
        try:
            rng_ga = np.random.default_rng(42)
            pop_size = 20
            max_gen = 30
            mutation_rate = 0.3
            crossover_rate = 0.7
            
            # Initialize population
            population = []
            # Seed with best so far
            if best_bary is not None:
                population.append(best_bary.copy())
            # Add lattice and random inits
            population.append(generate_lattice_init())
            for _ in range(pop_size - 2):
                population.append(generate_random_init(rng_ga.integers(0, 10000)))
            
            for gen in range(max_gen):
                if time.time() > deadline - 45:
                    break
                
                # Evaluate population
                scored = []
                for bary in population:
                    bary = clamp_bary(bary)
                    cart = bary_to_cart(bary)
                    val = calc_min_area_cartesian(cart)
                    scored.append((val, bary))
                
                scored.sort(key=lambda x: -x[0])
                top_val, top_bary = scored[0]
                if top_val > best_val:
                    best_val = top_val
                    best_bary = top_bary.copy()
                
                # Selection: keep top half
                survivors = [b for _, b in scored[:pop_size // 2]]
                
                # Generate offspring
                offspring = []
                while len(offspring) < pop_size // 2:
                    if time.time() > deadline - 45:
                        break
                    
                    if rng_ga.random() < crossover_rate and len(survivors) >= 2:
                        # Crossover
                        p1, p2 = rng_ga.choice(len(survivors), size=2, replace=False)
                        a, b = survivors[p1], survivors[p2]
                        mask = rng_ga.random(a.shape) < 0.5
                        child = np.where(mask, a, b)
                    else:
                        # Clone a random survivor
                        child = survivors[rng_ga.integers(0, len(survivors))].copy()
                    
                    # Mutation
                    if rng_ga.random() < mutation_rate:
                        child = child + rng_ga.normal(0, 0.1, child.shape)
                        child = clamp_bary(child)
                    
                    offspring.append(child)
                
                population = survivors + offspring
        except Exception:
            pass
    
    # 3. Basin Hopping (manual loop for time safety)
    if time.time() < deadline - 10 and best_bary is not None:
        current_bary = best_bary.copy()
        current_val = best_val
        rng_bh = np.random.default_rng(42)
        
        while time.time() < deadline - 5:
            # Adaptive noise scale: start large, shrink as we get closer to deadline
            remaining = deadline - 5 - time.time()
            noise_scale = 0.3 * (0.5 + 0.5 * remaining / max(1, deadline - 10 - start_time))
            noise = rng_bh.normal(0, noise_scale, current_bary.shape)
            new_bary = clamp_bary(current_bary + noise)
            
            refined = refine_local(new_bary, time_limit=min(4.0, deadline - 5 - time.time()))
            val = calc_min_area_cartesian(bary_to_cart(refined))
            
            if val > current_val:
                current_val = val
                current_bary = refined
                update_best(refined)
                track_good(refined)
            else:
                # Occasionally jump to a random good candidate
                if good_candidates and rng_bh.random() < 0.1:
                    _, cand_bary = good_candidates[rng_bh.integers(0, len(good_candidates))]
                    current_bary = cand_bary.copy()
                    current_val = calc_min_area_cartesian(bary_to_cart(current_bary))
    
    cart = bary_to_cart(best_bary)
    return cart
