import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 142.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_TOTAL = np.sqrt(3) / 4.0

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

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

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1]. copy()
    
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal-like lattice initialization.
    For 11 points, use a hexagonal packing arrangement.
    """
    candidates = []
    
    # Hexagonal lattice points within the triangle
    # Use a grid with hexagonal offset
    pts = []
    
    # Row 0: 4 points along the bottom
    for i in range(4):
        pts.append([i / 3.0, 0.0])
    
    # Row 1: 3 points offset
    for i in range(3):
        pts.append([(i + 0.5) / 3.0, H / 3.0])
    
    # Row 2: 2 points
    for i in range(2):
        pts.append([(i + 1.0) / 3.0, 2 * H / 3.0])
    
    # Row 3: 1 point at top
    pts.append([0.5, H])
    
    pts = np.array(pts)
    candidates.append(pts)
    
    # Add some variations with slight perturbations
    rng = np.random.default_rng(123)
    for _ in range(3):
        perturbed = pts + rng.normal(0, 0.02, size=pts.shape)
        perturbed = clamp_bary(perturbed)
        candidates.append(perturbed)
    
    return candidates

def generate_symmetric_init():
    """
    Generate a symmetric initialization respecting the 3-fold symmetry.
    This often leads to better local optima for Heilbronn problems.
    """
    candidates = []
    
    # Symmetric configuration:
    # 1 point at center
    # 3 points forming a small triangle around center
    # 6 points forming a larger triangle around center (or along edges)
    
    # A known good symmetric start for 11 points:
    # Center: (1/3, 1/3)
    # Ring 1 (radius r1, angles 0, 120, 240): 3 points
    # Ring 2 (radius r2, angles 60, 180, 300): 3 points
    # Vertices: 3 points
    
    # Let's try a specific symmetric layout
    c = np.array([1.0/3, 1.0/3])
    
    # 3 vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, H])
    
    # 3 points on mid-edges? No, that's 6.
    # Let's use 1 center, 3 inner, 6 outer?
    
    # Try:
    # 1 center
    # 3 points at distance 0.2 from center, angles 0, 120, 240
    # 6 points at distance 0.4 from center, angles 30, 90, 150, 210, 270, 330
    # Total 10? No, 1+3+6=10. Need 11.
    
    # Alternative:
    # 1 center
    # 3 points at distance 0.25
    # 6 points at distance 0.45
    # 1 point? No.
    
    # Let's just generate a symmetric perturbation of the hexagonal one,
    # or use a specific known good config if available.
    # For now, let's create a "star" pattern.
    
    pts = [c]
    
    # 3 points closer to center
    r1 = 0.2
    angles1 = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    for a in angles1:
        pts.append(c + r1 * np.array([np.cos(a), np.sin(a)]))
        
    # 6 points further out
    r2 = 0.4
    angles2 = [np.pi/2 + np.pi/6, np.pi/2 + np.pi/6 + np.pi/3, 
               np.pi/2 + np.pi/6 + 2*np.pi/3, np.pi/2 + np.pi/6 + 3*np.pi/3,
               np.pi/2 + np.pi/6 + 4*np.pi/3, np.pi/2 + np.pi/6 + 5*np.pi/3]
    for a in angles2:
        pts.append(c + r2 * np.array([np.cos(a), np.sin(a)]))
        
    # 1 point? We have 10. Add one more?
    # Maybe move one of the outer ones to a vertex?
    # Or just add a 4th inner point?
    
    # Let's stick to the hexagonal init for robustness, but add a "tight" symmetric one.
    # Actually, let's just return the hexagonal ones plus a perturbed symmetric one.
    
    return candidates

def generate_k3_lattice_plus():
    """
    Generate K=3 lattice (10 pts) + 11th point.
    """
    candidates = []
    K = 3
    lattice_pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            u = i / K
            v = j / K
            lattice_pts.append([u, v])
    
    lattice_pts = np.array(lattice_pts)
    
    # Add 11th point at various positions
    add_points = [
        [0.5, 0.25],
        [0.25, 0.5],
        [0.25, 0.25],
        [0.4, 0.4],
        [0.3, 0.3]
    ]
    
    for ap in add_points:
        pts = list(lattice_pts)
        pts.append(ap)
        candidates.append(np.array(pts))
        
    return candidates

def generate_random_init(rng, n_candidates=5):
    """Generate random initial configurations."""
    candidates = []
    for _ in range(n_candidates):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        candidates.append(np.column_stack([u, v]))
    return candidates

def active_triangles(bary, areas, tol=1e-8):
    """
    Identify triangles that are 'active' (close to the minimum area).
    Returns indices of triangles and the points involved.
    """
    min_area = np.min(areas)
    active_mask = areas < (min_area + tol)
    active_idx = np.where(active_mask)[0]
    
    # Extract points involved in these triangles
    points_involved = set()
    for idx in active_idx:
        points_involved.add(tri_idx[idx, 0])
        points_involved.add(tri_idx[idx, 1])
        points_involved.add(tri_idx[idx, 2])
        
    return active_idx, list(points_involved)

def coordinated_ascent(bary, time_limit=5.0, max_iter=500, step_size=0.05):
    """
    Structural change: Coordinated Ascent.
    Instead of generic gradient descent on -min(areas), we:
    1. Find active constraints (min area triangles).
    2. Propose small moves for points involved in these constraints.
    3. Accept move if it increases the minimum area.
    4. Decay step size if no improvement.
    """
    start_time = time.time()
    bary = bary.copy()
    areas = calc_areas_bary(bary)
    current_min = np.min(areas)
    
    rng = np.random.default_rng(42)
    
    no_improve_count = 0
    step = step_size
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        active_idx, active_pts = active_triangles(bary, areas, tol=1e-6)
        
        if not active_pts:
            break
            
        # Pick a random point from active points
        p = rng.choice(active_pts)
        
        # Generate a random direction in barycentric space (tangent to constraint u+v+w=1)
        # We can just perturb u and v and clamp.
        du = rng.normal(0, step)
        dv = rng.normal(0, step)
        
        new_bary = bary.copy()
        new_bary[p, 0] += du
        new_bary[p, 1] += dv
        new_bary = clamp_bary(new_bary)
        
        new_areas = calc_areas_bary(new_bary)
        new_min = np.min(new_areas)
        
        if new_min > current_min + 1e-12:
            bary = new_bary
            areas = new_areas
            current_min = new_min
            no_improve_count = 0
            # Slightly increase step if we found a good move? Or keep constant?
            # Keep constant to avoid oscillation, decay on stagnation.
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                step *= 0.8
                no_improve_count = 0
                if step < 1e-5:
                    break
                    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coordinated Ascent (New Structural Change)
    # This is more effective for max-min problems than generic gradient methods
    if time.time() - start_time < time_limit * 0.7:
        bary_current = coordinated_ascent(bary_current, time_limit=time_limit * 0.7)
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
        
    # Stage 2: SLSQP polish for precision
    # Use hard_min_obj which is non-smooth, but SLSQP handles it okay with finite diff
    # Or use soft_min_obj for smoothness? Soft min is better for SLSQP.
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,), # Lower beta for smoother landscape near optimum
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    # Final Coordinated Ascent with very small steps to polish
    if time.time() - start_time < time_limit:
        bary_current = coordinated_ascent(bary_current, time_limit=time.time() - start_time, step_size=0.01)

    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    candidates = []
    
    # 1. Hexagonal lattice based candidates (primary)
    candidates.extend(generate_hexagonal_init())
    
    # 2. K=3 Lattice based candidates (secondary)
    candidates.extend(generate_k3_lattice_plus())
    
    # 3. Random candidates
    rng = np.random.default_rng(42)
    candidates.extend(generate_random_init(rng, n_candidates=4))

    # Process candidates with time allocation
    n_cands = len(candidates)
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        remaining = deadline - 20 - time.time()
        if remaining < 3:
            break
            
        # Allocate more time to earlier (better structured) candidates
        if i < 4:  # Hexagonal candidates get more time
            t_limit = min(10.0, remaining)
        elif i < 9:  # K=3 candidates
            t_limit = min(5.0, remaining)
        else:  # Random candidates
            t_limit = min(3.0, remaining)
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # Global Search with Differential Evolution if time permits
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            
            # Run differential evolution with moderate settings
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            remaining = deadline - 20 - time.time()
            if remaining > 5:
                refined_de = refine_local(de_bary, time_limit=min(10.0, remaining))
                update_best(refined_de)
        except Exception:
            pass

    # Second DE run with different seed if time permits
    if time.time() < deadline - 25:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_de2 = differential_evolution(
                de_obj,
                bounds,
                seed=123,
                maxiter=20,
                popsize=10,
                tol=1e-6,
                polish=False
            )
            
            de_bary2 = clamp_bary(result_de2.x.reshape(N, 2))
            remaining = deadline - 15 - time.time()
            if remaining > 4:
                refined_de2 = refine_local(de_bary2, time_limit=min(8.0, remaining))
                update_best(refined_de2)
        except Exception:
            pass

    # Final polish on best solution
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # Use coordinated ascent for final polish as it is more robust to non-smoothness
            bounds = [(0, 1)] * (N * 2)
            # First a quick SLSQP
            result_slsqp = minimize(
                soft_min_obj,
                best_bary.flatten(),
                args=(50.0,),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                # Final coordinated ascent
                refined_ca = coordinated_ascent(best_bary, time_limit=2.0, step_size=0.005)
                update_best(refined_ca)
            except Exception:
                pass

    if best_bary is None:
        # Fallback
        best_bary = generate_hexagonal_init()[0]

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
