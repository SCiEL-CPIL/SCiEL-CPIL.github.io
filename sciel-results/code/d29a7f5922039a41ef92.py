import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
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

def generate_k3_lattice_plus():
    """
    Generate K=3 lattice (10 pts) + 11th point at various positions.
    K=3 lattice has points at i/3, j/3 for i,j >= 0, i+j <= 3.
    Points: (0,0), (1/3,0), (2/3,0), (1,0), (0,1/3), (1/3,1/3), (2/3,1/3), (0,2/3), (1/3,2/3), (0,1)
    Total 10 points.
    """
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
        [0.4, 0.3],
        [0.3, 0.4],
        [0.33, 0.33], # centroid, but might be close to (1/3,1/3)
        [0.5, 0.1],
        [0.1, 0.5],
    ]
    
    candidates = []
    for ap in add_points:
        pts = list(lattice_pts)
        pts.append(ap)
        candidates.append(np.array(pts))
        
    return candidates

def generate_k4_lattice_minus():
    """
    Generate K=4 lattice (15 pts) and remove 4 points.
    K=4 lattice has 15 points.
    Remove interior points to maximize spread.
    Interior points: i>0, j>0, i+j<4.
    (1,1), (1,2), (2,1) are interior.
    Also (1,3), (3,1), (2,2) are edge/corner?
    Let's list all:
    (0,0), (1,0), (2,0), (3,0), (4,0)
    (0,1), (1,1), (2,1), (3,1)
    (0,2), (1,2), (2,2)
    (0,3), (1,3)
    (0,4)
    
    Interior (u>0, v>0, w>0 => u<1, v<1, u+v<1):
    (1/4, 1/4), (1/4, 2/4), (2/4, 1/4).
    These are indices 6, 8, 11 in row-major?
    Row 0: 0-4
    Row 1: 5-8 -> (0,1/4)=5, (1/4,1/4)=6, (2/4,1/4)=7, (3/4,1/4)=8
    Row 2: 9-11 -> (0,2/4)=9, (1/4,2/4)=10, (2/4,2/4)=11
    Row 3: 12-13 -> (0,3/4)=12, (1/4,3/4)=13
    Row 4: 14 -> (0,1)=14
    
    Interior: 6, 10, 7. (1/4,1/4), (1/4,2/4), (2/4,1/4).
    Remove these 3. Need to remove 1 more.
    Remove a corner, e.g., (0,0) index 0.
    """
    K = 4
    lattice_pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            u = i / K
            v = j / K
            lattice_pts.append([u, v])
    
    lattice_pts = np.array(lattice_pts)
    
    # Indices to remove: 6, 10, 7, 0
    remove_idx = [6, 10, 7, 0]
    keep_idx = [i for i in range(len(lattice_pts)) if i not in remove_idx]
    kept_pts = lattice_pts[keep_idx]
    
    return [kept_pts]

def generate_hexagonal_template():
    """
    Generate a hexagonal lattice template with 11 points.
    """
    pts = []
    s = 0.25
    for i in range(5):
        for j in range(5):
            u = i * s
            v = j * s
            if u >= 0 and v >= 0 and u + v <= 1:
                pts.append([u, v])
    
    if len(pts) > N:
        pts = np.array(pts)
        selected = [0, 1, 2]
        remaining = list(range(3, len(pts)))
        while len(selected) < N and remaining:
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                min_dist = min(np.linalg.norm(pts[idx] - pts[s]) for s in selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        pts = pts[selected]
    elif len(pts) < N:
        while len(pts) < N:
            u = 0.5
            v = 0.5
            pts.append([u, v])
    
    return np.array(pts[:N])

def deformable_lattice_params_to_bary(params):
    """
    Convert deformation parameters to barycentric coordinates.
    """
    template = generate_hexagonal_template()
    
    scale = params[0]
    shear_x = params[1]
    shear_y = params[2]
    
    bary = template.copy()
    
    center_u = np.mean(bary[:, 0])
    center_v = np.mean(bary[:, 1])
    
    bary[:, 0] = center_u + scale * (bary[:, 0] - center_u)
    bary[:, 1] = center_v + scale * (bary[:, 1] - center_v)
    
    bary[:, 0] += shear_x * bary[:, 1]
    bary[:, 1] += shear_y * bary[:, 0]
    
    for i in range(N):
        bary[i, 0] += params[3 + 2*i]
        bary[i, 1] += params[3 + 2*i + 1]
    
    return bary

def deformable_obj(params):
    """
    Objective for deformable lattice optimization.
    """
    bary = deformable_lattice_params_to_bary(params)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    return -np.min(areas)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with graduated beta
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.6:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.9:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
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

    # 1. Generate multiple templates and refine
    templates = []
    
    # Template 1: Standard hexagonal
    t1 = generate_hexagonal_template()
    templates.append(t1)
    
    # Template 2: K=3 Lattice + 1 (from inspiration)
    k3_templates = generate_k3_lattice_plus()
    templates.extend(k3_templates)
    
    # Template 3: K=4 Lattice - 4 (from inspiration)
    k4_templates = generate_k4_lattice_minus()
    templates.extend(k4_templates)
    
    # Template 4: Random but structured
    rng = np.random.default_rng(42)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    templates.append(np.column_stack([u, v]))
    
    # Refine each template
    for i, template in enumerate(templates):
        if time.time() > deadline - 30:
            break
        refined = refine_local(template, time_limit=min(8.0, deadline - start_time - 20))
        update_best(refined)
    
    # 2. Deformable lattice optimization (low-dimensional global search)
    if time.time() < deadline - 40:
        try:
            n_params = 3 + 2 * N
            bounds_deform = [(0.5, 1.5), (-0.2, 0.2), (-0.2, 0.2)] + [(-0.1, 0.1)] * (2 * N)
            
            result_deform = differential_evolution(
                deformable_obj,
                bounds_deform,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-5,
                polish=False
            )
            
            best_params = result_deform.x
            bary_deform = deformable_lattice_params_to_bary(best_params)
            bary_deform = clamp_bary(bary_deform)
            
            refined_deform = refine_local(bary_deform, time_limit=8.0)
            update_best(refined_deform)
        except Exception:
            pass
    
    # 3. Basin hopping on best found
    if best_bary is not None and time.time() < deadline - 20:
        bh_time = min(20.0, deadline - start_time - 15.0)
        if bh_time > 5:
            try:
                start_bh = time.time()
                rng = np.random.default_rng(123)
                current_bary = best_bary.copy()
                best_bh_bary = best_bary.copy()
                best_bh_val = best_val
                
                temperature = 0.03
                pert_scale = 0.05
                
                while time.time() - start_bh < bh_time:
                    noise = rng.normal(0, pert_scale, current_bary.shape)
                    perturbed = clamp_bary(current_bary + noise)
                    
                    refined = refine_local(perturbed, time_limit=min(3.0, bh_time - (time.time() - start_bh)))
                    refined_val = -hard_min_obj(refined.flatten())
                    
                    if refined_val > best_bh_val:
                        best_bh_val = refined_val
                        best_bh_bary = refined.copy()
                        current_bary = refined.copy()
                    elif np.log(rng.random()) < (refined_val - best_bh_val) / temperature:
                        current_bary = refined.copy()
                    
                    temperature *= 0.97
                    pert_scale *= 0.99
                    
                    if time.time() - start_bh > bh_time * 0.95:
                        break
                
                update_best(best_bh_bary)
            except Exception:
                pass
    
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
