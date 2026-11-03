"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

def _bc_to_cartesian(bc):
    """Convert (b,c) pairs to cartesian. a = 1-b-c. x = b + 0.5*c, y = c*S3H."""
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])

def _cartesian_to_bc(pts):
    """Convert cartesian to (b,c). c = y/S3H, b = x - 0.5*c."""
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])

def _all_areas_from_bc(bc):
    """Compute all triangle areas from (b,c) representation."""
    pts = _bc_to_cartesian(bc)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _min_area_from_bc(bc):
    """Compute min triangle area from (b,c) representation."""
    return np.min(_all_areas_from_bc(bc))

def _soft_min_area_from_bc(bc, beta):
    """Soft minimum of triangle areas."""
    areas = _all_areas_from_bc(bc)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _bc_penalty(bc):
    """Penalty for barycentric coords outside valid range."""
    b = bc[:, 0]
    c = bc[:, 1]
    a = 1.0 - b - c
    pen = 0.0
    neg_b = np.minimum(b, 0.0)
    neg_c = np.minimum(c, 0.0)
    neg_a = np.minimum(a, 0.0)
    pos_b = np.maximum(b - 1.0, 0.0)
    pos_c = np.maximum(c - 1.0, 0.0)
    pos_a = np.maximum(a - 1.0, 0.0)
    pen = 1000.0 * (np.sum(neg_b**2) + np.sum(neg_c**2) + np.sum(neg_a**2) +
                    np.sum(pos_b**2) + np.sum(pos_c**2) + np.sum(pos_a**2))
    return pen

def _objective_bc(flat, beta):
    """Objective in (b,c) space: minimize -soft_min + penalty."""
    bc = flat.reshape(-1, 2)
    ma = _soft_min_area_from_bc(bc, beta)
    pen = _bc_penalty(bc)
    return -ma + pen

def _objective_bc_hard(flat):
    """Hard min objective in (b,c) space."""
    bc = flat.reshape(-1, 2)
    ma = _min_area_from_bc(bc)
    pen = _bc_penalty(bc)
    return -ma + pen

def _project_bc(bc):
    """Project (b,c) into valid barycentric region: b>=0, c>=0, b+c<=1."""
    bc = bc.copy()
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1)
    s = b + c
    mask = s > 1.0
    if np.any(mask):
        scale = 1.0 / s[mask]
        b[mask] *= scale
        c[mask] *= scale
    bc[:, 0] = b
    bc[:, 1] = c
    return bc

def _kinetic_anneal(bc, n_iter=5000, t0=0.08, t1=0.001, rng=None):
    """Kinetic Annealing with Topological Collision Resolution.
    
    This is a structural improvement: instead of generic perturbations, we
    explicitly identify the minimum-area triangle and apply a directed push
    to its vertices, combined with a periodic symmetry-snap heuristic.
    """
    if rng is None:
        rng = np.random.default_rng()
    bc = bc.copy()
    current_area = _min_area_from_bc(bc)
    best_bc = bc.copy()
    best_area = current_area
    
    # Symmetry axes for equilateral triangle in (b,c) space
    # The centroid is at (1/3, 1/3)
    centroid = np.array([1.0/3, 1.0/3])
    
    for it in range(n_iter):
        t = t0 * (t1 / t0) ** (it / n_iter)
        
        # Periodic symmetry snap: every 200 iterations, try snapping to 3-fold symmetry
        if it % 200 == 0 and it > 0:
            # Try to enforce 3-fold rotational symmetry around centroid
            # In barycentric coords, 120-degree rotation maps:
            # (a,b,c) -> (c,a,b) in full barycentric
            # In (b,c) coords: b_new = c, c_new = a = 1-b-c
            # So (b,c) -> (c, 1-b-c)
            bc_sym = bc.copy()
            b = bc[:, 0]
            c = bc[:, 1]
            # Apply rotation to each point
            bc_rot1 = np.column_stack([c, 1.0 - b - c])
            bc_rot2 = np.column_stack([1.0 - b - c, b])
            bc_rot1 = _project_bc(bc_rot1)
            bc_rot2 = _project_bc(bc_rot2)
            
            # Average the three orientations to get a more symmetric configuration
            bc_avg = (bc + bc_rot1 + bc_rot2) / 3.0
            bc_avg = _project_bc(bc_avg)
            avg_area = _min_area_from_bc(bc_avg)
            
            if avg_area > current_area or rng.random() < np.exp((avg_area - current_area) / max(t, 1e-10)):
                bc = bc_avg
                current_area = avg_area
                if current_area > best_area:
                    best_area = current_area
                    best_bc = bc.copy()
            continue
        
        # Compute all areas and find the minimum-area triangle
        areas = _all_areas_from_bc(bc)
        min_idx = np.argmin(areas)
        i, j, k = _TRIPLETS[min_idx]
        
        # Topological Collision Resolution: push the three vertices of the
        # minimum-area triangle apart
        # Compute the triangle's centroid in (b,c) space
        tri_centroid = (bc[i] + bc[j] + bc[k]) / 3.0
        
        # Direction from triangle centroid to each vertex
        disp = np.zeros_like(bc)
        for idx in [i, j, k]:
            d = bc[idx] - tri_centroid
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                # Push vertex away from triangle centroid
                force = 0.5 * t / (dist + 0.01)
                disp[idx] += d / dist * force
        
        # Also apply a small generic perturbation to a random point
        ridx = rng.integers(0, N)
        perturbation = rng.normal(0, 0.01 * t / t0, size=2)
        disp[ridx] += perturbation
        
        bc_new = bc + disp
        bc_new = _project_bc(bc_new)
        
        new_area = _min_area_from_bc(bc_new)
        delta = new_area - current_area
        
        if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
            bc = bc_new
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_bc = bc.copy()
    
    return best_bc

def _sa_refine(bc, n_iter=2000, t0=0.05, t1=0.001, rng=None):
    """Simulated annealing refinement in barycentric space."""
    if rng is None:
        rng = np.random.default_rng()
    bc = bc.copy()
    current_area = _min_area_from_bc(bc)
    best_bc = bc.copy()
    best_area = current_area
    
    for it in range(n_iter):
        t = t0 * (t1 / t0) ** (it / n_iter)
        
        i = rng.integers(0, N)
        bc_new = bc.copy()
        perturbation = rng.normal(0, 0.02, size=2)
        bc_new[i] = bc[i] + perturbation
        bc_new = _project_bc(bc_new)
        
        new_area = _min_area_from_bc(bc_new)
        delta = new_area - current_area
        
        if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
            bc = bc_new
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_bc = bc.copy()
    
    return best_bc

def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_bc = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bc):
        nonlocal best_bc, best_area
        bc = _project_bc(bc.copy())
        area = _min_area_from_bc(bc)
        if area > best_area:
            best_area = area
            best_bc = bc.copy()

    # ---- Initial configurations in barycentric (b,c) space ----
    init_bc = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.3, 0.1],
        [0.7, 0.1],
        [0.5, 0.4],
        [0.2, 0.3],
    ], dtype=float)
    init_bc = _project_bc(init_bc)
    _update_best(init_bc)

    def _random_bc():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        return np.array([u, v])

    def _make_ring_bc(scale=0.3):
        bc = np.zeros((N, 2))
        bc[0] = [0.0, 0.0]
        bc[1] = [1.0, 0.0]
        bc[2] = [0.0, 1.0]
        bc[3] = [0.5, 0.0]
        bc[4] = [0.5, 0.5]
        bc[5] = [0.0, 0.5]
        cx, cy = 1/3, 1/3
        for i in range(6, N):
            angle = 2 * np.pi * (i - 6) / (N - 6)
            r = scale * (0.5 + 0.5 * (i - 6) / (N - 6))
            b = cx + r * np.cos(angle)
            c = cy + r * np.sin(angle) * 0.7
            bc[i] = [b, c]
        return _project_bc(bc)

    bases_bc = [init_bc.copy()]
    bases_bc.append(_make_ring_bc(0.4))
    bases_bc.append(_make_ring_bc(0.25))
    for _ in range(6):
        bases_bc.append(init_bc + rng.normal(0, 0.15, size=(N, 2)))
    for _ in range(4):
        bases_bc.append(np.array([_random_bc() for _ in range(N)]))
    if best_bc is not None:
        bases_bc.append(best_bc.copy())

    # ---- Phase A: Differential Evolution in (b,c) space ----
    if _elapsed_ok(55):
        try:
            def _de_obj(flat_bc):
                bc = flat_bc.reshape(-1, 2)
                ma = _min_area_from_bc(bc)
                pen = _bc_penalty(bc)
                return -ma + pen

            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj,
                bounds_de,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            de_bc = _project_bc(de_result.x.reshape(-1, 2))
            _update_best(de_bc)
        except Exception:
            pass

    # ---- Phase B: Soft-min gradient optimization with adaptive beta ----
    n_starts = min(8, len(bases_bc))
    for s in range(n_starts):
        if not _elapsed_ok(45):
            break
        base = bases_bc[s].copy()
        bc = base + rng.normal(0, 0.02, size=(N, 2))
        bc = _project_bc(bc)
        _update_best(bc)

        betas = [30.0, 100.0, 300.0, 1000.0]
        for beta in betas:
            if not _elapsed_ok(40):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_bc(x, b),
                    bc.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                bc_new = _project_bc(res.x.reshape(-1, 2))
                _update_best(bc_new)
                bc = bc_new
            except Exception:
                pass

    # ---- Phase C: Kinetic Annealing with Topological Collision Resolution ----
    if best_bc is not None and _elapsed_ok(35):
        for ka_run in range(4):
            if not _elapsed_ok(20):
                break
            bc_ka = _kinetic_anneal(best_bc, n_iter=5000, t0=0.08, t1=0.001, rng=rng)
            _update_best(bc_ka)

    # ---- Phase D: Simulated Annealing refinement ----
    if best_bc is not None and _elapsed_ok(25):
        for sa_run in range(4):
            if not _elapsed_ok(15):
                break
            bc_sa = _sa_refine(best_bc, n_iter=3000, t0=0.03, t1=0.0005, rng=rng)
            _update_best(bc_sa)

    # ---- Phase E: Nelder-Mead polish ----
    if best_bc is not None and _elapsed_ok(20):
        for _ in range(6):
            if not _elapsed_ok(10):
                break
            try:
                res = minimize(
                    _objective_bc_hard,
                    best_bc.flatten(),
                    method='Nelder-Mead',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                bc_res = _project_bc(res.x.reshape(-1, 2))
                _update_best(bc_res)
            except Exception:
                pass

    # ---- Phase F: Final high-precision polish ----
    if best_bc is not None and _elapsed_ok(10):
        for _ in range(3):
            if not _elapsed_ok(5):
                break
            try:
                res = minimize(
                    _objective_bc_hard,
                    best_bc.flatten(),
                    method='Nelder-Mead',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                bc_res = _project_bc(res.x.reshape(-1, 2))
                _update_best(bc_res)
            except Exception:
                pass

    if best_bc is None:
        pts = np.array([_random_bc() for _ in range(N)])
        return _bc_to_cartesian(pts)

    best_bc = _project_bc(best_bc)
    return _bc_to_cartesian(best_bc)
# EVOLVE-BLOCK-END
