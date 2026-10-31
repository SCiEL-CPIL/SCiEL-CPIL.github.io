import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices for all C(11,3) = 165 triangles
TRIS = np.array(list(combinations(range(N), 3)))
TRI_A = TRIS[:, 0]
TRI_B = TRIS[:, 1]
TRI_C = TRIS[:, 2]


def bary_to_cart(bary):
    """bary (n,3) -> cartesian (n,2) for triangle (0,0),(1,0),(0.5,H)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = bary[:, 2] * H
    return np.column_stack([x, y])


def cart_to_bary(cart):
    c = cart[:, 1] / H
    b = cart[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def clamp_bary(bary):
    bary = np.clip(bary, 0.0, 1.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return bary / s


def all_areas_cart(cart):
    pa = cart[TRI_A]
    pb = cart[TRI_B]
    pc = cart[TRI_C]
    return 0.5 * np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                        - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))


def hard_min_bary(bary):
    bary = clamp_bary(bary)
    return float(np.min(all_areas_cart(bary_to_cart(bary))))


def soft_min_obj(flat, beta=120.0):
    """Smooth surrogate of min area (differentiable)."""
    bary = clamp_bary(flat.reshape(N, 3))
    areas = all_areas_cart(bary_to_cart(bary))
    m = areas.min()
    shifted = np.clip(-beta * (areas - m), -500, 500)
    logsum = np.log(np.sum(np.exp(shifted)))
    return m - logsum / beta


def _delaunay_smooth_obj(flat, beta=120.0, lam=0.03, gamma=40.0):
    """
    Hybrid objective: soft-min area PLUS a variance-regularization term on the
    smallest areas. Encourages a *balanced* set of critical triangles.
    """
    bary = clamp_bary(flat.reshape(N, 3))
    cart = bary_to_cart(bary)
    areas = all_areas_cart(cart)
    m = areas.min()
    shifted = np.clip(-beta * (areas - m), -500, 500)
    logsum = np.log(np.sum(np.exp(shifted)))
    softmin = m - logsum / beta

    # Delaunay-aware: penalize spread among the smallest triangles.
    try:
        tri = Delaunay(cart)
        d_areas = 0.5 * np.abs((cart[tri.simplices[:, 1], 0] - cart[tri.simplices[:, 0], 0])
                               * (cart[tri.simplices[:, 2], 1] - cart[tri.simplices[:, 0], 1])
                               - (cart[tri.simplices[:, 1], 1] - cart[tri.simplices[:, 0], 1])
                               * (cart[tri.simplices[:, 2], 0] - cart[tri.simplices[:, 0], 0]))
        k = min(20, len(d_areas))
        small = np.sort(d_areas)[:k]
        var_term = np.var(small)
    except Exception:
        small = np.sort(areas)[:20]
        var_term = np.var(small)

    # A soft-min of the *variance* (so it's smooth), scaled by lam.
    sm = small.min()
    sh2 = np.clip(-gamma * (small - sm), -500, 500)
    lse2 = np.log(np.sum(np.exp(sh2)))
    var_soft = sm - lse2 / gamma
    return softmin - lam * var_soft


def _neg_hybrid(flat, beta, lam, gamma):
    return -_delaunay_smooth_obj(flat, beta, lam, gamma)


def _neg_soft(flat, beta):
    return -soft_min_obj(flat, beta)


def _neg_hard(flat):
    return -hard_min_bary(flat)


def generate_hexagonal_init():
    pts = []
    pts.append([1.0, 0.0, 0.0])
    pts.append([0.0, 1.0, 0.0])
    pts.append([0.0, 0.0, 1.0])
    pts.append([0.5, 0.5, 0.0])
    pts.append([0.0, 0.5, 0.5])
    pts.append([0.5, 0.0, 0.5])
    pts.append([1 / 3, 1 / 3, 1 / 3])
    cx, cy, cz = 1 / 3, 1 / 3, 1 / 3
    r = 0.13
    for angle in [0, np.pi / 3, 2 * np.pi / 3, np.pi]:
        da = r * np.cos(angle)
        db = r * np.sin(angle)
        dc = -da - db
        p = [cx + da, cy + db, cz + dc]
        p = np.array(p)
        p = p / max(p.sum(), 1e-12)
        pts.append(p)
    return np.array(pts)


def generate_grid_init():
    pts = []
    for i in range(N):
        row = i // 4
        col = i % 4
        a = 1.0 - 0.25 * row - 0.18 * col
        b = 0.25 * row + 0.18 * col
        c = 0.25 * row + 0.18 * col
        p = np.array([a, b, c])
        p = np.clip(p, 0, 1)
        p = p / p.sum()
        pts.append(p)
    return np.array(pts)


def generate_random_init(rng):
    r1 = rng.random(N)
    r2 = rng.random(N)
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    bary = np.column_stack([1 - r1 - r2, r1, r2])
    return bary


def generate_perturbed_init(base, rng, scale):
    bary = base + rng.normal(0, scale, size=base.shape)
    return clamp_bary(bary)


def _local_polish_bary(bary, time_left_fn, budget=20.0):
    """L-BFGS-B on hybrid + Nelder-Mead on hard, from a bary starting point."""
    best_bary = bary.copy()
    best_score = hard_min_bary(best_bary)
    start = time.time()

    for beta in [80.0, 200.0]:
        if time_left_fn() < 8.0:
            break
        try:
            res = minimize(
                _neg_hybrid,
                best_bary.flatten(),
                args=(beta, 0.03, 40.0),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (N * 3),
                options={'maxiter': 120, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            cand = clamp_bary(res.x.reshape(N, 3))
            s = hard_min_bary(cand)
            if s > best_score:
                best_score = s
                best_bary = cand.copy()
        except Exception:
            pass

    # Nelder-Mead on the hard objective for exact polish.
    if time_left_fn() > 6.0:
        try:
            res = minimize(
                _neg_hard,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            cand = clamp_bary(res.x.reshape(N, 3))
            s = hard_min_bary(cand)
            if s > best_score:
                best_score = s
                best_bary = cand.copy()
        except Exception:
            pass

    return best_bary, best_score


def _simulated_annealing(bary, time_left_fn, budget=25.0):
    """SA on the hard objective with adaptive step size; accepts worse moves."""
    best_bary = bary.copy()
    best_score = hard_min_bary(best_bary)
    cur_bary = bary.copy()
    cur_score = best_score
    start = time.time()
    T = 0.02
    step = 0.06
    rng = np.random.default_rng(12345)

    while time_left_fn() > 3.0:
        for _ in range(300):
            if time_left_fn() < 3.0:
                break
            move = rng.normal(0, step, size=(N, 3))
            cand = clamp_bary(cur_bary + move)
            s = hard_min_bary(cand)
            delta = s - cur_score
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                cur_bary = cand
                cur_score = s
                if s > best_score:
                    best_score = s
                    best_bary = cand.copy()
        T *= 0.95
        step *= 0.97
    return best_bary, best_score


def _refined_local_search(bary, time_left_fn):
    """
    Multi-stage refinement:
    1. Gradient ascent on soft-min with adaptive beta (increasing sharpness)
    2. Coordinate descent on the hard objective
    3. Final Nelder-Mead polish with very tight tolerances
    """
    best_bary = bary.copy()
    best_score = hard_min_bary(best_bary)

    # Stage 1: Gradient ascent on soft-min with increasing beta
    for beta in [50.0, 100.0, 200.0, 400.0]:
        if time_left_fn() < 5.0:
            break
        try:
            res = minimize(
                _neg_soft,
                best_bary.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            cand = clamp_bary(res.x.reshape(N, 3))
            s = hard_min_bary(cand)
            if s > best_score:
                best_score = s
                best_bary = cand.copy()
        except Exception:
            pass

    # Stage 2: Coordinate descent on hard objective
    if time_left_fn() > 8.0:
        coord_bary = best_bary.copy()
        improved = True
        while improved and time_left_fn() > 5.0:
            improved = False
            for i in range(N):
                for j in range(3):
                    # Try small perturbations in each coordinate
                    for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                        cand = coord_bary.copy()
                        cand[i, j] += delta
                        cand = clamp_bary(cand)
                        s = hard_min_bary(cand)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_bary = cand.copy()
                            coord_bary = cand.copy()
                            improved = True
                            break

    # Stage 3: Final Nelder-Mead with very tight tolerances
    if time_left_fn() > 4.0:
        try:
            res = minimize(
                _neg_hard,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            cand = clamp_bary(res.x.reshape(N, 3))
            s = hard_min_bary(cand)
            if s > best_score:
                best_score = s
                best_bary = cand.copy()
        except Exception:
            pass

    return best_bary, best_score


def _spectral_optimization(bary_init, time_left_fn, n_modes=3):
    """
    Optimize both point positions and spectral field parameters jointly.
    The spectral field's critical points guide the point placement.
    """
    best_bary = bary_init.copy()
    best_score = hard_min_bary(best_bary)
    rng = np.random.default_rng(999)
    
    # Initial mode parameters
    modes_params = np.zeros(n_modes * 4)
    for i in range(n_modes):
        modes_params[i*4] = rng.uniform(3, 7)    # kx
        modes_params[i*4+1] = rng.uniform(3, 7)  # ky
        modes_params[i*4+2] = rng.uniform(1, 2)  # amp
        modes_params[i*4+3] = rng.uniform(0, 2*np.pi)  # phase
    
    # Joint optimization: [bary_coords (33), mode_params (12)] = 45 variables
    def joint_obj(flat):
        bary_flat = flat[:N*3]
        mode_flat = flat[N*3:]
        bary = clamp_bary(bary_flat.reshape(N, 3))
        cart = bary_to_cart(bary)
        areas = all_areas_cart(cart)
        m = areas.min()
        
        # Soft-min
        beta = 150.0
        shifted = np.clip(-beta * (areas - m), -500, 500)
        logsum = np.log(np.sum(np.exp(shifted)))
        softmin = m - logsum / beta
        
        # Gradient penalty
        x, y = cart[:, 0], cart[:, 1]
        grad_pen = 0.0
        for i in range(n_modes):
            kx = mode_flat[i*4]
            ky = mode_flat[i*4+1]
            amp = mode_flat[i*4+2]
            ph = mode_flat[i*4+3]
            gx = amp * kx * np.cos(kx * x + ky * y + ph)
            gy = amp * ky * np.cos(kx * x + ky * y + ph)
            grad_pen += np.mean(gx**2 + gy**2)
        
        return -(softmin - 0.15 * grad_pen)
    
    # Start from current best
    x0 = np.concatenate([best_bary.flatten(), modes_params])
    
    try:
        res = minimize(
            joint_obj,
            x0,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * (N*3) + [(1, 10), (1, 10), (0.1, 3), (0, 2*np.pi)] * n_modes,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        cand_bary = clamp_bary(res.x[:N*3].reshape(N, 3))
        s = hard_min_bary(cand_bary)
        if s > best_score:
            best_score = s
            best_bary = cand_bary.copy()
    except Exception:
        pass
    
    # Polish with standard methods
    if time_left_fn() > 10.0:
        try:
            pb, ps = _refined_local_search(best_bary, time_left_fn)
            if ps > best_score:
                best_score = ps
                best_bary = pb.copy()
        except Exception:
            pass
    
    return best_bary, best_score


def _rbf_optimization(bary_init, time_left_fn):
    """
    RBF-based optimization: represent points as a low-frequency radial basis function
    network with a small number of free center parameters, reducing search dimensionality.
    This imposes smooth spatial coherence that prevents degenerate clusters.
    """
    best_bary = bary_init.copy()
    best_score = hard_min_bary(best_bary)
    rng = np.random.default_rng(777)
    
    # Number of RBF centers (reduced dimensionality)
    n_centers = 8
    
    # Initial center parameters in barycentric space
    centers = rng.uniform(0.1, 0.9, size=(n_centers, 3))
    centers = clamp_bary(centers)
    
    # RBF bandwidth
    sigma = 0.3
    
    def rbf_to_points(centers_flat, sigma):
        """Convert RBF centers to point positions using nearest-neighbor assignment."""
        centers = clamp_bary(centers_flat.reshape(n_centers, 3))
        cart_centers = bary_to_cart(centers)
        
        # Assign each of N points to nearest center
        # Use a weighted combination for smoothness
        points_cart = np.zeros((N, 2))
        for i in range(N):
            dists = np.linalg.norm(cart_centers - cart_centers[i % n_centers], axis=1)
            weights = np.exp(-dists**2 / (2 * sigma**2))
            weights /= weights.sum()
            points_cart[i] = np.sum(weights[:, None] * cart_centers, axis=0)
        
        # Ensure points are inside triangle
        bary_pts = cart_to_bary(points_cart)
        return clamp_bary(bary_pts)
    
    def rbf_objective(params):
        """Objective for RBF optimization."""
        centers_flat = params[:n_centers * 3]
        sigma = params[n_centers * 3]
        sigma = np.clip(sigma, 0.1, 0.5)
        
        bary_pts = rbf_to_points(centers_flat, sigma)
        cart_pts = bary_to_cart(bary_pts)
        areas = all_areas_cart(cart_pts)
        m = areas.min()
        
        # Soft-min
        beta = 100.0
        shifted = np.clip(-beta * (areas - m), -500, 500)
        logsum = np.log(np.sum(np.exp(shifted)))
        softmin = m - logsum / beta
        
        # Penalize too-large sigma (points too close to centers)
        sigma_pen = 0.1 * (sigma - 0.3)**2
        
        return -(softmin - sigma_pen)
    
    # Start from current best
    x0 = np.concatenate([best_bary.flatten()[:n_centers*3], [0.3]])
    
    try:
        res = minimize(
            rbf_objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * (n_centers * 3) + [(0.1, 0.5)],
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        cand_bary = rbf_to_points(res.x[:n_centers*3], res.x[n_centers*3])
        s = hard_min_bary(cand_bary)
        if s > best_score:
            best_score = s
            best_bary = cand_bary.copy()
    except Exception:
        pass
    
    # Polish with standard methods
    if time_left_fn() > 10.0:
        try:
            pb, ps = _refined_local_search(best_bary, time_left_fn)
            if ps > best_score:
                best_score = ps
                best_bary = pb.copy()
        except Exception:
            pass
    
    return best_bary, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin under 150s

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)

    best_bary = None
    best_score = -1.0

    def consider(bary):
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        s = hard_min_bary(bary)
        if s > best_score:
            best_score = s
            best_bary = bary.copy()
        return s

    # --- Stage 1: build diverse initial pool ---
    inits = [generate_hexagonal_init(), generate_grid_init()]
    for scale in [0.03, 0.08, 0.15, 0.25]:
        inits.append(generate_perturbed_init(generate_hexagonal_init(), rng, scale))
    for _ in range(10):
        inits.append(generate_random_init(rng))

    scored = [(init, consider(init)) for init in inits]
    scored.sort(key=lambda x: x[1], reverse=True)

    # --- Stage 2: spectral optimization on top candidates ---
    for bary_init, init_score in scored[:6]:
        if time_left() < 20.0:
            break
        try:
            pb, ps = _spectral_optimization(bary_init, time_left, n_modes=3)
            consider(pb)
        except Exception:
            pass

    # --- Stage 3: RBF-based optimization on top candidates ---
    for bary_init, init_score in scored[:4]:
        if time_left() < 15.0:
            break
        try:
            pb, ps = _rbf_optimization(bary_init, time_left)
            consider(pb)
        except Exception:
            pass

    # --- Stage 4: standard refined search on best ---
    if best_bary is not None and time_left() > 15.0:
        try:
            pb, ps = _refined_local_search(best_bary, time_left)
            consider(pb)
        except Exception:
            pass

    # --- Stage 5: simulated annealing ---
    if best_bary is not None and time_left() > 15.0:
        try:
            sb, ss = _simulated_annealing(best_bary, time_left)
            consider(sb)
        except Exception:
            pass

    # --- Stage 6: final polish ---
    if best_bary is not None and time_left() > 8.0:
        try:
            pb, ps = _refined_local_search(best_bary, time_left)
            consider(pb)
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_hexagonal_init()
        best_score = consider(best_bary)

    return bary_to_cart(clamp_bary(best_bary))


if __name__ == "__main__":
    pts = construct()
    print("Best score:", hard_min_bary(cart_to_bary(pts)))
