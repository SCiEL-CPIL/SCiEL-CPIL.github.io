"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
S3 = np.sqrt(3.0)
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])
TRI_AREA = S3 / 4.0  # area of the equilateral triangle

# Precompute all pairs of indices for the C(11,3) = 165 triangles
import itertools
TRI_INDICES = list(itertools.combinations(range(N), 3))
N_TRI = len(TRI_INDICES)


def bary_to_cart(u, v):
    """Convert barycentric (u,v) to Cartesian. w = 1-u-v."""
    w = 1.0 - u - v
    x = u * V1[0] + w * V0[0]  # u*1 + w*0
    y = u * V1[1] + w * V0[1] + v * V2[1]  # u*0 + w*0 + v*(sqrt(3)/2)
    return np.column_stack([x, y])


def cart_to_bary(pts):
    """Convert Cartesian points to barycentric (u,v)."""
    # Solve: p = u*V1 + v*V2 (since V0=origin)
    # x = u + 0.5*v
    # y = (sqrt(3)/2)*v
    # => v = 2*y/sqrt(3), u = x - 0.5*v
    x = pts[:, 0]
    y = pts[:, 1]
    v = 2.0 * y / S3
    u = x - 0.5 * v
    return np.column_stack([u, v])


def compute_areas(bary_pts):
    """Compute all 165 triangle areas from barycentric points."""
    cart = bary_to_cart(bary_pts[:, 0], bary_pts[:, 1])
    areas = np.empty(N_TRI)
    for i, (a, b, c) in enumerate(TRI_INDICES):
        ax, ay = cart[a]
        bx, by = cart[b]
        cx, cy = cart[c]
        areas[i] = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    return areas


def soft_min_objective(bary_flat, beta):
    """Soft-min of triangle areas (negative for minimization)."""
    bary = bary_flat.reshape(N, 2)
    # Clip to feasible region
    u = bary[:, 0]
    v = bary[:, 1]
    # For now, just compute areas; constraints handled by DE bounds
    areas = compute_areas(bary)
    # Soft min: -(1/beta) * log(sum(exp(-beta * areas)))
    # This is a smooth approximation to min(areas)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta  # negative soft-min


def exact_min_area(bary):
    """Exact minimum triangle area."""
    areas = compute_areas(bary)
    return np.min(areas)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    best_bary = None
    best_score = 0.0

    # Phase 1: Differential Evolution for global search
    # Bounds: u in [0.01, 0.98], v in [0.01, 0.98], u+v <= 0.98
    # We'll use bounds and then project
    bounds_de = [(0.01, 0.98)] * (2 * N)
    
    def de_objective(bary_flat):
        bary = bary_flat.reshape(N, 2)
        # Project to feasible region: clip u,v >= 0, u+v <= 1
        u = np.clip(bary[:, 0], 0.0, 1.0)
        v = np.clip(bary[:, 1], 0.0, 1.0)
        # Ensure u+v <= 1
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            scale = 1.0 / s[mask]
            u[mask] *= scale
            v[mask] *= scale
        bary = np.column_stack([u, v])
        areas = compute_areas(bary)
        return -np.min(areas)  # maximize min area

    # Run DE with a moderate population
    try:
        result_de = differential_evolution(
            de_objective,
            bounds=bounds_de,
            seed=42,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            polish=False,  # We'll do our own polish
            init='sobol',
            updating='immediate',
            workers=1,
        )
        if time.time() < deadline:
            bary_de = result_de.x.reshape(N, 2)
            # Project
            u = np.clip(bary_de[:, 0], 0.0, 1.0)
            v = np.clip(bary_de[:, 1], 0.0, 1.0)
            s = u + v
            mask = s > 1.0
            if np.any(mask):
                scale = 1.0 / s[mask]
                u[mask] *= scale
                v[mask] *= scale
            bary_de = np.column_stack([u, v])
            score_de = exact_min_area(bary_de)
            if score_de > best_score:
                best_score = score_de
                best_bary = bary_de.copy()
    except Exception:
        pass

    # Phase 2: Multi-start L-BFGS-B local optimization with soft-min
    n_restarts = 15
    rng = np.random.RandomState(123)

    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break

        # Generate initial point
        if restart == 0 and best_bary is not None:
            bary_init = best_bary + rng.normal(0, 0.05, size=(N, 2))
        elif restart == 1:
            # Grid-like initialization
            bary_init = np.zeros((N, 2))
            for i in range(N):
                # Distribute roughly
                u = (i % 3) / 2.5 + 0.1
                v = (i // 3) / 3.5 + 0.1
                bary_init[i] = [u, v]
        else:
            # Random feasible points
            bary_init = np.zeros((N, 2))
            for i in range(N):
                u = rng.uniform(0.02, 0.95)
                v = rng.uniform(0.02, min(0.95, 1.0 - u - 0.02))
                bary_init[i] = [u, v]

        # Project to feasible region
        u = np.clip(bary_init[:, 0], 0.01, 0.98)
        v = np.clip(bary_init[:, 1], 0.01, 0.98)
        s = u + v
        mask = s > 0.98
        if np.any(mask):
            scale = 0.98 / s[mask]
            u[mask] *= scale
            v[mask] *= scale
        bary_init = np.column_stack([u, v])

        # L-BFGS-B optimization with soft-min objective
        beta = 20.0  # temperature for soft-min

        def objective_and_grad(bary_flat):
            bary = bary_flat.reshape(N, 2)
            # Soft penalty for infeasibility
            u = bary[:, 0]
            v = bary[:, 1]
            penalty = 0.0
            # u >= 0, v >= 0, u+v <= 1
            pen_u = np.maximum(0, -u + 0.005)
            pen_v = np.maximum(0, -v + 0.005)
            pen_s = np.maximum(0, u + v - 0.995)
            penalty = 100.0 * (np.sum(pen_u**2) + np.sum(pen_v**2) + np.sum(pen_s**2))
            
            areas = compute_areas(bary)
            # Soft min
            scaled = -beta * areas
            max_scaled = np.max(scaled)
            log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            soft_min_val = -log_sum / beta
            return penalty - soft_min_val

        # Run L-BFGS-B
        try:
            res = minimize(
                objective_and_grad,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=[(0.01, 0.98)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if time.time() > deadline - 5.0:
                break

            bary_opt = res.x.reshape(N, 2)
            # Project
            u = np.clip(bary_opt[:, 0], 0.0, 1.0)
            v = np.clip(bary_opt[:, 1], 0.0, 1.0)
            s = u + v
            mask = s > 1.0
            if np.any(mask):
                scale = 1.0 / s[mask]
                u[mask] *= scale
                v[mask] *= scale
            bary_opt = np.column_stack([u, v])

            score_opt = exact_min_area(bary_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
        except Exception:
            pass

    # Phase 3: Final polish with tighter soft-min and more iterations
    if best_bary is not None and time.time() < deadline - 5.0:
        beta_final = 50.0

        def objective_final(bary_flat):
            bary = bary_flat.reshape(N, 2)
            u = bary[:, 0]
            v = bary[:, 1]
            penalty = 0.0
            pen_u = np.maximum(0, -u + 0.001)
            pen_v = np.maximum(0, -v + 0.001)
            pen_s = np.maximum(0, u + v - 0.999)
            penalty = 1000.0 * (np.sum(pen_u**2) + np.sum(pen_v**2) + np.sum(pen_s**2))
            areas = compute_areas(bary)
            scaled = -beta_final * areas
            max_scaled = np.max(scaled)
            log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            soft_min_val = -log_sum / beta_final
            return penalty - soft_min_val

        try:
            res_final = minimize(
                objective_final,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0.001, 0.999)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            bary_final = res_final.x.reshape(N, 2)
            u = np.clip(bary_final[:, 0], 0.0, 1.0)
            v = np.clip(bary_final[:, 1], 0.0, 1.0)
            s = u + v
            mask = s > 1.0
            if np.any(mask):
                scale = 1.0 / s[mask]
                u[mask] *= scale
                v[mask] *= scale
            bary_final = np.column_stack([u, v])
            score_final = exact_min_area(bary_final)
            if score_final > best_score:
                best_score = score_final
                best_bary = bary_final.copy()
        except Exception:
            pass

    # Convert to Cartesian and return
    if best_bary is None:
        # Fallback: random points
        rng = np.random.RandomState(0)
        bary = np.zeros((N, 2))
        for i in range(N):
            u = rng.uniform(0.05, 0.9)
            v = rng.uniform(0.05, min(0.9, 1.0 - u - 0.05))
            bary[i] = [u, v]
        best_bary = bary

    cart = bary_to_cart(best_bary[:, 0], best_bary[:, 1])
    return cart


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    # Compute min area for reporting
    areas = compute_areas(cart_to_bary(pts))
    print("Min area:", np.min(areas))
    print("Normalized min area (divided by triangle area):", np.min(areas) / TRI_AREA)
