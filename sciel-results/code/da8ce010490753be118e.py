"""Improved packer for 50 equal circles in a unit circle."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 50
# Best known radius for reference (Graham et al. 1998)
R_REF = 0.1258254895304


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 140s to be safe within 150s

    # Precompute pair indices for distance constraints
    pair_idx = _pairwise_indices(N)
    n_pairs = len(pair_idx)

    # Precompute one-hot style matrix for linear constraints on squared distances
    # Actually, for the reformulation with auxiliary variables, we need a different approach.
    # Let's use a direct optimization with a smooth penalty or constraint.

    # Strategy:
    # 1. Start with a good initial guess (hexagonal packing or random with local opt)
    # 2. Use scipy.optimize.minimize with a smooth objective
    # 3. Objective: minimize -r where r is the minimum of (distances/2) and (1 - |c_i|)
    #    To make it smooth, we can use a soft-min approximation or just optimize the
    #    minimum directly with SLSQP which handles constraints.

    # Better approach: use the fact that at optimum, the minimum is achieved by
    # some constraints being active. We can use a penalty method or direct SLSQP.

    # Let's try a different formulation:
    # Maximize r subject to:
    #   |c_i - c_j| >= 2r for all i < j
    #   |c_i| <= 1 - r for all i
    #
    # This is a non-convex problem. We'll use multiple random starts with SLSQP.

    def objective(x, r):
        """x is (N,2) flattened, r is scalar. We want to maximize r.
        Here we minimize -r with constraints."""
        return -r

    def make_constraints(r):
        """Create constraint functions for given r."""
        def dist_constraints(x):
            """|c_i - c_j|^2 >= (2r)^2 for all i<j"""
            c = x.reshape(N, 2)
            ci = c[pair_idx[:, 0]]
            cj = c[pair_idx[:, 1]]
            d2 = np.sum((ci - cj) ** 2, axis=1)
            return d2 - (2 * r) ** 2

        def boundary_constraints(x):
            """|c_i|^2 <= (1 - r)^2 for all i"""
            c = x.reshape(N, 2)
            r2 = np.sum(c ** 2, axis=1)
            return (1 - r) ** 2 - r2

        return [dist_constraints, boundary_constraints]

    # Initial guess: hexagonal packing
    def hexagonal_init():
        # Generate hexagonal lattice points and select N that fit in unit circle
        spacing = 0.25
        pts = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if x * x + y * y <= 0.8:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) < N:
            # Fallback to random
            return _random_init()
        # Select N points closest to origin
        r = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(r)[:N]
        return pts[idx]

    def _random_init():
        """Random points inside unit circle."""
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 0.8, N))
        x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return x

    # Start with hexagonal
    c0 = hexagonal_init()

    # Try to improve using SLSQP with a sequence of r values
    # Start with a small r and increase

    best_c = c0.copy()
    best_r = 0.0

    # Compute current r
    def compute_r(c):
        c = c.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
        r2 = np.hypot(c[:, 0], c[:, 1])
        return min(np.min(d) / 2, np.min(1 - r2))

    best_r = compute_r(best_c)

    # Try to increase r iteratively
    r_try = best_r * 1.01
    max_iter_outer = 50

    for outer_iter in range(max_iter_outer):
        if time.time() > deadline:
            break
        if r_try > R_REF * 1.001:
            break

        # Define constraints for this r_try
        def dist_cons(x):
            c = x.reshape(N, 2)
            ci = c[pair_idx[:, 0]]
            cj = c[pair_idx[:, 1]]
            d2 = np.sum((ci - cj) ** 2, axis=1)
            return d2 - (2 * r_try) ** 2

        def bound_cons(x):
            c = x.reshape(N, 2)
            r2 = np.sum(c ** 2, axis=1)
            return (1 - r_try) ** 2 - r2

        cons = [
            {'type': 'ineq', 'fun': dist_cons},
            {'type': 'ineq', 'fun': bound_cons},
        ]

        x0 = best_c.flatten()

        res = minimize(
            lambda x: -r_try,  # constant objective, we just want feasibility
            x0,
            method='SLSQP',
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12},
        )

        if res.success or res.status == 0:
            # Check if the solution is feasible
            c_new = res.x.reshape(N, 2)
            r_new = compute_r(c_new)
            if r_new >= r_try - 1e-8:
                best_c = c_new
                best_r = r_new
                r_try = r_new * 1.005
            else:
                # Infeasible, try smaller increment
                r_try = (best_r + r_try) / 2
        else:
            # Failed, try smaller increment
            r_try = (best_r + r_try) / 2

        # If we've been stuck, try a random restart
        if outer_iter % 10 == 9:
            c_rand = _random_init()
            r_rand = compute_r(c_rand)
            if r_rand > best_r:
                best_c = c_rand
                best_r = r_rand
                r_try = best_r * 1.01

    # Final polish: try to fine-tune
    if time.time() < deadline - 10:
        # Use a more precise local optimization
        c = best_c.reshape(N, 2)

        def smooth_objective(x):
            """Minimize negative soft-min of distances/2 and (1-|c_i|)"""
            c = x.reshape(N, 2)
            ci = c[pair_idx[:, 0]]
            cj = c[pair_idx[:, 1]]
            d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
            b = 1 - np.hypot(c[:, 0], c[:, 1])
            # Soft min using log-sum-exp
            all_vals = np.concatenate([d, b])
            beta = 500.0
            min_val = -np.log(np.sum(np.exp(-beta * all_vals))) / beta
            return min_val

        # Actually, we want to MAXIMIZE the minimum, so minimize the negative
        def neg_smooth_objective(x):
            c = x.reshape(N, 2)
            ci = c[pair_idx[:, 0]]
            cj = c[pair_idx[:, 1]]
            d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
            b = 1 - np.hypot(c[:, 0], c[:, 1])
            all_vals = np.concatenate([d, b])
            beta = 500.0
            # We want to maximize min(all_vals), so minimize -soft_min
            # soft_min = -log(sum(exp(-beta*x)))/beta
            # We want to minimize -soft_min = log(sum(exp(-beta*x)))/beta
            return np.log(np.sum(np.exp(-beta * all_vals))) / beta

        res = minimize(
            neg_smooth_objective,
            best_c.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15},
        )

        if res.success:
            c_new = res.x.reshape(N, 2)
            r_new = compute_r(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new

    # Final check: ensure all points are valid
    c_final = best_c.reshape(N, 2)
    # Clamp points that might be slightly outside
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_r:
            scale = (1 - best_r) / r2[i]
            c_final[i] *= scale

    return c_final
