"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Start with a simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()

    def exact_radius(x):
        """Compute the exact radius for a given configuration."""
        x = x.reshape(N, 3)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist / 2.0, min_wall_dist)

    def soft_min_score(x, beta=20.0):
        """
        Smooth surrogate: maximize soft-min of (pairwise dists/2, wall dists).
        We minimize negative soft-min.
        Uses log-sum-exp for numerical stability.
        """
        x = x.reshape(N, 3)
        # Pairwise squared distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(sq_dists, 1e10)  # ignore self
        pair_terms = sq_dists / 4.0  # (dist/2)^2

        # Wall distances squared
        wall_sq = np.min((0.5 - np.abs(x)) ** 2, axis=1)

        # Combine all terms into one vector
        all_terms = np.concatenate([pair_terms.ravel(), wall_sq.ravel()])
        # Soft-min via log-sum-exp: -log(sum(exp(-beta * x))) / beta
        # For stability: let m = min(all_terms), then soft_min = m - log(sum(exp(-beta*(all_terms - m))))/beta
        m = np.min(all_terms)
        # Clip for numerical stability
        shifted = all_terms - m
        # exp(-beta * shifted) where shifted >= 0
        exp_vals = np.exp(-beta * np.clip(shifted, 0, 50))
        s = np.sum(exp_vals)
        soft_min_sq = m - np.log(s) / beta
        # We want to maximize soft_min_sq, so minimize negative
        return -soft_min_sq

    def soft_min_grad(x, beta=20.0):
        """Gradient of soft_min_score."""
        x = x.reshape(N, 3)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(sq_dists, 1e10)
        pair_terms = sq_dists / 4.0

        wall_sq = np.min((0.5 - np.abs(x)) ** 2, axis=1)
        all_terms = np.concatenate([pair_terms.ravel(), wall_sq.ravel()])

        m = np.min(all_terms)
        shifted = all_terms - m
        exp_vals = np.exp(-beta * np.clip(shifted, 0, 50))
        s = np.sum(exp_vals)
        weights = exp_vals / s  # softmax weights

        # Gradient of soft_min_sq = m - log(s)/beta
        # d(soft_min_sq)/dx = -d(log(s))/dx / beta
        # d(log(s))/dx = sum_j w_j * d(terms_j)/dx
        # where w_j = exp(-beta*(terms_j - m)) / s

        # For pairwise terms: term_ij = (x_i - x_j)^2 / 4
        # d(term_ij)/dx_i = (x_i - x_j) / 2
        # d(term_ij)/dx_j = (x_j - x_i) / 2

        grad = np.zeros((N, 3))

        # Pairwise contributions
        # weights for pairwise: first N*(N-1)/2 * 2 entries (symmetric)
        # Actually pair_terms is NxN with diagonal set to large, so we have N*N entries
        # but diagonal is ignored. Let's handle it properly.
        # pair_terms.ravel() has N*N entries, diagonal is 1e10/4
        pw = weights[:N * N].reshape(N, N)
        # For each pair (i,j), contribution to grad[i] and grad[j]
        # d(pair_terms[i,j])/dx_i = (x_i - x_j) / 2
        # d(pair_terms[i,j])/dx_j = (x_j - x_i) / 2
        # grad[i] -= beta * sum_j pw[i,j] * (x_i - x_j) / 2
        # grad[j] -= beta * sum_i pw[j,i] * (x_j - x_i) / 2

        # Vectorized:
        # sum_j pw[i,j] * (x_i - x_j) = x_i * sum_j pw[i,j] - sum_j pw[i,j] * x_j
        row_sums = np.sum(pw, axis=1)
        weighted_x = pw @ x  # shape (N, 3)
        pair_grad_i = x * row_sums[:, np.newaxis] - weighted_x  # shape (N, 3)
        # d(soft_min)/dx = -1/beta * sum_j w_j * d(term_j)/dx
        # d(term_ij)/dx_i = (x_i - x_j)/2, so contribution is pair_grad_i / 2
        grad -= beta * pair_grad_i / 2.0

        # Wall contributions
        # wall_sq[i] = min_k (0.5 - |x[i,k]|)^2
        # We need to know which coordinate is the minimum for each point
        abs_x = np.abs(x)
        wall_dists_per_coord = (0.5 - abs_x) ** 2  # shape (N, 3)
        min_idx = np.argmin(wall_dists_per_coord, axis=1)  # shape (N,)

        ww = weights[N * N:]  # shape (N,)
        # d(wall_sq[i])/dx[i,k] for the min coordinate k:
        # if x[i,k] > 0: d/dx = 2*(0.5 - x[i,k])*(-1) = -2*(0.5 - x[i,k])
        # if x[i,k] < 0: d/dx = 2*(0.5 + x[i,k])*(1) = 2*(0.5 + x[i,k])
        # Actually: wall_sq[i] = (0.5 - |x[i,k]|)^2
        # d/dx = 2*(0.5 - |x[i,k]|) * d(0.5 - |x[i,k]|)/dx
        # d(0.5 - |x|)/dx = -sign(x)
        # So d(wall_sq)/dx = -2*(0.5 - |x[i,k]|) * sign(x[i,k])
        # = -2*(0.5 - |x[i,k]|) * sign(x[i,k])

        for i in range(N):
            k = min_idx[i]
            if ww[i] > 1e-15:
                sign = np.sign(x[i, k])
                if sign == 0:
                    sign = 1.0
                dval = -2.0 * (0.5 - abs(x[i, k])) * sign
                grad[i, k] += beta * dval

        return grad.flatten()

    best_x = x0.copy()
    best_score = exact_radius(best_x)

    # Try multiple restarts with different initial configurations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break

        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            # Slightly perturbed grid
            rng = np.random.default_rng(42)
            x_init = x0 + rng.normal(0, 0.015, size=(N, 3))
        elif restart == 2:
            # More perturbation
            rng = np.random.default_rng(123)
            x_init = x0 + rng.normal(0, 0.03, size=(N, 3))
        elif restart == 3:
            # Different grid: 4x4x4 with some removal
            m2 = 4
            ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
            grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
            rng = np.random.default_rng(777)
            x_init = grid2[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
        elif restart == 4:
            # 3x4x5 grid
            m3, m4, m5 = 3, 4, 5
            tx = (np.arange(m3) + 0.5) / m3 - 0.5
            ty = (np.arange(m4) + 0.5) / m4 - 0.5
            tz = (np.arange(m5) + 0.5) / m5 - 0.5
            grid3 = np.array([(x, y, z) for z in tz for y in ty for x in tx])
            rng = np.random.default_rng(999)
            x_init = grid3[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
        else:
            rng = np.random.default_rng(restart * 1000 + 555)
            x_init = x0 + rng.normal(0, 0.02 + 0.005 * restart, size=(N, 3))

        x_init = np.clip(x_init, -0.48, 0.48)

        # Use L-BFGS-B with soft-min objective
        try:
            result = minimize(
                lambda v: soft_min_score(v, beta=30.0),
                x_init.flatten(),
                jac=lambda v: soft_min_grad(v, beta=30.0),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-9, 'maxfun': 5000}
            )
            x_candidate = result.x.reshape(N, 3)
            x_candidate = np.clip(x_candidate, -0.5, 0.5)
            sc = exact_radius(x_candidate)
            if sc > best_score:
                best_score = sc
                best_x = x_candidate.copy()
        except Exception:
            pass

        if time.time() > deadline - 10.0:
            break

    # Final polish: do a tight local optimization on the exact (nonsmooth) objective
    # using a few iterations of gradient-free or small-step L-BFGS
    if time.time() < deadline - 2.0:
        # Try a few small perturbations around best_x and re-optimize
        for polish in range(3):
            if time.time() > deadline - 2.0:
                break
            rng = np.random.default_rng(polish * 77 + 11)
            x_init = best_x + rng.normal(0, 0.005, size=(N, 3))
            x_init = np.clip(x_init, -0.49, 0.49)
            try:
                result = minimize(
                    lambda v: soft_min_score(v, beta=50.0),
                    x_init.flatten(),
                    jac=lambda v: soft_min_grad(v, beta=50.0),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 3000}
                )
                x_candidate = result.x.reshape(N, 3)
                x_candidate = np.clip(x_candidate, -0.5, 0.5)
                sc = exact_radius(x_candidate)
                if sc > best_score:
                    best_score = sc
                    best_x = x_candidate.copy()
            except Exception:
                pass

    # Ensure validity: no coincident centers, all inside cube
    best_x = np.clip(best_x, -0.5, 0.5)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                best_x[j] += 1e-5 * np.array([1, 0, 0])
                best_x[j] = np.clip(best_x[j], -0.5, 0.5)

    return best_x
# EVOLVE-BLOCK-END
