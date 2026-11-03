"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii_vectorized(centers):
        """
        Vectorized computation of maximum non-overlapping radii from centers.
        Uses the constraint: r_i <= min(boundary_dists_i, min_{j!=i} d_ij/2)
        but with the additional coupling that r_i + r_j <= d_ij.
        We solve this iteratively.
        """
        centers = np.asarray(centers)
        n = len(centers)

        # Boundary constraints
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))

        # Pairwise constraints: for each pair (i,j), r_i + r_j <= d_ij
        # This is a system of linear inequalities. We solve iteratively.
        # Precompute all pairwise distances
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)

        # Iterate: for each pair, if r_i + r_j > d_ij, scale down
        for _ in range(50):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    d = dists[i, j]
                    s = radii[i] + radii[j]
                    if s > d:
                        # Scale both proportionally
                        if s > 0:
                            scale = d / s
                            radii[i] *= scale
                            radii[j] *= scale
                            changed = True
            if not changed:
                break

        return np.maximum(0, radii)

    def eval_packing(centers):
        """Evaluate the sum of radii for given centers."""
        radii = compute_radii_vectorized(centers)
        return np.sum(radii), radii

    def objective_smooth(centers_flat, beta=200.0):
        """
        Fully vectorized smooth objective using soft-min via log-sum-exp.
        For each circle i, radius_i = soft_min(boundary_dists_i, {d_ij/2 for j!=i})
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)

        # Boundary constraints per circle (shape: n x 4)
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]

        # Pairwise distances (shape: n x n)
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0

        # For each circle i, collect all constraints: 4 boundary + (n-1) pairwise
        # Build matrix of all constraints: shape (n, n+3)
        constraints = np.empty((n, n + 3))
        constraints[:, 0] = d_left
        constraints[:, 1] = d_right
        constraints[:, 2] = d_bottom
        constraints[:, 3] = d_top
        constraints[:, 4:] = half_dists

        # Soft-min along axis=1 using log-sum-exp trick
        c_min = np.min(constraints, axis=1, keepdims=True)
        # For numerical stability, subtract c_min before exp
        shifted = -beta * (constraints - c_min)
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -500, 500)
        log_sum = np.log(np.sum(np.exp(shifted), axis=1))
        soft_min_vals = c_min.squeeze() - log_sum / beta

        return -np.sum(soft_min_vals)

    def objective_smooth_grad(centers_flat, beta=200.0):
        """
        Vectorized smooth objective AND gradient.
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)

        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]

        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0

        constraints = np.empty((n, n + 3))
        constraints[:, 0] = d_left
        constraints[:, 1] = d_right
        constraints[:, 2] = d_bottom
        constraints[:, 3] = d_top
        constraints[:, 4:] = half_dists

        c_min = np.min(constraints, axis=1, keepdims=True)
        shifted = -beta * (constraints - c_min)
        shifted = np.clip(shifted, -500, 500)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        # Weights for soft-min: w_k = exp(-beta*(c_k - c_min)) / sum(...)
        weights = exp_vals / sum_exp  # shape (n, n+3)

        soft_min_vals = c_min.squeeze() - np.log(sum_exp.squeeze()) / beta

        # Gradient of -sum(soft_min_i) w.r.t. centers
        # d(soft_min_i)/d(c_k) = -w_ik (for k-th constraint of circle i)
        # d(soft_min_i)/dx_j = sum_k w_ik * d(c_ik)/dx_j
        # So d(-sum soft_min_i)/dx_j = -sum_i sum_k w_ik * d(c_ik)/dx_j

        grad = np.zeros_like(centers)

        # Boundary contributions
        # d_left_i = x_i, d_right_i = 1-x_i, d_bottom_i = y_i, d_top_i = 1-y_i
        # d(soft_min_i)/dx_i = w_i0 * 1 + w_i1 * (-1)
        # d(-soft_min_i)/dx_i = -w_i0 + w_i1
        grad[:, 0] += -(weights[:, 0] - weights[:, 1])
        # d(soft_min_i)/dy_i = w_i2 * 1 + w_i3 * (-1)
        grad[:, 1] += -(weights[:, 2] - weights[:, 3])

        # Pairwise contributions
        # For constraint (i, j): c_ij = d_ij / 2 = sqrt((x_i-x_j)^2 + (y_i-y_j)^2) / 2
        # d(c_ij)/dx_i = (x_i - x_j) / (2 * d_ij)
        # d(c_ij)/dx_j = -(x_i - x_j) / (2 * d_ij)
        # Contribution to grad from soft_min_i: -w_ij * d(c_ij)/dx_i
        # Contribution to grad from soft_min_j: -w_ji * d(c_ji)/dx_j
        # Note d(c_ij) = d(c_ji), and w_ij and w_ji are different

        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 1e-12:
                    continue
                w_ij = weights[i, j + 4]  # weight of constraint j in soft-min of i
                w_ji = weights[j, i + 4]  # weight of constraint i in soft-min of j

                # d(c_ij)/dx_i = (x_i - x_j) / (2*d)
                # d(c_ij)/dx_j = -(x_i - x_j) / (2*d)
                # grad_i += -w_ij * (x_i-x_j)/(2*d)
                # grad_j += -w_ji * (-(x_i-x_j)/(2*d)) = +w_ji * (x_i-x_j)/(2*d)
                factor_x = (centers[i, 0] - centers[j, 0]) / (2.0 * d)
                factor_y = (centers[i, 1] - centers[j, 1]) / (2.0 * d)

                grad[i, 0] += -w_ij * factor_x
                grad[i, 1] += -w_ij * factor_y
                grad[j, 0] += w_ji * factor_x
                grad[j, 1] += w_ji * factor_y

        return -np.sum(soft_min_vals), grad.flatten()

    def gradient_polish(centers, max_iters=300, beta=200.0):
        """
        Final high-precision polish using L-BFGS-B with smooth objective.
        """
        try:
            result = minimize(
                lambda x: objective_smooth(x, beta=beta),
                centers.flatten(),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            opt_centers = result.x.reshape(-1, 2)
            radii = compute_radii_vectorized(opt_centers)
            x_full = np.column_stack([opt_centers, radii])
            if is_valid(x_full, tol=1e-6):
                return x_full
        except Exception:
            pass
        return None

    def pairwise_gradient_polish(centers, max_iters=500, lr=0.001):
        """
        Direct subgradient polish on the nonsmooth objective.
        Uses the fact that at optimum, the active constraints determine the gradient.
        """
        current = centers.copy()
        current_score, current_radii = eval_packing(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Compute active constraints for each circle
            n = len(current)
            radii = current_radii.copy()

            # Determine which constraints are active (within tolerance)
            d_left = current[:, 0]
            d_right = 1.0 - current[:, 0]
            d_bottom = current[:, 1]
            d_top = 1.0 - current[:, 1]

            # Subgradient: for each circle, the direction to increase radius
            # is determined by the active constraints
            grad = np.zeros_like(current)

            for i in range(n):
                # Boundary active constraints
                tol_act = 1e-7
                if abs(d_left[i] - radii[i]) < tol_act:
                    grad[i, 0] += -1.0  # moving right increases r
                if abs(d_right[i] - radii[i]) < tol_act:
                    grad[i, 0] += 1.0  # moving left increases r
                if abs(d_bottom[i] - radii[i]) < tol_act:
                    grad[i, 1] += -1.0
                if abs(d_top[i] - radii[i]) < tol_act:
                    grad[i, 1] += 1.0

                # Pairwise active constraints
                for j in range(i+1, n):
                    dx = current[i, 0] - current[j, 0]
                    dy = current[i, 1] - current[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-12:
                        continue
                    if abs(radii[i] + radii[j] - d) < tol_act:
                        # Active: moving i away from j increases r_i
                        # and moving j away from i increases r_j
                        ux = dx / d
                        uy = dy / d
                        grad[i, 0] += ux
                        grad[i, 1] += uy
                        grad[j, 0] -= ux
                        grad[j, 1] -= uy

            # Normalize gradient
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break

            # Adaptive step size
            step = lr / (1.0 + 0.1 * it)
            new_centers = current + step * grad / gnorm
            new_centers = np.clip(new_centers, 0.01, 0.99)

            new_score, new_radii = eval_packing(new_centers)

            if new_score > current_score:
                current = new_centers
                current_score = new_score
                current_radii = new_radii
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
            else:
                # Reduce step size
                lr *= 0.9

        return best, best_score

    def simulated_annealing(centers, max_iters=50000, T_start=0.05, T_end=0.001, seed=0):
        """
        Simulated annealing to optimize circle center positions.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)

        best = current.copy()
        best_score = current_score

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break

            # Cooling schedule: geometric
            T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))

            # Pick a random circle to move
            i = rng.randint(N)
            # Adaptive step size
            step = 0.02 * (T / T_start) ** 0.5 + 0.002
            dx = rng.normal(0, step)
            dy = rng.normal(0, step)

            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _ = eval_packing(new_centers)
            delta = new_score - current_score

            # Acceptance criterion
            if delta > 0 or rng.uniform() < np.exp(delta / max(T, 1e-10)):
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def local_search(centers, max_iters=3000, step_size=0.01, seed=0):
        """
        Best-fit local search: only accept improvements.
        """
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)

            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)

            new_score, _ = eval_packing(new_centers)

            if new_score > current_score + 1e-12:
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        return best, best_score

    def grid_init():
        """Structured grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def hex_init():
        """Hexagonal-like initialization."""
        centers = []
        for i in range(6):
            for j in range(5):
                if len(centers) < N:
                    x = 0.1 + i * 0.16 + (0.08 if j % 2 else 0)
                    y = 0.1 + j * 0.17
                    if x < 0.9 and y < 0.9:
                        centers.append([x, y])
        if len(centers) < N:
            rng = np.random.RandomState(42)
            while len(centers) < N:
                centers.append(list(rng.uniform(0.1, 0.9, 2)))
        return np.array(centers[:N])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    # Main optimization loop
    best_result = None
    best_score = -np.inf

    # Multiple initialization strategies
    inits = []
    inits.append(('grid', grid_init()))
    inits.append(('hex', hex_init()))
    for s in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131, 151, 173]:
        inits.append((f'rand_{s}', random_init(s)))

    # Run SA + local search + gradient polish from each init
    for init_name, init_centers in inits:
        if time.time() > deadline - 10:
            break

        # Phase 1: Simulated Annealing
        sa_centers, sa_score = simulated_annealing(
            init_centers,
            max_iters=20000,
            T_start=0.04,
            T_end=0.002,
            seed=hash(init_name) % 10000
        )

        # Phase 2: Local search polish
        ls_centers, ls_score = local_search(
            sa_centers,
            max_iters=2000,
            step_size=0.008,
            seed=hash(init_name) % 10000 + 1
        )

        # Phase 3: Gradient polish with smooth objective
        polished = gradient_polish(ls_centers, max_iters=200, beta=200.0)

        if polished is not None:
            p_score = np.sum(polished[:, 2])
            if p_score > best_score:
                best_score = p_score
                best_result = polished.copy()
        else:
            # Use local search result if polish failed
            if ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

        # Also check SA result directly
        if sa_score > best_score:
            best_score = sa_score
            radii = compute_radii_vectorized(sa_centers)
            best_result = np.column_stack([sa_centers, radii])

    # If no valid result, fall back to grid
    if best_result is None:
        centers = grid_init()
        radii = compute_radii_vectorized(centers)
        best_result = np.column_stack([centers, radii])

    # Final refinement: multiple SA restarts from best solution
    if best_result is not None and time.time() < deadline - 20:
        for trial in range(15):
            if time.time() > deadline - 5:
                break

            # Perturb best solution
            rng = np.random.RandomState(500 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.012, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # SA from perturbed
            sa_centers, sa_score = simulated_annealing(
                perturbed,
                max_iters=10000,
                T_start=0.02,
                T_end=0.001,
                seed=1000 + trial
            )

            # Local search
            ls_centers, ls_score = local_search(
                sa_centers,
                max_iters=1000,
                step_size=0.005,
                seed=2000 + trial
            )

            # Pairwise gradient polish (direct on nonsmooth)
            pg_centers, pg_score = pairwise_gradient_polish(ls_centers, max_iters=300, lr=0.002)
            if pg_score > ls_score:
                ls_centers, ls_score = pg_centers, pg_score

            # Smooth gradient polish
            polished = gradient_polish(ls_centers, max_iters=150, beta=200.0)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

    # Second pass: more local search from best result with finer steps
    if best_result is not None and time.time() < deadline - 10:
        for trial in range(8):
            if time.time() > deadline - 5:
                break

            # Slight perturbation
            rng = np.random.RandomState(3000 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.006, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # Fine local search
            ls_centers, ls_score = local_search(
                perturbed,
                max_iters=1200,
                step_size=0.003,
                seed=4000 + trial
            )

            # Pairwise gradient polish
            pg_centers, pg_score = pairwise_gradient_polish(ls_centers, max_iters=400, lr=0.001)
            if pg_score > ls_score:
                ls_centers, ls_score = pg_centers, pg_score

            # Smooth gradient polish
            polished = gradient_polish(ls_centers, max_iters=150, beta=300.0)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            elif ls_score > best_score:
                best_score = ls_score
                radii = compute_radii_vectorized(ls_centers)
                best_result = np.column_stack([ls_centers, radii])

    # Ensure final validity with small margin
    if best_result is not None:
        # Apply tiny margin to ensure strict validity
        radii = compute_radii_vectorized(best_result[:, :2])
        radii = np.maximum(0, radii - 1e-9)
        best_result = np.column_stack([best_result[:, :2], radii])

        if not is_valid(best_result, tol=1e-6):
            # Fallback: conservative grid
            centers = grid_init()
            radii = compute_radii_vectorized(centers)
            radii = np.maximum(0, radii - 0.001)
            best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
