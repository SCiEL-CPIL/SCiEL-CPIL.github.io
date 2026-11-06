"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Hard deadline: 135 seconds (15s safety margin for 150s kill)
DEADLINE = 135.0


def _covering_radius_rad(points):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    Uses convex hull facets and spherical triangle circumradius.
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        dots = np.clip(points @ points.T, -1.0, 1.0)
        dists = np.arccos(dots)
        return np.max(dists)

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]

        ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))

        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            r = max(ab, bc, ca) / 2
            max_r = max(max_r, r)
            continue
        n = n / n_norm

        cos_a, cos_b, cos_c = np.cos(ab), np.cos(bc), np.cos(ca)

        numerator = cos_a * cos_b * cos_c - cos_a - cos_b - cos_c + 1
        denominator = 2 * (1 - cos_a - cos_b - cos_c + cos_a * cos_b + cos_b * cos_c + cos_c * cos_a)

        if abs(denominator) < 1e-10:
            r = max(ab, bc, ca) / 2
            max_r = max(max_r, r)
            continue

        cos_2r = numerator / denominator
        cos_2r = np.clip(cos_2r, -1.0, 1.0)
        r = 0.5 * np.arccos(cos_2r)

        max_r = max(max_r, r)

    return max_r


def _objective_and_grad(x_flat, query_points, beta):
    """
    Smooth objective with analytical gradient for covering radius approximation.
    """
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    x = x / norms

    # cos_dists: (M, N)
    cos_dists = query_points @ x.T
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    
    # Chordal distance squared
    chordal_sq = 2.0 * (1.0 - cos_dists)  # (M, N)
    
    # Min over N for each query point
    min_idx = np.argmin(chordal_sq, axis=1)
    min_vals = chordal_sq[np.arange(len(min_idx)), min_idx]
    
    # Soft-min approximation using log-sum-exp
    scaled = beta * chordal_sq  # (M, N)
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled, axis=1)
    
    # lse = max_scaled + log(sum_exp), then divide by beta
    lse_per_query = (max_scaled.squeeze() + np.log(sum_exp)) / beta
    
    obj = np.mean(lse_per_query)
    
    # Gradient
    # d/dx_j [min_i (2(1 - q·x_i))] approx via soft min
    # For each query q, weight w_j = exp(beta*chordal_sq[q,j]) / sum
    weights = exp_scaled / sum_exp[:, np.newaxis]  # (M, N)
    
    # d chordal_sq[q,j]/dx_j = 2 * q  (since d(1 - q·x_j)/dx_j = -q, d(2(1-q·x_j))/dx_j = -2q)
    # Wait: chordal_sq = 2(1 - q·x), so d/dx = -2q
    # But we have soft min of chordal_sq, and we want gradient of mean of soft min
    
    # grad_obj = (1/M) * sum_q sum_j weights[q,j] * d(chordal_sq[q,j])/dx_j
    # = (1/M) * sum_q sum_j weights[q,j] * (-2 * query_points[q])
    # = (1/M) * sum_q (-2 * query_points[q]) * sum_j weights[q,j]
    # = (1/M) * sum_q (-2 * query_points[q]) * 1  (since weights sum to 1)
    # = (1/M) * sum_q (-2 * query_points[q])
    
    # Wait, that doesn't depend on x! Let me reconsider.
    
    # Actually the soft min is: LSE(x) = (1/beta) * log(sum_j exp(beta * chordal_sq[q,j]))
    # d LSE / dx_j = (1/beta) * (1/sum) * exp(beta*chordal_sq[q,j]) * beta * d(chordal_sq[q,j])/dx_j
    # = weights[q,j] * d(chordal_sq[q,j])/dx_j
    # = weights[q,j] * (-2 * query_points[q])
    
    # So grad_obj = (1/M) * sum_q sum_j weights[q,j] * (-2 * query_points[q])
    # But this is a vector of shape (N, 3)
    # grad_obj[j] = (1/M) * sum_q weights[q,j] * (-2 * query_points[q])
    
    # This can be computed as: -2/M * (weights.T @ query_points)
    grad = (-2.0 / len(query_points)) * (weights.T @ query_points)  # (N, 3)
    
    # Now we need to account for the normalization x = x/norm
    # Let x_raw be the raw variable, x_norm = x_raw / ||x_raw||
    # dx_norm/dx_raw involves the Jacobian of normalization
    
    # For simplicity, use the chain rule:
    # If f(x_norm) where x_norm = x_raw / ||x_raw||
    # grad_x_raw = J^T * grad_x_norm
    # where J is the Jacobian of normalization
    
    # The Jacobian of x/||x|| w.r.t. x is (I - x x^T / ||x||^2) / ||x||
    # So grad_x_raw[j] = (I - x_norm[j] x_norm[j]^T) * grad_x_norm[j] / ||x_raw[j]||
    
    grad_raw = np.zeros_like(x_flat)
    for j in range(N):
        n_j = np.linalg.norm(x_raw_j := x_flat[3*j:3*j+3])
        if n_j < 1e-10:
            continue
        proj = np.outer(x, x)[j]
        proj = np.outer(x[j], x[j])
        grad_raw[3*j:3*j+3] = ((np.eye(3) - proj) @ grad[j]) / n_j
    
    return obj, grad_raw.flatten()


def construct():
    start_time = time.time()

    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = points.copy()
    best_radius = _covering_radius_rad(best_points)

    # Generate random query points for smooth objective
    np.random.seed(42)
    M = 1500
    u = np.random.uniform(-1, 1, (M, 3))
    norms_u = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / norms_u

    # Phase 1: Gradient-based optimization with smooth objective
    try:
        result = minimize(
            lambda x: _objective_and_grad(x, u, beta=15.0)[0],
            lambda x: _objective_and_grad(x, u, beta=15.0)[1],
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )

        if result.x is not None:
            opt_points = result.x.reshape(N, 3)
            opt_radius = _covering_radius_rad(opt_points)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 40:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return best_points / norms

    # Phase 2: Powell with raw objective
    try:
        result = minimize(
            lambda x: _covering_radius_rad(x.reshape(N, 3)),
            best_points.flatten(),
            method='Powell',
            options={'maxiter': 1000, 'maxfev': 2000, 'disp': False}
        )

        if result.x is not None:
            opt_points = result.x.reshape(N, 3)
            opt_radius = _covering_radius_rad(opt_points)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 25:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return best_points / norms

    # Phase 3: Simulated annealing with exact hull evaluation
    if time.time() - start_time < DEADLINE - 15:
        T_init = 0.01
        T_min = 0.001
        cooling = 0.995
        T = T_init
        current_points = best_points.copy()
        current_radius = best_radius
        n_iter_max = 200
        
        for it in range(n_iter_max):
            if time.time() - start_time > DEADLINE - 10:
                break
            if T < T_min:
                break
            
            # Propose a move: perturb a random point
            idx = np.random.randint(N)
            step = 0.05 * T / T_init + 0.01
            noise = np.random.randn(3) * step
            new_points = current_points.copy()
            new_points[idx] = new_points[idx] + noise
            new_norm = np.linalg.norm(new_points[idx])
            if new_norm > 1e-10:
                new_points[idx] = new_points[idx] / new_norm
            
            new_radius = _covering_radius_rad(new_points)
            delta = new_radius - current_radius
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_points = new_points
                current_radius = new_radius
                if current_radius < best_radius:
                    best_points = current_points.copy()
                    best_radius = current_radius
            
            T *= cooling

    # Phase 4: Final fine-tuning with Powell
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                lambda x: _covering_radius_rad(x.reshape(N, 3)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 1000, 'disp': False}
            )

            if result.x is not None:
                opt_points = result.x.reshape(N, 3)
                opt_radius = _covering_radius_rad(opt_points)
                if opt_radius < best_radius:
                    best_points = opt_points
                    best_radius = opt_radius
        except Exception:
            pass

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms

    return best_points
# EVOLVE-BLOCK-END
