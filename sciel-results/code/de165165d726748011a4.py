"""Uncertainty bound minimization via spherical reparameterization and adaptive stochastic local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195.0


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
    # Precompute up to n=12 (need x^{24})
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        # (2n)! / (4^n * n!) * sqrt(pi)
        # Recurrence: I[n] = I[n-1] * (2n-1) / 2
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _bound_from_sphere(theta1, theta2, radius):
    """
    Convert spherical coordinates to (a0, a4, a8) and compute bound.
    a0 = radius * sin(theta2) * cos(theta1)
    a4 = radius * sin(theta2) * sin(theta1)
    a8 = radius * cos(theta2)
    """
    st2 = np.sin(theta2)
    a0 = radius * st2 * np.cos(theta1)
    a4 = radius * st2 * np.sin(theta1)
    a8 = radius * np.cos(theta2)
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses spherical reparameterization with adaptive stochastic local search.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    # Convert seed to spherical coordinates
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        theta1_seed = np.arctan2(seed[1], seed[0])
        theta2_seed = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0))
    else:
        theta1_seed = 0.0
        theta2_seed = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Broad exploration with multiple starting points on the sphere
    # Try different radii and angles
    radii = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    n_angles = 50

    for r in radii:
        if time.time() > _DEADLINE - 60:
            break
        for _ in range(n_angles):
            if time.time() > _DEADLINE - 60:
                break
            t1 = rng.uniform(-np.pi, np.pi)
            t2 = rng.uniform(0, np.pi)
            b = _bound_from_sphere(t1, t2, r)
            if b < best_bound:
                best_bound = b
                st2 = np.sin(t2)
                best_params = np.array([
                    r * st2 * np.cos(t1),
                    r * st2 * np.sin(t1),
                    r * np.cos(t2)
                ])

    # Also try the seed point's neighborhood
    for r in [0.8 * r_seed, 0.9 * r_seed, 1.0 * r_seed, 1.1 * r_seed, 1.2 * r_seed]:
        if time.time() > _DEADLINE - 60:
            break
        for _ in range(30):
            dt1 = rng.normal(0, 0.3)
            dt2 = rng.normal(0, 0.3)
            t1 = theta1_seed + dt1
            t2 = theta2_seed + dt2
            b = _bound_from_sphere(t1, t2, r)
            if b < best_bound:
                best_bound = b
                st2 = np.sin(t2)
                best_params = np.array([
                    r * st2 * np.cos(t1),
                    r * st2 * np.sin(t1),
                    r * np.cos(t2)
                ])

    # Phase 2: Adaptive stochastic local search (simulated annealing style)
    # Work in spherical coordinates (theta1, theta2, log_radius)
    r_best = np.linalg.norm(best_params)
    if r_best < 1e-15:
        r_best = 1.0
    log_r_best = np.log(r_best)
    t1_best = np.arctan2(best_params[1], best_params[0])
    t2_best = np.arccos(np.clip(best_params[2] / r_best, -1.0, 1.0))

    # State: (theta1, theta2, log_r)
    state = np.array([t1_best, t2_best, log_r_best])
    best_state = state.copy()
    best_val = best_bound

    # Adaptive step sizes
    steps = np.array([0.1, 0.1, 0.1])
    accept_count = 0
    total_count = 0
    temperature = 1.0
    cooling = 0.9995

    max_iters = 30000
    iter_count = 0

    while iter_count < max_iters:
        if time.time() > _DEADLINE - 10:
            break

        # Generate perturbation
        delta = rng.normal(0, 1, 3) * steps
        new_state = state + delta
        # Wrap theta1
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        # Clamp theta2
        new_state[1] = np.clip(new_state[1], 0.01, np.pi - 0.01)
        # Clamp log_r
        new_state[2] = np.clip(new_state[2], -3.0, 4.0)

        new_val = _bound_from_sphere(new_state[0], new_state[1], np.exp(new_state[2]))

        if not np.isfinite(new_val):
            new_val = 1e18

        delta_val = new_val - best_val
        total_count += 1

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            accept_count += 1
            # Shrink steps slightly on improvement
            steps *= 0.98
        else:
            # Accept with probability exp(-delta_val / temperature)
            if delta_val < 1e-10 or np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                steps *= 1.02
            else:
                steps *= 1.05

        # Adaptive step size adjustment
        if total_count % 100 == 0:
            ratio = accept_count / total_count
            if ratio > 0.6:
                steps *= 1.1
            elif ratio < 0.3:
                steps *= 0.8
            accept_count = 0
            total_count = 0

        # Cool down
        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

        iter_count += 1

    # Phase 3: Fine-tuning with smaller steps
    steps = np.array([0.01, 0.01, 0.01])
    state = best_state.copy()
    temperature = 1e-3
    cooling = 0.999

    for _ in range(20000):
        if time.time() > _DEADLINE - 5:
            break

        delta = rng.normal(0, 1, 3) * steps
        new_state = state + delta
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        new_state[1] = np.clip(new_state[1], 0.001, np.pi - 0.001)
        new_state[2] = np.clip(new_state[2], -3.0, 4.0)

        new_val = _bound_from_sphere(new_state[0], new_state[1], np.exp(new_state[2]))
        if not np.isfinite(new_val):
            new_val = 1e18

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            steps *= 0.97
        else:
            if new_val - best_val < 1e-12 or np.exp(-(new_val - best_val) / temperature) > rng.random():
                state = new_state
                steps *= 1.01
            else:
                steps *= 1.03

        temperature *= cooling
        if temperature < 1e-12:
            temperature = 1e-12

    # Convert back to (a0, a4, a8)
    r_final = np.exp(best_state[2])
    t1_final = best_state[0]
    t2_final = best_state[1]
    st2 = np.sin(t2_final)
    result = np.array([
        r_final * st2 * np.cos(t1_final),
        r_final * st2 * np.sin(t1_final),
        r_final * np.cos(t2_final)
    ])

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        # Fall back to seed
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
