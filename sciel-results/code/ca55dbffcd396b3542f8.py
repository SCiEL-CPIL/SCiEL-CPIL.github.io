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

    rng = np.random.default_rng(42)

    # Phase 1: Systematic grid exploration on sphere with multiple radii
    # The bound is homogeneous in the coefficients, so we can fix radius=1
    # and search over the unit sphere (theta1, theta2)
    
    # First, do a coarse grid on the unit sphere
    n_theta1 = 36
    n_theta2 = 18
    
    for i in range(n_theta1):
        if time.time() > _DEADLINE - 80:
            break
        for j in range(n_theta2):
            if time.time() > _DEADLINE - 80:
                break
            t1 = -np.pi + (2 * np.pi * i) / n_theta1
            t2 = np.pi * j / (n_theta2 - 1) if n_theta2 > 1 else np.pi/2
            
            # Try multiple radii
            for r in [0.5, 1.0, 2.0, 5.0, 10.0]:
                b = _bound_from_sphere(t1, t2, r)
                if b < best_bound:
                    best_bound = b
                    st2 = np.sin(t2)
                    best_params = np.array([
                        r * st2 * np.cos(t1),
                        r * st2 * np.sin(t1),
                        r * np.cos(t2)
                    ])

    # Phase 2: Random sampling on sphere with various radii
    n_random = 500
    for _ in range(n_random):
        if time.time() > _DEADLINE - 70:
            break
        t1 = rng.uniform(-np.pi, np.pi)
        t2 = rng.uniform(0, np.pi)
        r = 10 ** rng.uniform(-1.0, 1.5)  # radius from 0.1 to ~30
        
        b = _bound_from_sphere(t1, t2, r)
        if b < best_bound:
            best_bound = b
            st2 = np.sin(t2)
            best_params = np.array([
                r * st2 * np.cos(t1),
                r * st2 * np.sin(t1),
                r * np.cos(t2)
            ])

    # Phase 3: Convert best to spherical coords and do local optimization
    r_best = np.linalg.norm(best_params)
    if r_best < 1e-15:
        r_best = 1.0
    log_r_best = np.log(r_best)
    t1_best = np.arctan2(best_params[1], best_params[0])
    t2_best = np.arccos(np.clip(best_params[2] / r_best, -1.0, 1.0))

    # Nelder-Mead style local search in (theta1, theta2, log_r)
    def _local_optimize(start_state, max_iters=5000):
        """Nelder-Mead optimization in spherical coordinates."""
        # Initial simplex
        step = 0.1
        simplex = np.array([
            start_state,
            start_state + np.array([step, 0, 0]),
            start_state + np.array([0, step, 0]),
            start_state + np.array([0, 0, step])
        ])
        
        def _eval_state(s):
            s = s.copy()
            s[0] = (s[0] + np.pi) % (2 * np.pi) - np.pi
            s[1] = np.clip(s[1], 0.001, np.pi - 0.001)
            s[2] = np.clip(s[2], -3.0, 4.0)
            val = _bound_from_sphere(s[0], s[1], np.exp(s[2]))
            return val if np.isfinite(val) else 1e18
        
        vals = np.array([_eval_state(s) for s in simplex])
        best_val = np.min(vals)
        best_state = simplex[np.argmin(vals)].copy()
        
        alpha, gamma, rho = 1.0, 2.0, 0.5  # reflection, expansion, contraction
        
        for it in range(max_iters):
            if time.time() > _DEADLINE - 20:
                break
                
            # Sort simplex
            order = np.argsort(vals)
            simplex = simplex[order]
            vals = vals[order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            reflected = centroid + alpha * (centroid - worst)
            reflected[0] = (reflected[0] + np.pi) % (2 * np.pi) - np.pi
            reflected[1] = np.clip(reflected[1], 0.001, np.pi - 0.001)
            reflected[2] = np.clip(reflected[2], -3.0, 4.0)
            ref_val = _eval_state(reflected)
            
            if ref_val < vals[0]:
                # Expansion
                expanded = centroid + gamma * (reflected - centroid)
                expanded[0] = (expanded[0] + np.pi) % (2 * np.pi) - np.pi
                expanded[1] = np.clip(expanded[1], 0.001, np.pi - 0.001)
                expanded[2] = np.clip(expanded[2], -3.0, 4.0)
                exp_val = _eval_state(expanded)
                
                if exp_val < ref_val:
                    simplex[-1] = expanded
                    vals[-1] = exp_val
                else:
                    simplex[-1] = reflected
                    vals[-1] = ref_val
            elif ref_val < vals[-1]:
                simplex[-1] = reflected
                vals[-1] = ref_val
            else:
                # Contraction
                if ref_val < vals[-1]:
                    contracted = centroid + rho * (reflected - centroid)
                    contract_type = 'outer'
                else:
                    contracted = centroid - rho * (worst - centroid)
                    contract_type = 'inner'
                
                contracted[0] = (contracted[0] + np.pi) % (2 * np.pi) - np.pi
                contracted[1] = np.clip(contracted[1], 0.001, np.pi - 0.001)
                contracted[2] = np.clip(contracted[2], -3.0, 4.0)
                con_val = _eval_state(contracted)
                
                if con_val < vals[-1]:
                    simplex[-1] = contracted
                    vals[-1] = con_val
                else:
                    # Shrink
                    for i in range(1, 4):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i][0] = (simplex[i][0] + np.pi) % (2 * np.pi) - np.pi
                        simplex[i][1] = np.clip(simplex[i][1], 0.001, np.pi - 0.001)
                        simplex[i][2] = np.clip(simplex[i][2], -3.0, 4.0)
                        vals[i] = _eval_state(simplex[i])
            
            # Track best
            cur_best_idx = np.argmin(vals)
            if vals[cur_best_idx] < best_val:
                best_val = vals[cur_best_idx]
                best_state = simplex[cur_best_idx].copy()
        
        return best_state, best_val

    # Run local optimization from multiple starting points
    n_local_starts = 8
    for _ in range(n_local_starts):
        if time.time() > _DEADLINE - 30:
            break
        
        # Start near current best with small perturbation
        pert = rng.normal(0, 0.2, 3)
        start = np.array([t1_best, t2_best, log_r_best]) + pert
        start[0] = (start[0] + np.pi) % (2 * np.pi) - np.pi
        start[1] = np.clip(start[1], 0.01, np.pi - 0.01)
        start[2] = np.clip(start[2], -3.0, 4.0)
        
        opt_state, opt_val = _local_optimize(start, max_iters=3000)
        
        if opt_val < best_bound:
            best_bound = opt_val
            best_params = np.array([
                np.exp(opt_state[2]) * np.sin(opt_state[1]) * np.cos(opt_state[0]),
                np.exp(opt_state[2]) * np.sin(opt_state[1]) * np.sin(opt_state[0]),
                np.exp(opt_state[2]) * np.cos(opt_state[1])
            ])
            t1_best = opt_state[0]
            t2_best = opt_state[1]
            log_r_best = opt_state[2]

    # Phase 4: Fine-tuning with very small steps
    fine_steps = 0.01
    for _ in range(2000):
        if time.time() > _DEADLINE - 10:
            break
        
        delta = rng.normal(0, fine_steps, 3)
        new_state = np.array([t1_best, t2_best, log_r_best]) + delta
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        new_state[1] = np.clip(new_state[1], 0.001, np.pi - 0.001)
        new_state[2] = np.clip(new_state[2], -3.0, 4.0)
        
        new_val = _bound_from_sphere(new_state[0], new_state[1], np.exp(new_state[2]))
        if np.isfinite(new_val) and new_val < best_bound:
            best_bound = new_val
            t1_best = new_state[0]
            t2_best = new_state[1]
            log_r_best = new_state[2]
            best_params = np.array([
                np.exp(new_state[2]) * np.sin(new_state[1]) * np.cos(new_state[0]),
                np.exp(new_state[2]) * np.sin(new_state[1]) * np.sin(new_state[0]),
                np.exp(new_state[2]) * np.cos(new_state[1])
            ])
        else:
            fine_steps *= 0.995

    # Convert back to (a0, a4, a8)
    r_final = np.exp(log_r_best)
    st2 = np.sin(t2_best)
    result = np.array([
        r_final * st2 * np.cos(t1_best),
        r_final * st2 * np.sin(t1_best),
        r_final * np.cos(t2_best)
    ])

    # Verify the result
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        # Fall back to seed
        return seed.copy()

    return result
# EVOLVE-BLOCK-END
