# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0

# Precompute integrals and matrices
_sp = np.sqrt(np.pi)
_I = np.zeros(13)
_I[0] = _sp
for _n in range(1, 13):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

_M = np.zeros((3, 7, 7))
for _k in range(3):
    for _i in range(7):
        for _j in range(7):
            _idx = _k + _i + _j
            if _idx < 13:
                _M[_k, _i, _j] = _I[_idx]

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_bound(a0, a4, a8):
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    m = np.einsum('i,kij,j->k', p, _M, p)
    q = p * _SIGNS
    n = np.einsum('i,kij,j->k', q, _M, q) * 2.0 * np.pi

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
    st2 = np.sin(theta2)
    a0 = radius * st2 * np.cos(theta1)
    a4 = radius * st2 * np.sin(theta1)
    a8 = radius * np.cos(theta2)
    return _compute_bound(a0, a4, a8)


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        theta1_seed = np.arctan2(seed[1], seed[0])
        theta2_seed = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0))
    else:
        theta1_seed = 0.0
        theta2_seed = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Broad exploration
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

    # Neighborhood of seed
    for r in [0.8 * r_seed, 0.9 * r_seed, 1.0 * r_seed, 1.1 * r_seed, 1.2 * r_seed]:
        if time.time() > _DEADLINE - 60:
            break
        for _ in range(30):
            dt1 = rng.normal(0, 0.3)
            dt2 = rng.normal(0, 0.3)
            t1 = theta1_seed + dt1
            t2 = theta1_seed + dt2 # Bug in original: should be theta2_seed
            # Fixing the bug from original code where t2 was assigned theta1_seed + dt2
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

    # Phase 2: Adaptive stochastic local search
    r_best = np.linalg.norm(best_params)
    if r_best < 1e-15:
        r_best = 1.0
    log_r_best = np.log(r_best)
    t1_best = np.arctan2(best_params[1], best_params[0])
    t2_best = np.arccos(np.clip(best_params[2] / r_best, -1.0, 1.0))

    state = np.array([t1_best, t2_best, log_r_best])
    best_state = state.copy()
    best_val = best_bound

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

        delta = rng.normal(0, 1, 3) * steps
        new_state = state + delta
        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        new_state[1] = np.clip(new_state[1], 0.01, np.pi - 0.01)
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
            steps *= 0.98
        else:
            if delta_val < 1e-10 or np.exp(-delta_val / temperature) > rng.random():
                state = new_state
                accept_count += 1
                steps *= 1.02
            else:
                steps *= 1.05

        if total_count % 100 == 0:
            ratio = accept_count / total_count
            if ratio > 0.6:
                steps *= 1.1
            elif ratio < 0.3:
                steps *= 0.8
            accept_count = 0
            total_count = 0

        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

        iter_count += 1

    # Phase 3: Fine-tuning
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

    # Phase 4: Polish with Nelder-Mead
    if time.time() < _DEADLINE - 5:
        r_final = np.exp(best_state[2])
        t1_final = best_state[0]
        t2_final = best_state[1]
        st2 = np.sin(t2_final)
        x0 = np.array([
            r_final * st2 * np.cos(t1_final),
            r_final * st2 * np.sin(t1_final),
            r_final * np.cos(t2_final)
        ])
        
        def objective(x):
            return _compute_bound(x[0], x[1], x[2])
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Verify and return
    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()
    return best_params
# EVOLVE-BLOCK-END
