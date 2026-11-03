import numpy as np
import time
import sys

N = 50
DELTA = 0.5 / N


def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return l2_sq / (l1 * autoconv_max)


def _smooth_objective(h, tau):
    """Returns (-score, gradient) for smooth max surrogate."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0, np.zeros_like(h)
    l2_sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    max_c = np.max(conv)
    if max_c < 1e-12:
        return 0.0, np.zeros_like(h)
    # log-sum-exp for smooth max
    shifted = (conv - max_c) / tau
    log_sum_exp = max_c + tau * np.log(np.sum(np.exp(shifted)))
    smooth_max = log_sum_exp
    if smooth_max < 1e-12:
        return 0.0, np.zeros_like(h)

    score = l2_sq / (l1 * DELTA * smooth_max)

    # Gradient: d(score)/d(h_i)
    # score = l2_sq / (l1 * DELTA * smooth_max)
    # d(score)/d(h_i) = [2*h_i * (l1*DELTA*smooth_max) - l2_sq * (DELTA*smooth_max + l1*DELTA*d(smooth_max)/d(h_i))] / (l1*DELTA*smooth_max)^2
    # d(smooth_max)/d(h_i) = sum_j conv_j * d(conv_j)/d(h_i) / sum(exp(shifted))
    # conv_j = sum_k h_k * h_{j-k}, d(conv_j)/d(h_i) = h_{j-i} + h_{j-(N-1-i)} (with bounds)

    denom = l1 * DELTA * smooth_max
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    # weights for smooth max
    w = exp_shifted / sum_exp  # shape (2N-1,)

    # d(smooth_max)/d(h_i) = sum_j w_j * (h_{j-i} + h_{j-(N-1-i)})
    # conv index j corresponds to sum over k: h[k]*h[j-k] where j-k in [0,N-1], k in [0,N-1]
    # d(conv_j)/d(h_i) = h[j-i] if 0<=j-i<N, plus h[j-(N-1-i)] if 0<=j-(N-1-i)<N
    # Note: conv is full mode, index j = 0..2N-2
    # conv[j] = sum_{k: 0<=k<N, 0<=j-k<N} h[k]*h[j-k]
    # d(conv[j])/d(h[i]) = h[j-i] (if 0<=j-i<N) + h[j-(N-1-i)] (if 0<=j-(N-1-i)<N)

    dsm = np.zeros(N)
    for i in range(N):
        # term 1: h[j-i] where j ranges 0..2N-2, j-i in [0,N-1] => j in [i, i+N-1]
        j1_start = i
        j1_end = min(i + N, 2 * N - 1)
        if j1_start <= j1_end:
            idx1 = j1_start - i  # 0..N-1
            dsm[i] += np.sum(w[j1_start:j1_end + 1] * h[idx1])
        # term 2: h[j-(N-1-i)] where j-(N-1-i) in [0,N-1] => j in [N-1-i, 2N-2-i]
        j2_start = max(N - 1 - i, 0)
        j2_end = min(2 * N - 2 - i, 2 * N - 2)
        if j2_start <= j2_end:
            idx2 = j2_start - (N - 1 - i)
            dsm[i] += np.sum(w[j2_start:j2_end + 1] * h[idx2])

    # d(score)/d(h_i)
    num = 2.0 * h * denom - l2_sq * (DELTA * smooth_max + l1 * DELTA * dsm)
    grad = num / (denom * denom)

    return -score, -grad


def _lbfgs_local(h_init, tau, max_iter=2000, deadline=None):
    """L-BFGS-B with analytic gradients."""
    from scipy.optimize import minimize
    h_init = np.clip(h_init, 0, None)
    s = np.sum(h_init)
    if s < 1e-12:
        return h_init, 0.0
    h_init = h_init / s

    def neg_obj(h):
        val, _ = _smooth_objective(h, tau)
        return val

    def neg_grad(h):
        _, g = _smooth_objective(h, tau)
        return g

    res = minimize(neg_obj, h_init, jac=neg_grad, method='L-BFGS-B',
                   bounds=[(0, None)] * N,
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                            'maxls': 50})
    h_out = np.clip(res.x, 0, None)
    s_out = np.sum(h_out)
    if s_out > 1e-12:
        h_out = h_out / s_out
    return h_out, _score_raw(h_out)


def _differential_evolution(h_init, tau, pop_size=40, max_gen=80,
                            crossover=0.8, mutation=(0.5, 1.0),
                            deadline=None):
    """Differential evolution for global search."""
    h_init = np.clip(h_init, 0, None)
    s = np.sum(h_init)
    if s < 1e-12:
        return h_init, 0.0
    h_init = h_init / s

    rng = np.random.RandomState(42)
    pop = np.zeros((pop_size, N))
    pop[0] = h_init
    for i in range(1, pop_size):
        # diverse random initializations
        k = rng.randint(2, 20)
        idx = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = rng.rand(k)
        s_h = np.sum(h)
        if s_h > 0:
            h /= s_h
        else:
            h = np.ones(N) / N
        pop[i] = h

    def eval_h(h):
        h_c = np.clip(h, 0, None)
        s_c = np.sum(h_c)
        if s_c < 1e-12:
            return -1e10
        h_c = h_c / s_c
        val, _ = _smooth_objective(h_c, tau)
        return val

    fitness = np.array([eval_h(p) for p in pop])
    best_idx = np.argmin(fitness)  # minimizing -score
    best_h = pop[best_idx].copy()
    best_val = fitness[best_idx]

    for gen in range(max_gen):
        if deadline is not None and time.time() > deadline:
            break
        for i in range(pop_size):
            if deadline is not None and time.time() > deadline:
                break
            # select 3 distinct random individuals
            a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            # mutation
            F = rng.uniform(*mutation)
            mutant = pop[a] + F * (pop[b] - pop[c])
            # crossover
            cross_mask = rng.random(N) < crossover
            trial = np.where(cross_mask, mutant, pop[i])
            # normalize and clip
            trial = np.clip(trial, 0, None)
            s_t = np.sum(trial)
            if s_t < 1e-12:
                trial = pop[i].copy()
            else:
                trial = trial / s_t
            # evaluate
            trial_val = eval_h(trial)
            if trial_val < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_val
                if trial_val < best_val:
                    best_val = trial_val
                    best_h = trial.copy()
        # elitism: keep best
        bi = np.argmin(fitness)
        if fitness[bi] < best_val:
            best_val = fitness[bi]
            best_h = pop[bi].copy()

    return best_h, _score_raw(best_h)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Initial diverse candidates
    x_idx = np.arange(N)
    inits = []

    # Uniform
    inits.append(np.ones(N) / N)

    # Gaussian peaks
    for c in np.linspace(3, N - 3, 10):
        for w in [2, 4, 7, 12, 20]:
            h_g = np.exp(-0.5 * ((x_idx - c) / w) ** 2)
            s = np.sum(h_g)
            if s > 0:
                inits.append(h_g / s)

    # Triangular
    for c in np.linspace(3, N - 3, 8):
        for w in [3, 5, 10, 15]:
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            s = np.sum(h_t)
            if s > 0:
                inits.append(h_t / s)

    # Sparse random
    rng = np.random.RandomState(123)
    for _ in range(50):
        k = rng.randint(3, 18)
        idx = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = rng.rand(k)
        s = np.sum(h)
        if s > 0:
            inits.append(h / s)

    # Alternating
    for period in [2, 3, 4, 5, 7]:
        h = np.zeros(N)
        h[x_idx % period == 0] = 1.0
        s = np.sum(h)
        if s > 0:
            inits.append(h / s)

    # Two/three peaks
    for _ in range(30):
        k = rng.randint(2, 5)
        idx = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        s = np.sum(h)
        if s > 0:
            inits.append(h / s)

    # Evaluate all initializations
    best_h = inits[0].copy()
    best_score = _score_raw(best_h)
    for h_i in inits:
        s_i = _score_raw(h_i)
        if s_i > best_score:
            best_score = s_i
            best_h = h_i.copy()

    # Sort by score, take top for optimization
    scored = [(_score_raw(h), h) for h in inits]
    scored.sort(key=lambda x: -x[0])
    top_inits = [h for s, h in scored[:10]]

    # Phase 1: Differential evolution for global search (coarse tau)
    tau_coarse = 0.1
    for h_init in top_inits[:5]:
        if time.time() > deadline - 30:
            break
        h_de, s_de = _differential_evolution(h_init, tau_coarse,
                                             pop_size=30, max_gen=60,
                                             deadline=deadline - 20)
        if s_de > best_score:
            best_score = s_de
            best_h = h_de

    # Phase 2: L-BFGS-B local optimization with decreasing tau
    # Start from best and a few diverse points
    local_starts = [best_h.copy()]
    for h_i in top_inits[:3]:
        if not any(np.allclose(h_i, s) for s in local_starts):
            local_starts.append(h_i.copy())

    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for h_start in local_starts[:4]:
        if time.time() > deadline - 15:
            break
        current = h_start.copy()
        for tau in taus:
            if time.time() > deadline - 10:
                break
            max_iter = 1500 if tau > 0.01 else 3000
            h_loc, s_loc = _lbfgs_local(current, tau, max_iter=max_iter,
                                        deadline=deadline - 5)
            if s_loc > best_score:
                best_score = s_loc
                best_h = h_loc
            current = h_loc

    # Phase 3: Final polish with very tight tau
    if time.time() < deadline - 5:
        h_final, s_final = _lbfgs_local(best_h, 1e-4, max_iter=5000,
                                        deadline=deadline - 1)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final

    # Ensure valid output
    best_h = np.clip(best_h, 0, None)
    s_total = np.sum(best_h)
    if s_total < 1e-12:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s_total

    return list(best_h)
