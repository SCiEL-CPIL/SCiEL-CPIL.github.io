import numpy as np
import time

def priority(adjacency):
    """
    Priority scores for a greedy Maximum Independent Set decoder.
    Uses continuous soft surrogate optimization followed by discrete swap polish.
    """
    start_time = time.time()
    DEADLINE = 140.0

    A = np.asarray(adjacency)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    A = (A + A.T)
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    A_bool = (A_bool | A_bool.T)
    np.fill_diagonal(A_bool, False)

    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    deg = A_bool.sum(axis=1).astype(float)

    def simulate(s):
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs.size:
                    active[nbrs] = False
        return count

    def normalize(x):
        lo, hi = x.min(), x.max()
        if hi > lo:
            return (x - lo) / (hi - lo)
        return np.zeros_like(x)

    neighbor_deg_sum = A_bool.astype(float).dot(deg)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    base = -(0.7 * normalize(deg) + 0.3 * normalize(avg_neighbor_deg))
    base = normalize(base)

    best_val = simulate(base)
    best_scores = base.copy()

    def soft_value(s, beta):
        order = np.argsort(-s, kind='stable')
        log_surv = np.zeros(N)
        pos = np.empty(N, dtype=int)
        pos[order] = np.arange(N)
        pick = np.zeros(N)
        for v in order:
            nbrs = neighbors_list[v]
            if nbrs.size:
                earlier = nbrs[pos[nbrs] < pos[v]]
                if earlier.size:
                    log_surv[v] = np.log(np.clip(1.0 - pick[earlier], 1e-9, 1.0)).sum()
            w = np.exp(beta * s[v] + log_surv[v])
            pick[v] = w
        total = (pick * np.exp(log_surv)).sum()
        return total

    def optimize_continuous(s0, beta, iters, lr, deadline):
        s = s0.copy()
        cur = soft_value(s, beta)
        for it in range(iters):
            if time.time() - start_time > deadline:
                break
            h = 1e-3
            grad = np.zeros(N)
            for k in range(N):
                sp = s.copy(); sp[k] += h
                sm = s.copy(); sm[k] -= h
                grad[k] = (soft_value(sp, beta) - soft_value(sm, beta)) / (2 * h)
            s = s + lr * grad
            s = normalize(s)
            new = soft_value(s, beta)
            if new <= cur:
                s = normalize(s0 if it == 0 else s)
                break
            s0 = s.copy()
            cur = new
        return s

    rng = np.random.default_rng(123)

    beta = 6.0
    for r in range(4):
        if time.time() - start_time > DEADLINE - 40:
            break
        if r == 0:
            s0 = best_scores.copy()
        else:
            s0 = best_scores + rng.normal(0, 0.15, N)
            s0 = normalize(s0)
        s_opt = optimize_continuous(s0, beta, iters=60, lr=0.05, deadline=DEADLINE - 40)
        val = simulate(s_opt)
        if val > best_val:
            best_val = val
            best_scores = s_opt.copy()

    score_order = np.argsort(-best_scores)
    current = best_scores.copy()
    cur_val = best_val
    no_improve = 0
    max_no_improve = 4000
    while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
        if rng.random() < 0.6:
            base_idx = rng.integers(0, N - 1)
            offset = rng.integers(1, min(20, N - 1))
            i = score_order[base_idx]
            j = score_order[min(base_idx + offset, N - 1)]
        else:
            i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        current[i], current[j] = current[j], current[i]
        new_val = simulate(current)
        if new_val > cur_val:
            cur_val = new_val
            no_improve = 0
            if cur_val > best_val:
                best_val = cur_val
                best_scores = current.copy()
                score_order = np.argsort(-current)
        else:
            current[i], current[j] = current[j], current[i]
            no_improve += 1

    return best_scores
