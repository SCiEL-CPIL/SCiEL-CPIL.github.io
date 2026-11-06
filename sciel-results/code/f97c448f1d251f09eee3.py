# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _topo_order(n, pred_lists):
    """Return topological order and successor adjacency."""
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order, succ


def _downstream_stats(n, order, succ, times):
    """Compute longest_down and total_descendant_time via reverse topo."""
    ld = [0.0] * n
    td = [0.0] * n
    for u in reversed(order):
        best_ld = 0.0
        tot = 0.0
        for v in succ[u]:
            if times[v] + ld[v] > best_ld:
                best_ld = times[v] + ld[v]
            tot += times[v] + td[v]
        ld[u] = best_ld
        td[u] = tot
    return ld, td


def _greedy_stations(n, times, pred_lists, succ, cycle_time, priority_vec):
    """
    Simulate the greedy station-filling: repeatedly pick the highest-priority
    eligible task (all preds done) that fits in remaining cycle time.
    Returns number of stations.
    """
    remaining = set(range(n))
    done = set()
    stations = 0
    # Precompute eligibility: a task is eligible when all its preds are in done
    while remaining:
        stations += 1
        cap = cycle_time
        # Greedily pick from eligible tasks in priority order
        # Build eligible set
        eligible = []
        for t in remaining:
            ok = True
            for p in pred_lists[t]:
                if p not in done:
                    ok = False
                    break
            if ok and times[t] <= cap:
                eligible.append(t)
        if not eligible:
            # Should not happen if feasible; pick any with no pending preds
            for t in remaining:
                ok = all(p in done for p in pred_lists[t])
                if ok:
                    eligible.append(t)
                    break
            if not eligible:
                # infeasible; take smallest time
                eligible = [min(remaining, key=lambda x: times[x])]
        # Sort by priority descending, break ties by time desc then index
        eligible.sort(key=lambda t: (-priority_vec[t], -times[t], t))
        for t in eligible:
            if times[t] <= cap:
                cap -= times[t]
                done.add(t)
                remaining.discard(t)
    return stations


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    pred_lists = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    if n == 0:
        return np.array([], dtype=float)

    t0 = time.time()
    DEADLINE = 130.0  # safety margin under 150s

    order, succ = _topo_order(n, pred_lists, pred_lists)
    ld, td = _downstream_stats(n, order, succ, times)

    # Upstream longest path
    lu = [0.0] * n
    for u in order:
        for v in succ[u]:
            c = lu[u] + times[u]
            if c > lu[v]:
                lu[v] = c

    # Candidate priority vectors (smooth surrogates for "best packing")
    # Each vector is a smooth function of task attributes. We simulate the
    # greedy with each and pick the one yielding fewest stations.
    cp_len = max(lu[i] + times[i] + ld[i] for i in range(n)) + 1e-9
    max_t = max(times) + 1e-9
    max_ld = max(ld) + 1e-9 if max(ld) > 0 else 1.0
    max_td = max(td) + 1e-9 if max(td) > 0 else 1.0
    max_lu = max(lu) + 1e-9 if max(lu) > 0 else 1.0
    max_succ = max([len(succ[i]) for i in range(n)]) + 1e-9
    max_pred = max([len(pred_lists[i]) for i in range(n)]) + 1e-9

    lu_a = np.array(lu, dtype=float)
    ld_a = np.array(ld, dtype=float)
    td_a = np.array(td, dtype=float)
    succ_a = np.array([len(succ[i]) for i in range(n)], dtype=float)
    pred_a = np.array([len(pred_lists[i]) for i in range(n)], dtype=float)
    cp_a = (lu_a + times + ld_a) / cp_len

    # Family of smooth composite priorities (soft-max style emphasis)
    # We use a log-sum-exp-like soft emphasis on the dominant feature.
    def _soft(x, k=5.0):
        # smooth max-like: k * log(sum(exp(x_i/k))) broadcast... but here per-task.
        # Instead: use tanh-based soft ranking for a single vector.
        m = np.max(x)
        return x  # placeholder; real soft ranking done via sorting below

    # Build a small set of candidate priority vectors
    candidates = []

    # 1. Pure time
    candidates.append(times.copy())
    # 2. Time + downstream
    candidates.append(times / max_t * 0.5 + ld_a / max_ld * 0.5)
    # 3. Time + CP proximity
    candidates.append(times / max_t * 0.4 + cp_a * 0.4 + ld_a / max_ld * 0.2)
    # 4. Downstream-dominant
    candidates.append(ld_a / max_ld * 0.5 + td_a / max_td * 0.3 + times / max_t * 0.2)
    # 5. Successor-count emphasis (bottleneck tasks first)
    candidates.append(succ_a / max_succ * 0.3 + times / max_t * 0.3 + cp_a * 0.2 + ld_a / max_ld * 0.2)
    # 6. Inverse predecessor count (independent tasks first)
    candidates.append((1.0 - pred_a / max_pred) * 0.3 + times / max_t * 0.4 + cp_a * 0.3)
    # 7. Smooth "rank by longest_down then time" via soft ordering
    #    Use a temperature-sorted soft rank:
    combined = ld_a / max_ld * 0.6 + times / max_t * 0.4
    # soft-rank: sort indices, assign smoothed ranks
    idx = np.argsort(-combined)
    ranks = np.zeros(n)
    for r, t in enumerate(idx):
        ranks[t] = r
    # smooth: normalize ranks to [0,1] with a soft inverse
    max_r = max(n - 1, 1)
    soft_rank = 1.0 - (ranks / max_r) ** 0.7  # concave soft-rank
    candidates.append(soft_rank)
    # 8. Hybrid: time * cp-proximity (emphasize big tasks on critical path)
    candidates.append((times / max_t) * cp_a * 0.5 + (times / max_t) * 0.3 + ld_a / max_ld * 0.2)
    # 9. Total descendant work emphasis
    candidates.append(td_a / max_td * 0.4 + times / max_t * 0.3 + cp_a * 0.3)
    # 10. Pure CP proximity
    candidates.append(cp_a.copy())

    # Evaluate each candidate via greedy simulation; pick best (fewest stations)
    best_priority = candidates[0]
    best_stations = _greedy_stations(n, times, pred_lists, succ, cycle_time, candidates[0])

    for c in candidates[1:]:
        if time.time() - t0 > DEADLINE:
            break
        s = _greedy_stations(n, times, pred_lists, succ, cycle_time, c)
        if s < best_stations:
            best_stations = s
            best_priority = c

    # Optional: local refinement around the best candidate via coordinate
    # perturbation of weights (a smooth polish pass).
    if time.time() - t0 < DEADLINE - 10:
        # Represent the best candidate as a linear combination of base features
        # and do a small grid of weight perturbations.
        base_feats = np.column_stack([
            times / max_t,
            ld_a / max_ld,
            td_a / max_td,
            cp_a,
            succ_a / max_succ,
            1.0 - pred_a / max_pred,
            lu_a / max_lu,
        ])
        # Find weights that best approximate best_priority via least squares
        # Then perturb.
        w0, *_ = np.linalg.lstsq(base_feats, best_priority, rcond=None)
        # Small local search over weight perturbations
        best_w = w0
        for _ in range(20):
            if time.time() - t0 > DEADLINE:
                break
            for i in range(len(w0)):
                for delta in (-0.1, 0.1):
                    w_new = w0.copy()
                    w_new[i] += delta
                    # Renormalize to non-negative and sum to 1
                    w_pos = np.clip(w_new, 0, None)
                    if w_pos.sum() < 1e-9:
                        continue
                    w_pos = w_pos / w_pos.sum()
                    p = base_feats @ w_pos
                    s = _greedy_stations(n, times, pred_lists, succ, cycle_time, p)
                    if s < best_stations:
                        best_stations = s
                        best_priority = p
                        best_w = w_pos
    return best_priority
# EVOLVE-BLOCK-END
