# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _START) < _DEADLINE


def priority(instance):
    # Robust extraction of fields
    try:
        durations = np.asarray(instance["durations"], dtype=float)
    except Exception:
        return np.asarray(instance.get("durations", [1.0]), dtype=float)
    n = int(durations.size)

    # Predecessors: list of lists (or arrays)
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            p = int(p)
            if 0 <= p < n and i not in succ[p]:
                succ[p].append(i)

    # Resource demands & capacities
    demands = instance.get("demands", None)
    caps = instance.get("capacities", None)

    # ---------- CPM forward / backward pass for slack ----------
    # ES, EF
    ES = np.zeros(n)
    EF = np.zeros(n)
    # topological order (Kahn)
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            p = int(p)
            if 0 <= p < n:
                indeg[i] += 1
    order = []
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    indeg2 = list(indeg)
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)
    if len(order) < n:  # cycle fallback
        order = list(range(n))

    for u in order:
        for v in succ[u]:
            if ES[v] < ES[u] + durations[u]:
                ES[v] = ES[u] + durations[u]
    for u in order:
        EF[u] = ES[u] + durations[u]

    makespan = float(np.max(EF)) if n else 0.0

    # LS, LF
    LF = np.full(n, makespan)
    LS = np.full(n, makespan)
    for u in reversed(order):
        for v in succ[u]:
            if LF[u] > LS[v]:
                LF[u] = LS[v]
        LS[u] = LF[u] - durations[u]

    slack = np.maximum(0.0, LS - ES)

    # ---------- Resource pressure (static peak demand heuristic) ----------
    # For each resource, estimate how "scarce" it is by the ratio of
    # (sum of demand*duration weighted by criticality) to capacity.
    resource_pressure = np.zeros(n)
    if demands is not None and caps is not None:
        try:
            D = np.asarray(demands, dtype=float)
            C = np.asarray(caps, dtype=float)
        except Exception:
            D = None
        if D is not None and D.ndim == 2 and D.shape[0] == n:
            nr = D.shape[1]
            for r in range(nr):
                cap = C[r] if r < len(C) else 1.0
                if cap <= 0:
                    cap = 1.0
                # total work on this resource
                work = float(np.sum(D[:, r] * durations))
                # scarcity factor: work relative to capacity * makespan
                scarcity = work / (cap * max(makespan, 1.0))
                # activities using this resource get pressure scaled by their demand
                resource_pressure += D[:, r] * (1.0 + scarcity)

    # ---------- Successor bottleneck (out-degree) ----------
    outdeg = np.array([len(succ[i]) for i in range(n)], dtype=float)
    outdeg_norm = outdeg / (max(outdeg.max(), 1.0))

    # ---------- Combine ----------
    # Primary: low slack (high criticality). Normalize slack to [0,1].
    s_max = float(np.max(slack)) if n else 0.0
    if s_max > 0:
        crit = 1.0 - slack / s_max
    else:
        crit = np.ones(n)

    # Duration normalized
    d_max = float(np.max(durations)) if n else 0.0
    d_norm = durations / d_max if d_max > 0 else np.zeros(n)

    # Resource pressure normalized
    rp_max = float(np.max(resource_pressure)) if n else 0.0
    rp_norm = resource_pressure / rp_max if rp_max > 0 else np.zeros(n)

    score = (
        1.0 * crit
        + 0.4 * d_norm
        + 0.5 * rp_norm
        + 0.3 * outdeg_norm
    )
    return score
# EVOLVE-BLOCK-END
