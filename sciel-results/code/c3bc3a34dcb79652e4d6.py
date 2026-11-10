"""Planted 3-SAT solver (150 vars): planted-value finder + WalkSAT + DPLL, deadline-guarded."""
# EVOLVE-BLOCK-START
import time
import random

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def _find_planted(n, clauses, deadline):
    """Recover the planted assignment via value-count majority.

    Planted instances are generated from a fixed assignment, so each variable's
    true value appears in more clauses than its negation (statistical bias).
    """
    if time.time() > deadline:
        return None
    vals = [0.0] * n
    for c in clauses:
        for l in c:
            i = abs(l) - 1
            vals[i] += 1.0 if l > 0 else -1.0
    best, best_cost = None, None
    for s in (1.0, -1.0):
        a = [vals[i] * s > 0 for i in range(n)]
        cost = sum(1 for c in clauses
                   if not any(a[abs(l) - 1] == (l > 0) for l in c))
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, a
        if cost == 0:
            return a
    return best


def _cost(a, clauses):
    return sum(1 for c in clauses
               if not any(a[abs(l) - 1] == (l > 0) for l in c))


def _walksat(n, clauses, init, rng, deadline, max_iter):
    """WalkSAT with random restarts handled by the caller; clause-weight heuristics."""
    if _HAS_NUMPY:
        return _walksat_np(n, clauses, init, rng, deadline, max_iter)
    return _walksat_py(n, clauses, init, rng, deadline, max_iter)


def _walksat_py(n, clauses, init, rng, deadline, max_iter):
    a = init[:]
    m = len(clauses)
    unsat = [not any(a[abs(l) - 1] == (l > 0) for l in c) for c in clauses]
    weights = [1.0] * m
    for _ in range(max_iter):
        if time.time() > deadline:
            return a
        if not any(unsat):
            return a
        # weighted random choice of an unsat clause
        while True:
            ci = rng.randrange(m)
            if unsat[ci]:
                break
        c = clauses[ci]
        l = c[rng.randrange(len(c))]
        i = abs(l) - 1
        a[i] = not a[i]
        # update unsat/weights for clauses touching i
        for cj in range(m):
            cj_c = clauses[cj]
            old = unsat[cj]
            new = not any(a[abs(x) - 1] == (x > 0) for x in cj_c)
            if old and not new:
                weights[cj] = 1.0
            elif new and not old:
                weights[cj] = 2.0
            unsat[cj] = new
    return a


def _walksat_np(n, clauses, init, rng, deadline, max_iter):
    m = len(clauses)
    arr = np.array(clauses, dtype=np.int32).reshape(m, -1)
    signs = np.sign(arr).astype(np.int8)
    idx = np.abs(arr).astype(np.int32) - 1
    a = np.array(init, dtype=np.int8)
    for _ in range(max_iter):
        if time.time() > deadline:
            return a.astype(bool).tolist()
        sat = (a[idx] == (signs > 0).astype(np.int8)).any(axis=1)
        if sat.all():
            return a.astype(bool).tolist()
        unsat_idx = np.nonzero(~sat)[0]
        ci = unsat_idx[rng.randrange(len(unsat_idx))]
        l = arr[ci, rng.randrange(len(arr[ci]))]
        a[abs(l) - 1] = 1 - a[abs(l) - 1]
    return a.astype(bool).tolist()


def _dpll(n, clauses, rng, deadline):
    """DPLL with unit propagation + pure-literal elimination (safety net)."""
    remaining = [list(c) for c in clauses]
    values = [None] * n

    def propagate():
        changed = True
        while changed:
            changed = False
            new_rem = []
            for c in remaining:
                un = [l for l in c if values[abs(l) - 1] is None]
                if len(un) == 0:
                    if any(values[abs(l) - 1] == (l > 0) for l in c):
                        continue
                    return False
                elif len(un) == 1:
                    l = un[0]
                    values[abs(l) - 1] = (l > 0)
                    changed = True
                    continue
                new_rem.append(c)
            remaining[:] = new_rem
            if not remaining:
                return True
            pos, neg = set(), set()
            for c in remaining:
                for l in c:
                    if values[abs(l) - 1] is None:
                        (pos if l > 0 else neg).add(abs(l) - 1)
            for v in (pos - neg):
                values[v] = True
                changed = True
            for v in (neg - pos):
                values[v] = False
                changed = True
            if not remaining:
                return True
        return True

    def all_assigned():
        return all(values[i] is not None for i in range(n))

    if not propagate():
        return None
    if all_assigned():
        return values[:]
    unassigned = [i for i in range(n) if values[i] is None]
    if not unassigned:
        return values[:]
    v = unassigned[rng.randrange(len(unassigned))]
    for val in (True, False):
        if time.time() > deadline:
            return None
        save = values[:]
        rem_save = [c[:] for c in remaining]
        values[v] = val
        if propagate():
            if all_assigned():
                return values[:]
            r = _dpll(n, clauses, rng, deadline)
            if r is not None:
                return r
        values[:] = save
        remaining[:] = rem_save
    return None


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    start = time.time()
    deadline = start + 270.0  # safety margin under 300s

    # 1) Fast planted-solution recovery.
    planted = _find_planted(n, clauses, deadline)
    if planted is not None and _cost(planted, clauses) == 0:
        return planted

    # 2) WalkSAT with restarts from planted guess / random.
    best = planted if planted is not None else [rng.random() < 0.5 for _ in range(n)]
    best_cost = _cost(best, clauses)
    attempts = 0
    while time.time() < deadline - 30.0:
        init = planted if (attempts % 3 == 0 and planted is not None) \
            else [rng.random() < 0.5 for _ in range(n)]
        res = _walksat(n, clauses, init, rng, deadline - 30.0, max_iter=150000)
        cost = _cost(res, clauses)
        if cost < best_cost:
            best_cost, best = cost, res
        if cost == 0:
            return res
        attempts += 1

    # 3) DPLL safety net.
    if time.time() < deadline - 5.0:
        d = _dpll(n, clauses, rng, deadline - 5.0)
        if d is not None and all(d[i] is not None for i in range(n)):
            if _cost(d, clauses) == 0:
                return d

    return best
# EVOLVE-BLOCK-END
