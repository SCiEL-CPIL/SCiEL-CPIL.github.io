import time
import random

_DEADLINE = 325.0  # safety margin before the 330s kill


def _dpll(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    var_count = {}
    for c in clauses:
        for l in c:
            v = abs(l)
            var_count[v] = var_count.get(v, 0) + 1

    var_val = {}

    def unit_vars():
        forced = {}
        for c in clauses:
            unassigned = []
            for l in c:
                v = abs(l)
                if v not in var_val:
                    unassigned.append(l)
            if len(unassigned) == 1:
                l = unassigned[0]
                forced[abs(l)] = (l > 0)
        return forced

    def any_falsified():
        for c in clauses:
            ok = False
            for l in c:
                v = abs(l)
                if v in var_val:
                    if var_val[v] == (l > 0):
                        ok = True
                        break
            if not ok:
                return True
        return False

    def dpll():
        changed = True
        while changed:
            changed = False
            for v, b in unit_vars().items():
                if v not in var_val:
                    var_val[v] = b
                    changed = True
        if any_falsified():
            return False
        best_v = None
        best_c = -1
        for v in range(1, n + 1):
            if v in var_val:
                continue
            c = var_count.get(v, 0)
            if c > best_c:
                best_c = c
                best_v = v
        if best_v is None:
            return True
        for b in (True, False):
            var_val[best_v] = b
            if dpll():
                return True
            del var_val[best_v]
        return False

    if dpll():
        return [var_val.get(v, False) for v in range(1, n + 1)]
    return None


def _walksat(instance, rng, time_limit):
    n = instance["n"]
    clauses = instance["clauses"]
    assignment = [rng.random() < 0.5 for _ in range(n)]

    def count_unsat(c):
        cnt = 0
        for l in c:
            if assignment[abs(l) - 1] != (l > 0):
                cnt += 1
        return cnt

    start = time.time()
    for _ in range(500000):
        if time.time() - start > time_limit:
            break
        unsat_idx = -1
        min_unsat = 4
        for ci, c in enumerate(clauses):
            u = count_unsat(c)
            if u == 0:
                continue
            if u < min_unsat:
                min_unsat = u
                unsat_idx = ci
                if u == 1:
                    break
        if unsat_idx == -1:
            break
        clause = clauses[unsat_idx]
        best_l = None
        best_score = -1
        for l in clause:
            score = 0
            for c2 in clauses:
                if c2 is clause:
                    continue
                for l2 in c2:
                    if l2 == -l:
                        score += 1
            if score > best_score:
                best_score = score
                best_l = l
        if best_l is None:
            best_l = rng.choice(clause)
        assignment[abs(best_l) - 1] = not assignment[abs(best_l) - 1]
    return assignment


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + _DEADLINE

    result = _dpll(instance)
    if result is not None:
        ok = True
        for c in clauses:
            if not any(result[abs(l) - 1] == (l > 0) for l in c):
                ok = False
                break
        if ok:
            return result

    remaining = deadline - time.time()
    if remaining < 1:
        return [False] * n
    rng = random.Random(0)
    return _walksat(instance, rng, remaining)
