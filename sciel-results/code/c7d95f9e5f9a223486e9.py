"""Spectral-init + weighted min-conflicts local search for planted 3-SAT."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280  # safety margin before 300s/330s kill

    # ---------- Spectral initialization ----------
    def lit_idx(l):
        return l - 1 if l > 0 else 2 * n - l - 1

    A = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for c in clauses:
        idxs = [lit_idx(l) for l in c]
        for i in idxs:
            A[i, i] += 1
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                A[idxs[a], idxs[b]] += 1
                A[idxs[b], idxs[a]] += 1

    v = np.ones(2 * n, dtype=np.float64)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(60):
        w = A.dot(v)
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            break
        w /= nw
        if np.abs(np.dot(w, v)) > 0.9999999:
            v = w
            break
        v = w

    assignment = np.zeros(n, dtype=bool)
    for i in range(n):
        assignment[i] = abs(v[i]) >= abs(v[n + i])

    # ---------- Local search structures ----------
    clause_vars = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]

    # Reverse map: var -> list of clause indices containing it
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        seen = set()
        for (var, is_pos) in clause_vars[ci]:
            if var not in seen:
                var_clauses[var].append(ci)
                seen.add(var)

    clause_sat = np.zeros(m, dtype=bool)
    conflict = np.zeros(n, dtype=np.int64)

    def clause_ok(ci, assign):
        for (var, is_pos) in clause_vars[ci]:
            if assign[var] == is_pos:
                return True
        return False

    for ci in range(m):
        ok = clause_ok(ci, assignment)
        clause_sat[ci] = ok
        if not ok:
            seen = set()
            for (var, is_pos) in clause_vars[ci]:
                if var not in seen:
                    conflict[var] += 1
                    seen.add(var)

    unsat_count = int(np.sum(~clause_sat))
    best_unsat = unsat_count
    best_assign = assignment.copy()

    if unsat_count == 0:
        return assignment.tolist()

    clause_freq = np.ones(m, dtype=np.float64)

    rng_state = 123456789
    def rand():
        nonlocal rng_state
        rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
        return rng_state / 0x7fffffff

    step = 0
    while time.time() < deadline:
        if unsat_count == 0:
            return assignment.tolist()
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assignment.copy()

        unsat_indices = np.nonzero(~clause_sat)[0]
        if len(unsat_indices) == 0:
            return assignment.tolist()

        weights = clause_freq[unsat_indices]
        total_w = weights.sum()
        if total_w <= 0:
            ci = int(unsat_indices[0])
        else:
            r = rand() * total_w
            acc = 0.0
            ci = int(unsat_indices[-1])
            for j in range(len(unsat_indices)):
                acc += weights[j]
                if r <= acc:
                    ci = int(unsat_indices[j])
                    break

        cvars = clause_vars[ci]
        min_conf = np.inf
        best_vars = []
        seen = set()
        for (var, is_pos) in cvars:
            if var in seen:
                continue
            seen.add(var)
            cf = conflict[var]
            if cf < min_conf:
                min_conf = cf
                best_vars = [var]
            elif cf == min_conf:
                best_vars.append(var)
        var = best_vars[int(rand() * len(best_vars))]

        old = assignment[var]
        assignment[var] = not old

        # Update all clauses containing var
        for ci2 in var_clauses[var]:
            new_ok = clause_ok(ci2, assignment)
            if new_ok and not clause_sat[ci2]:
                # became satisfied: decrement conflict for its vars
                clause_sat[ci2] = True
                unsat_count -= 1
                seen2 = set()
                for (v2, p2) in clause_vars[ci2]:
                    if v2 not in seen2:
                        conflict[v2] -= 1
                        seen2.add(v2)
            elif (not new_ok) and clause_sat[ci2]:
                # became unsatisfied
                clause_sat[ci2] = False
                unsat_count += 1
                seen2 = set()
                for (v2, p2) in clause_vars[ci2]:
                    if v2 not in seen2:
                        conflict[v2] += 1
                        seen2.add(v2)
            # else no change in sat status

        step += 1

    return best_assign.tolist()
# EVOLVE-BLOCK-END
