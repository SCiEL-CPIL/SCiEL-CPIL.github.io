"""Planted 3-SAT solver (150 vars). Vectorized WalkSAT with clause-weight
adaptation, noise, and multi-restart, under a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start = time.time()
    deadline = start + 300.0  # evaluator kills at 330s; keep 30s safety margin

    # Precompute literal arrays (row-major: clause i uses [3i, 3i+2]).
    lit_var = np.empty(m * 3, dtype=np.int64)
    lit_sign = np.empty(m * 3, dtype=np.int8)
    for i, cl in enumerate(clauses):
        base = i * 3
        for j, l in enumerate(cl):
            lit_var[base + j] = abs(l) - 1
            lit_sign[base + j] = 1 if l > 0 else 0

    # Clause -> (vars, signs) for quick per-clause recompute.
    cl_vars = [np.array([abs(l) - 1 for l in cl], dtype=np.int64) for cl in clauses]
    cl_sign = [np.array([1 if l > 0 else 0 for l in cl], dtype=np.int8) for cl in clauses]

    # Variable -> list of clause indices that contain it (for O(1) flip update).
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in cl_vars[i]:
            var_to_clauses[v].append(i)
    var_to_clauses = [np.array(lst, dtype=np.int64) for lst in var_to_clauses]

    rng = random.Random(0)
    best_assign = None
    best_unsat = m + 1

    def is_deadline():
        return (time.time() - start) >= deadline

    def recompute(assign):
        sat = (assign[lit_var] == lit_sign).reshape(m, 3)
        unsat_counts = np.sum(1 - sat, axis=1)
        return sat, unsat_counts

    def make_assign():
        return np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)

    assign = make_assign()
    sat, unsat_counts = recompute(assign)
    cur_unsat = int(np.sum(unsat_counts > 0))
    weights = np.ones(m, dtype=np.float64)

    if cur_unsat == 0:
        return assign.astype(bool).tolist()
    best_unsat = cur_unsat
    best_assign = assign.copy()

    budgets = [30000, 60000, 120000, 240000, 480000]
    noise = 0.30

    for ri, budget in enumerate(budgets):
        if is_deadline():
            break
        if ri > 0:
            assign = make_assign()
            sat, unsat_counts = recompute(assign)
            cur_unsat = int(np.sum(unsat_counts > 0))
            weights = np.ones(m, dtype=np.float64)
            if cur_unsat == 0:
                return assign.astype(bool).tolist()
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assign = assign.copy()
            noise = min(0.55, 0.25 + 0.05 * ri)

        prev_unsat = cur_unsat
        stagnate = 0
        for step in range(budget):
            if step % 500 == 0 and is_deadline():
                break
            if cur_unsat == 0:
                return assign.astype(bool).tolist()

            # Weighted random unsatisfied clause.
            uidx = np.nonzero(unsat_counts > 0)[0]
            w = weights[uidx]
            total = w.sum()
            if total <= 0:
                total = 1.0
                w = np.ones(len(uidx), dtype=np.float64)
            r = rng.random() * total
            cidx = int(uidx[np.searchsorted(np.cumsum(w), r)])

            # Choose a literal to flip: min-unsat heuristic with noise.
            csat = sat[cidx]
            cvars = cl_vars[cidx]
            if rng.random() < noise:
                var = int(cvars[rng.randrange(3)])
            else:
                uvars = cvars[csat == 0]
                if len(uvars) == 0:
                    uvars = cvars
                # For each candidate var, count other unsatisfied clauses it appears in.
                best_var = int(uvars[0])
                best_score = None
                for v in uvars:
                    occ = var_to_clauses[v]
                    score = int(np.sum(unsat_counts[occ]))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_var = int(v)
                var = best_var

            old = assign[var]
            assign[var] = 1 - old
            # Update affected clauses (O(1) per clause via incidence).
            for i in var_to_clauses[var]:
                s = cl_sign[i]
                v = cl_vars[i]
                row = (assign[v] == s)
                sat[i] = row
                unsat_counts[i] = int(np.sum(1 - row))
            cur_unsat = int(np.sum(unsat_counts > 0))

            # Weight adaptation: raise persistently-unsatisfied clauses.
            if cur_unsat > prev_unsat:
                weights[unsat_counts > 0] *= 1.05
            prev_unsat = cur_unsat

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assign = assign.copy()

            if cur_unsat == prev_unsat:
                stagnate += 1
            else:
                stagnate = 0
            if stagnate > 2000:
                break

    if best_assign is not None:
        return best_assign.astype(bool).tolist()
    return assign.astype(bool).tolist()
# EVOLVE-BLOCK-END
