"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    t0 = time.time()
    deadline = 315.0  # safety margin before 330s kill

    m = len(clauses)
    if m == 0:
        return [True] * n

    # Parse clauses into numpy arrays
    clause_sgn = np.empty((m, 3), dtype=np.int8)
    clause_var = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_sgn[i, j] = 1 if lit > 0 else -1
            clause_var[i, j] = abs(lit) - 1

    # Build variable-to-clause incidence
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_var[i, j]
            var_clauses[v].append((i, j))

    var_clause_idx = []
    var_clause_pos = []
    for lst in var_clauses:
        if lst:
            idx = np.array([x[0] for x in lst], dtype=np.int32)
            pos = np.array([x[1] for x in lst], dtype=np.int32)
        else:
            idx = np.array([], dtype=np.int32)
            pos = np.array([], dtype=np.int32)
        var_clause_idx.append(idx)
        var_clause_pos.append(pos)

    # Literal frequency for heuristic
    pos_count = np.zeros(n, dtype=np.int32)
    neg_count = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for j in range(3):
            v = clause_var[i, j]
            if clause_sgn[i, j] == 1:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    def evaluate_full(assignment):
        vals = clause_sgn * (2.0 * assignment[clause_var] - 1.0)
        sat = np.any(vals > 0, axis=1)
        return np.where(~sat)[0]

    def make_heuristic_initial(rng):
        assignment = np.zeros(n, dtype=np.float64)
        for v in range(n):
            p = pos_count[v]
            q = neg_count[v]
            if p == q:
                assignment[v] = 1.0 if rng.rand() < 0.5 else 0.0
            elif p > q:
                prob_true = 0.7 + 0.2 * (p - q) / (p + q)
                assignment[v] = 1.0 if rng.rand() < prob_true else 0.0
            else:
                prob_true = 0.3 - 0.2 * (q - p) / (q + p)
                assignment[v] = 1.0 if rng.rand() < max(0.05, prob_true) else 0.0
        return assignment

    def compute_delta(v, assignment):
        """Compute delta (change in satisfied count) if we flip variable v."""
        cidx = var_clause_idx[v]
        cpos = var_clause_pos[v]
        if len(cidx) == 0:
            return 0
        signs = clause_sgn[cidx, cpos]
        other_vars = clause_var[cidx]
        all_vals = clause_sgn[cidx] * (2.0 * assignment[other_vars] - 1.0)
        currently_sat = np.any(all_vals > 0, axis=1)
        # Flip: negate the value at position cpos
        # new_sat: a clause is satisfied if any literal (excluding the flipped one) is satisfied,
        # or the flipped literal was the only satisfied one and flipping makes it unsat -> not sat
        # Actually: new_all_vals at cpos is negated. Clause is sat if any new_val > 0.
        new_all_vals = all_vals.copy()
        rows = np.arange(len(cidx))
        new_all_vals[rows, cpos] *= -1
        new_sat = np.any(new_all_vals > 0, axis=1)
        return int(np.sum(new_sat) - np.sum(currently_sat))

    def update_unsat_after_flip(v, assignment, unsat):
        """Update unsat set after flipping variable v."""
        cidx = var_clause_idx[v]
        if len(cidx) == 0:
            return evaluate_full(assignment)
        all_vals = clause_sgn[cidx] * (2.0 * assignment[clause_var[cidx]] - 1.0)
        sat = np.any(all_vals > 0, axis=1)
        # Vectorized set update
        unsat_set = set(unsat.tolist())
        for k in range(len(cidx)):
            ci = int(cidx[k])
            if sat[k]:
                unsat_set.discard(ci)
            else:
                unsat_set.add(ci)
        return np.array(list(unsat_set), dtype=np.int32)

    rng = np.random.RandomState(42)
    best_assignment = None
    best_unsat_count = m + 1

    max_restarts = 800
    base_steps = 6000

    for restart in range(max_restarts):
        if time.time() - t0 > deadline:
            break

        if restart < 15:
            assignment = make_heuristic_initial(rng)
        else:
            assignment = (rng.rand(n) < 0.5).astype(np.float64)

        unsat = evaluate_full(assignment)
        unsat_count = len(unsat)

        if unsat_count == 0:
            return assignment.astype(bool).tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        steps = base_steps + restart * 400

        for step in range(steps):
            if time.time() - t0 > deadline:
                break

            if len(unsat) == 0:
                return assignment.astype(bool).tolist()

            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment.copy()

            clause_idx = unsat[rng.randint(len(unsat))]

            # Compute deltas for all 3 variables in the clause
            deltas = []
            for j in range(3):
                v = clause_var[clause_idx, j]
                deltas.append((compute_delta(v, assignment), v))

            deltas.sort(key=lambda x: x[0], reverse=True)
            best_delta, best_v = deltas[0]

            # Random tie-breaking among equal best deltas
            candidates = [d[1] for d in deltas if d[0] == best_delta]
            v = candidates[rng.randint(len(candidates))]

            assignment[v] = 1.0 - assignment[v]
            unsat = update_unsat_after_flip(v, assignment, unsat)

        if len(unsat) == 0:
            return assignment.astype(bool).tolist()
        if len(unsat) < best_unsat_count:
            best_unsat_count = len(unsat)
            best_assignment = assignment.copy()

    if best_assignment is not None:
        return best_assignment.astype(bool).tolist()
    return (rng.rand(n) < 0.5).tolist()
# EVOLVE-BLOCK-END
