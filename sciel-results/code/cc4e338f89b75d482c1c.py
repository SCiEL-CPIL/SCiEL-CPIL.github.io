"""Planted 3-SAT solver: vectorized WalkSAT + DLS with restarts and deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    # Deadline guard: evaluator kills at 330s; keep a safety margin.
    start = time.time()
    deadline = start + 290.0

    m = len(clauses)

    # Precompute literal arrays for vectorized evaluation.
    # lit_vars: (m,3) 0-based variable index
    # lit_sign: (m,3) +1 for positive literal, -1 for negative
    lit_vars = np.empty((m, 3), dtype=np.int64)
    lit_sign = np.empty((m, 3), dtype=np.int64)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            lit_vars[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    rng = random.Random(0)

    def evaluate(assignment):
        """Return (clause_satisfied bool array, unsat_count)."""
        # assignment: bool array length n
        # For each literal: satisfied if assignment[var] == (sign==1)
        # i.e. assignment[var] * sign + (1 - assignment[var]) * (1 - sign) == 1
        # Simpler: (assignment[var] & (sign>0)) | (~assignment[var] & (sign<0))
        vals = np.where(lit_sign > 0, assignment, ~assignment)  # (m,3)
        clause_sat = vals.any(axis=1)
        return clause_sat, int((~clause_sat).sum())

    def walksat_step(assignment, clause_sat, unsat_count, dls_scores):
        """One WalkSAT step. Returns (assignment, clause_sat, unsat_count, dls_scores)."""
        if unsat_count == 0:
            return assignment, clause_sat, unsat_count, dls_scores
        # Pick a random unsat clause
        unsat_indices = np.where(~clause_sat)[0]
        ci = unsat_indices[rng.randrange(len(unsat_indices))]
        clause = clauses[ci]
        # WalkSAT: with prob p pick literal from clause (to satisfy), else random var
        p = 0.55
        if rng.random() < p:
            li = rng.randrange(3)
            var = abs(clause[li]) - 1
        else:
            var = rng.randrange(n)
        assignment[var] = not assignment[var]
        # Update affected clauses (those containing var)
        affected = np.where(lit_vars == var)[0]
        for idx in affected:
            c = clauses[idx]
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if sat:
                clause_sat[idx] = True
            else:
                clause_sat[idx] = False
                dls_scores[idx] += 1
        unsat_count = int((~clause_sat).sum())
        return assignment, clause_sat, unsat_count, dls_scores

    def dls_step(assignment, clause_sat, unsat_count, dls_scores):
        """One DLS step: flip variable that satisfies most unsat clauses."""
        if unsat_count == 0:
            return assignment, clause_sat, unsat_count, dls_scores
        # For each variable, count how many unsat clauses it would satisfy if flipped
        # An unsat clause has all 3 literals false. Flipping var v satisfies a clause
        # if v appears in it (in either polarity).
        unsat_mask = ~clause_sat  # (m,)
        # For each var, count unsat clauses containing it
        var_unsat_count = np.zeros(n, dtype=np.int64)
        # Vectorized: for each unsat clause, add 1 to each of its 3 vars
        unsat_vars = lit_vars[unsat_mask]  # (k,3)
        np.add.at(var_unsat_count, unsat_vars.ravel(), 1)
        # Pick var with max count; tie-break randomly
        max_count = var_unsat_count.max()
        candidates = np.where(var_unsat_count == max_count)[0]
        var = candidates[rng.randrange(len(candidates))]
        assignment[var] = not assignment[var]
        # Update affected clauses
        affected = np.where(lit_vars == var)[0]
        for idx in affected:
            c = clauses[idx]
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if sat:
                clause_sat[idx] = True
            else:
                clause_sat[idx] = False
                dls_scores[idx] += 1
        unsat_count = int((~clause_sat).sum())
        return assignment, clause_sat, unsat_count, dls_scores

    best_assignment = None
    best_unsat = m + 1

    # Multiple restarts
    max_steps_per_restart = 5000
    while time.time() < deadline:
        # Random initial assignment
        assignment = (rng.random(n) < 0.5).astype(bool)
        clause_sat, unsat_count = evaluate(assignment)
        dls_scores = np.zeros(m, dtype=np.int64)

        if unsat_count == 0:
            return assignment.tolist()

        # Track best in this restart
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        step_budget = max_steps_per_restart
        for step in range(step_budget):
            if time.time() >= deadline:
                break
            if unsat_count == 0:
                return assignment.tolist()

            # Mix: mostly WalkSAT, occasionally DLS
            if rng.random() < 0.85:
                assignment, clause_sat, unsat_count, dls_scores = walksat_step(
                    assignment, clause_sat, unsat_count, dls_scores
                )
            else:
                assignment, clause_sat, unsat_count, dls_scores = dls_step(
                    assignment, clause_sat, unsat_count, dls_scores
                )

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

            # Periodic DLS reset of scores (prevent score explosion)
            if step % 500 == 499:
                dls_scores *= 0.5

    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()
# EVOLVE-BLOCK-END
