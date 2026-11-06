# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit

    n, clauses = instance["n"], instance["clauses"]

    # Convert clauses to 0-based signed literal arrays
    # Each literal: (var_idx, sign) where sign is +1 or -1
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = [abs(l) - 1 for l in c]
        signs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_)
        clause_signs.append(signs)

    clause_vars = np.array(clause_vars, dtype=np.int32)
    clause_signs = np.array(clause_signs, dtype=np.int32)
    m = len(clauses)

    # Build variable-to-clause index for quick lookup
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clause_vars):
        for v in c:
            var_to_clauses[v].append(i)

    # Preprocessing: Pure literal elimination and unit propagation
    def preprocess(assignments, clause_vars, clause_signs, var_to_clauses):
        """Apply pure literal elimination and unit propagation iteratively."""
        changed = True
        while changed:
            changed = False
            # Pure literal elimination
            for v in range(len(assignments)):
                if assignments[v] is not None:
                    continue
                # Check if all occurrences of var v have same sign
                signs_for_v = []
                for ci in var_to_clauses[v]:
                    # Find position of v in clause ci
                    for j, cv in enumerate(clause_vars[ci]):
                        if cv == v:
                            signs_for_v.append(clause_signs[ci][j])
                            break
                if signs_for_v:
                    if all(s == 1 for s in signs_for_v):
                        assignments[v] = True
                        changed = True
                    elif all(s == -1 for s in signs_for_v):
                        assignments[v] = False
                        changed = True
            # Unit propagation
            if changed:
                continue
            for ci in range(len(clause_vars)):
                # Count unassigned literals in this clause
                unassigned = 0
                satisfied = False
                for j, cv in enumerate(clause_vars[ci]):
                    if assignments[cv] is not None:
                        val = assignments[cv]
                        sign = clause_signs[ci][j]
                        if (sign == 1 and val) or (sign == -1 and not val):
                            satisfied = True
                            break
                    else:
                        unassigned += 1
                if satisfied:
                    continue
                if unassigned == 1:
                    # Find the unassigned variable
                    for j, cv in enumerate(clause_vars[ci]):
                        if assignments[cv] is None:
                            sign = clause_signs[ci][j]
                            assignments[cv] = (sign == 1)
                            changed = True
                            break
        return assignments

    # Initial random assignment
    rng = random.Random(42)
    assignments = [None] * n
    for i in range(n):
        assignments[i] = rng.random() < 0.5

    # Apply preprocessing
    assignments = preprocess(assignments, clause_vars, clause_signs, var_to_clauses)

    # Convert to numpy array for vectorized operations
    # Use 1.0 for True, 0.0 for False
    assign_arr = np.array([1.0 if a else 0.0 for a in assignments], dtype=np.float64)

    # Track which variables are fixed by preprocessing
    fixed_mask = np.array([1.0 if a is not None else 0.0 for a in assignments], dtype=np.float64)

    def check_satisfaction(assign):
        """Check if all clauses are satisfied. Returns True if SAT."""
        # For each clause, check if any literal is satisfied
        # A literal (var, sign) is satisfied if:
        #   sign == 1 and assign[var] == 1
        #   sign == -1 and assign[var] == 0
        # Compute satisfaction for all clauses at once
        # assign_expanded[i, j] = assign[clause_vars[i, j]]
        assign_expanded = assign[clause_vars]  # shape (m, max_clause_len)
        # sign_expanded[i, j] = clause_signs[i, j]
        # Satisfied if: (sign == 1 and assign == 1) or (sign == -1 and assign == 0)
        # = (sign * (2*assign - 1) == 1)
        # sign * (2*assign - 1): if sign=1, assign=1 -> 1; sign=1, assign=0 -> -1
        #                        if sign=-1, assign=0 -> 1; sign=-1, assign=1 -> -1
        val = clause_signs * (2.0 * assign_expanded - 1.0)
        # A clause is satisfied if any literal has val == 1
        # But we need to handle padding - actually all clauses are length 3
        sat = np.any(val == 1.0, axis=1)
        return np.all(sat)

    def get_unsatisfied_clauses(assign):
        """Return indices of unsatisfied clauses."""
        assign_expanded = assign[clause_vars]
        val = clause_signs * (2.0 * assign_expanded - 1.0)
        sat = np.any(val == 1.0, axis=1)
        return np.where(~sat)[0]

    # Check if already solved
    if check_satisfaction(assign_arr):
        return [bool(a) for a in assignments]

    # Clause-weighted WalkSAT with restarts
    best_assign = assign_arr.copy()
    best_unsat_count = len(get_unsatisfied_clauses(assign_arr))

    def walksat_step(assign, unsat_clauses, noise):
        """One step of clause-weighted WalkSAT. Returns new assign, new unsat_clauses."""
        if len(unsat_clauses) == 0:
            return assign, unsat_clauses

        # Weight each variable by how many unsatisfied clauses it appears in
        var_weights = np.zeros(n, dtype=np.float64)
        for ci in unsat_clauses:
            for v in clause_vars[ci]:
                var_weights[v] += 1.0

        # Pick an unsatisfied clause (randomly, but weighted by weight)
        # For efficiency, pick random unsatisfied clause
        ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
        clause = clause_vars[ci]
        signs = clause_signs[ci]

        # Choose variable to flip
        # With probability (1-noise), pick the variable that appears in most unsatisfied clauses
        # With probability noise, pick randomly from the clause
        if rng.random() < (1.0 - noise):
            # Pick variable with max weight among those in this clause
            max_w = -1
            best_v = clause[0]
            for v in clause:
                if var_weights[v] > max_w:
                    max_w = var_weights[v]
                    best_v = v
            v = best_v
        else:
            v = clause[rng.randint(0, len(clause) - 1)]

        # Don't flip fixed variables
        if fixed_mask[v] > 0:
            return assign, unsat_clauses

        # Flip variable v
        new_assign = assign.copy()
        new_assign[v] = 1.0 - new_assign[v]

        # Recompute unsatisfied clauses
        new_unsat = get_unsatisfied_clauses(new_assign)
        return new_assign, new_unsat

    # Multiple restarts
    num_restarts = 20
    steps_per_restart = 5000

    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reset to initial assignment (or a new random one for later restarts)
        if restart == 0:
            assign = best_assign.copy()
        else:
            # New random assignment, but respect fixed variables
            assign = assign_arr.copy()
            for i in range(n):
                if fixed_mask[i] == 0:
                    assign[i] = 1.0 if rng.random() < 0.5 else 0.0

        unsat = get_unsatisfied_clauses(assign)
        if len(unsat) == 0:
            best_assign = assign.copy()
            break

        # Adaptive noise: start low, increase over time
        for step in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break

            if len(unsat) == 0:
                best_assign = assign.copy()
                break

            # Noise schedule: increase from 0.1 to 0.5
            progress = step / float(steps_per_restart)
            noise = 0.1 + 0.4 * progress

            assign, unsat = walksat_step(assign, unsat, noise)

            # Track best
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assign = assign.copy()

        if best_unsat_count == 0:
            break

    # Final check
    if check_satisfaction(best_assign):
        return [bool(a) for a in best_assign]

    # Return the best assignment found
    return [bool(a) for a in best_assign]
# EVOLVE-BLOCK-END
