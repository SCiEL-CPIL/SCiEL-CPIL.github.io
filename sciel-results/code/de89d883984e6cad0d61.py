"""Improved planted 3-SAT solver with efficient WalkSAT + restarts."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: 330s wall-clock limit, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0

    # Preprocess: convert to 0-based variable indices and sign (+1/-1)
    # clause_vars[c] = list of (var_idx, sign)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
        clause_vars.append(cv)

    # Build variable -> list of (clause_idx, position_in_clause) for fast updates
    # We'll track for each variable which clauses it appears in and its sign there
    var_occurrences = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for pi, (v, s) in enumerate(cv):
            var_occurrences[v].append((ci, pi, s))

    # We'll maintain:
    #   assignment: numpy bool array of length n
    #   clause_sat_count: numpy int array of length m, number of satisfied literals per clause
    #   For each variable, we track how flipping it affects each clause

    rng = random.Random(42)

    def is_satisfied(assignment, clause):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def full_check(assignment):
        """Check if all clauses are satisfied."""
        for ci in range(m):
            cv = clause_vars[ci]
            sat = False
            for v, s in cv:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    sat = True
                    break
            if not sat:
                return False, ci
        return True, -1

    def init_state(assignment):
        """Initialize clause_sat_count and related structures."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for ci, cv in enumerate(clause_vars):
            cnt = 0
            for v, s in cv:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    cnt += 1
            clause_sat_count[ci] = cnt
        return clause_sat_count

    def pick_clause(clause_sat_count, rng):
        """Pick a random unsatisfied clause."""
        unsat_indices = np.where(clause_sat_count == 0)[0]
        if len(unsat_indices) == 0:
            return -1
        return int(rng.choice(unsat_indices.tolist()))

    def flip_variable(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count."""
        assignment[v] = not assignment[v]
        for (ci, pi, s) in var_occurrences[v]:
            # Before flip: was this literal satisfied?
            old_val = assignment[v]  # now flipped, so old was not assignment[v]
            # Actually, we just flipped, so old value is not assignment[v]
            old_val = not assignment[v]
            was_sat = (old_val and s == 1) or (not old_val and s == -1)
            is_sat = (assignment[v] and s == 1) or (not assignment[v] and s == -1)
            if was_sat and not is_sat:
                clause_sat_count[ci] -= 1
            elif not was_sat and is_sat:
                clause_sat_count[ci] += 1

    # Main solver loop with restarts
    best_assignment = None
    best_unsat_count = m + 1

    max_restarts = 5000
    steps_per_restart = 5000
    p_flip_good = 0.55  # probability to flip a literal that satisfies the clause

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        clause_sat_count = init_state(assignment)
        unsat_count = int(np.sum(clause_sat_count == 0))

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline - 1.0:
                break

            if unsat_count == 0:
                return assignment.tolist()

            # Pick an unsatisfied clause
            ci = pick_clause(clause_sat_count, rng)
            if ci < 0:
                return assignment.tolist()

            cv = clause_vars[ci]

            # Try to flip a literal that satisfies the clause (with probability p_flip_good)
            if rng.random() < p_flip_good:
                # Pick a random literal from the clause and flip it
                pi = rng.randint(0, len(cv) - 1)
                v = cv[pi][0]
            else:
                # Pick a random variable from the clause
                pi = rng.randint(0, len(cv) - 1)
                v = cv[pi][0]

            # Compute the change in unsat count if we flip v
            # This is a heuristic: just flip and track
            old_unsat = unsat_count

            # Track which clauses change
            affected = []
            for (aci, api, s) in var_occurrences[v]:
                old_val = assignment[v]
                was_sat = (old_val and s == 1) or (not old_val and s == -1)
                is_sat = (not old_val and s == 1) or (old_val and s == -1)
                if was_sat != is_sat:
                    affected.append((aci, was_sat, is_sat))

            # Calculate new unsat count
            delta_unsat = 0
            for (aci, was_sat, is_sat) in affected:
                if was_sat and not is_sat:
                    # Clause becomes unsatisfied
                    if clause_sat_count[aci] == 1:  # only this literal was satisfying
                        delta_unsat += 1
                elif not was_sat and is_sat:
                    # Clause becomes satisfied
                    if clause_sat_count[aci] == 0:
                        delta_unsat -= 1

            # With some probability, accept even if it increases unsat count (for exploration)
            if delta_unsat <= 0 or (delta_unsat > 0 and rng.random() < 0.3):
                # Perform the flip
                flip_variable(v, assignment, clause_sat_count)
                unsat_count = old_unsat + delta_unsat

                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment.copy()
            else:
                # Skip this flip
                continue

        # End of restart
        if time.time() > deadline - 1.0:
            break

    # Return best found
    if best_assignment is not None:
        # Final check
        ok, _ = full_check(best_assignment)
        if ok:
            return best_assignment.tolist()
        return best_assignment.tolist()
    else:
        return [False] * n
# EVOLVE-BLOCK-END
