"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Precompute: for each clause, which variable indices it contains,
    # and the sign (+1 if literal is positive, -1 if negative).
    clause_vars = np.empty((m, 3), dtype=np.int64)
    clause_signs = np.empty((m, 3), dtype=np.int8)
    for ci, c in enumerate(clauses):
        for li, lit in enumerate(c):
            clause_vars[ci, li] = abs(lit) - 1
            clause_signs[ci, li] = 1 if lit > 0 else -1

    # For each variable, list of (clause_index, position_in_clause, sign)
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            v = int(clause_vars[ci, li])
            var_clauses[v].append((ci, li, int(clause_signs[ci, li])))

    # For each variable, list of (clause_index, position_in_clause, sign)
    # grouped by sign: var_pos_clauses[v] = list of (ci, li) where sign=+1
    # var_neg_clauses[v] = list of (ci, li) where sign=-1
    var_pos_clauses = [[] for _ in range(n)]
    var_neg_clauses = [[] for _ in range(n)]
    for v in range(n):
        for (ci, li, s) in var_clauses[v]:
            if s == 1:
                var_pos_clauses[v].append((ci, li))
            else:
                var_neg_clauses[v].append((ci, li))

    # clause_satisfied_counts[ci] = number of literals satisfied in clause ci
    # unsat_counts[v] = number of clauses where variable v is the ONLY
    #   satisfied literal (i.e., flipping v would make that clause unsat)

    deadline = time.time() + 320.0

    def evaluate_and_solve(rng, assignment):
        """Run WalkSAT with clause count tracking. Returns (solved, assignment)."""
        # assignment: numpy bool array of length n
        # Compute initial satisfied counts
        # A literal (v, sign) is satisfied iff sign * (1 if assignment[v] else -1) > 0
        # i.e., if sign==1: assignment[v] is True; if sign==-1: assignment[v] is False

        # clause_sat_count[ci] = sum over li of is_satisfied(clause_vars[ci,li], clause_signs[ci,li])
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            for li in range(3):
                v = int(clause_vars[ci, li])
                s = int(clause_signs[ci, li])
                if s == 1:
                    if assignment[v]:
                        clause_sat_count[ci] += 1
                else:
                    if not assignment[v]:
                        clause_sat_count[ci] += 1

        unsat_clauses = np.where(clause_sat_count == 0)[0]
        if len(unsat_clauses) == 0:
            return True, assignment

        # unsat_only_count[v] = number of clauses where v is the only satisfied literal
        # For a clause ci with sat_count == 1, the single satisfied literal's variable
        # gets +1 to its unsat_only_count.
        unsat_only_count = np.zeros(n, dtype=np.int32)
        for ci in range(m):
            if clause_sat_count[ci] == 1:
                # Find the single satisfied literal
                for li in range(3):
                    v = int(clause_vars[ci, li])
                    s = int(clause_signs[ci, li])
                    if s == 1:
                        if assignment[v]:
                            unsat_only_count[v] += 1
                            break
                    else:
                        if not assignment[v]:
                            unsat_only_count[v] += 1
                            break

        # WalkSAT loop
        # Parameters: noise parameter for literal selection
        step = 0
        while time.time() < deadline - 0.5:
            if len(unsat_clauses) == 0:
                return True, assignment

            # Pick a random unsatisfied clause
            ci = int(unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)])

            # Score each literal in the clause:
            # If we flip variable v of literal (v, s):
            #   - The literal becomes satisfied (since clause is unsat, no lit is satisfied)
            #   - We gain: number of OTHER clauses where v is the only satisfied literal
            #     (these become unsat)
            #   - We lose: 0 for this clause (it becomes sat)
            # Actually: flipping v changes satisfaction of v in all clauses containing v.
            # For clause ci (currently unsat): v's literal becomes sat -> clause ci gains 1 sat.
            # For other clauses containing v:
            #   If v's literal was sat in that clause: it becomes unsat.
            #     If that clause had sat_count == 1, it becomes unsat (bad).
            #     If that clause had sat_count > 1, it loses one but stays sat (neutral).
            #   If v's literal was unsat in that clause: it becomes sat.
            #     That clause gains one sat (good, but we're focusing on unsat clause).

            # Standard WalkSAT scoring: for each literal in the chosen clause,
            # score = number of other clauses that would become unsatisfied
            #         minus number of other clauses that would become satisfied
            # We want to minimize "harm" (clauses becoming unsat)
            # and maximize "help" (clauses becoming sat).

            # Simpler: score = - (number of clauses where v is the only satisfied literal)
            #           + (number of other unsat clauses that would become sat)
            # But computing the second term is expensive. Standard WalkSAT just
            # uses: score = number of clauses that would become UNSAT (we want to minimize)

            # Let's use: for each variable v in clause ci,
            #   harm = unsat_only_count[v]  (clauses that become unsat)
            #   This is the standard "min-conflicts" style score.

            # Pick literal with minimum harm, with some randomness
            # (classic WalkSAT: with prob p, pick random; else pick min-harm)
            noise = 0.5
            if rng.random() < noise:
                li = rng.randint(0, 2)
            else:
                # Find min harm
                min_harm = 999999
                best_li = 0
                for li in range(3):
                    v = int(clause_vars[ci, li])
                    harm = int(unsat_only_count[v])
                    if harm < min_harm:
                        min_harm = harm
                        best_li = li
                li = best_li

            v = int(clause_vars[ci, li])
            s = int(clause_signs[ci, li])

            # Update counts before flipping
            # Current state of literal (v, s): is it satisfied?
            # Since clause ci is unsat, this literal is NOT satisfied.
            # After flip, it WILL be satisfied.
            # So clause ci: sat_count goes from 0 to 1.
            #   unsat_clauses: remove ci.
            #   unsat_only_count: ci now has 1 sat literal (v), so v gets +1 to unsat_only_count.

            # For other clauses containing v:
            #   v's literal sign in clause ci is s.
            #   In other clauses, v may appear with sign +1 or -1.
            #   Currently: v's literal is unsatisfied in clause ci (we're flipping).
            #   After flip: v's truth value changes.
            #   For a clause cj containing v with sign s2:
            #     Before: literal satisfied iff (s2==1 and assignment[v]) or (s2==-1 and not assignment[v])
            #     After: assignment[v] flips, so literal satisfaction flips.
            #     If it was satisfied -> becomes unsatisfied: clause cj loses 1 sat.
            #       If cj had sat_count == 1 -> becomes unsat (bad), unsat_only_count[v] decreases by 1
            #       (because v was the only sat literal in cj, and now cj is unsat, so v no longer
            #        "supports" any clause as sole sat literal... wait, we need to be careful).

            # Let me re-derive carefully.
            # Let a = assignment[v] (current truth value).
            # After flip: a' = not a.
            #
            # For clause ci (the chosen unsat clause), literal (v, s):
            #   Currently unsatisfied (since ci is unsat).
            #   After flip: satisfied.
            #   clause_sat_count[ci]: 0 -> 1
            #   ci leaves unsat_clauses.
            #   unsat_only_count[v]: +1 (v is now the sole satisfied literal in ci)
            #
            # For other clauses cj containing v:
            #   Let s2 be the sign of v in cj.
            #   Current satisfaction of (v, s2): sat2 = (s2==1 and a) or (s2==-1 and not a)
            #   After flip: sat2' = not sat2
            #
            #   Case 1: sat2 was True (literal was satisfied)
            #     clause_sat_count[cj] decreases by 1.
            #     If clause_sat_count[cj] was 1: becomes 0 -> cj becomes unsat.
            #       unsat_only_count[v] decreases by 1 (v was sole sat literal, now gone).
            #       cj enters unsat_clauses.
            #     If clause_sat_count[cj] was > 1: stays satisfied.
            #       No change to unsat_only_count.
            #
            #   Case 2: sat2 was False (literal was unsatisfied)
            #     clause_sat_count[cj] increases by 1.
            #     If clause_sat_count[cj] was 0: becomes 1 -> cj leaves unsat_clauses.
            #       unsat_only_count[v] increases by 1 (v is now sole sat literal in cj).
            #     If clause_sat_count[cj] was > 0: stays satisfied (or was already sat).
            #       No change to unsat_only_count.

            # Apply changes:
            old_a = assignment[v]

            # Clause ci: 0 -> 1
            clause_sat_count[ci] = 1
            unsat_only_count[v] += 1

            # Other clauses containing v
            for (cj, lj) in var_pos_clauses[v]:
                if cj == ci:
                    continue
                # sign = +1, currently: sat iff a
                if old_a:  # was satisfied
                    clause_sat_count[cj] -= 1
                    if clause_sat_count[cj] == 0:
                        # becomes unsat
                        unsat_only_count[v] -= 1
                        # cj enters unsat_clauses - we'll rebuild later
                else:  # was unsatisfied
                    clause_sat_count[cj] += 1
                    if clause_sat_count[cj] == 1:
                        # was unsat, now sat
                        unsat_only_count[v] += 1
                        # cj leaves unsat_clauses

            for (cj, lj) in var_neg_clauses[v]:
                if cj == ci:
                    continue
                # sign = -1, currently: sat iff not a
                if not old_a:  # was satisfied
                    clause_sat_count[cj] -= 1
                    if clause_sat_count[cj] == 0:
                        unsat_only_count[v] -= 1
                else:  # was unsatisfied
                    clause_sat_count[cj] += 1
                    if clause_sat_count[cj] == 1:
                        unsat_only_count[v] += 1

            # Flip
            assignment[v] = not old_a

            # Rebuild unsat_clauses periodically (every 100 steps) for efficiency
            step += 1
            if step % 100 == 0 or step == 1:
                unsat_clauses = np.where(clause_sat_count == 0)[0]
                if len(unsat_clauses) == 0:
                    return True, assignment

        # Time's up, return current best
        return False, assignment

    # Multiple restarts with different random seeds
    best_assignment = None
    rng = random.Random(0)

    while time.time() < deadline - 1.0:
        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        solved, result = evaluate_and_solve(rng, assignment)
        if solved:
            return result.tolist()

        if result is not None:
            # Count unsatisfied clauses as a quality measure
            # (we don't easily have the final count, so just keep last)
            best_assignment = result

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
