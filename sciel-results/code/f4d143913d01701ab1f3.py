"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Preprocess: 0-based variable indices and sign booleans per clause
    cvars = []
    csigns = []
    for cl in clauses:
        vs = [abs(l) - 1 for l in cl]
        ss = [l > 0 for l in cl]
        cvars.append(vs)
        csigns.append(ss)

    # Variable -> list of clause indices containing that variable
    var_clauses = [[] for _ in range(n)]
    for ci, vs in enumerate(cvars):
        for v in vs:
            var_clauses[v].append(ci)

    # Clause -> count of occurrences of each variable (for tie-breaking)
    # Since clauses are 3-CNF, each variable appears at most once per clause
    # We'll just use the number of clauses containing the variable

    deadline = time.time() + 325.0  # safety margin before 330s kill

    def evaluate_assignment(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for j in range(3):
                if assignment[vs[j]] == ss[j]:
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat

    def is_fully_satisfied(assignment):
        """Quick check if all clauses are satisfied."""
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for j in range(3):
                if assignment[vs[j]] == ss[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def walk_sat_restart(assignment, max_flips, rng, noise_p):
        """
        Perform WalkSAT from given assignment with incremental unsat tracking.
        Returns (success: bool, final_assignment, flips_used).
        """
        # Compute initial unsat set
        unsat = evaluate_assignment(assignment)
        if not unsat:
            return True, assignment, 0

        # Track unsat count per variable for scoring
        # unsat_count[v] = number of unsat clauses containing variable v
        unsat_count = [0] * n
        for ci in unsat:
            vs = cvars[ci]
            for v in vs:
                unsat_count[v] += 1

        flips_used = 0
        m_unsat = len(unsat)

        for _ in range(max_flips):
            if m_unsat == 0:
                return True, assignment, flips_used

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            vs = cvars[ci]
            ss = csigns[ci]

            # Choose which variable to flip in this clause
            # With probability noise_p, pick randomly; otherwise pick the one
            # that minimizes the number of newly unsatisfied clauses
            if rng.random() < noise_p:
                idx = rng.randint(0, 2)
            else:
                best_idx = 0
                best_score = float('inf')
                for j in range(3):
                    v = vs[j]
                    # If we flip v, how many currently-satisfied clauses become unsat?
                    # And how many currently-unsat clauses become sat?
                    # Score = number of clauses that would become newly unsatisfied
                    # A clause containing v that is currently satisfied becomes unsat
                    # only if v was the only satisfied literal in it.
                    
                    # Count clauses containing v that are currently satisfied
                    # and where v is the only satisfied literal
                    newly_unsat = 0
                    for cj in var_clauses[v]:
                        if cj == ci:
                            continue  # skip the clause we're already flipping in
                        cjs = csigns[cj]
                        cjv = cvars[cj]
                        # Check if clause cj is currently satisfied
                        cj_sat = False
                        v_is_only = True
                        for k in range(3):
                            cv = cjv[k]
                            if assignment[cv] == cjs[k]:
                                cj_sat = True
                                if cv != v:
                                    v_is_only = False
                        if cj_sat and v_is_only:
                            newly_unsat += 1
                    
                    # Also consider: flipping v might satisfy other unsat clauses
                    # We want to minimize net increase in unsat count
                    # Score = newly_unsat - (number of unsat clauses containing v that would be fixed)
                    # But the clause ci itself is one such clause
                    fixed = 1  # at least ci gets fixed
                    for cj in var_clauses[v]:
                        if cj == ci:
                            continue
                        # Is cj currently unsatisfied?
                        cjv2 = cvars[cj]
                        cjs2 = csigns[cj]
                        cj_is_unsat = True
                        for k in range(3):
                            if assignment[cjv2[k]] == cjs2[k]:
                                cj_is_unsat = False
                                break
                        if not cj_is_unsat:
                            continue
                        # cj is unsat; if flipping v makes v satisfy cj, then cj becomes sat
                        # v satisfies cj if assignment[v] after flip == cjs2[k] for the position of v
                        for k in range(3):
                            if cjv2[k] == v:
                                # After flip, assignment[v] becomes not assignment[v]
                                if (not assignment[v]) == cjs2[k]:
                                    fixed += 1
                                break

                    score = newly_unsat - (fixed - 1)  # -1 because ci is always fixed
                    if score < best_score:
                        best_score = score
                        best_idx = j
                idx = best_idx

            v = vs[idx]
            old_val = assignment[v]
            assignment[v] = not old_val
            flips_used += 1

            # Update unsat set incrementally
            # Clauses affected: all clauses containing v
            for cj in var_clauses[v]:
                cjv = cvars[cj]
                cjs = csigns[cj]
                # Determine if cj is satisfied before and after the flip
                # Before: we had old_val for v
                # After: we have not old_val for v
                
                # Check satisfaction before flip (using old_val)
                cj_sat_before = False
                for k in range(3):
                    cv = cjv[k]
                    if cv == v:
                        if old_val == cjs[k]:
                            cj_sat_before = True
                            break
                    else:
                        if assignment[cv] == cjs[k]:  # note: assignment[v] is already flipped, but cv != v here
                            cj_sat_before = True
                            break
                
                # Check satisfaction after flip (using not old_val for v)
                cj_sat_after = False
                for k in range(3):
                    cv = cjv[k]
                    if cv == v:
                        if (not old_val) == cjs[k]:
                            cj_sat_after = True
                            break
                    else:
                        if assignment[cv] == cjs[k]:
                            cj_sat_after = True
                            break
                
                if cj_sat_before and not cj_sat_after:
                    # cj became unsatisfied
                    if cj not in unsat:
                        unsat.append(cj)
                        m_unsat += 1
                        for vv in cjv:
                            unsat_count[vv] += 1
                elif not cj_sat_before and cj_sat_after:
                    # cj became satisfied
                    if cj in unsat:
                        unsat.remove(cj)
                        m_unsat -= 1
                        for vv in cjv:
                            unsat_count[vv] -= 1

        return False, assignment, flips_used

    # Multiple restarts with increasing flip budgets and adaptive noise
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Restart schedule: start with moderate budget, increase over time
    restart_budgets = [5000, 10000, 20000, 40000, 80000, 150000, 300000, 500000]
    noise_levels = [0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]

    restart_idx = 0
    while time.time() < deadline:
        # Choose restart parameters
        if restart_idx < len(restart_budgets):
            max_flips = restart_budgets[restart_idx]
            noise_p = noise_levels[restart_idx]
        else:
            # After exhausting scheduled restarts, use larger budgets
            max_flips = min(1000000, restart_budgets[-1] * 2)
            noise_p = 0.5 + 0.3 * rng.random()

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        success, final_assignment, flips_used = walk_sat_restart(
            assignment, max_flips, rng, noise_p
        )

        if success:
            return final_assignment

        # Track best partial solution
        unsat_list = evaluate_assignment(final_assignment)
        if len(unsat_list) < best_unsat_count:
            best_unsat_count = len(unsat_list)
            best_assignment = final_assignment[:]

        restart_idx += 1

    # Return best partial solution found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
