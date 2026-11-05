"""
Hybrid WalkSAT with incremental clause tracking and adaptive restarts.
Structural change: maintain per-clause satisfaction counts and per-variable
clause incidence to enable O(1) incremental updates instead of O(m) recomputation.
"""
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285.0  # Safety margin before 300s kill

    # Precompute clause structure: for each clause, list of (var_idx, polarity)
    # polarity: 1 if literal is positive (var must be True), -1 if negative
    clause_data = []
    for c in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])

    # Precompute variable-to-clause incidence
    # var_clauses[var_idx] = list of (clause_idx, polarity_in_clause)
    var_clauses = [[] for _ in range(n)]
    for ci, cdata in enumerate(clause_data):
        for var_idx, pol in cdata:
            var_clauses[var_idx].append((ci, pol))

    def initial_state(assignment):
        """Compute initial sat_counts and unsat_clauses."""
        sat_counts = [0] * m
        unsat = []
        for ci, cdata in enumerate(clause_data):
            cnt = 0
            for var_idx, pol in cdata:
                if (assignment[var_idx] == True) == (pol == 1):
                    cnt += 1
            sat_counts[ci] = cnt
            if cnt == 0:
                unsat.append(ci)
        return sat_counts, unsat

    def flip_variable(assignment, var_idx, sat_counts, unsat):
        """
        Flip variable var_idx and update sat_counts and unsat incrementally.
        Returns new unsat list.
        """
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val

        # Update all clauses containing this variable
        for ci, pol in var_clauses[var_idx]:
            old_sat = sat_counts[ci]
            # Determine if this literal was satisfied before and after flip
            # Literal satisfied if (assignment[var] == True) == (pol == 1)
            was_sat = (old_val == True) == (pol == 1)
            is_now_sat = (new_val == True) == (pol == 1)

            if was_sat and not is_now_sat:
                # Was satisfied, now unsatisfied
                new_sat = old_sat - 1
                sat_counts[ci] = new_sat
                if new_sat == 0:
                    unsat.append(ci)
            elif not was_sat and is_now_sat:
                # Was unsatisfied, now satisfied
                new_sat = old_sat + 1
                sat_counts[ci] = new_sat
                if new_sat > 0:
                    # Remove from unsat if it was there
                    try:
                        unsat.remove(ci)
                    except ValueError:
                        pass

        return unsat

    def is_fully_satisfied(unsat):
        return len(unsat) == 0

    def walksat_run(assignment, rng, max_steps, time_limit):
        """
        Run WalkSAT with incremental updates.
        Returns (assignment, solved).
        """
        sat_counts, unsat = initial_state(assignment)
        best_unsat_count = len(unsat)
        best_assignment = assignment[:]
        noise = 0.45

        for step in range(max_steps):
            if time.time() > time_limit:
                break
            if not unsat:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            clause = clause_data[ci]

            if rng.random() < noise:
                # Random flip
                var_idx, pol = rng.choice(clause)
            else:
                # Choose variable that minimizes damage (breaks fewest clauses)
                best_var = -1
                best_damage = float('inf')
                for var_idx, pol in clause:
                    # Estimate damage: count clauses that would become unsatisfied
                    # A clause c is broken if:
                    #   - it contains this variable
                    #   - it is currently satisfied (sat_counts[c] > 0)
                    #   - the only satisfied literal in c is this variable's literal
                    #     (i.e., sat_counts[c] == 1 AND this literal is currently satisfied)
                    # But wait: after flip, this literal becomes unsatisfied.
                    # So if sat_counts[c] == 1 and this literal is currently satisfied,
                    # flipping makes sat_counts[c] == 0 -> broken.
                    # If sat_counts[c] > 1, flipping this literal (if it was satisfied)
                    # still leaves c satisfied.
                    # If this literal was NOT satisfied, flipping it makes it satisfied,
                    # so it can only help.
                    
                    # Current state of this literal:
                    current_val = assignment[var_idx]
                    lit_was_sat = (current_val == True) == (pol == 1)
                    
                    damage = 0
                    if lit_was_sat:
                        # Flipping will make this literal unsatisfied
                        # Clauses that are currently satisfied ONLY by this literal will break
                        for c_idx, c_pol in var_clauses[var_idx]:
                            if c_idx == ci:
                                continue
                            if sat_counts[c_idx] == 1:
                                # Check if the satisfied literal in c_idx is this one
                                # c_pol is the polarity of this variable in clause c_idx
                                # The literal is satisfied if (assignment[var] == True) == (c_pol == 1)
                                # Since lit_was_sat is True for pol, and c_pol may differ
                                # Actually, lit_was_sat is for the specific polarity in THIS clause
                                # For clause c_idx, the polarity is c_pol
                                # The literal in c_idx is satisfied if (assignment[var] == True) == (c_pol == 1)
                                lit_in_c_sat = (current_val == True) == (c_pol == 1)
                                if lit_in_c_sat:
                                    damage += 1
                    
                    if damage < best_damage:
                        best_damage = damage
                        best_var = var_idx
                
                if best_var == -1:
                    var_idx, pol = rng.choice(clause)
                else:
                    # Find the polarity for best_var in this clause
                    for v, p in clause:
                        if v == best_var:
                            var_idx, pol = v, p
                            break

            # Perform the flip
            unsat = flip_variable(assignment, var_idx, sat_counts, unsat)

            # Track best
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment[:]

        # Check if solved
        if not unsat:
            return assignment, True

        # Return best found
        if best_unsat_count < len(unsat):
            return best_assignment, False
        return assignment, False

    # Main loop with adaptive restarts
    best_assignment = None
    best_unsat_count = m + 1
    rng = random.Random(42)

    # Adaptive restart strategy: start with short runs, increase if making progress
    base_steps = 500
    max_steps_cap = 50000
    trial = 0
    steps = base_steps
    stagnant_count = 0

    while time.time() < deadline - 10.0:
        trial += 1
        # Adaptive step count
        if stagnant_count >= 3:
            steps = min(steps * 2, max_steps_cap)
            stagnant_count = 0
        elif trial % 5 == 0 and steps > base_steps:
            steps = max(base_steps, steps // 2)

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Quick check
        sat_counts, unsat = initial_state(assignment)
        if not unsat:
            return assignment

        # Run WalkSAT
        time_limit = min(deadline - 5.0, time.time() + 2.0)
        result, solved = walksat_run(assignment, rng, steps, time_limit)

        if solved:
            return result

        # Track best
        _, final_unsat = initial_state(result)
        unsat_count = len(final_unsat)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = result[:]
            stagnant_count = 0
        else:
            stagnant_count += 1

        # If close to deadline, do one final long run with best assignment
        if time.time() > deadline - 30.0 and best_assignment is not None:
            time_limit = deadline - 5.0
            result, solved = walksat_run(best_assignment, rng, 100000, time_limit)
            if solved:
                return result
            _, final_unsat = initial_state(result)
            if len(final_unsat) < best_unsat_count:
                best_assignment = result
            break

    if best_assignment is not None:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
