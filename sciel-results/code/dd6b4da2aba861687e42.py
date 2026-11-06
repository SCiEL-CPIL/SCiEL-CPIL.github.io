# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin before 300s limit

    # Preprocess clauses to 0-based signed literals
    # Each clause: list of (var_idx, sign) where sign is 1 for positive, -1 for negative
    clauses_pre = []
    for c in clauses:
        pre_c = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pre_c.append((var_idx, sign))
        clauses_pre.append(pre_c)

    m = len(clauses_pre)

    # Build variable-to-clause index for efficient updates
    # var_clause[var] = list of (clause_idx, sign_in_clause)
    var_clause = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_pre):
        for (var, sign) in c:
            var_clause[var].append((ci, sign))

    def count_unsat(clause_idx, assignment):
        """Count how many literals in the clause are unsatisfied."""
        c = clauses_pre[clause_idx]
        cnt = 0
        for (var, sign) in c:
            if assignment[var] != (sign == 1):
                cnt += 1
        return cnt

    def is_satisfied(clause_idx, assignment):
        """Check if clause is satisfied."""
        c = clauses_pre[clause_idx]
        for (var, sign) in c:
            if assignment[var] == (sign == 1):
                return True
        return False

    def total_unsat(assignment):
        """Count total unsatisfied clauses."""
        cnt = 0
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                cnt += 1
        return cnt

    def walksat(assignment, max_steps, rng, p_flip):
        """WalkSAT with random restarts. Returns (solved, final_assignment, steps_used)."""
        current = assignment[:]
        best_unsat = total_unsat(current)
        best_assignment = current[:]

        if best_unsat == 0:
            return True, current, 0

        for step in range(max_steps):
            # Find unsatisfied clauses
            unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci, current)]
            if not unsat_clauses:
                return True, current, step + 1

            # Update best
            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = current[:]

            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clauses_pre[clause_idx]

            # Compute flip scores: for each variable in clause, how many clauses would
            # be satisfied/unsatisfied if we flip it
            best_var = -1
            best_score = -10**9
            best_flip_val = None

            for (var, sign) in clause:
                # If we flip var, what happens to all clauses containing var?
                # Current state of var
                cur_val = current[var]
                new_val = not cur_val

                # Score: number of clauses that become satisfied minus number that become unsatisfied
                # For each clause containing var:
                # - If clause currently satisfied and var is the only true literal, flipping makes it unsat
                # - If clause currently unsat and var is the only false literal, flipping makes it sat
                score = 0
                for (ci2, sign2) in var_clause[var]:
                    # Check current satisfaction status of clause ci2
                    c2 = clauses_pre[ci2]
                    # Count true literals in c2 currently
                    true_cnt = 0
                    for (v2, s2) in c2:
                        if current[v2] == (s2 == 1):
                            true_cnt += 1

                    # After flip, var's value changes
                    # var contributes (new_val == (sign2 == 1)) to the clause
                    var_was_true = (cur_val == (sign2 == 1))
                    var_will_be_true = (new_val == (sign2 == 1))

                    if true_cnt == 0:
                        # Clause is unsatisfied
                        if var_will_be_true:
                            score += 1  # becomes satisfied
                    elif true_cnt == 1 and var_was_true:
                        # Clause is satisfied, and var is the only true literal
                        if not var_will_be_true:
                            score -= 1  # becomes unsatisfied

                # Add randomness
                score += rng.random() * 0.1

                if score > best_score:
                    best_score = score
                    best_var = var
                    best_flip_val = new_val

            if best_var >= 0:
                current[best_var] = best_flip_val

            # Check if solved
            if all(is_satisfied(ci, current) for ci in range(m)):
                return True, current, step + 1

        # Return best found
        if total_unsat(best_assignment) < total_unsat(current):
            return False, best_assignment, max_steps
        return False, current, max_steps

    def simulated_annealing(assignment, max_steps, rng):
        """Simulated annealing for 3-SAT. Returns (solved, final_assignment, steps_used)."""
        current = assignment[:]
        best_unsat = total_unsat(current)
        best_assignment = current[:]

        if best_unsat == 0:
            return True, current, 0

        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        cooling = (T_end / T_start) ** (1.0 / max(1, max_steps - 1))

        T = T_start
        for step in range(max_steps):
            T *= cooling

            # Find unsatisfied clauses
            unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci, current)]
            if not unsat_clauses:
                return True, current, step + 1

            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = current[:]

            # Pick a random variable to flip
            var = rng.randint(0, n - 1)
            new_val = not current[var]

            # Compute delta: change in number of unsatisfied clauses
            # For each clause containing var, check if flipping changes satisfaction
            delta = 0
            for (ci2, sign2) in var_clause[var]:
                c2 = clauses_pre[ci2]
                true_cnt = 0
                for (v2, s2) in c2:
                    if current[v2] == (s2 == 1):
                        true_cnt += 1

                var_was_true = (current[var] == (sign2 == 1))
                var_will_be_true = (new_val == (sign2 == 1))

                if true_cnt == 0:
                    # Unsatisfied
                    if var_will_be_true:
                        delta -= 1  # becomes satisfied, good
                elif true_cnt == 1 and var_was_true:
                    # Satisfied, var is only true
                    if not var_will_be_true:
                        delta += 1  # becomes unsatisfied, bad

            # Accept with probability exp(-delta/T) if delta > 0
            if delta <= 0:
                current[var] = new_val
            else:
                if rng.random() < np.exp(-delta / max(T, 1e-10)):
                    current[var] = new_val

            # Check if solved
            if all(is_satisfied(ci, current) for ci in range(m)):
                return True, current, step + 1

        if total_unsat(best_assignment) < total_unsat(current):
            return False, best_assignment, max_steps
        return False, current, max_steps

    def local_search_with_tabu(assignment, max_steps, rng, tabu_len=10):
        """Local search with tabu list. Returns (solved, final_assignment, steps_used)."""
        current = assignment[:]
        best_unsat = total_unsat(current)
        best_assignment = current[:]

        if best_unsat == 0:
            return True, current, 0

        tabu = {}  # var -> step when it became tabu-free
        for step in range(max_steps):
            unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci, current)]
            if not unsat_clauses:
                return True, current, step + 1

            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = current[:]

            # Pick random unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clauses_pre[clause_idx]

            # Find best flip among variables in clause (excluding tabu)
            best_var = -1
            best_score = -10**9

            for (var, sign) in clause:
                if var in tabu and step < tabu[var]:
                    continue  # Tabu, skip

                cur_val = current[var]
                new_val = not cur_val

                score = 0
                for (ci2, sign2) in var_clause[var]:
                    c2 = clauses_pre[ci2]
                    true_cnt = 0
                    for (v2, s2) in c2:
                        if current[v2] == (s2 == 1):
                            true_cnt += 1

                    var_was_true = (cur_val == (sign2 == 1))
                    var_will_be_true = (new_val == (sign2 == 1))

                    if true_cnt == 0:
                        if var_will_be_true:
                            score += 1
                    elif true_cnt == 1 and var_was_true:
                        if not var_will_be_true:
                            score -= 1

                score += rng.random() * 0.5

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var < 0:
                # All tabu, pick random
                best_var = rng.randint(0, n - 1)

            current[best_var] = not current[best_var]
            tabu[best_var] = step + tabu_len

            if all(is_satisfied(ci, current) for ci in range(m)):
                return True, current, step + 1

        if total_unsat(best_assignment) < total_unsat(current):
            return False, best_assignment, max_steps
        return False, current, max_steps

    # Main search loop with multiple strategies and restarts
    rng = random.Random(0)
    best_overall = None
    best_unsat_overall = float('inf')

    # Initial random assignments
    initial_assignments = []
    for _ in range(5):
        init = [rng.random() < 0.5 for _ in range(n)]
        initial_assignments.append(init)

    # Also try biased initializations
    # For planted SAT, the planted solution might have certain bias
    # Try all-true and all-false
    initial_assignments.append([True] * n)
    initial_assignments.append([False] * n)

    strategies = [
        ("walksat", walksat),
        ("sa", simulated_annealing),
        ("tabu", local_search_with_tabu),
    ]

    steps_per_call = 5000
    p_flip_values = [0.5, 0.3, 0.7, 0.4, 0.6]

    attempt = 0
    while time.time() < deadline:
        # Pick a strategy and initial assignment
        strat_name, strat_func = strategies[attempt % len(strategies)]
        init = initial_assignments[attempt % len(initial_assignments)]

        if strat_name == "walksat":
            p = p_flip_values[attempt % len(p_flip_values)]
            solved, result, steps = strat_func(init, steps_per_call, rng, p)
        elif strat_name == "sa":
            solved, result, steps = strat_func(init, steps_per_call, rng)
        elif strat_name == "tabu":
            solved, result, steps = strat_func(init, steps_per_call, rng, tabu_len=10)
        else:
            solved, result, steps = strat_func(init, steps_per_call, rng)

        unsat_count = total_unsat(result)
        if unsat_count < best_unsat_overall:
            best_unsat_overall = unsat_count
            best_overall = result[:]

        if solved:
            return result

        attempt += 1

        # Occasionally try a fresh random start
        if attempt % 10 == 0:
            fresh = [rng.random() < 0.5 for _ in range(n)]
            initial_assignments.append(fresh)

    # Final polish: if we have a near-solution, do intensive local search
    if best_overall is not None and best_unsat_overall > 0 and best_unsat_overall <= 10:
        # Intensive walkSAT from the best solution
        while time.time() < deadline:
            solved, result, steps = walksat(best_overall, steps_per_call, rng, 0.5)
            if solved:
                return result
            unsat_count = total_unsat(result)
            if unsat_count < best_unsat_overall:
                best_unsat_overall = unsat_count
                best_overall = result[:]

    return best_overall if best_overall is not None else [False] * n
# EVOLVE-BLOCK-END
