"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """Simulated Annealing SAT solver with incremental clause tracking."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 45s safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build variable-to-clauses index for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(ci)

    def is_clause_satisfied(clause, assignment):
        """Check if a single clause is satisfied."""
        for var_idx, sign in clause:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False

    def init_unsat_and_counts(assignment):
        """Initialize unsat clause set and per-variable unsat clause counts."""
        unsat_clauses = set()
        var_unsat_count = [0] * n
        for ci, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                unsat_clauses.add(ci)
                for var_idx, _ in clause:
                    var_unsat_count[var_idx] += 1
        return unsat_clauses, var_unsat_count

    def sa_one(assignment, rng, max_steps, T_init=10.0, T_min=0.01, cooling_rate=0.9995):
        """Run simulated annealing with maintained counts."""
        unsat_clauses, var_unsat_count = init_unsat_and_counts(assignment)
        current_uc = len(unsat_clauses)

        if current_uc == 0:
            return assignment, True

        T = T_init
        best_uc_in_run = current_uc
        best_assignment_snapshot = assignment[:]

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if current_uc == 0:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            clause = clause_data[ci]

            # Choose which literal to flip
            # With some probability: random literal (exploration)
            # Otherwise: pick literal whose variable appears in most unsat clauses (exploitation)
            if rng.random() < 0.3:
                chosen_idx = rng.randint(0, len(clause) - 1)
            else:
                best_var = clause[0][0]
                best_count = var_unsat_count[best_var]
                for i in range(1, len(clause)):
                    v = clause[i][0]
                    c = var_unsat_count[v]
                    if c > best_count:
                        best_count = c
                        best_var = v
                for i in range(len(clause)):
                    if clause[i][0] == best_var:
                        chosen_idx = i
                        break

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Incrementally update unsat_clauses and var_unsat_count
            delta_uc = 0
            for affected_ci in var_to_clauses[var_idx]:
                was_unsat = affected_ci in unsat_clauses
                is_now_sat = is_clause_satisfied(clause_data[affected_ci], assignment)

                if was_unsat and is_now_sat:
                    unsat_clauses.discard(affected_ci)
                    for v, _ in clause_data[affected_ci]:
                        var_unsat_count[v] -= 1
                    delta_uc -= 1
                elif not was_unsat and not is_now_sat:
                    unsat_clauses.add(affected_ci)
                    for v, _ in clause_data[affected_ci]:
                        var_unsat_count[v] += 1
                    delta_uc += 1

            new_uc = current_uc + delta_uc

            # Accept or reject based on simulated annealing
            if delta_uc <= 0:
                # Improvement or neutral: always accept
                current_uc = new_uc
            else:
                # Worse: accept with probability exp(-delta_uc / T)
                if T > T_min:
                    accept_prob = math.exp(-delta_uc / T)
                    if rng.random() < accept_prob:
                        current_uc = new_uc
                    else:
                        # Revert the flip
                        assignment[var_idx] = old_val
                        # Revert incremental updates
                        for affected_ci in var_to_clauses[var_idx]:
                            was_unsat_now = affected_ci in unsat_clauses
                            # After revert, check state
                            is_sat_after_revert = is_clause_satisfied(clause_data[affected_ci], assignment)
                            if was_unsat_now and is_sat_after_revert:
                                unsat_clauses.discard(affected_ci)
                                for v, _ in clause_data[affected_ci]:
                                    var_unsat_count[v] -= 1
                            elif not was_unsat_now and not is_sat_after_revert:
                                unsat_clauses.add(affected_ci)
                                for v, _ in clause_data[affected_ci]:
                                    var_unsat_count[v] += 1

            # Track best solution
            if current_uc < best_uc_in_run:
                best_uc_in_run = current_uc
                best_assignment_snapshot = assignment[:]

            # Cool down
            T = max(T_min, T * cooling_rate)

        return best_assignment_snapshot, (best_uc_in_run == 0)

    # Phase saving: remember which value of each variable led to more progress
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters
    max_steps_per_restart = 50000
    num_restarts = 3000  # max restarts, will stop at deadline

    # Phase history: for each variable, track which value was "good"
    phase_good = [None] * n  # None means undecided

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Initial assignment: use phase history if available, else random
        assignment = []
        for i in range(n):
            if phase_good[i] is not None:
                assignment.append(phase_good[i])
            else:
                assignment.append(rng.random() < 0.5)

        # Try to solve
        assignment, solved = sa_one(assignment, rng, max_steps_per_restart)

        if solved:
            return assignment

        # Update phase history
        current_uc = sum(1 for ci, c in enumerate(clause_data) if not is_clause_satisfied(c, assignment))
        if current_uc < best_unsat_count:
            best_unsat_count = current_uc
            best_assignment = assignment[:]
            for i in range(n):
                if best_assignment and assignment[i] != best_assignment[i]:
                    phase_good[i] = assignment[i]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
