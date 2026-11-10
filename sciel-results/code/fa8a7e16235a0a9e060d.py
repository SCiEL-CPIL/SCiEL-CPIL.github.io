"""Improved 3-SAT solver using optimized WalkSAT with literal-count tracking."""

import random
import time

# Module-level state for time management
_start_time = time.time()
# Assume 330s total limit, leave 15s buffer for safety/overhead
_DEADLINE = _start_time + 315.0


def solve(instance):
    """
    Solves a 3-SAT instance.
    Returns a list of booleans representing the assignment.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Pre-process: Map variable indices to clauses containing them
    var_to_clauses = [[] for _ in range(n)]
    # For each clause, store the position of each variable's literal
    # clause_var_pos[c_idx][var_idx] = position in clause (0,1,2) or -1 if not present
    clause_var_pos = []
    # clause_literals[c_idx] = list of (var_idx, is_pos)
    clause_literals = []

    for idx, clause in enumerate(clauses):
        var_pos = {}
        lits = []
        for pos, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            var_pos[var_idx] = pos
            lits.append((var_idx, is_pos))
        clause_var_pos.append(var_pos)
        clause_literals.append(lits)
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(idx)

    def compute_state(assignment):
        """Compute clause_satisfied, literal_true_counts, and unsat_indices for a given assignment."""
        clause_satisfied = [False] * m
        lit_true_counts = [0] * m  # number of true literals per clause
        unsat_indices = []
        for i, lits in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in lits:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            lit_true_counts[i] = cnt
            if cnt > 0:
                clause_satisfied[i] = True
            else:
                unsat_indices.append(i)
        return clause_satisfied, lit_true_counts, len(unsat_indices), unsat_indices

    # Pre-compute for each variable, the list of (clause_idx, position_in_clause)
    var_clause_pos = [[] for _ in range(n)]
    for c_idx in range(m):
        for var_idx, pos in clause_var_pos[c_idx].items():
            var_clause_pos[var_idx].append((c_idx, pos))

    # Initialize random assignment
    assignment = [random.random() < 0.5 for _ in range(n)]
    clause_satisfied, lit_true_counts, unsat_count, unsat_indices = compute_state(assignment)

    # Best solution tracking
    best_assignment = assignment.copy()
    best_unsat_count = unsat_count

    if unsat_count == 0:
        return assignment

    # WalkSAT with restarts and adaptive parameters
    max_restarts = 500
    base_steps = 5000
    flip_threshold = 0.30  # 30% random, 70% best

    for restart in range(max_restarts):
        # Check time
        if time.time() > _DEADLINE - 3:
            return best_assignment

        if restart == 0:
            # First iteration already initialized above
            pass
        elif restart % 10 == 0:
            # Every 10th restart: full random restart
            assignment = [random.random() < 0.5 for _ in range(n)]
            clause_satisfied, lit_true_counts, unsat_count, unsat_indices = compute_state(assignment)
        else:
            # Partial restart: perturb best assignment
            assignment = best_assignment.copy()
            # Flip a few random variables
            num_flips = max(1, n // 10)
            for _ in range(num_flips):
                v = random.randrange(n)
                assignment[v] = not assignment[v]
            clause_satisfied, lit_true_counts, unsat_count, unsat_indices = compute_state(assignment)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # Adaptive steps
        steps = base_steps

        # Track consecutive steps without improvement for adaptive randomness
        stuck_count = 0
        local_best = unsat_count

        for step in range(steps):
            # Check time every 50 steps
            if step % 50 == 0 and time.time() > _DEADLINE - 3:
                return best_assignment

            if unsat_count == 0:
                return assignment

            # Pick random unsatisfied clause from maintained list
            clause_idx = random.choice(unsat_indices)
            clause = clause_literals[clause_idx]

            # Adaptive flip threshold: increase randomness when stuck
            if stuck_count > 500:
                eff_threshold = min(0.6, flip_threshold + 0.01 * (stuck_count - 500) // 100)
            else:
                eff_threshold = flip_threshold

            # Decide: random flip or best flip
            if random.random() < eff_threshold:
                # Random flip: pick random variable from clause
                var_idx, is_pos = random.choice(clause)
            else:
                # Best flip: pick variable that minimizes increase in unsat count
                best_var = clause[0][0]
                best_delta = float('inf')

                for var_idx, is_pos in clause:
                    # Calculate delta if we flip var_idx
                    delta = 0
                    for c_idx, pos in var_clause_pos[var_idx]:
                        old_cnt = lit_true_counts[c_idx]

                        # Was the literal for var_idx in this clause true?
                        v, p = clause_literals[c_idx][pos]
                        lit_was_true = (assignment[v] == p)

                        if lit_was_true:
                            new_cnt = old_cnt - 1
                        else:
                            new_cnt = old_cnt + 1

                        if old_cnt > 0 and new_cnt == 0:
                            delta += 1
                        elif old_cnt == 0 and new_cnt > 0:
                            delta -= 1

                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                    elif delta == best_delta and random.random() < 0.5:
                        best_var = var_idx

                var_idx = best_var

            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]

            # Update clause satisfaction and maintain unsat_indices
            for c_idx, pos in var_clause_pos[var_idx]:
                v, p = clause_literals[c_idx][pos]
                if assignment[v] == p:
                    # Literal is now true
                    lit_true_counts[c_idx] += 1
                    if not clause_satisfied[c_idx]:
                        clause_satisfied[c_idx] = True
                        unsat_count -= 1
                        # Remove from unsat_indices (mark as -1 for lazy deletion)
                        # We'll rebuild periodically
                else:
                    # Literal is now false
                    lit_true_counts[c_idx] -= 1
                    if lit_true_counts[c_idx] == 0:
                        clause_satisfied[c_idx] = False
                        unsat_count += 1

            # Track improvement
            if unsat_count < local_best:
                local_best = unsat_count
                stuck_count = 0
            else:
                stuck_count += 1

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

            # Periodically rebuild unsat_indices to avoid stale entries
            if step % 200 == 199 or len(unsat_indices) > unsat_count * 2:
                unsat_indices = [i for i in range(m) if not clause_satisfied[i]]

    return best_assignment
