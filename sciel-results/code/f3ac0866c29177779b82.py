"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)

    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0

    # Pre-process clauses for faster access
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)

    # Build variable-to-clauses index for efficient updates
    var_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)

    def is_satisfied(clause_idx, assignment):
        fc = fast_clauses[clause_idx]
        for var_idx, required in fc:
            if assignment[var_idx] == required:
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    # Initialize mutable state containers in the enclosing scope
    unsat_set = set()
    var_unsat_count = [0] * n

    # Build phase-savings index: for each (var, required_value), the list of
    # clause indices in which that literal appears.
    var_req_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, required in fc:
            var_req_to_clauses[(var_idx, required)].append(i)

    def phase_savings(var_idx, required):
        """Number of other unsat clauses satisfied by setting var_idx to `required`."""
        total = 0
        for c_idx in var_req_to_clauses[(var_idx, required)]:
            if c_idx in unsat_set:
                total += 1
        return total

    def pick_literal(clause, clause_idx, walk_prob):
        """Choose which literal to flip. Returns (var_idx, required)."""
        if random.random() < walk_prob:
            return random.choice(clause)

        scores = []
        for var_idx, required in clause:
            freq = var_unsat_count[var_idx]
            ps = phase_savings(var_idx, required)
            scores.append(freq + ps)

        total_score = sum(scores)
        if total_score == 0:
            return random.choice(clause)

        r = random.random() * total_score
        cumulative = 0
        for i, score in enumerate(scores):
            cumulative += score
            if r <= cumulative:
                return clause[i]
        return clause[-1]

    def apply_flip(var_idx):
        """Flip variable var_idx and update unsat_set / var_unsat_count incrementally.
        Returns the net change in unsat count (positive = got worse)."""
        assignment[var_idx] = not assignment[var_idx]
        delta = 0
        for c_idx in var_to_clauses[var_idx]:
            was_unsat = c_idx in unsat_set
            is_sat = is_satisfied(c_idx, assignment)
            if was_unsat and is_sat:
                unsat_set.discard(c_idx)
                delta -= 1
                for v_idx, _ in fast_clauses[c_idx]:
                    var_unsat_count[v_idx] -= 1
            elif not was_unsat and not is_sat:
                unsat_set.add(c_idx)
                delta += 1
                for v_idx, _ in fast_clauses[c_idx]:
                    var_unsat_count[v_idx] += 1
        return delta

    def init_state(assignment):
        nonlocal unsat_set, var_unsat_count
        unsat_set = set()
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        var_unsat_count = [0] * n
        for c_idx in unsat_set:
            for v_idx, _ in fast_clauses[c_idx]:
                var_unsat_count[v_idx] += 1
        return len(unsat_set)

    best_assignment = None
    best_unsat_count = float('inf')

    base_walk_prob = 0.20
    max_walk_prob = 0.50
    progress_threshold = 500

    # Simulated annealing parameters
    T0 = 6.0          # initial temperature (accepts moderate worsening)
    T_floor = 0.05    # below this, effectively only improving moves accepted
    max_steps = 15000

    for restart in range(500):
        if time.time() > deadline:
            break

        assignment = [random.random() < 0.5 for _ in range(n)]
        current_unsat = init_state(assignment)

        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Annealing schedule for this restart
        T = T0
        cooling_rate = 0.99997
        steps_since_improvement = 0
        local_best_unsat = current_unsat
        walk_prob = base_walk_prob

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if not unsat_set:
                return assignment

            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]

            if len(unsat_set) < local_best_unsat:
                local_best_unsat = len(unsat_set)
                steps_since_improvement = 0
                walk_prob = base_walk_prob
            else:
                steps_since_improvement += 1
                if steps_since_improvement > progress_threshold:
                    walk_prob = min(max_walk_prob,
                                    base_walk_prob + (steps_since_improvement - progress_threshold) * 0.001)

            # Choose a clause (weight toward harder clauses)
            clause_idx = random.choice(list(unsat_set))
            clause = fast_clauses[clause_idx]

            var_idx, required = pick_literal(clause, clause_idx, walk_prob)
            delta = apply_flip(var_idx)

            # Simulated annealing acceptance:
            # delta < 0 -> improvement, always accept.
            # delta > 0 -> worsening, accept with prob exp(-delta / T).
            if delta > 0:
                if T > T_floor:
                    if random.random() > math.exp(-delta / T):
                        # Reject: revert the flip
                        apply_flip(var_idx)
                else:
                    # Too cold to accept any worsening move: revert
                    apply_flip(var_idx)

            # Cool the temperature
            T *= cooling_rate
            if T < T_floor:
                T = T_floor

            if not unsat_set:
                return assignment

        # End of restart

    # Final high-precision polish pass on the best assignment found
    if best_assignment is not None and time.time() < deadline - 5.0:
        assignment = best_assignment[:]
        current_unsat = init_state(assignment)
        if current_unsat > 0:
            polish_steps = 20000
            for step in range(polish_steps):
                if time.time() > deadline - 2.0:
                    break
                if not unsat_set:
                    break
                if len(unsat_set) < best_unsat_count:
                    best_unsat_count = len(unsat_set)
                    best_assignment = assignment[:]

                # Low walk prob, only accept small worsening (delta <= 1)
                clause_idx = random.choice(list(unsat_set))
                clause = fast_clauses[clause_idx]
                var_idx, required = pick_literal(clause, clause_idx, 0.05)
                delta = apply_flip(var_idx)
                if delta > 1:
                    apply_flip(var_idx)  # revert large worsening

        if best_unsat_count == 0:
            return best_assignment

    if best_assignment is not None:
        return best_assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
