"""Planted 3-SAT solver: incremental WalkSAT + SA escape + clause-weight learning + polish."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]

    # 0-based clause literals: list of (var_idx, is_pos)
    clause_literals = []
    for c in clauses_raw:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_literals.append(cv)

    m = len(clause_literals)

    # variable -> list of (clause_idx, is_pos)
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_literals):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((i, is_pos))

    rng = random.Random(42)
    deadline = time.monotonic() + 290.0

    def init_clause_counts(assignment):
        counts = [0] * m
        for i, cv in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts

    def get_unsat_from_counts(counts):
        return [i for i in range(m) if counts[i] == 0]

    def belief_propagation_init(num_iterations=100):
        var_bias = [0.0] * n
        for it in range(num_iterations):
            new_var_bias = [0.0] * n
            for c_idx, cv in enumerate(clause_literals):
                for j, (var_idx, is_pos) in enumerate(cv):
                    prod_complement = 1.0
                    for k, (v2, p2) in enumerate(cv):
                        if k == j:
                            continue
                        bias = var_bias[v2]
                        if p2:
                            if bias > 0:
                                p_false = 1.0 / (1.0 + math.exp(bias))
                            else:
                                p_false = math.exp(bias) / (1.0 + math.exp(bias))
                        else:
                            if bias > 0:
                                p_true = 1.0 / (1.0 + math.exp(-bias))
                            else:
                                p_true = 1.0 / (1.0 + math.exp(bias))
                            p_false = p_true
                        if p_false > 1.0:
                            p_false = 1.0
                        elif p_false < 0.0:
                            p_false = 0.0
                        prod_complement *= p_false
                    p_sat_given_false = 1.0 - prod_complement
                    if p_sat_given_false < 1e-15:
                        p_sat_given_false = 1e-15
                    msg = -math.log(p_sat_given_false)
                    if is_pos:
                        new_var_bias[var_idx] += msg
                    else:
                        new_var_bias[var_idx] -= msg
            var_bias = new_var_bias
        return [bias > 0 for bias in var_bias]

    def weighted_pick(unsat_list, weights):
        total = 0.0
        for ci in unsat_list:
            total += weights[ci]
        if total <= 0.0:
            return rng.choice(unsat_list)
        r = rng.random() * total
        acc = 0.0
        for ci in unsat_list:
            acc += weights[ci]
            if r <= acc:
                return ci
        return unsat_list[-1]

    def optimized_walksat(assignment, max_steps, sa_temp, weights):
        counts = init_clause_counts(assignment)
        unsat_set = set(get_unsat_from_counts(counts))
        unsat_list = list(unsat_set)

        # weight learning: bump clauses that stay unsat, decay others
        for step in range(max_steps):
            if time.monotonic() > deadline - 2.0:
                return False, assignment, unsat_set, counts, unsat_list
            if not unsat_set:
                return True, assignment, unsat_set, counts, unsat_list

            # Occasionally refresh the unsat list from the set
            if step % 32 == 0:
                unsat_list = list(unsat_set)

            # Select unsatisfied clause weighted by clause weights
            c_idx = weighted_pick(unsat_list, weights)
            cv = clause_literals[c_idx]

            # Score each variable in the clause: gain (satisfies unsat) - loss (breaks sat),
            # weighted by clause weights, plus a small random tiebreak (smooth surrogate).
            best_var = None
            best_score = float('-inf')
            for j, (var_idx, is_pos) in enumerate(cv):
                gain = 0.0
                loss = 0.0
                for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                    if counts[c_aff] == 0:
                        # currently unsat: does flipping satisfy it?
                        if assignment[var_idx] == is_pos_aff:
                            # flipping makes this literal true -> satisfies
                            gain += weights[c_aff]
                        else:
                            # flipping makes it false; clause stays unsat unless another lit is true
                            # (for unsat clause, no other lit is true) -> no change
                            pass
                    else:
                        # currently sat: does flipping break it?
                        if assignment[var_idx] == is_pos_aff:
                            # this literal is true now; after flip it becomes false.
                            # clause breaks only if no other literal is satisfied.
                            still_sat = False
                            for k2, (v2, p2) in enumerate(clause_literals[c_aff]):
                                if v2 == var_idx:
                                    continue
                                if assignment[v2] == p2:
                                    still_sat = True
                                    break
                            if not still_sat:
                                loss += weights[c_aff]
                        else:
                            # literal false now; flip makes it true -> clause stays sat
                            pass

                score = gain - 1.5 * loss + rng.random() * 0.5
                if score > best_score:
                    best_score = score
                    best_var = var_idx

            if best_var is None:
                best_var = cv[0][0]

            var_idx = best_var
            old_unsat = len(unsat_set)

            # Flip variable and update counts incrementally
            assignment[var_idx] = not assignment[var_idx]
            for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                if assignment[var_idx] == is_pos_aff:
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)

            new_unsat = len(unsat_set)

            # SA: accept worse moves with decreasing probability
            if new_unsat > old_unsat:
                delta = new_unsat - old_unsat
                T = sa_temp * max(0.01, 1.0 - step / max_steps)
                if rng.random() > math.exp(-delta / max(T, 0.001)):
                    # Reject: flip back
                    assignment[var_idx] = not assignment[var_idx]
                    for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                        if assignment[var_idx] == is_pos_aff:
                            if counts[c_aff] == 0:
                                unsat_set.discard(c_aff)
                            counts[c_aff] += 1
                        else:
                            counts[c_aff] -= 1
                            if counts[c_aff] == 0:
                                unsat_set.add(c_aff)

            # Weight learning: bump the selected clause if still unsat; gentle decay
            if counts[c_idx] == 0:
                weights[c_idx] *= 1.08
            else:
                weights[c_idx] *= 0.97

        return False, assignment, unsat_set, counts, unsat_list

    def polish(assignment, max_iters=200):
        """Final high-precision greedy polish: flip the variable that reduces unsat most."""
        counts = init_clause_counts(assignment)
        unsat_set = set(get_unsat_from_counts(counts))
        for _ in range(max_iters):
            if time.monotonic() > deadline - 1.0:
                break
            if not unsat_set:
                return assignment, counts, unsat_set
            # Find single flip with best (satisfy_unsat - break_sat)
            best_var = None
            best_delta = 1  # require at least 1 improvement
            for var_idx in range(n):
                # compute net effect of flipping var_idx
                net = 0
                for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                    if assignment[var_idx] == is_pos_aff:
                        # currently satisfies; flip breaks
                        if counts[c_aff] == 1:
                            net -= 1  # becomes unsat
                    else:
                        # currently not satisfied by this lit; flip satisfies
                        if counts[c_aff] == 0:
                            net += 1  # becomes sat
                if net > best_delta:
                    best_delta = net
                    best_var = var_idx
            if best_var is None:
                break
            assignment[best_var] = not assignment[best_var]
            for c_aff, is_pos_aff in var_to_clauses[best_var]:
                if assignment[best_var] == is_pos_aff:
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)
        return assignment, counts, unsat_set

    best_assignment = [False] * n
    best_unsat_count = m + 1

    restart_budget = 2500
    base_steps = 4000

    for restart in range(restart_budget):
        if time.monotonic() > deadline - 5.0:
            break

        if restart < 5:
            assignment = belief_propagation_init(num_iterations=50 + restart * 10)
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]

        sa_temp = 0.5 + rng.random() * 1.5
        weights = [1.0] * m

        remaining = deadline - time.monotonic()
        if remaining < 10.0:
            break
        elif remaining < 30.0:
            steps = 2000
        elif remaining < 60.0:
            steps = 3500
        else:
            steps = base_steps

        solved, assignment, unsat_set, counts, _ = optimized_walksat(assignment, steps, sa_temp, weights)
        if solved:
            return assignment

        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

    # Final polish from best partial assignment
    if best_unsat_count > 0 and time.monotonic() < deadline - 1.0:
        a, counts, unsat_set = polish(best_assignment[:], max_iters=300)
        if not unsat_set:
            return a
        if len(unsat_set) < best_unsat_count:
            best_assignment = a
            best_unsat_count = len(unsat_set)

    return best_assignment
# EVOLVE-BLOCK-END
