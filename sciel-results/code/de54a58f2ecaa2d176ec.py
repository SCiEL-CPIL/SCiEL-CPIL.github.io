# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin before 330s limit

    # Convert to 0-based
    clauses_internal = []
    for c in clauses:
        clauses_internal.append(tuple((abs(l) - 1, l > 0) for l in c))

    m = len(clauses_internal)

    # Build index: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses_internal):
        for var, _ in c:
            var_to_clauses[var].append(i)

    # Pure literal elimination preprocessing
    pure_assignment = [None] * n
    active_clauses = set(range(m))
    changed = True
    while changed:
        changed = False
        lit_counts = [[0, 0] for _ in range(n)]
        for ci in active_clauses:
            c = clauses_internal[ci]
            for var, is_pos in c:
                idx = 1 if is_pos else 0
                lit_counts[var][idx] += 1

        for var in range(n):
            if pure_assignment[var] is not None:
                continue
            neg, pos = lit_counts[var]
            if neg == 0 and pos > 0:
                pure_assignment[var] = True
                changed = True
                for ci in var_to_clauses[var]:
                    if ci in active_clauses:
                        c = clauses_internal[ci]
                        for v, p in c:
                            if pure_assignment[v] is not None and pure_assignment[v] == p:
                                active_clauses.discard(ci)
                                break
            elif pos == 0 and neg > 0:
                pure_assignment[var] = False
                changed = True
                for ci in var_to_clauses[var]:
                    if ci in active_clauses:
                        c = clauses_internal[ci]
                        for v, p in c:
                            if pure_assignment[v] is not None and pure_assignment[v] == p:
                                active_clauses.discard(ci)
                                break

    # Check if all clauses are satisfied by pure assignment
    all_satisfied = True
    for ci in range(m):
        c = clauses_internal[ci]
        sat = False
        for var, is_pos in c:
            if pure_assignment[var] is not None and pure_assignment[var] == is_pos:
                sat = True
                break
        if not sat:
            all_satisfied = False
            break

    if all_satisfied:
        assignment = []
        for var in range(n):
            if pure_assignment[var] is not None:
                assignment.append(pure_assignment[var])
            else:
                assignment.append(False)
        return assignment

    non_pure_vars = [var for var in range(n) if pure_assignment[var] is None]
    np_count = len(non_pure_vars)
    var_map = {var: i for i, var in enumerate(non_pure_vars)}

    reduced_clauses = []
    for ci in range(m):
        c = clauses_internal[ci]
        sat = False
        new_clause = []
        for var, is_pos in c:
            if pure_assignment[var] is not None:
                if pure_assignment[var] == is_pos:
                    sat = True
                    break
            else:
                new_clause.append((var_map[var], is_pos))
        if not sat and new_clause:
            reduced_clauses.append(tuple(new_clause))

    rm = len(reduced_clauses)

    if rm == 0:
        assignment = []
        for var in range(n):
            if pure_assignment[var] is not None:
                assignment.append(pure_assignment[var])
            else:
                assignment.append(False)
        return assignment

    # Build index for reduced problem
    rvar_to_clauses = [[] for _ in range(np_count)]
    for i, c in enumerate(reduced_clauses):
        for var, _ in c:
            rvar_to_clauses[var].append(i)

    # Precompute clause literal arrays for faster access
    clause_vars = [np.array([v for v, _ in c], dtype=np.int32) for c in reduced_clauses]
    clause_pos = [np.array([p for _, p in c], dtype=np.bool_) for c in reduced_clauses]

    rng = random.Random(42)

    def build_full_assignment(reduced_assign):
        full = [False] * n
        for var in range(n):
            if pure_assignment[var] is not None:
                full[var] = pure_assignment[var]
            else:
                full[var] = reduced_assign[var_map[var]]
        return full

    def is_clause_satisfied(ci, assign):
        """Check if clause ci is satisfied by assignment."""
        cvars = clause_vars[ci]
        cpos = clause_pos[ci]
        vals = np.array(assign[cvars], dtype=np.bool_)
        return bool(np.any(vals == cpos))

    def get_unsat_clauses(assign):
        """Get list of unsatisfied clause indices."""
        unsat = []
        for ci in range(rm):
            if not is_clause_satisfied(ci, assign):
                unsat.append(ci)
        return unsat

    def unit_propagate(assign, unsat_list):
        """Perform unit propagation: satisfy any clause with exactly one false literal."""
        changed = True
        while changed:
            changed = False
            for ci in unsat_list:
                c = reduced_clauses[ci]
                false_lits = [(var, is_pos) for var, is_pos in c if assign[var] != is_pos]
                if len(false_lits) == 1:
                    var, is_pos = false_lits[0]
                    if assign[var] != is_pos:
                        assign[var] = is_pos
                        changed = True
                        # Recompute unsat list
                        unsat_list = get_unsat_clauses(assign)
                        if not unsat_list:
                            return []
                        break
        return unsat_list

    best_full_assignment = None
    best_unsat_count = m + 1

    max_restarts = 2000
    base_steps = 3000
    step_growth = 1.15

    restart = 0
    while restart < max_restarts and time.time() < deadline:
        reduced_assign = [rng.random() < 0.5 for _ in range(np_count)]
        unsat_list = get_unsat_clauses(reduced_assign)

        if not unsat_list:
            return build_full_assignment(reduced_assign)

        unsat_count = len(unsat_list)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_full_assignment = build_full_assignment(reduced_assign)

        # Unit propagation at start
        unsat_list = unit_propagate(reduced_assign, unsat_list)
        if not unsat_list:
            return build_full_assignment(reduced_assign)
        unsat_count = len(unsat_list)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_full_assignment = build_full_assignment(reduced_assign)

        initial_noise = 0.6
        min_noise = 0.15
        noise_decay = 0.997

        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 80000)

        noise = initial_noise
        prev_unsat = unsat_count
        stagnant_count = 0
        stagnant_threshold = 500

        for step in range(steps):
            if time.time() >= deadline:
                break

            if not unsat_list:
                return build_full_assignment(reduced_assign)

            cur_unsat = len(unsat_list)
            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_full_assignment = build_full_assignment(reduced_assign)

            if cur_unsat < prev_unsat:
                noise = max(min_noise, noise * 0.92)
                stagnant_count = 0
            else:
                stagnant_count += 1

            prev_unsat = cur_unsat
            noise = max(min_noise, noise * noise_decay)

            # If stuck for too long, increase noise to escape local minimum
            if stagnant_count > stagnant_threshold:
                noise = min(0.8, noise + 0.1)
                stagnant_count = 0

            # Select variable to flip
            if rng.random() < noise:
                ci = rng.choice(unsat_list)
                c = reduced_clauses[ci]
                false_vars = [var for var, is_pos in c if reduced_assign[var] != is_pos]
                if false_vars:
                    var_to_flip = rng.choice(false_vars)
                else:
                    var_to_flip = rng.randint(0, np_count - 1)
            else:
                # Weighted selection based on variable occurrence in unsat clauses
                var_weights = [0] * np_count
                for ci in unsat_list:
                    c = reduced_clauses[ci]
                    for var, is_pos in c:
                        if reduced_assign[var] != is_pos:
                            var_weights[var] += 1

                # Pick a clause weighted by sum of weights of its false literals
                clause_weights = []
                for ci in unsat_list:
                    c = reduced_clauses[ci]
                    w = 0
                    for var, is_pos in c:
                        if reduced_assign[var] != is_pos:
                            w += var_weights[var]
                    clause_weights.append(w)

                total_w = sum(clause_weights)
                if total_w == 0:
                    ci = rng.choice(unsat_list)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    ci = unsat_list[0]
                    for i, w in enumerate(clause_weights):
                        cum += w
                        if r <= cum:
                            ci = unsat_list[i]
                            break

                c = reduced_clauses[ci]
                false_vars = [(var, var_weights[var]) for var, is_pos in c if reduced_assign[var] != is_pos]
                if not false_vars:
                    var_to_flip = rng.randint(0, np_count - 1)
                else:
                    total_fw = sum(w for _, w in false_vars)
                    r = rng.random() * total_fw
                    cum = 0
                    var_to_flip = false_vars[0][0]
                    for var, w in false_vars:
                        cum += w
                        if r <= cum:
                            var_to_flip = var
                            break

            # Flip the variable
            old_val = reduced_assign[var_to_flip]
            reduced_assign[var_to_flip] = not reduced_assign[var_to_flip]

            # Incrementally update unsat list
            # Clauses that were satisfied may become unsatisfied, and vice versa
            affected_clauses = set(rvar_to_clauses[var_to_flip])
            new_unsat = []
            for ci in unsat_list:
                if ci in affected_clauses:
                    # Recheck this clause
                    if is_clause_satisfied(ci, reduced_assign):
                        pass  # Now satisfied, remove from unsat
                    else:
                        new_unsat.append(ci)
                else:
                    new_unsat.append(ci)
            # Check if any previously satisfied clauses became unsatisfied
            for ci in affected_clauses:
                if ci not in set(new_unsat) and not is_clause_satisfied(ci, reduced_assign):
                    new_unsat.append(ci)
            unsat_list = new_unsat

            # Unit propagation after flip
            if unsat_list:
                unsat_list = unit_propagate(reduced_assign, unsat_list)
                if not unsat_list:
                    return build_full_assignment(reduced_assign)

        restart += 1

    if best_full_assignment is not None:
        return best_full_assignment

    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
