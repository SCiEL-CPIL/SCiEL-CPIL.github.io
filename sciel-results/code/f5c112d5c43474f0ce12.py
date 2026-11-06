"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # 300s limit, 15s safety margin

    # Precompute variable occurrences for efficient updates
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occurrences[var].append((ci, lit > 0))

    # Convert clauses to numpy arrays for vectorized operations
    # Each clause: array of (var, sign) where sign is +1 or -1
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    for ci, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[ci, j] = abs(lit) - 1
            clause_signs[ci, j] = 1 if lit > 0 else -1

    def count_unsat(assignment):
        """Count unsatisfied clauses using numpy vectorization."""
        # assignment is a numpy array of int8: +1 or -1
        # For each clause, check if any literal is satisfied
        # sat[ci, j] = 1 if assignment[clause_vars[ci,j]] == clause_signs[ci,j]
        # A clause is satisfied if sum(sat[ci,:]) >= 1
        lit_vals = assignment[clause_vars]  # shape (m, 3)
        satisfied = (lit_vals == clause_signs)  # shape (m, 3), bool
        clause_sat = satisfied.sum(axis=1)  # shape (m,)
        return int((clause_sat == 0).sum())

    def check_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        lit_vals = assignment[clause_vars]
        satisfied = (lit_vals == clause_signs)
        return bool((satisfied.sum(axis=1) == 0).sum() == 0)

    def dpll_with_unit_prop(seed, time_budget=15):
        """DPLL with unit propagation."""
        rng = random.Random(seed)
        start_time = time.time()

        def propagate(clauses_list, assignment):
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses_list:
                    sat = False
                    unit_lit = None
                    open_lits = []
                    for lit in clause:
                        var = abs(lit) - 1
                        if var in assignment:
                            val = assignment[var]
                            if (lit > 0 and val) or (lit < 0 and not val):
                                sat = True
                                break
                        else:
                            open_lits.append(lit)

                    if sat:
                        continue
                    elif len(open_lits) == 0:
                        return assignment, clauses_list, True
                    elif len(open_lits) == 1:
                        unit_lit = open_lits[0]
                        var = abs(unit_lit) - 1
                        val = unit_lit > 0
                        if var in assignment:
                            if assignment[var] != val:
                                return assignment, clauses_list, True
                        else:
                            assignment[var] = val
                            changed = True
                        continue
                    else:
                        new_clauses.append(open_lits)
            return assignment, new_clauses, False

        best_result = None
        best_unsat = float('inf')

        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break

            assignment = {}
            num_initial = min(n // 5, 25)
            for var in rng.sample(range(n), num_initial):
                assignment[var] = rng.random() < 0.5

            clauses_copy = [list(c) for c in clauses]
            assignment, clauses_copy, contradiction = propagate(clauses_copy, assignment)

            if contradiction:
                continue

            if not clauses_copy:
                result = [False] * n
                for var in range(n):
                    result[var] = assignment.get(var, rng.random() < 0.5)
                return result

            result = [assignment.get(var, rng.random() < 0.5) for var in range(n)]
            unsat_count = 0
            for clause in clauses_copy:
                if not any(result[abs(l)-1] == (l > 0) for l in clause):
                    unsat_count += 1

            if unsat_count == 0:
                return result

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_result = result[:]

        return best_result if best_result else [False] * n

    def walksat_vectorized(seed, max_steps=200000):
        """Vectorized WalkSAT with numpy for fast evaluation."""
        rng = random.Random(seed)
        # assignment as numpy int8 array: +1 for True, -1 for False
        assignment = np.array([1 if rng.random() < 0.5 else -1 for _ in range(n)], dtype=np.int8)

        # Compute initial sat_counts
        lit_vals = assignment[clause_vars]
        satisfied = (lit_vals == clause_signs)
        sat_count = satisfied.sum(axis=1).astype(np.int32)
        unsat_mask = (sat_count == 0)
        unsat_indices = np.where(unsat_mask)[0]

        if len(unsat_indices) == 0:
            return (assignment + 1) // 2  # Convert to bool list

        noise = 0.3
        last_check = time.time()

        for step in range(max_steps):
            if step % 5000 == 0:
                if time.time() > deadline - 5:
                    break
                if time.time() - last_check > 1.0:
                    last_check = time.time()

            if len(unsat_indices) == 0:
                break

            # Pick random unsatisfied clause
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            clause = clauses[ci]

            # Score each literal in the clause
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = 1 if lit > 0 else -1

                if assignment[var] == want_val:
                    # Already satisfied, flipping would make it unsat
                    # Score is very bad
                    candidates.append((-100, var, want_val))
                    continue

                # Compute gain and loss
                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val:  # +var
                        if want_val == 1:
                            # Would satisfy this literal
                            if sat_count[oc] == 0:
                                gain += 1
                        else:
                            # Would unsatisfy this literal
                            if sat_count[oc] == 1:
                                loss += 1
                    else:  # -var
                        if want_val == -1:
                            if sat_count[oc] == 0:
                                gain += 1
                        else:
                            if sat_count[oc] == 1:
                                loss += 1

                score = gain - loss
                candidates.append((score, var, want_val))

            # Choose variable to flip
            if rng.random() < noise:
                _, var, want_val = candidates[rng.randrange(len(candidates))]
            else:
                best_score = max(c[0] for c in candidates)
                best_cands = [c for c in candidates if c[0] == best_score]
                _, var, want_val = best_cands[rng.randrange(len(best_cands))]

            # Apply flip
            if assignment[var] == want_val:
                continue
            assignment[var] = want_val

            # Update sat_counts
            for (oc, lit_val) in var_occurrences[var]:
                expected_val = 1 if lit_val else -1
                if assignment[var] == expected_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_mask[oc] = False
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_mask[oc] = True

            # Update unsat_indices efficiently
            if step % 1000 == 0:
                unsat_indices = np.where(unsat_mask)[0]

        # Final check
        unsat_indices = np.where(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return (assignment + 1) // 2

        return (assignment + 1) // 2

    def walksat_from_start(seed, max_steps=200000):
        """WalkSAT with vectorized unsat detection, starting from random."""
        rng = random.Random(seed)
        assignment = np.array([1 if rng.random() < 0.5 else -1 for _ in range(n)], dtype=np.int8)

        lit_vals = assignment[clause_vars]
        satisfied = (lit_vals == clause_signs)
        sat_count = satisfied.sum(axis=1).astype(np.int32)
        unsat_mask = (sat_count == 0)
        unsat_indices = np.where(unsat_mask)[0]

        if len(unsat_indices) == 0:
            return (assignment + 1) // 2

        noise = 0.3
        last_check = time.time()

        for step in range(max_steps):
            if step % 5000 == 0:
                if time.time() > deadline - 5:
                    break
                if time.time() - last_check > 1.0:
                    last_check = time.time()

            if len(unsat_indices) == 0:
                break

            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            clause = clauses[ci]

            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = 1 if lit > 0 else -1

                if assignment[var] == want_val:
                    candidates.append((-100, var, want_val))
                    continue

                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val:
                        if want_val == 1:
                            if sat_count[oc] == 0:
                                gain += 1
                        else:
                            if sat_count[oc] == 1:
                                loss += 1
                    else:
                        if want_val == -1:
                            if sat_count[oc] == 0:
                                gain += 1
                        else:
                            if sat_count[oc] == 1:
                                loss += 1

                score = gain - loss
                candidates.append((score, var, want_val))

            if rng.random() < noise:
                _, var, want_val = candidates[rng.randrange(len(candidates))]
            else:
                best_score = max(c[0] for c in candidates)
                best_cands = [c for c in candidates if c[0] == best_score]
                _, var, want_val = best_cands[rng.randrange(len(best_cands))]

            if assignment[var] == want_val:
                continue
            assignment[var] = want_val

            for (oc, lit_val) in var_occurrences[var]:
                expected_val = 1 if lit_val else -1
                if assignment[var] == expected_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_mask[oc] = False
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_mask[oc] = True

            if step % 1000 == 0:
                unsat_indices = np.where(unsat_mask)[0]

        unsat_indices = np.where(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return (assignment + 1) // 2
        return (assignment + 1) // 2

    # Strategy:
    # 1. Try DPLL with unit propagation (good for easy instances)
    # 2. Try multiple WalkSAT restarts (good for planted instances near threshold)
    # 3. Use vectorized operations for speed

    best_assignment = None
    best_unsat = float('inf')

    # Phase 1: DPLL
    if time.time() < deadline - 120:
        for seed in range(5):
            if time.time() > deadline - 120:
                break
            result = dpll_with_unit_prop(seed, time_budget=20)
            result_np = np.array([1 if r else -1 for r in result], dtype=np.int8)
            unsat_count = count_unsat(result_np)
            if unsat_count == 0:
                return result
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = result

    # Phase 2: WalkSAT with multiple restarts
    if time.time() < deadline - 10:
        # Try WalkSAT from random starts
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            result = walksat_from_start(seed, max_steps=100000)
            result_np = np.array([1 if r else -1 for r in result], dtype=np.int8)
            unsat_count = count_unsat(result_np)
            if unsat_count == 0:
                return result
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = result

    # Phase 3: If we have a best assignment, try to polish it
    if best_assignment is not None and time.time() < deadline - 5:
        # Try WalkSAT starting from best_assignment
        best_np = np.array([1 if r else -1 for r in best_assignment], dtype=np.int8)
        rng = random.Random(12345)

        lit_vals = best_np[clause_vars]
        satisfied = (lit_vals == clause_signs)
        sat_count = satisfied.sum(axis=1).astype(np.int32)
        unsat_mask = (sat_count == 0)
        unsat_indices = np.where(unsat_mask)[0]

        if len(unsat_indices) == 0:
            return best_assignment

        assignment = best_np.copy()
        noise = 0.25
        max_steps = 300000

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if len(unsat_indices) == 0:
                break

            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            clause = clauses[ci]

            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = 1 if lit > 0 else -1

                if assignment[var] == want_val:
                    candidates.append((-100, var, want_val))
                    continue

                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val:
                        if want_val == 1:
                            if sat_count[oc] == 0:
                                gain += 1
                        else:
                            if sat_count[oc] == 1:
                                loss += 1
                    else:
                        if want_val == -1:
                            if sat_count[oc] == 0:
                                gain += 1
                        else:
                            if sat_count[oc] == 1:
                                loss += 1

                score = gain - loss
                candidates.append((score, var, want_val))

            if rng.random() < noise:
                _, var, want_val = candidates[rng.randrange(len(candidates))]
            else:
                best_score = max(c[0] for c in candidates)
                best_cands = [c for c in candidates if c[0] == best_score]
                _, var, want_val = best_cands[rng.randrange(len(best_cands))]

            if assignment[var] == want_val:
                continue
            assignment[var] = want_val

            for (oc, lit_val) in var_occurrences[var]:
                expected_val = 1 if lit_val else -1
                if assignment[var] == expected_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_mask[oc] = False
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_mask[oc] = True

            if step % 1000 == 0:
                unsat_indices = np.where(unsat_mask)[0]

        unsat_indices = np.where(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return (assignment + 1) // 2

        # Check if this polish improved things
        final_result = (assignment + 1) // 2
        final_np = np.array([1 if r else -1 for r in final_result], dtype=np.int8)
        final_unsat = count_unsat(final_np)
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = final_result

    if best_assignment is None:
        return [False] * n

    return best_assignment
# EVOLVE-BLOCK-END
