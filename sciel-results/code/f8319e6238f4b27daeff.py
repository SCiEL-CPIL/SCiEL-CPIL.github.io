# EVOLVE-BLOCK-START
import random
import time
import sys


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0  # safety margin before 330s kill

    # Pre-compute clause literal indices and signs
    clause_idx = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[l > 0 for l in c] for c in clauses]
    m = len(clauses)

    def dpll_solve(clauses_list, n_vars):
        """Complete DPLL with watched literals and unit propagation."""
        if time.time() > deadline:
            return None

        # Build watched literals structure
        # For each clause, watch two literals
        # Literal: (var, sign) where sign=True means positive
        def lit_key(var, sign):
            return (var, sign)

        # Build watches: for each literal, list of (clause_idx, other_watch_idx)
        watches = {}
        for ci in range(len(clauses_list)):
            c = clauses_list[ci]
            # Watch first two literals
            for wi in range(2):
                var = abs(c[wi]) - 1
                sign = c[wi] > 0
                key = (var, sign)
                if key not in watches:
                    watches[key] = []
                other_idx = 1 - wi
                watches[key].append((ci, other_idx))

        # Assignment: dict var -> bool
        assignment = {}

        def propagate():
            """Unit propagation with watched literals."""
            queue = []
            # Find unit clauses initially
            for ci in range(len(clauses_list)):
                c = clauses_list[ci]
                unassigned = [l for l in c if abs(l) - 1 not in assignment]
                if len(unassigned) == 0:
                    return False  # Conflict
                if len(unassigned) == 1:
                    queue.append((unassigned[0], ci))

            while queue:
                lit, ci = queue.pop(0)
                var = abs(lit) - 1
                sign = lit > 0

                if var in assignment:
                    if assignment[var] != sign:
                        return False  # Conflict
                    continue

                assignment[var] = sign

                # Check watched literals for this variable
                # For each clause watching (var, not sign), if other watch is unassigned, move watch
                other_sign = not sign
                key = (var, other_sign)
                if key in watches:
                    to_remove = []
                    for (c_idx, other_w) in watches[key][:]:
                        c = clauses_list[c_idx]
                        other_var = abs(c[other_w]) - 1
                        other_lit_sign = c[other_w] > 0

                        # Check if other watch is satisfied
                        if other_var in assignment:
                            if assignment[other_var] == other_lit_sign:
                                continue  # Clause satisfied, no action
                            # Other watch is falsified, check if clause is unit or empty
                            unassigned_in_clause = []
                            for l in c:
                                v = abs(l) - 1
                                if v not in assignment:
                                    unassigned_in_clause.append(l)
                            if len(unassigned_in_clause) == 0:
                                return False  # Conflict
                            elif len(unassigned_in_clause) == 1:
                                queue.append((unassigned_in_clause[0], c_idx))
                            # No watch needed since clause is satisfied or unit handled
                        else:
                            # Move watch to unassigned literal
                            to_remove.append((c_idx, other_w))
                            new_key = (other_var, other_lit_sign)
                            if new_key not in watches:
                                watches[new_key] = []
                            watches[new_key].append((c_idx, 0))  # Watch the unassigned literal at position 0

                    if to_remove:
                        watches[key] = [w for w in watches[key] if w not in to_remove]

            return True

        if not propagate():
            return None

        # Check if all variables assigned
        if len(assignment) == n_vars:
            return [assignment[i] for i in range(n_vars)]

        # Check if all clauses satisfied
        def all_satisfied():
            for ci in range(len(clauses_list)):
                c = clauses_list[ci]
                for l in c:
                    var = abs(l) - 1
                    if var in assignment and assignment[var] == (l > 0):
                        break
                else:
                    return False
            return True

        if all_satisfied():
            # Fill in remaining variables arbitrarily
            for i in range(n_vars):
                if i not in assignment:
                    assignment[i] = False
            return [assignment[i] for i in range(n_vars)]

        # Pick a branch variable (heuristic: most frequent in unassigned clauses)
        var_freq = [0] * n_vars
        for ci in range(len(clauses_list)):
            c = clauses_list[ci]
            for l in c:
                var = abs(l) - 1
                if var not in assignment:
                    var_freq[var] += 1

        # Pick variable with highest frequency among unassigned
        best_var = -1
        best_freq = -1
        for i in range(n_vars):
            if i not in assignment and var_freq[i] > best_freq:
                best_freq = var_freq[i]
                best_var = i

        if best_var == -1:
            return None

        # Try True first
        assignment[best_var] = True
        # Save state
        saved_watches = {k: v[:] for k, v in watches.items()}
        saved_assignment = dict(assignment)

        if propagate():
            # Check if solved
            if len(assignment) == n_vars:
                return [assignment[i] for i in range(n_vars)]
            if all_satisfied():
                for i in range(n_vars):
                    if i not in assignment:
                        assignment[i] = False
                return [assignment[i] for i in range(n_vars)]
            # Recurse
            result = dpll_solve(clauses_list, n_vars)
            if result is not None:
                return result

        # Restore state
        assignment.clear()
        assignment.update(saved_assignment)
        watches.clear()
        watches.update(saved_watches)
        del assignment[best_var]

        # Try False
        assignment[best_var] = False
        saved_watches2 = {k: v[:] for k, v in watches.items()}
        saved_assignment2 = dict(assignment)

        if propagate():
            if len(assignment) == n_vars:
                return [assignment[i] for i in range(n_vars)]
            if all_satisfied():
                for i in range(n_vars):
                    if i not in assignment:
                        assignment[i] = False
                return [assignment[i] for i in range(n_vars)]
            result = dpll_solve(clauses_list, n_vars)
            if result is not None:
                return result

        return None

    # Try DPLL first with a time limit
    dpll_result = dpll_solve(clauses, n)
    if dpll_result is not None:
        # Verify
        for c in clauses:
            sat = False
            for l in c:
                var = abs(l) - 1
                if dpll_result[var] == (l > 0):
                    sat = True
                    break
            if not sat:
                break
        else:
            return dpll_result

    # Fallback to optimized WalkSAT
    best_assignment = None
    best_unsat_count = float('inf')

    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            cidx = clause_idx[i]
            csigns = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cidx[j]] == csigns[j]:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def is_satisfied(assignment):
        for i in range(m):
            cidx = clause_idx[i]
            csigns = clause_signs[i]
            sat = False
            for j in range(3):
                if assignment[cidx[j]] == csigns[j]:
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Build variable-to-clause index for faster gain/cost calculation
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_idx[i][j]].append(i)

    for restart in range(1000):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        if is_satisfied(assignment):
            return assignment

        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        steps = 0
        max_steps = 20000
        while steps < max_steps:
            if time.time() > deadline:
                break

            # Find unsatisfied clauses
            unsat_clauses = []
            for i in range(m):
                cidx = clause_idx[i]
                csigns = clause_signs[i]
                sat = False
                for j in range(3):
                    if assignment[cidx[j]] == csigns[j]:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)

            if not unsat_clauses:
                return assignment

            # Weighted clause selection
            clause_idx_chosen = rng.choice(unsat_clauses)
            cidx = clause_idx[clause_idx_chosen]
            csigns = clause_signs[clause_idx_chosen]

            # For each literal in the chosen clause, compute gain and cost
            best_var = -1
            best_score = -float('inf')

            for j in range(3):
                var = cidx[j]
                new_val = not assignment[var]

                # Gain: number of unsat clauses satisfied by setting var = new_val
                gain = 0
                for ci in unsat_clauses:
                    ci_idx = clause_idx[ci]
                    ci_signs = clause_signs[ci]
                    for k in range(3):
                        if ci_idx[k] == var:
                            if ci_signs[k] == new_val:
                                gain += 1
                            break

                # Cost: number of currently satisfied clauses that become unsatisfied
                cost = 0
                for ci in var_to_clauses[var]:
                    if ci in unsat_clauses:
                        continue
                    ci_idx = clause_idx[ci]
                    ci_signs = clause_signs[ci]
                    # Check if clause is currently satisfied
                    sat = False
                    for k in range(3):
                        if ci_idx[k] == var:
                            if ci_signs[k] == new_val:
                                # This literal would become false, check if clause still satisfied
                                for k2 in range(3):
                                    if k2 != k and assignment[ci_idx[k2]] == ci_signs[k2]:
                                        sat = True
                                        break
                                break
                            else:
                                sat = True
                                break
                        elif assignment[ci_idx[k]] == ci_signs[k]:
                            sat = True
                            break
                    if not sat:
                        cost += 1

                # Score: gain - cost + random noise
                noise = rng.random() * 0.1
                score = gain - cost + noise

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var == -1:
                best_var = cidx[rng.randint(0, 2)]

            assignment[best_var] = not assignment[best_var]
            steps += 1

        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment

    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
