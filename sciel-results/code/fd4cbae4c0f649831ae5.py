import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 315.0  # 15s margin before 330s kill

    m = len(clauses)
    if m == 0:
        return [False] * n

    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_list = []
        signs_list = []
        for lit in clause:
            var_idx = abs(lit) - 1
            vars_list.append(var_idx)
            signs_list.append(lit > 0)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)

    # Build var_to_clauses: for each variable, list of (clause_idx, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            var_to_clauses[v].append((i, clause_signs[i][j]))

    # Precompute pure literal info
    pos_count = [0] * n
    neg_count = [0] * n
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            if clause_signs[i][j]:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    def init_state(assignment):
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            count = 0
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == clause_signs[i][j]:
                    count += 1
            clause_sat_count[i] = count
        return clause_sat_count

    def is_satisfied(clause_sat_count):
        return bool(np.all(clause_sat_count > 0))

    def flip_var(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count."""
        old_val = assignment[v]
        assignment[v] = not old_val
        for ci, sign in var_to_clauses[v]:
            if assignment[v] == sign:
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1

    def pure_literal_eliminate(assignment, clause_sat_count):
        """Find pure literals and assign them."""
        changed = False
        for v in range(n):
            if pos_count[v] > 0 and neg_count[v] == 0:
                if not assignment[v]:
                    flip_var(v, assignment, clause_sat_count)
                    changed = True
            elif neg_count[v] > 0 and pos_count[v] == 0:
                if assignment[v]:
                    flip_var(v, assignment, clause_sat_count)
                    changed = True
        return changed

    def vectorized_delta(assignment, clause_sat_count, unsat_indices):
        """
        For each variable in the union of variables in unsatisfied clauses,
        compute the net change in unsatisfied clause count if we flip that variable.
        Returns: dict var -> delta (negative is good)
        """
        # Get all variables that appear in unsatisfied clauses
        candidate_vars = set()
        for ci in unsat_indices:
            for v in clause_vars[ci]:
                candidate_vars.add(v)
        
        if not candidate_vars:
            return {}
        
        candidate_list = list(candidate_vars)
        n_cand = len(candidate_list)
        
        # For each candidate variable, compute delta
        # delta = (new_unsat_count - old_unsat_count)
        # When we flip var v:
        #   For each clause c containing v:
        #     old_sat = clause_sat_count[c]
        #     new_sat = old_sat + 1 if assignment[v] != sign else old_sat - 1
        #     old_unsat_contribution = 1 if old_sat == 0 else 0
        #     new_unsat_contribution = 1 if new_sat == 0 else 0
        #     delta_c = new_unsat_contribution - old_unsat_contribution
        #   delta = sum of delta_c over all clauses containing v
        
        deltas = {}
        for v in candidate_list:
            delta = 0
            for ci, sign in var_to_clauses[v]:
                old_sat = clause_sat_count[ci]
                if assignment[v] == sign:
                    new_sat = old_sat - 1
                else:
                    new_sat = old_sat + 1
                old_unsat = 1 if old_sat == 0 else 0
                new_unsat = 1 if new_sat == 0 else 0
                delta += new_unsat - old_unsat
            deltas[v] = delta
        
        return deltas

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    # WalkSAT with adaptive weight and vectorized scoring
    base_weight = 0.5
    max_flips = 30000

    # Track time for periodic checks
    while time.time() - start_time < timeout:
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = init_state(assignment)

        if is_satisfied(clause_sat_count):
            return assignment

        # Try pure literal elimination
        pure_literal_eliminate(assignment, clause_sat_count)

        if is_satisfied(clause_sat_count):
            return assignment

        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()

        # WalkSAT
        for flip_num in range(max_flips):
            if time.time() - start_time > timeout - 10.0:
                break

            # Find unsatisfied clauses
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return assignment

            # Adaptive weight: start high, decay over flips
            weight = base_weight * (1.0 - 0.3 * (flip_num / max_flips))

            if rng.random() < weight:
                # Random walk: pick random unsat clause, pick random var
                ci = int(rng.choice(unsat_indices))
                vars_in_clause = clause_vars[ci]
                v = vars_in_clause[rng.randrange(len(vars_in_clause))]
            else:
                # Score-based: use vectorized delta to find best variable
                # Among all variables in unsatisfied clauses, pick the one with minimum delta
                deltas = vectorized_delta(assignment, clause_sat_count, unsat_indices)
                
                if not deltas:
                    # Fallback: random
                    ci = int(rng.choice(unsat_indices))
                    vars_in_clause = clause_vars[ci]
                    v = vars_in_clause[rng.randrange(len(vars_in_clause))]
                else:
                    # Find variable with minimum delta (most negative is best)
                    # Add small random tie-breaking
                    min_delta = min(deltas.values())
                    best_vars = [var for var, d in deltas.items() if d == min_delta]
                    v = rng.choice(best_vars)

            flip_var(v, assignment, clause_sat_count)

            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat == 0:
                return assignment

            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()

            # Periodically check time
            if flip_num % 100 == 0 and time.time() - start_time > timeout - 5.0:
                break

    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
