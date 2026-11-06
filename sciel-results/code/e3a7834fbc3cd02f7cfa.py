# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit

    n, clauses = instance["n"], instance["clauses"]

    # Convert to 0-based literals: var index 0..n-1, sign: +1 or -1
    # Store as (var_idx, sign) where sign=1 means positive literal, -1 means negative
    clause_list = []
    for c in clauses:
        parsed = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            parsed.append((var_idx, sign))
        clause_list.append(parsed)

    m = len(clause_list)

    # Build occurrence maps for efficient preprocessing
    # pos_occ[var] = list of clause indices where var appears positively
    # neg_occ[var] = list of clause indices where var appears negatively
    pos_occ = [[] for _ in range(n)]
    neg_occ = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_list):
        for (v, s) in clause:
            if s == 1:
                pos_occ[v].append(ci)
            else:
                neg_occ[v].append(ci)

    # Assignment array: None = unassigned, True/False = assigned
    assignment = [None] * n
    # Clause status: 0 = active, 1 = satisfied, 2 = unit, 3 = empty (conflict)
    clause_status = np.zeros(m, dtype=np.int8)
    # For unit clauses, track which literal remains
    unit_literal = [None] * m

    # Helper: propagate a variable assignment
    def propagate(var_idx, value):
        """Assign var_idx to value, propagate through unit clauses. Returns False on conflict."""
        assignment[var_idx] = value
        sign = 1 if value else -1
        # The literal (var_idx, sign) is now true
        # Clauses containing this literal become satisfied
        occ = pos_occ[var_idx] if sign == 1 else neg_occ[var_idx]
        for ci in occ:
            if clause_status[ci] == 0:  # only active clauses
                clause_status[ci] = 1  # satisfied
        # The opposite literal is now false
        opp_sign = -sign
        opp_occ = pos_occ[var_idx] if opp_sign == 1 else neg_occ[var_idx]
        for ci in opp_occ:
            if clause_status[ci] == 0:  # active clause, remove false literal
                # Check if clause becomes unit or empty
                remaining = 0
                rem_lit = None
                for (v, s) in clause_list[ci]:
                    if v == var_idx and s == opp_sign:
                        continue  # this literal is false
                    if assignment[v] is not None:
                        lit_val = assignment[v] == (s == 1)
                        if not lit_val:
                            continue  # this literal is also false
                        else:
                            # This literal is true, clause satisfied
                            clause_status[ci] = 1
                            break
                    else:
                        remaining += 1
                        rem_lit = (v, s)
                else:
                    if remaining == 0:
                        clause_status[ci] = 3  # conflict
                        return False
                    elif remaining == 1:
                        clause_status[ci] = 2
                        unit_literal[ci] = rem_lit
                        # Recursively propagate this unit
                        uv, us = rem_lit
                        return propagate(uv, us == 1)
                    # else: still active with >1 unassigned literals
        return True

    # Initial unit propagation
    # Find all unit clauses
    for ci in range(m):
        if len(clause_list[ci]) == 1:
            clause_status[ci] = 2
            unit_literal[ci] = clause_list[ci][0]

    # Process units
    changed = True
    while changed:
        changed = False
        for ci in range(m):
            if clause_status[ci] == 2:
                v, s = unit_literal[ci]
                if assignment[v] is None:
                    if not propagate(v, s == 1):
                        # Conflict - shouldn't happen for satisfiable planted instances
                        # Fall back to random assignment
                        for i in range(n):
                            if assignment[i] is None:
                                assignment[i] = random.random() < 0.5
                        return assignment
                    changed = True
            elif clause_status[ci] == 3:
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = random.random() < 0.5
                return assignment

    # Pure literal elimination
    for v in range(n):
        if assignment[v] is not None:
            continue
        # Count if var appears only positively or only negatively in active clauses
        pos_count = 0
        neg_count = 0
        for ci in pos_occ[v]:
            if clause_status[ci] == 0:
                pos_count += 1
        for ci in neg_occ[v]:
            if clause_status[ci] == 0:
                neg_count += 1
        if pos_count > 0 and neg_count == 0:
            # Pure positive
            if not propagate(v, True):
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = random.random() < 0.5
                return assignment
        elif neg_count > 0 and pos_count == 0:
            # Pure negative
            if not propagate(v, False):
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = random.random() < 0.5
                return assignment

    # Assign remaining unassigned variables randomly
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = random.random() < 0.5

    # Check if already satisfied
    def is_satisfied():
        for ci in range(m):
            sat = False
            for (v, s) in clause_list[ci]:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    if is_satisfied():
        return assignment

    # WalkSAT on the reduced problem
    # Build list of active (unsatisfied) clauses for search
    # We'll work with the full clause list but focus on unsatisfied ones

    # Precompute for each clause: list of (var_idx, sign)
    # For speed, use numpy where possible

    # Convert assignment to numpy bool array
    assign_np = np.array(assignment, dtype=bool)

    # For each clause, precompute var indices and signs as arrays
    clause_vars = np.array([[(v, s) for (v, s) in c] for c in clause_list], dtype=object)

    def count_unsat(assign_arr):
        """Count unsatisfied clauses."""
        count = 0
        for ci in range(m):
            sat = False
            for (v, s) in clause_list[ci]:
                if assign_arr[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # Vectorized satisfaction check
    def get_unsat_clauses(assign_arr):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for ci in range(m):
            sat = False
            for (v, s) in clause_list[ci]:
                if assign_arr[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat.append(ci)
        return unsat

    best_assign = assign_np.copy()
    best_unsat = count_unsat(assign_np)

    rng = random.Random(42)

    # Dynamic WalkSAT with restarts
    max_restarts = 200
    steps_per_restart = 5000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random restart: assign unassigned vars randomly, keep assigned
        if restart > 0:
            # Perturbation: flip some random variables
            num_flips = rng.randint(1, max(1, n // 10))
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                assign_np[v] = not assign_np[v]

        current_unsat = count_unsat(assign_np)
        if current_unsat == 0:
            best_assign = assign_np.copy()
            break
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assign = assign_np.copy()

        # Noise parameter: starts high, decays
        base_noise = 0.5
        noise = base_noise

        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            unsat_clauses = get_unsat_clauses(assign_np)
            if not unsat_clauses:
                best_assign = assign_np.copy()
                best_unsat = 0
                break

            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assign = assign_np.copy()

            # Decay noise over time within restart
            noise = base_noise * (1.0 - step / steps_per_restart * 0.5)

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clause_list[ci]

            # Count damage for each literal in the clause
            # A literal (v, s) is satisfied if assign_np[v] == (s==1)
            # Flipping v changes its truth value
            best_var = -1
            best_damage = float('inf')
            candidates = []
            for (v, s) in clause:
                # Current state of literal
                lit_sat = assign_np[v] == (s == 1)
                # If we flip v, this literal changes
                # Damage = change in number of satisfied clauses
                # For each clause containing v, check if flipping v changes satisfaction
                damage = 0
                for cj in range(m):
                    clause_j = clause_list[cj]
                    # Check if v is in clause_j
                    v_in_j = False
                    for (vj, sj) in clause_j:
                        if vj == v:
                            v_in_j = True
                            break
                    if not v_in_j:
                        continue
                    # Check if clause_j is currently satisfied
                    cur_sat = False
                    for (vj, sj) in clause_j:
                        if assign_np[vj] == (sj == 1):
                            cur_sat = True
                            break
                    # After flip, check if clause_j is satisfied
                    new_sat = False
                    for (vj, sj) in clause_j:
                        if vj == v:
                            # v is flipped
                            if (not assign_np[v]) == (sj == 1):
                                new_sat = True
                                break
                        else:
                            if assign_np[vj] == (sj == 1):
                                new_sat = True
                                break
                    if cur_sat and not new_sat:
                        damage += 1
                    elif not cur_sat and new_sat:
                        damage -= 1
                candidates.append((damage, v))

            # Add noise
            noise_vals = []
            for (d, v) in candidates:
                noise_vals.append(d + rng.gauss(0, 1) * noise * 3)

            # Pick the one with minimum noisy damage
            min_idx = 0
            min_val = noise_vals[0]
            for i in range(1, len(noise_vals)):
                if noise_vals[i] < min_val:
                    min_val = noise_vals[i]
                    min_idx = i

            flip_var = candidates[min_idx][1]
            assign_np[flip_var] = not assign_np[flip_var]

            # Update current_unsat approximately
            # Recompute if needed (expensive but safe)
            if step % 50 == 0:
                current_unsat = count_unsat(assign_np)

        if best_unsat == 0:
            break

    # Final check
    final_unsat = count_unsat(best_assign)
    if final_unsat == 0:
        return best_assign.tolist()

    # Try a few more restarts with different seeds if time permits
    if time.time() < deadline - 10:
        for seed in range(10):
            if time.time() > deadline:
                break
            assign_np2 = np.array(assignment, dtype=bool)
            rng2 = random.Random(seed * 1000 + 7)
            for step in range(3000):
                if time.time() > deadline:
                    break
                unsat_clauses = get_unsat_clauses(assign_np2)
                if not unsat_clauses:
                    return assign_np2.tolist()
                ci = rng2.choice(unsat_clauses)
                clause = clause_list[ci]
                # Random flip from clause
                v, s = rng2.choice(clause)
                assign_np2[v] = not assign_np2[v]
            final_unsat2 = count_unsat(assign_np2)
            if final_unsat2 < best_unsat:
                best_unsat = final_unsat2
                best_assign = assign_np2.copy()
            if best_unsat == 0:
                return best_assign.tolist()

    return best_assign.tolist()
# EVOLVE-BLOCK-END
