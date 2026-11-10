"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit

    # Preprocess: convert to 0-based variable indices and signs
    clause_vars = []
    clause_sigs = []
    for c in clauses:
        cv = [abs(l) - 1 for l in c]
        cs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(cv)
        clause_sigs.append(cs)

    # Build var -> list of (clause_idx, pos_in_clause) for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append((i, j))

    # Precompute variable degrees for static heuristic
    var_degree = [0] * n
    for i in range(m):
        for j in range(3):
            var_degree[clause_vars[i][j]] += 1

    # Precompute clause weights (sum of degrees of its variables)
    clause_weights = [0] * m
    for i in range(m):
        w = 0
        for j in range(3):
            w += var_degree[clause_vars[i][j]]
        clause_weights[i] = w

    # Precompute numpy arrays for fast vectorized operations
    cv_np = np.array(clause_vars, dtype=np.int32)  # (m, 3)
    cs_np = np.array(clause_sigs, dtype=np.int8)   # (m, 3)
    # lit_val[i, j] = 1.0 if literal (i,j) is satisfied given assignment, else 0.0
    # We'll compute this on the fly using boolean assignment

    rng = random.Random(42)

    def make_random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def walk_sat_run(assignment, max_flips, p=0.55):
        """WalkSAT with optimized list-based unsat clause tracking and static clause weighting."""
        
        # Initialize clause satisfaction status
        clause_sat_count = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            cnt = 0
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    cnt += 1
            clause_sat_count[i] = cnt

        # Initialize unsat_list and unsat_pos for O(1) selection and updates
        unsat_list = []
        unsat_pos = [-1] * m  # -1 if not in unsat_list, else index in unsat_list
        
        for i in range(m):
            if clause_sat_count[i] == 0:
                unsat_pos[i] = len(unsat_list)
                unsat_list.append(i)

        for flip in range(max_flips):
            if time.time() - start_time > DEADLINE:
                return None, assignment

            if len(unsat_list) == 0:
                return assignment[:], assignment

            # Pick an unsatisfied clause
            # Strategy: With high probability, pick the clause with highest static weight.
            # With low probability, pick randomly to escape local minima.
            if rng.random() < 0.8:
                # Find max weight in unsat_list
                best_ci = -1
                best_w = -1
                for idx in unsat_list:
                    if clause_weights[idx] > best_w:
                        best_w = clause_weights[idx]
                        best_ci = idx
                ci = best_ci
            else:
                idx = rng.randint(0, len(unsat_list) - 1)
                ci = unsat_list[idx]

            # Choose literal to flip
            if rng.random() < p:
                # Random literal in the clause
                li = rng.randint(0, 2)
            else:
                # Weighted: prefer literals that satisfy more other unsat clauses
                best_li = 0
                best_weight = -1
                for li in range(3):
                    var = clause_vars[ci][li]
                    weight = 0
                    for (oi, oj) in var_to_clauses[var]:
                        if oi == ci:
                            continue
                        # Check if oi is currently unsat
                        if unsat_pos[oi] != -1:
                            weight += 1
                    if weight > best_weight:
                        best_weight = weight
                        best_li = li
                li = best_li

            var = clause_vars[ci][li]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val

            # Update clause_sat_count and unsat_list for all clauses containing this var
            for (oi, oj) in var_to_clauses[var]:
                osig = clause_sigs[oi][oj]
                was_sat = (old_val == (osig > 0))
                now_sat = (new_val == (osig > 0))
                
                if was_sat and not now_sat:
                    # Clause became unsat
                    clause_sat_count[oi] -= 1
                    if clause_sat_count[oi] == 0:
                        # Add to unsat_list
                        unsat_pos[oi] = len(unsat_list)
                        unsat_list.append(oi)
                elif not was_sat and now_sat:
                    # Clause became sat
                    clause_sat_count[oi] += 1
                    if clause_sat_count[oi] > 0:
                        # Remove from unsat_list if it was there
                        if unsat_pos[oi] != -1:
                            pos = unsat_pos[oi]
                            last_idx = len(unsat_list) - 1
                            last_clause = unsat_list[last_idx]
                            
                            # Swap with last
                            unsat_list[pos] = last_clause
                            unsat_pos[last_clause] = pos
                            
                            # Pop last
                            unsat_list.pop()
                            unsat_pos[oi] = -1

        # Return None if not solved, but return the final assignment for tracking
        if len(unsat_list) == 0:
            return assignment[:], assignment
        return None, assignment

    def count_unsat(assignment):
        """Count unsatisfied clauses using numpy for speed."""
        # Convert assignment to numpy bool array
        a_np = np.array(assignment, dtype=bool)
        # For each literal (i, j): satisfied if a_np[cv_np[i,j]] == (cs_np[i,j] > 0)
        # cs_np[i,j] > 0 is a bool array (m, 3)
        lit_sat = a_np[cv_np] == (cs_np > 0)  # shape (m, 3)
        # Clause is satisfied if any literal is satisfied
        clause_sat = lit_sat.any(axis=1)  # shape (m,)
        return int((~clause_sat).sum())

    def polish(assignment, unsat_clauses):
        """Try to fix a small number of unsat clauses by flipping variables.
        Uses multi-pass local search with restarts for better escape from local minima."""
        if not unsat_clauses:
            return assignment
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        
        # Collect unique variables from unsat clauses
        vars_to_try = set()
        for ci in unsat_clauses:
            for j in range(3):
                vars_to_try.add(clause_vars[ci][j])
        
        best_assign = assignment[:]
        best_count = current_unsat
        
        # Multi-pass local search
        for _pass in range(3):
            improved = False
            for var in vars_to_try:
                if time.time() - start_time > DEADLINE:
                    return best_assign
                # Flip var
                assignment[var] = not assignment[var]
                new_count = count_unsat(assignment)
                if new_count < best_count:
                    best_count = new_count
                    best_assign = assignment[:]
                    improved = True
                else:
                    # Unflip var
                    assignment[var] = not assignment[var]
            
            if not improved:
                break
            
            # If we made progress, refresh vars_to_try from current best
            if best_count < current_unsat:
                current_unsat = best_count
                assignment = best_assign[:]
                # Recompute unsat clauses
                unsat_clauses = []
                for i in range(m):
                    cv = clause_vars[i]
                    cs = clause_sigs[i]
                    sat = False
                    for j in range(3):
                        if best_assign[cv[j]] == (cs[j] > 0):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(i)
                vars_to_try = set()
                for ci in unsat_clauses:
                    for j in range(3):
                        vars_to_try.add(clause_vars[ci][j])
        
        return best_assign

    def continuous_energy_descent(assignment, max_iters=500, lr=0.1):
        """
        Continuous relaxation with soft-min surrogate for smooth gradients.
        Minimizes E_soft = sum_c log(1 + exp(beta * (1 - sum_j lit_val_cj))) / beta
        which approximates sum_c max(0, 1 - sum_j lit_val_cj) as beta -> inf.
        The gradient is smooth everywhere, enabling better optimization.
        Returns a boolean assignment derived from the continuous solution.
        """
        x = np.array(assignment, dtype=np.float64)
        
        beta = 5.0  # Start moderate, increase for sharper approximation
        lr_schedule = [0.15, 0.1, 0.05, 0.02, 0.01]  # Learning rate per phase
        phase_len = max_iters // len(lr_schedule)
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            # Adaptive beta: increase as we progress for sharper approximation
            progress = it / max_iters
            beta = 5.0 + 25.0 * progress  # 5 to 30
            
            # Learning rate schedule
            phase = it // max(1, phase_len)
            lr = lr_schedule[min(phase, len(lr_schedule) - 1)]
            
            # Compute literal values
            x_vars = x[cv_np]  # shape (m, 3)
            sig_mask = cs_np > 0  # shape (m, 3)
            lit_vals = np.where(sig_mask, x_vars, 1.0 - x_vars)  # shape (m, 3)
            
            # Clause satisfaction sum
            clause_sum = lit_vals.sum(axis=1)  # shape (m,)
            
            # Soft deficit using log-sum-exp for numerical stability
            # E_i = log(1 + exp(beta * (1 - clause_sum_i))) / beta
            # Gradient: dE_i/d(clause_sum_i) = -exp(beta*(1-clause_sum_i)) / (1 + exp(beta*(1-clause_sum_i)))
            # This is -1/(1 + exp(-beta*(1-clause_sum_i))) = -sigmoid(beta*(1-clause_sum_i))
            
            z = beta * (1.0 - clause_sum)  # shape (m,)
            # sigmoid(z) = 1 / (1 + exp(-z)), computed stably
            sigmoid_z = np.where(z >= 0, 
                                  1.0 / (1.0 + np.exp(-z)),
                                  np.exp(z) / (1.0 + np.exp(z)))
            
            # Gradient weight for each clause: -sigmoid_z
            # dE/dx_v = sum_{(i,j): v in clause i} (-sigmoid_z[i]) * sig[i,j]
            
            grad = np.zeros(n, dtype=np.float64)
            
            # Vectorized gradient accumulation
            # For each clause i, contribution to grad is -sigmoid_z[i] * sig[i,j] for each var
            weighted_sigs = -sigmoid_z[:, np.newaxis] * cs_np.astype(np.float64)  # (m, 3)
            
            # Scatter add: for each (i, j), add weighted_sigs[i,j] to grad[cv_np[i,j]]
            np.add.at(grad, cv_np.flatten(), weighted_sigs.flatten())
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            # Gradient descent step
            x -= lr * grad / grad_norm
            
            # Clip to [0, 1]
            x = np.clip(x, 0.0, 1.0)
            
            # Periodic threshold and check
            if it % 20 == 0 or it == max_iters - 1:
                bool_assign = x > 0.5
                if count_unsat(bool_assign) == 0:
                    return bool_assign.tolist()
        
        # Final threshold
        bool_assign = x > 0.5
        return bool_assign.tolist()

    best_assignment = None
    best_unsat = m + 1

    # Luby-like restart sequence
    multipliers = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    base_flips = 100
    max_restarts = 1000

    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        # Determine flip budget
        mult = multipliers[restart % len(multipliers)]
        current_base = base_flips + (restart // 10) * 100
        flips = current_base * mult
        
        if flips > 5000:
            flips = 5000

        assignment = make_random_assignment()
        
        p = 0.55
        result, final_assignment = walk_sat_run(assignment, max_flips=flips, p=p)
        
        if result is not None:
            return result

        # Track best
        unsat_count = count_unsat(final_assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = final_assignment[:]
            
            # If we are very close, try polishing
            if unsat_count <= 3:
                unsat_clauses = []
                for i in range(m):
                    cv = clause_vars[i]
                    cs = clause_sigs[i]
                    sat = False
                    for j in range(3):
                        if final_assignment[cv[j]] == (cs[j] > 0):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(i)
                
                polished = polish(final_assignment, unsat_clauses)
                polished_unsat = count_unsat(polished)
                if polished_unsat == 0:
                    return polished
                if polished_unsat < best_unsat:
                    best_unsat = polished_unsat
                    best_assignment = polished[:]
        
        # Occasionally try continuous energy descent from the current best
        if restart % 5 == 0 and best_assignment is not None and best_unsat <= 10:
            cont_result = continuous_energy_descent(best_assignment, max_iters=300, lr=0.1)
            cont_unsat = count_unsat(cont_result)
            if cont_unsat == 0:
                return cont_result
            if cont_unsat < best_unsat:
                best_unsat = cont_unsat
                best_assignment = cont_result[:]
        
        # Also try from random assignments periodically
        if restart % 10 == 5 and best_assignment is not None:
            rand_assign = make_random_assignment()
            cont_result = continuous_energy_descent(rand_assign, max_iters=150, lr=0.1)
            cont_unsat = count_unsat(cont_result)
            if cont_unsat == 0:
                return cont_result
            if cont_unsat < best_unsat:
                best_unsat = cont_unsat
                best_assignment = cont_result[:]

    if best_assignment is not None:
        return best_assignment

    return [True] * n
# EVOLVE-BLOCK-END
