import random
import time
import math
import numpy as np

def solve(instance):
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s kill

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # --- Preprocessing: Pure Literal + Unit Propagation ---
    work_clauses = [list(c) for c in clauses]
    assigned = {}
    active = [True] * m

    def update_active():
        for ci in range(m):
            if not active[ci]:
                continue
            for lit in work_clauses[ci]:
                v = abs(lit) - 1
                if v in assigned:
                    if (lit > 0 and assigned[v]) or (lit < 0 and not assigned[v]):
                        active[ci] = False
                        break

    changed = True
    while changed:
        changed = False
        # Pure Literal
        pos_vars = set()
        neg_vars = set()
        for ci in range(m):
            if not active[ci]:
                continue
            for lit in work_clauses[ci]:
                v = abs(lit) - 1
                if v in assigned:
                    continue
                if lit > 0:
                    pos_vars.add(v)
                else:
                    neg_vars.add(v)

        for v in pos_vars - neg_vars:
            assigned[v] = True
            changed = True
        for v in neg_vars - pos_vars:
            assigned[v] = False
            changed = True

        if changed:
            update_active()
            continue

        # Unit Propagation
        unit_found = False
        for ci in range(m):
            if not active[ci]:
                continue
            unassigned_lits = [lit for lit in work_clauses[ci] if (abs(lit) - 1) not in assigned]
            if len(unassigned_lits) == 0:
                return [False] * n
            elif len(unassigned_lits) == 1:
                lit = unassigned_lits[0]
                v = abs(lit) - 1
                assigned[v] = (lit > 0)
                unit_found = True
                changed = True

        if unit_found:
            update_active()

    # Build Reduced Problem
    unassigned_vars = [v for v in range(n) if v not in assigned]
    var_map = {v: i for i, v in enumerate(unassigned_vars)}
    n_red = len(unassigned_vars)

    if n_red == 0:
        return [assigned.get(v, False) for v in range(n)]

    red_clauses = []
    for ci in range(m):
        if not active[ci]:
            continue
        lits = []
        for lit in work_clauses[ci]:
            v = abs(lit) - 1
            if v not in assigned:
                lits.append(lit)
        if len(lits) <= 1:
            if len(lits) == 1:
                assigned[abs(lits[0]) - 1] = (lits[0] > 0)
            continue
        red_clauses.append(lits)

    m_red = len(red_clauses)
    if m_red == 0:
        return [assigned.get(v, False) for v in range(n)]

    # --- New: DPLL with VSIDS ---
    def dpll_vsids():
        if time.time() - start_time > deadline - 10:
            return None
        
        # Map original literals to reduced indices
        # red_clauses contains original literals
        # We need to work with indices 0..n_red-1
        # Let's convert red_clauses to use new indices
        rc = []
        for c in red_clauses:
            new_c = []
            for lit in c:
                v_old = abs(lit) - 1
                v_new = var_map[v_old]
                new_lit = v_new + 1 if lit > 0 else -(v_new + 1)
                new_c.append(new_lit)
            rc.append(new_c)
        
        n_vars = n_red
        # VSIDS activities
        activity = [0.0] * n_vars
        # Assignment: 0 = False, 1 = True, -1 = Unassigned
        assignment = [-1] * n_vars
        
        def propagate():
            # Returns True if success, False if conflict
            # Repeatedly find unit clauses and assign
            changed = True
            while changed:
                changed = False
                for ci, clause in enumerate(rc):
                    # Check if clause is satisfied
                    sat = False
                    unassigned_lits = []
                    for lit in clause:
                        v = abs(lit) - 1
                        val = assignment[v]
                        if val == -1:
                            unassigned_lits.append(lit)
                        else:
                            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                                sat = True
                                break
                    if sat:
                        continue
                    if len(unassigned_lits) == 0:
                        return False # Conflict
                    if len(unassigned_lits) == 1:
                        lit = unassigned_lits[0]
                        v = abs(lit) - 1
                        assignment[v] = 1 if lit > 0 else 0
                        # Boost activity
                        activity[v] *= 1.05
                        changed = True
            return True

        def choose_var():
            # Pick unassigned var with highest activity
            max_act = -1
            best_v = -1
            for v in range(n_vars):
                if assignment[v] == -1:
                    if activity[v] > max_act:
                        max_act = activity[v]
                        best_v = v
            return best_v

        def search(depth):
            if time.time() - start_time > deadline - 10:
                return None # Timeout
            
            if not propagate():
                return False
            
            # Check if all vars assigned
            all_assigned = True
            for v in range(n_vars):
                if assignment[v] == -1:
                    all_assigned = False
                    break
            
            if all_assigned:
                return True
            
            v = choose_var()
            if v == -1:
                return True # Should be caught by all_assigned
            
            # Try True
            assignment[v] = 1
            if search(depth + 1):
                return True
            
            # Try False
            assignment[v] = 0
            if search(depth + 1):
                return True
            
            # Backtrack
            assignment[v] = -1
            # Decay activities
            for i in range(n_vars):
                activity[i] *= 0.95
            return False

        result = search(0)
        if result:
            # Convert back to original variable assignments
            sol = [False] * n_vars
            for v in range(n_vars):
                sol[v] = (assignment[v] == 1)
            return sol
        return None

    dpll_result = dpll_vsids()
    
    if dpll_result is not None:
        # Verify
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = dpll_result[i]
        final_assign = [full_assign.get(v, False) for v in range(n)]
        
        sat_count = 0
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (lit > 0 and final_assign[v]) or (lit < 0 and not final_assign[v]):
                    sat = True
                    break
            if sat:
                sat_count += 1
        
        if sat_count == m:
            return final_assign

    # Prepare structures for WalkSAT
    var_occ = [[] for _ in range(n_red)]
    clause_lits = []
    pos_counts = [0] * n_red
    neg_counts = [0] * n_red

    for ci, c in enumerate(red_clauses):
        lits = []
        for lit in c:
            v_old = abs(lit) - 1
            v_new = var_map[v_old]
            p = 1 if lit > 0 else 0
            lits.append((v_new, p))
            var_occ[v_new].append((ci, p))
            if p == 1:
                pos_counts[v_new] += 1
            else:
                neg_counts[v_new] += 1
        clause_lits.append(lits)

    def get_initial_assign(strategy):
        assign = [False] * n_red
        if strategy == 'polarity':
            for v in range(n_red):
                assign[v] = pos_counts[v] >= neg_counts[v]
        elif strategy == 'random':
            for v in range(n_red):
                assign[v] = random.random() < 0.5
        return assign

    best_unsat_global = float('inf')
    best_assign_global = None
    phase = [False] * n_red

    def weighted_walk_sat(max_restarts, base_steps, noise_init, noise_decay, weight_factor):
        nonlocal best_unsat_global, best_assign_global, phase

        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break

            # Initialize
            if restart == 0:
                assign = get_initial_assign('polarity')
            elif restart == 1:
                assign = get_initial_assign('random')
            else:
                assign = phase[:]
                noise_frac = 0.05 + 0.15 * (restart / max_restarts)
                for v in range(n_red):
                    if random.random() < noise_frac:
                        assign[v] = not assign[v]

            # Compute initial state
            clause_sat_count = [0] * m_red
            unsat_clauses = []
            unsat_index = {}
            for ci in range(m_red):
                cnt = 0
                for v, p in clause_lits[ci]:
                    if assign[v] == (p == 1):
                        cnt += 1
                clause_sat_count[ci] = cnt
                if cnt == 0:
                    unsat_clauses.append(ci)
                    unsat_index[ci] = len(unsat_clauses) - 1

            unsat_count = len(unsat_clauses)

            if unsat_count == 0:
                return assign

            if unsat_count < best_unsat_global:
                best_unsat_global = unsat_count
                best_assign_global = assign[:]
                phase = assign[:]

            # Clause weights
            weights = [1.0] * m_red
            total_weight = 1.0

            steps = 0
            max_steps = base_steps + restart * 50
            noise = noise_init * (noise_decay ** restart)

            while steps < max_steps and time.time() - start_time < deadline:
                if unsat_count == 0:
                    return assign

                # Pick unsat clause using weighted selection
                if len(unsat_clauses) == 1:
                    ci = unsat_clauses[0]
                else:
                    r = random.random() * total_weight
                    cum = 0.0
                    ci = unsat_clauses[0]
                    for uc in unsat_clauses:
                        cum += weights[uc]
                        if r <= cum:
                            ci = uc
                            break

                clause = clause_lits[ci]

                # Choose variable to flip using min-break heuristic
                # Balance: minimize damage to other clauses, maximize benefit
                if random.random() < noise:
                    v, p = random.choice(clause)
                else:
                    best_v = -1
                    best_score = float('inf')
                    for v, p in clause:
                        old_val = assign[v]
                        new_val = not old_val
                        damage = 0
                        benefit = 0
                        for (cci, cp) in var_occ[v]:
                            if cci == ci:
                                continue
                            old_sat = old_val == (cp == 1)
                            new_sat = new_val == (cp == 1)
                            if old_sat and not new_sat:
                                # We break this clause
                                if clause_sat_count[cci] == 1:
                                    damage += 1
                            elif not old_sat and new_sat:
                                # We satisfy this clause
                                benefit += 1
                        # Score: damage is bad, benefit is good
                        # Use damage - benefit as the criterion
                        score = damage - benefit
                        if score < best_score:
                            best_score = score
                            best_v = v
                    v = best_v

                # Flip
                old_val = assign[v]
                assign[v] = not old_val
                new_val = assign[v]

                # Update counts
                for (cci, cp) in var_occ[v]:
                    old_sat = old_val == (cp == 1)
                    new_sat = new_val == (cp == 1)
                    if old_sat and not new_sat:
                        clause_sat_count[cci] -= 1
                        if clause_sat_count[cci] == 0:
                            unsat_count += 1
                            unsat_clauses.append(cci)
                            unsat_index[cci] = len(unsat_clauses) - 1
                            weights[cci] *= weight_factor
                            total_weight += weights[cci] - 1.0
                    elif not old_sat and new_sat:
                        clause_sat_count[cci] += 1
                        if clause_sat_count[cci] == 1:
                            unsat_count -= 1
                            idx = unsat_index[cci]
                            last = unsat_clauses[-1]
                            unsat_clauses[idx] = last
                            unsat_index[last] = idx
                            unsat_clauses.pop()
                            del unsat_index[cci]
                            old_w = weights[cci]
                            weights[cci] /= weight_factor
                            total_weight += weights[cci] - old_w

                steps += 1

                if unsat_count < best_unsat_global:
                    best_unsat_global = unsat_count
                    best_assign_global = assign[:]
                    phase = assign[:]
                    if unsat_count == 0:
                        return assign

        return best_assign_global

    # --- New: Spectral/Gradient-based approach as a secondary method ---
    def spectral_gradient_solve(max_iters=5000):
        """
        Use a smooth relaxation approach:
        Represent assignment as real-valued vector x in [-1, 1]^n_red
        Minimize sum over clauses of soft-min of literal values
        Use gradient descent with momentum
        """
        if time.time() - start_time > deadline:
            return None

        # Convert to real-valued representation
        # x[v] > 0 means True, x[v] < 0 means False
        # For a literal (v, p): value = x[v] if p==1 else -x[v]
        # Clause is satisfied if max(literal_values) > 0
        # We want to maximize sum of clause values, or minimize negative
        
        # Build clause matrix for fast evaluation
        # clause_mat[ci] = list of (v, sign) where sign = +1 if positive lit, -1 if negative
        clause_mat = []
        for c in red_clauses:
            mat = []
            for lit in c:
                v_old = abs(lit) - 1
                v_new = var_map[v_old]
                sign = 1.0 if lit > 0 else -1.0
                mat.append((v_new, sign))
            clause_mat.append(mat)

        # Initialize with polarity bias
        x = np.zeros(n_red)
        for v in range(n_red):
            x[v] = 1.0 if pos_counts[v] >= neg_counts[v] else -1.0

        # Gradient descent with momentum
        lr = 0.5
        momentum = 0.9
        velocity = np.zeros(n_red)
        
        best_x = x.copy()
        best_score = -float('inf')
        
        beta = 2.0  # Softness parameter for log-sum-exp approximation
        
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            # Compute gradient of smooth objective
            # Objective: sum_c softmax(max_lit_c) * beta
            # Use log-sum-exp for stability
            grad = np.zeros(n_red)
            
            total_obj = 0.0
            for ci, mat in enumerate(clause_mat):
                vals = np.array([x[v] * sign for v, sign in mat])
                # Soft max: beta * log(sum(exp(v/beta)))
                max_v = np.max(vals)
                exp_vals = np.exp((vals - max_v) / beta)
                log_sum = max_v / beta + np.log(np.sum(exp_vals))
                obj_c = beta * log_sum
                total_obj += obj_c
                
                # Gradient: d/dx [beta * log(sum exp(v_l/beta))]
                # = sum_l exp(v_l/beta) / sum exp(v_l/beta) * v_l_sign
                probs = exp_vals / np.sum(exp_vals)
                for i, (v, sign) in enumerate(mat):
                    grad[v] += probs[i] * sign
            
            # Update with momentum
            velocity = momentum * velocity + lr * grad
            x = x + velocity
            
            # Project to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            
            # Track best
            if total_obj > best_score:
                best_score = total_obj
                best_x = x.copy()
            
            # Check if we've found a satisfying assignment
            # Convert to boolean and check
            assign_bool = x > 0
            unsat = 0
            for ci, mat in enumerate(clause_mat):
                sat = False
                for v, sign in mat:
                    if assign_bool[v] == (sign > 0):
                        sat = True
                        break
                if not sat:
                    unsat += 1
                    if unsat > 0:
                        break
            
            if unsat == 0:
                return assign_bool.tolist()
            
            # Decay learning rate
            lr *= 0.999
            # Increase beta for sharper max
            beta *= 1.01

        return best_x > 0

    # Run WalkSAT first
    result = weighted_walk_sat(
        max_restarts=300,
        base_steps=200,
        noise_init=0.3,
        noise_decay=0.99,
        weight_factor=1.05
    )
    
    # Local polish: try to fix remaining unsatisfied clauses
    if result is not None and time.time() - start_time < deadline - 5:
        # Count unsat clauses
        assign = result[:]
        unsat_list = []
        for ci in range(m_red):
            sat = False
            for v, p in clause_lits[ci]:
                if assign[v] == (p == 1):
                    sat = True
                    break
            if not sat:
                unsat_list.append(ci)
        
        # Try to fix each unsat clause by flipping one of its literals
        for _ in range(min(len(unsat_list), 50)):
            if not unsat_list:
                break
            if time.time() - start_time > deadline - 5:
                break
            ci = unsat_list.pop()
            # Try each literal in the clause
            found = False
            for v, p in clause_lits[ci]:
                old_val = assign[v]
                assign[v] = not old_val
                # Check if this breaks any other clause
                breaks = 0
                for (cci, cp) in var_occ[v]:
                    if cci == ci:
                        continue
                    old_sat = old_val == (cp == 1)
                    new_sat = assign[v] == (cp == 1)
                    if old_sat and not new_sat:
                        # Check if clause cci is now unsat
                        sat_count = 0
                        for vv, pp in clause_lits[cci]:
                            if assign[vv] == (pp == 1):
                                sat_count += 1
                                if sat_count > 0:
                                    break
                        if sat_count == 0:
                            breaks += 1
                if breaks == 0:
                    found = True
                    break
                else:
                    assign[v] = old_val
            if not found:
                # Re-add to list
                unsat_list.append(ci)
        
        result = assign

    def verify_and_return(assign_list):
        if assign_list is None:
            return None
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = assign_list[i]
        final_assign = [full_assign.get(v, False) for v in range(n)]
        
        sat_count = 0
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (lit > 0 and final_assign[v]) or (lit < 0 and not final_assign[v]):
                    sat = True
                    break
            if sat:
                sat_count += 1
        
        if sat_count == m:
            return final_assign
        return None

    # Try spectral method
    spectral_result = spectral_gradient_solve(max_iters=3000)
    verified = verify_and_return(spectral_result)
    if verified is not None:
        return verified

    # If WalkSAT found something, verify it
    verified = verify_and_return(result)
    if verified is not None:
        return verified

    # Try WalkSAT with different parameters
    result2 = weighted_walk_sat(
        max_restarts=200,
        base_steps=300,
        noise_init=0.5,
        noise_decay=0.98,
        weight_factor=1.1
    )
    verified = verify_and_return(result2)
    if verified is not None:
        return verified
    
    # Polish result2 as well
    if result2 is not None and time.time() - start_time < deadline - 5:
        assign = result2[:]
        unsat_list = []
        for ci in range(m_red):
            sat = False
            for v, p in clause_lits[ci]:
                if assign[v] == (p == 1):
                    sat = True
                    break
            if not sat:
                unsat_list.append(ci)
        
        for _ in range(min(len(unsat_list), 50)):
            if not unsat_list:
                break
            if time.time() - start_time > deadline - 5:
                break
            ci = unsat_list.pop()
            found = False
            for v, p in clause_lits[ci]:
                old_val = assign[v]
                assign[v] = not old_val
                breaks = 0
                for (cci, cp) in var_occ[v]:
                    if cci == ci:
                        continue
                    old_sat = old_val == (cp == 1)
                    new_sat = assign[v] == (cp == 1)
                    if old_sat and not new_sat:
                        sat_count = 0
                        for vv, pp in clause_lits[cci]:
                            if assign[vv] == (pp == 1):
                                sat_count += 1
                                if sat_count > 0:
                                    break
                        if sat_count == 0:
                            breaks += 1
                if breaks == 0:
                    found = True
                    break
                else:
                    assign[v] = old_val
            if not found:
                unsat_list.append(ci)
        
        result2 = assign
        verified = verify_and_return(result2)
        if verified is not None:
            return verified

    # Try spectral again with different params
    spectral_result2 = spectral_gradient_solve(max_iters=5000)
    verified = verify_and_return(spectral_result2)
    if verified is not None:
        return verified

    # Fallback: use best from WalkSAT
    if result is not None:
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = result[i]
        return [full_assign.get(v, False) for v in range(n)]

    # Final fallback
    full_assign = dict(assigned)
    for v in range(n):
        if v not in full_assign:
            full_assign[v] = random.random() < 0.5
    return [full_assign.get(v, False) for v in range(n)]
