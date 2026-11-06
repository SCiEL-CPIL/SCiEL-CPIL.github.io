import random
import time
import math
import numpy as np

def solve(instance):
    start_time = time.time()
    deadline = 325.0  # Safety margin

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # --- Preprocessing: Pure Literal + Unit Propagation ---
    # We use a slightly optimized version of the original preprocessing
    work_clauses = [list(c) for c in clauses]
    assigned = {}
    active = [True] * m
    
    # Helper to update active clauses
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
            if not active[ci]: continue
            for lit in work_clauses[ci]:
                v = abs(lit) - 1
                if v in assigned: continue
                if lit > 0: pos_vars.add(v)
                else: neg_vars.add(v)
        
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
            if not active[ci]: continue
            unassigned_lits = [lit for lit in work_clauses[ci] if (abs(lit) - 1) not in assigned]
            if len(unassigned_lits) == 0:
                return [False] * n # Contradiction, shouldn't happen for SAT
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
        if not active[ci]: continue
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

    # Prepare structures for WalkSAT
    # var_occ[v] = list of (clause_idx, polarity)
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
            if p == 1: pos_counts[v_new] += 1
            else: neg_counts[v_new] += 1
        clause_lits.append(lits)

    def check_satisfied(assign):
        unsat = 0
        for ci in range(m_red):
            sat = False
            for v, p in clause_lits[ci]:
                if assign[v] == (p == 1):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat

    def get_initial_assign(strategy):
        assign = [False] * n_red
        if strategy == 'polarity':
            for v in range(n_red):
                assign[v] = pos_counts[v] >= neg_counts[v]
        elif strategy == 'random':
            for v in range(n_red):
                assign[v] = random.random() < 0.5
        else: # phase
            pass
        return assign

    best_unsat_global = float('inf')
    best_assign_global = None
    phase = [False] * n_red # Global phase memory

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
                # Phase saving: use phase from best previous run
                assign = phase[:]
                # Add some random noise to phase
                noise_frac = 0.05 + 0.15 * (restart / max_restarts)
                for v in range(n_red):
                    if random.random() < noise_frac:
                        assign[v] = not assign[v]
            
            # Compute initial state
            clause_sat_count = [0] * m_red
            unsat_clauses = []
            for ci in range(m_red):
                cnt = 0
                for v, p in clause_lits[ci]:
                    if assign[v] == (p == 1):
                        cnt += 1
                clause_sat_count[ci] = cnt
                if cnt == 0:
                    unsat_clauses.append(ci)
            
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
                    # Weighted random selection from unsat clauses
                    r = random.random() * total_weight
                    cum = 0.0
                    ci = unsat_clauses[0]
                    for uc in unsat_clauses:
                        cum += weights[uc]
                        if r <= cum:
                            ci = uc
                            break
                
                clause = clause_lits[ci]
                
                # Choose variable to flip
                if random.random() < noise:
                    v, p = random.choice(clause)
                else:
                    # Greedy: minimize damage
                    best_v = -1
                    best_damage = float('inf')
                    for v, p in clause:
                        old_val = assign[v]
                        damage = 0
                        for (cci, cp) in var_occ[v]:
                            if cci == ci:
                                continue
                            if clause_sat_count[cci] == 1:
                                if old_val == (cp == 1):
                                    damage += 1
                        if damage < best_damage:
                            best_damage = damage
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
                            weights[cci] *= weight_factor
                            total_weight += weights[cci] - 1.0
                    elif not old_sat and new_sat:
                        clause_sat_count[cci] += 1
                        if clause_sat_count[cci] == 1:
                            unsat_count -= 1
                            # Remove from unsat_clauses (swap with last for O(1))
                            idx = unsat_clauses.index(cci)
                            unsat_clauses[idx] = unsat_clauses[-1]
                            unsat_clauses.pop()
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

    def check_full_solution(assign):
        """Check if assign (list of bools for all n vars) satisfies all original clauses."""
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def try_walk_sat(params, time_limit):
        """Run WalkSAT with given params, return assignment or None."""
        t_start = time.time()
        result = weighted_walk_sat(
            max_restarts=params['max_restarts'], 
            base_steps=params['base_steps'], 
            noise_init=params['noise_init'], 
            noise_decay=params['noise_decay'], 
            weight_factor=params['weight_factor']
        )
        
        if result is None:
            return None
        
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = result[i]
        final_assign = [full_assign.get(v, False) for v in range(n)]
        
        if check_full_solution(final_assign):
            return final_assign
        return None

    # Phase 1: Standard WalkSAT
    params1 = {
        'max_restarts': 500, 
        'base_steps': 200, 
        'noise_init': 0.3, 
        'noise_decay': 0.99, 
        'weight_factor': 1.05
    }
    
    result = try_walk_sat(params1, deadline - (time.time() - start_time))
    if result is not None:
        return result

    # Phase 2: Vectorized Max-SAT relaxation via gradient descent
    if n_red > 0 and n_red <= 200:
        try:
            # Build clause matrix: shape (m_red, n_red) with values in {-1, 0, 1}
            # For each clause, for each variable, the entry is:
            # 1 if variable appears positively in clause
            # -1 if variable appears negatively in clause
            # 0 if variable doesn't appear in clause
            
            clause_matrix = np.zeros((m_red, n_red), dtype=np.float32)
            for ci, c in enumerate(red_clauses):
                for lit in c:
                    v_old = abs(lit) - 1
                    v_new = var_map[v_old]
                    clause_matrix[ci, v_new] = 1.0 if lit > 0 else -1.0
            
            # Convert to x in [-1, 1] space where x = 1 means True, x = -1 means False
            # A clause is satisfied if sum of (clause_matrix[i,j] * x_j) > 0 for at least one term
            # Actually for 3-SAT: clause (l1 ∨ l2 ∨ l3) is satisfied if any literal is true
            # In x-space: literal l_j is satisfied if clause_matrix[i,j] * x_j > 0
            # But we want to maximize the number of satisfied clauses
            
            # Use a smooth surrogate: for each clause, use soft-min of the literal values
            # Or better: maximize sum of min(1, max(lit_1, lit_2, lit_3))
            # Simpler: use -sum of exp(-beta * max(lit_values)) as a smooth max approximation
            
            # Even simpler: use the fact that for planted SAT, the solution should have
            # most clauses satisfied. Use gradient ascent on a smooth objective.
            
            # Objective: maximize sum over clauses of max(0, min(1, sum_of_literals))
            # This is non-differentiable, so use a smooth approximation.
            
            # Alternative: use Max-SAT formulation with weights
            # Maximize sum_i w_i * s_i where s_i = 1 if clause i satisfied, 0 otherwise
            # Smooth: s_i ≈ sigmoid(beta * sum_j clause_matrix[i,j] * x_j)
            
            beta = 5.0
            lr = 0.1
            n_iter = 5000
            
            # Initialize with polarity bias
            x = np.zeros(n_red, dtype=np.float32)
            for v in range(n_red):
                if pos_counts[v] >= neg_counts[v]:
                    x[v] = 1.0
                else:
                    x[v] = -1.0
            
            # Add small random perturbation
            x += (np.random.randn(n_red) * 0.1).astype(np.float32)
            x = np.clip(x, -1.0, 1.0)
            
            best_x = x.copy()
            best_score = -1.0
            
            for it in range(n_iter):
                if time.time() > deadline - 5.0:
                    break
                
                # Compute clause scores: score_i = sum_j clause_matrix[i,j] * x_j
                scores = clause_matrix @ x  # shape (m_red,)
                
                # Smooth satisfaction: s_i = sigmoid(beta * score_i)
                # Clip to avoid overflow
                scores_clipped = np.clip(scores, -50.0, 50.0)
                s = 1.0 / (1.0 + np.exp(-beta * scores_clipped))  # shape (m_red,)
                
                # Objective: sum of s
                obj = np.sum(s)
                
                if obj > best_score:
                    best_score = obj
                    best_x = x.copy()
                
                # Gradient: d(obj)/d(x_j) = sum_i beta * s_i * (1 - s_i) * clause_matrix[i,j]
                # = beta * (clause_matrix.T @ (s * (1 - s)))
                grad = beta * (clause_matrix.T @ (s * (1.0 - s)))
                
                # Add L2 regularization to encourage extreme values (±1)
                # This is like a ferromagnetic term: -lambda * sum x_j^2
                # Actually we want to push toward ±1, so use -lambda * sum (x_j^2 - 1)^2
                # Simpler: add a term that pushes x away from 0
                reg_grad = 0.1 * x  # This actually pulls toward 0, we want the opposite
                # For pushing toward ±1: -d/dx [(x^2-1)^2] = -4x(x^2-1) = 4x(1-x^2)
                reg_grad = 4.0 * x * (1.0 - x * x) * 0.05
                
                # Update
                x = x + lr * (grad + reg_grad)
                x = np.clip(x, -1.0, 1.0)
            
            # Convert best_x to boolean assignment
            x_binary = (best_x > 0).astype(int)
            
            # Map back to full assignment
            full_assign = dict(assigned)
            for i, v in enumerate(unassigned_vars):
                full_assign[v] = bool(x_binary[i])
            final_assign = [full_assign.get(v, False) for v in range(n)]
            
            if check_full_solution(final_assign):
                return final_assign
            
            # If not satisfied, use this as a starting point for a short WalkSAT
            # Convert to 0/1 array for WalkSAT
            walk_start = x_binary.tolist()
            
            # Run a short WalkSAT from this starting point
            assign = walk_start[:]
            clause_sat_count = [0] * m_red
            unsat_clauses = []
            for ci in range(m_red):
                cnt = 0
                for v, p in clause_lits[ci]:
                    if assign[v] == (p == 1):
                        cnt += 1
                clause_sat_count[ci] = cnt
                if cnt == 0:
                    unsat_clauses.append(ci)
            
            unsat_count = len(unsat_clauses)
            weights = [1.0] * m_red
            total_weight = 1.0
            
            max_steps = 1000
            for step in range(max_steps):
                if time.time() > deadline - 3.0:
                    break
                if unsat_count == 0:
                    break
                
                # Pick random unsat clause
                ci = random.choice(unsat_clauses)
                clause = clause_lits[ci]
                
                # Pick variable to flip with some noise
                if random.random() < 0.3:
                    v, p = random.choice(clause)
                else:
                    best_v = -1
                    best_damage = float('inf')
                    for v, p in clause:
                        old_val = assign[v]
                        damage = 0
                        for (cci, cp) in var_occ[v]:
                            if cci == ci:
                                continue
                            if clause_sat_count[cci] == 1:
                                if old_val == (cp == 1):
                                    damage += 1
                        if damage < best_damage:
                            best_damage = damage
                            best_v = v
                    v = best_v
                
                old_val = assign[v]
                assign[v] = not old_val
                new_val = assign[v]
                
                for (cci, cp) in var_occ[v]:
                    old_sat = old_val == (cp == 1)
                    new_sat = new_val == (cp == 1)
                    if old_sat and not new_sat:
                        clause_sat_count[cci] -= 1
                        if clause_sat_count[cci] == 0:
                            unsat_count += 1
                            unsat_clauses.append(cci)
                            weights[cci] *= 1.05
                            total_weight += weights[cci] - 1.0
                    elif not old_sat and new_sat:
                        clause_sat_count[cci] += 1
                        if clause_sat_count[cci] == 1:
                            unsat_count -= 1
                            idx = unsat_clauses.index(cci)
                            unsat_clauses[idx] = unsat_clauses[-1]
                            unsat_clauses.pop()
                            old_w = weights[cci]
                            weights[cci] /= 1.05
                            total_weight += weights[cci] - old_w
            
            if unsat_count == 0:
                full_assign = dict(assigned)
                for i, v in enumerate(unassigned_vars):
                    full_assign[v] = assign[i]
                final_assign = [full_assign.get(v, False) for v in range(n)]
                if check_full_solution(final_assign):
                    return final_assign
        except Exception:
            pass

    # Phase 3: Different WalkSAT parameters
    params2 = {
        'max_restarts': 200, 
        'base_steps': 300, 
        'noise_init': 0.5, 
        'noise_decay': 0.98, 
        'weight_factor': 1.1
    }
    result2 = try_walk_sat(params2, deadline - (time.time() - start_time))
    if result2 is not None:
        return result2
    
    # Fallback
    full_assign = dict(assigned)
    for v in range(n):
        if v not in full_assign:
            full_assign[v] = random.random() < 0.5
    return [full_assign.get(v, False) for v in range(n)]
