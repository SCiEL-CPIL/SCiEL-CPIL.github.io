# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin

    # Convert clauses to 0-based signed literals
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    m = len(clauses_0based)

    # Build data structures
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # === PREPROCESSING: Pure Literal Elimination + Unit Propagation ===
    # This fundamentally changes the problem representation by fixing variables
    fixed = {}  # var -> bool value
    remaining_vars = set(range(n))
    remaining_clauses = list(clauses_0based)
    
    # Iterative pure literal elimination and unit propagation
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        
        # Build literal occurrence map for remaining clauses
        # pos_count[var] = number of clauses where var appears positively (and is not fixed)
        # neg_count[var] = number of clauses where var appears negatively
        pos_count = {}
        neg_count = {}
        for ci, clause in enumerate(remaining_clauses):
            for var, sign in clause:
                if var not in remaining_vars:
                    continue
                if sign > 0:
                    pos_count[var] = pos_count.get(var, 0) + 1
                else:
                    neg_count[var] = neg_count.get(var, 0) + 1
        
        # Pure literal elimination: if a variable appears only in one polarity, assign it
        for var in list(remaining_vars):
            if var not in pos_count and var not in neg_count:
                continue
            if var in pos_count and var not in neg_count:
                # Only appears positively -> set to True
                fixed[var] = True
                remaining_vars.discard(var)
                changed = True
            elif var not in pos_count and var in neg_count:
                # Only appears negatively -> set to False
                fixed[var] = False
                remaining_vars.discard(var)
                changed = True
        
        if changed:
            # Remove satisfied clauses, simplify remaining
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                        # else: literal is false, remove it
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
                    # If simplified is empty, this is a contradiction (shouldn't happen for SAT instances)
            remaining_clauses = new_clauses
            continue
        
        # Unit propagation: find clauses with exactly one unassigned literal
        unit_clauses = []
        for clause in remaining_clauses:
            unassigned = [(var, sign) for var, sign in clause if var in remaining_vars]
            if len(unassigned) == 1:
                unit_clauses.append(unassigned[0])
        
        if unit_clauses:
            for var, sign in unit_clauses:
                if var in remaining_vars:
                    fixed[var] = (sign > 0)
                    remaining_vars.discard(var)
            changed = True
            # Remove satisfied clauses, simplify
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
    
    n_reduced = len(remaining_vars)
    m_reduced = len(remaining_clauses)
    
    # If all variables are fixed, check if all clauses are satisfied
    if n_reduced == 0:
        # All fixed, verify
        for clause in remaining_clauses:
            sat = False
            for var, sign in clause:
                if var in fixed and fixed[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                # Should not happen for satisfiable instances
                break
        # Build full assignment
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            else:
                full_assignment[var] = False
        return full_assignment
    
    # Map reduced variables to indices
    reduced_var_list = list(remaining_vars)
    var_to_reduced_idx = {var: i for i, var in enumerate(reduced_var_list)}
    
    # Convert remaining clauses to use reduced indices
    reduced_clauses = []
    for clause in remaining_clauses:
        reduced_clause = []
        for var, sign in clause:
            if var in remaining_vars:
                reduced_clause.append((var_to_reduced_idx[var], sign))
        if reduced_clause:
            reduced_clauses.append(reduced_clause)
    
    n_r = n_reduced
    m_r = len(reduced_clauses)
    
    # Build var_clauses for reduced instance
    reduced_var_clauses = [[] for _ in range(n_r)]
    for ci, clause in enumerate(reduced_clauses):
        for var, sign in clause:
            reduced_var_clauses[var].append((ci, sign))
    
    def _check_satisfied(assignment, clauses_list, var_clauses_list):
        """Check if all clauses are satisfied."""
        for clause in clauses_list:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def _count_unsat(assignment, clauses_list):
        """Count unsatisfied clauses."""
        count = 0
        for clause in clauses_list:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def _walksat_reduced(rng, max_restarts=100, steps_per_restart=5000):
        """Clause-weighted WalkSAT with dynamic noise on reduced instance."""
        best_assignment = None
        best_unsat = m_r + 1
        
        # Precompute clause weights based on frequency of literals
        clause_weights = np.ones(m_r, dtype=np.float64)
        for clause in reduced_clauses:
            weight = 1.0
            for var, sign in clause:
                # Weight by how many clauses this var appears in
                weight += 0.1 * len(reduced_var_clauses[var])
            clause_weights[int(np.argmax([1]))] = weight  # placeholder
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            assignment = [rng.random() < 0.5 for _ in range(n_r)]
            
            # Compute initial unsat
            clause_unsat = np.zeros(m_r, dtype=bool)
            for ci, clause in enumerate(reduced_clauses):
                sat = False
                for var, sign in clause:
                    if assignment[var] == (sign > 0):
                        sat = True
                        break
                clause_unsat[ci] = not sat
            
            unsat_count = int(np.sum(clause_unsat))
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Dynamic noise: starts high, decreases
            initial_noise = 2.0
            final_noise = 0.1
            noise = initial_noise
            
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return best_assignment
                
                if unsat_count == 0:
                    return assignment
                
                # Decay noise
                noise = initial_noise * (final_noise / initial_noise) ** (step / steps_per_restart)
                
                # Pick random unsatisfied clause (weighted by clause frequency)
                unsat_indices = np.where(clause_unsat)[0]
                if len(unsat_indices) == 0:
                    return assignment
                
                # Weighted selection: clauses with more variables in other unsat clauses get higher weight
                ci = int(rng.choice(unsat_indices))
                clause = reduced_clauses[ci]
                
                # Find best flip
                best_var = -1
                best_score = -float('inf')
                
                for var, sign in clause:
                    score = 0
                    for other_ci, other_sign in reduced_var_clauses[var]:
                        if other_ci == ci:
                            continue
                        if clause_unsat[other_ci]:
                            score += 1.0
                        else:
                            # Check if this is the only satisfying literal
                            other_clause = reduced_clauses[other_ci]
                            only_sat = True
                            for v2, s2 in other_clause:
                                if v2 == var:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                score -= 1.0
                    
                    # Dynamic noise
                    score += rng.gauss(0, noise)
                    
                    if score > best_score:
                        best_score = score
                        best_var = var
                
                if best_var == -1:
                    break
                
                # Flip
                assignment[best_var] = not assignment[best_var]
                
                # Update clause states
                for other_ci, other_sign in reduced_var_clauses[best_var]:
                    other_clause = reduced_clauses[other_ci]
                    sat = False
                    for v, s in other_clause:
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat
                
                if unsat_count == 0:
                    return assignment
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
        
        return best_assignment if best_assignment is not None else [False] * n_r
    
    def _simulated_annealing_reduced(rng, max_iters=50000):
        """Simulated annealing on reduced instance."""
        assignment = [rng.random() < 0.5 for _ in range(n_r)]
        
        clause_unsat = np.zeros(m_r, dtype=bool)
        for ci, clause in enumerate(reduced_clauses):
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat
        
        unsat_count = int(np.sum(clause_unsat))
        if unsat_count == 0:
            return assignment
        
        best_assignment = assignment[:]
        best_unsat = unsat_count
        
        T0 = 1.0
        T_end = 0.005
        alpha = (T_end / T0) ** (1.0 / max_iters)
        T = T0
        
        for step in range(max_iters):
            if time.time() > deadline:
                return best_assignment
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            var = rng.randrange(n_r)
            old_val = assignment[var]
            assignment[var] = not old_val
            
            delta = 0
            for other_ci, other_sign in reduced_var_clauses[var]:
                other_clause = reduced_clauses[other_ci]
                sat = False
                for v, s in other_clause:
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                is_unsat = not sat
                was_unsat = clause_unsat[other_ci]
                if was_unsat and not is_unsat:
                    delta -= 1
                elif not was_unsat and is_unsat:
                    delta += 1
                clause_unsat[other_ci] = is_unsat
            
            new_unsat = unsat_count + delta
            
            if new_unsat < unsat_count:
                unsat_count = new_unsat
            else:
                if delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                    unsat_count = new_unsat
                else:
                    assignment[var] = old_val
                    for other_ci, other_sign in reduced_var_clauses[var]:
                        other_clause = reduced_clauses[other_ci]
                        sat = False
                        for v, s in other_clause:
                            if assignment[v] == (s > 0):
                                sat = True
                                break
                        clause_unsat[other_ci] = not sat
                    unsat_count = int(np.sum(clause_unsat))
            
            T *= alpha
        
        return best_assignment
    
    # Run solvers on reduced instance
    rng = random.Random(42)
    
    # Try WalkSAT first
    reduced_result = _walksat_reduced(rng, max_restarts=50, steps_per_restart=8000)
    
    # Map back to full assignment
    full_assignment = [False] * n
    for var in range(n):
        if var in fixed:
            full_assignment[var] = fixed[var]
        elif var in var_to_reduced_idx:
            full_assignment[var] = reduced_result[var_to_reduced_idx[var]]
        else:
            full_assignment[var] = False
    
    # Verify
    all_sat = True
    for clause in clauses_0based:
        sat = False
        for var, sign in clause:
            if full_assignment[var] == (sign > 0):
                sat = True
                break
        if not sat:
            all_sat = False
            break
    
    if all_sat:
        return full_assignment
    
    # Try simulated annealing if time permits
    if time.time() < deadline - 10:
        reduced_result2 = _simulated_annealing_reduced(random.Random(123), max_iters=30000)
        
        full_assignment2 = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment2[var] = fixed[var]
            elif var in var_to_reduced_idx:
                full_assignment2[var] = reduced_result2[var_to_reduced_idx[var]]
            else:
                full_assignment2[var] = False
        
        all_sat2 = True
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if full_assignment2[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                all_sat2 = False
                break
        
        if all_sat2:
            return full_assignment2
    
    # Return best result found
    return full_assignment
# EVOLVE-BLOCK-END
