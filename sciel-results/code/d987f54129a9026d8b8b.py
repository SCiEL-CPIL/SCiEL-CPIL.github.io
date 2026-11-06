import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 325.0  # 5s margin before 330s kill
    
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
    
    # State: clause_sat_count[i] = number of satisfied literals in clause i
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
        new_val = assignment[v]
        for ci, sign in var_to_clauses[v]:
            if new_val == sign:
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    def unit_propagate(assignment, clause_sat_count):
        """Set variables in clauses with exactly 0 satisfied literals to satisfy one literal.
        Returns True if no contradiction (all clauses handled)."""
        changed = True
        max_rounds = 50
        rounds = 0
        while changed and rounds < max_rounds:
            changed = False
            rounds += 1
            unsat_mask = (clause_sat_count == 0)
            if not np.any(unsat_mask):
                break
            unsat_indices = np.where(unsat_mask)[0]
            for i in unsat_indices:
                if clause_sat_count[i] != 0:
                    continue
                best_v = clause_vars[i][0]
                best_damage = float('inf')
                for v in clause_vars[i]:
                    damage = 0
                    for other_ci, sign in var_to_clauses[v]:
                        if other_ci != i and clause_sat_count[other_ci] > 0:
                            damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_v = v
                sign = None
                for j, v in enumerate(clause_vars[i]):
                    if v == best_v:
                        sign = clause_signs[i][j]
                        break
                if assignment[best_v] != sign:
                    flip_var(best_v, assignment, clause_sat_count)
                    changed = True
        return True
    
    def pure_literal(assignment, clause_sat_count):
        """Find pure literals and assign them."""
        pos_count = [0] * n
        neg_count = [0] * n
        for i in range(m):
            for j, v in enumerate(clause_vars[i]):
                if clause_signs[i][j]:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        
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
    
    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1
    
    base_weight = 0.3
    flip_limit_base = 5000
    
    while time.time() - start_time < timeout:
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = init_state(assignment)
        
        if is_satisfied(clause_sat_count):
            return assignment
        
        # Try pure literal elimination
        pure_literal(assignment, clause_sat_count)
        
        if is_satisfied(clause_sat_count):
            return assignment
        
        # Unit propagation
        unit_propagate(assignment, clause_sat_count)
        
        if is_satisfied(clause_sat_count):
            return assignment
        
        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
        
        # WalkSAT with adaptive flip limit
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time < 5.0:
            break
        
        if current_unsat <= 3:
            max_flips = flip_limit_base * 3
            weight = base_weight * 0.5
        elif current_unsat <= 10:
            max_flips = flip_limit_base * 2
            weight = base_weight
        else:
            max_flips = flip_limit_base
            weight = base_weight * 1.5
        
        for flip_num in range(max_flips):
            if time.time() - start_time > timeout - 3.0:
                break
            
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return assignment
            
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            vars_in_clause = clause_vars[ci]
            
            if rng.random() < weight:
                v = vars_in_clause[rng.randrange(len(vars_in_clause))]
            else:
                best_v = vars_in_clause[0]
                best_damage = float('inf')
                for v in vars_in_clause:
                    damage = 0
                    for other_ci, sign in var_to_clauses[v]:
                        if other_ci != ci and clause_sat_count[other_ci] > 0:
                            damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_v = v
                v = best_v
            
            flip_var(v, assignment, clause_sat_count)
            
            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
    
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
