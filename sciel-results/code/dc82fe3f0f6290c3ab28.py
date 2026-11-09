import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # 1. DPLL with Unit Propagation
    def simplify(clauses, assignment):
        new_clauses = []
        unit_clauses = []
        
        for clause in clauses:
            new_clause = []
            has_true = False
            for lit in clause:
                var = abs(lit) - 1
                val = assignment[var]
                if val == 1 and lit > 0:
                    has_true = True
                    break
                if val == 1 and lit < 0:
                    has_true = True
                    break
                if val == 0 and lit > 0:
                    continue
                if val == 0 and lit < 0:
                    continue
                new_clause.append(lit)
            
            if has_true:
                continue
            if not new_clause:
                return None, True, []
            if len(new_clause) == 1:
                unit_clauses.append(new_clause[0])
            else:
                new_clauses.append(new_clause)
                
        return new_clauses, False, unit_clauses

    def dpll(clauses, n, assignment, node_limit):
        if node_limit <= 0:
            return False, assignment
            
        new_clauses, is_unsat, unit_clauses = simplify(clauses, assignment)
        if is_unsat:
            return False, assignment
            
        if not new_clauses:
            final_assign = assignment.copy()
            for i in range(n):
                if final_assign[i] == -1:
                    final_assign[i] = random.randint(0, 1)
            return True, final_assign
            
        changed = True
        while changed:
            changed = False
            for lit in unit_clauses:
                var = abs(lit) - 1
                if assignment[var] == -1:
                    assignment[var] = 1 if lit > 0 else 0
                    changed = True
            if changed:
                new_clauses, is_unsat, unit_clauses = simplify(clauses, assignment)
                if is_unsat:
                    return False, assignment
                if not new_clauses:
                    final_assign = assignment.copy()
                    for i in range(n):
                        if final_assign[i] == -1:
                            final_assign[i] = random.randint(0, 1)
                    return True, final_assign

        var_counts = {}
        for c in new_clauses:
            for lit in c:
                var = abs(lit)
                var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            final_assign = assignment.copy()
            for i in range(n):
                if final_assign[i] == -1:
                    final_assign[i] = random.randint(0, 1)
            return True, final_assign

        best_var = max(var_counts, key=var_counts.get)
        best_var_idx = best_var - 1
        
        assignment[best_var_idx] = 1
        success, result = dpll(new_clauses, n, assignment, node_limit - 1)
        if success:
            return True, result
        
        assignment[best_var_idx] = 0
        success, result = dpll(new_clauses, n, assignment, node_limit - 1)
        if success:
            return True, result
        
        assignment[best_var_idx] = -1
        return False, assignment

    initial_assignment = [-1] * n
    success, dpll_result = dpll(clauses, n, initial_assignment, 5000)
    if success:
        return [bool(x) for x in dpll_result]

    # 2. SA-WalkSAT
    assignment = [random.randint(0, 1) for _ in range(n)]
    
    clause_lits = []
    for c in clauses:
        lits = []
        for lit in c:
            lits.append((abs(lit)-1, lit > 0))
        clause_lits.append(lits)
        
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clause_lits):
        for var, _ in c:
            var_to_clauses[var].append(i)

    m = len(clauses)
    clause_sat = [False] * m
    unsat_clauses = []
    
    def check_clause(i):
        for var, is_pos in clause_lits[i]:
            if assignment[var] == (1 if is_pos else 0):
                return True
        return False

    for i in range(m):
        if check_clause(i):
            clause_sat[i] = True
        else:
            clause_sat[i] = False
            unsat_clauses.append(i)
            
    current_unsat = len(unsat_clauses)
    best_assignment = assignment.copy()
    best_unsat = current_unsat
    
    start_time = time.time()
    deadline = start_time + 290
    
    temperature = 1.0
    cooling_rate = 0.995
    min_temp = 0.001
    
    restarts = 0
    max_restarts = 100
    steps_since_improvement = 0
    max_steps_no_improve = 1000
    
    while time.time() < deadline:
        if current_unsat == 0:
            break
            
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            
        if steps_since_improvement > max_steps_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            assignment = [random.randint(0, 1) for _ in range(n)]
            # Re-initialize clause_sat and unsat_clauses
            for i in range(m):
                if check_clause(i):
                    clause_sat[i] = True
                else:
                    clause_sat[i] = False
            unsat_clauses = [i for i in range(m) if not clause_sat[i]]
            current_unsat = len(unsat_clauses)
            temperature = 1.0
            steps_since_improvement = 0
            continue
            
        if not unsat_clauses:
            break
            
        clause_idx = random.choice(unsat_clauses)
        clause = clause_lits[clause_idx]
        var_idx, is_pos = random.choice(clause)
        
        old_val = assignment[var_idx]
        new_val = 1 - old_val
        
        delta = 0
        for c_idx in var_to_clauses[var_idx]:
            was_sat = clause_sat[c_idx]
            
            # Check if clause is satisfied by other literals
            sat_by_other = False
            for v, p in clause_lits[c_idx]:
                if v == var_idx:
                    continue
                if assignment[v] == (1 if p else 0):
                    sat_by_other = True
                    break
            
            if was_sat:
                if not sat_by_other:
                    delta += 1
            else:
                if (is_pos and new_val == 1) or (not is_pos and new_val == 0):
                    delta -= 1

        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = True
        else:
            r = random.random()
            if r < np.exp(-delta / temperature):
                accept = True

        if accept:
            assignment[var_idx] = new_val
            # Update clause_sat and unsat_clauses
            for c_idx in var_to_clauses[var_idx]:
                was_sat = clause_sat[c_idx]
                # Re-evaluate
                is_sat_now = False
                for v, p in clause_lits[c_idx]:
                    if assignment[v] == (1 if p else 0):
                        is_sat_now = True
                        break
                
                if was_sat and not is_sat_now:
                    unsat_clauses.append(c_idx)
                    current_unsat += 1
                elif not was_sat and is_sat_now:
                    # Remove from unsat_clauses
                    # This is O(K) to remove from list. 
                    # Since K is small, it's fine.
                    try:
                        unsat_clauses.remove(c_idx)
                    except ValueError:
                        pass
                    current_unsat -= 1
                    
                clause_sat[c_idx] = is_sat_now

        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp

    return [bool(x) for x in best_assignment]
