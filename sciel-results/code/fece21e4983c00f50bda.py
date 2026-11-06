import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Precompute clause structure
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
            
    # Build var_to_clauses
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            if v >= 0:
                var_to_clauses[v].append(i)
                
    def is_lit_satisfied(assign, var, sign):
        return (assign[var] == 1) if sign == 1 else (assign[var] == 0)

    def is_clause_satisfied(assign, i):
        for j in range(3):
            v = int(clause_vars[i, j])
            s = int(clause_signs[i, j])
            if is_lit_satisfied(assign, v, s):
                return True
        return False

    def get_unsat_count(assign):
        count = 0
        for i in range(m):
            if not is_clause_satisfied(assign, i):
                count += 1
        return count

    rng = random.Random(42)
    
    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 300
    base_steps = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        assign = np.zeros(n, dtype=np.bool_)
        assign[:] = rng.random(n) < 0.5
        
        clause_sat = np.zeros(m, dtype=np.bool_)
        unsat_list = []
        unsat_count = 0
        
        for i in range(m):
            if is_clause_satisfied(assign, i):
                clause_sat[i] = True
            else:
                clause_sat[i] = False
                unsat_list.append(i)
                unsat_count += 1
                
        if unsat_count == 0:
            return assign.tolist()
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
            
        walk_prob = 0.3
        max_steps = base_steps + restart * 2000
        if restart > 100:
            max_steps *= 2
            
        step = 0
        no_improve = 0
        last_unsat = unsat_count
        
        while step < max_steps:
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assign.tolist()
                
            if not unsat_list:
                return assign.tolist()
                
            # Select unsatisfied clause
            c_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            
            # Get variables in this clause
            c_vars = clause_vars[c_idx]
            
            # Calculate gain for each variable in the clause
            best_gain = float('inf')
            best_var = -1
            
            for j in range(3):
                v = int(c_vars[j])
                
                # Calculate gain of flipping v
                gain = 0
                for c2 in var_to_clauses[v]:
                    if clause_sat[c2]:
                        # Clause is satisfied. Check if v is the ONLY satisfied literal.
                        only_sat = True
                        for k in range(3):
                            if k == j:
                                continue
                            v2 = int(clause_vars[c2, k])
                            s2 = int(clause_signs[c2, k])
                            if is_lit_satisfied(assign, v2, s2):
                                only_sat = False
                                break
                        if only_sat:
                            gain -= 1
                    else:
                        # Clause is unsatisfied. 
                        # Flipping v satisfies this clause.
                        gain += 1
                        
                if gain < best_gain:
                    best_gain = gain
                    best_var = v
                    
            # Decide which variable to flip
            if rng.random() < walk_prob:
                # Walk step
                j = rng.randint(0, 2)
                var_idx = int(c_vars[j])
            else:
                var_idx = best_var
                
            # Flip var_idx
            assign[var_idx] = not assign[var_idx]
            
            # Update clause statuses
            for c2 in var_to_clauses[var_idx]:
                old_sat = clause_sat[c2]
                new_sat = is_clause_satisfied(assign, c2)
                
                if old_sat and not new_sat:
                    # Became unsatisfied
                    unsat_list.append(c2)
                    unsat_count += 1
                elif not old_sat and new_sat:
                    # Became satisfied
                    try:
                        unsat_list.remove(c2)
                    except ValueError:
                        pass
                    unsat_count -= 1
                    
                clause_sat[c2] = new_sat
                
            step += 1
            
            # Adapt walk_prob
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve = 0
                if walk_prob > 0.2:
                    walk_prob = max(0.2, walk_prob * 0.95)
            else:
                no_improve += 1
                if no_improve > 100:
                    walk_prob = min(0.8, walk_prob * 1.1)
                if no_improve > 500:
                    break # Restart
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
                
    # Final check
    if best_assign is not None:
        if get_unsat_count(best_assign) == 0:
            return best_assign.tolist()
            
    return best_assign.tolist() if best_assign is not None else [False] * n
