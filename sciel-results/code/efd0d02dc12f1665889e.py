import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285
    m = len(clauses)
    
    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_ = []
        signs = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            vars_.append(var_idx)
            signs.append(sign)
        clause_vars.append(np.array(vars_, dtype=np.int32))
        clause_signs.append(np.array(signs, dtype=np.int32))
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)
            
    def is_clause_satisfied(ci, assignment):
        vals = assignment[clause_vars[ci]]
        signs = clause_signs[ci]
        return np.any(vals == (signs == 1))
        
    def initial_eval(assignment):
        clause_sat = np.zeros(m, dtype=bool)
        unsat_indices = []
        unsat_set = set()
        for i in range(m):
            if is_clause_satisfied(i, assignment):
                clause_sat[i] = True
            else:
                unsat_indices.append(i)
                unsat_set.add(i)
        return clause_sat, unsat_indices, unsat_set, m - len(unsat_indices)
        
    rng = random.Random(12345)
    
    best_assignment = None
    best_score = -1
    
    max_restarts = 500 
    max_steps = 20000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        clause_sat, unsat_indices, unsat_set, current_score = initial_eval(assignment)
        
        if not unsat_indices:
            return [bool(x) for x in assignment]
            
        local_best = assignment.copy()
        local_best_score = current_score
        
        initial_noise = 0.5
        final_noise = 0.05
        cooling_rate = (final_noise / initial_noise) ** (1.0 / max_steps)
        noise = initial_noise
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not unsat_indices:
                return [bool(x) for x in assignment]
            
            clause_idx = unsat_indices[rng.randrange(len(unsat_indices))]
            c_vars = clause_vars[clause_idx]
            
            best_gain = -1000
            best_k = -1
            
            for k in range(3):
                var_idx = c_vars[k]
                
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                delta = 0
                changes = []
                
                for ci in var_to_clauses[var_idx]:
                    new_sat = is_clause_satisfied(ci, assignment)
                    if new_sat != clause_sat[ci]:
                        changes.append((ci, clause_sat[ci], new_sat))
                        if new_sat:
                            delta += 1
                        else:
                            delta -= 1
                            
                new_score = current_score + delta
                assignment[var_idx] = old_val
                
                if new_score > best_gain:
                    best_gain = new_score
                    best_k = k
                    
            if rng.random() < noise:
                k = rng.randrange(3)
            else:
                k = best_k
                
            var_idx = c_vars[k]
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            delta = 0
            for ci in var_to_clauses[var_idx]:
                new_sat = is_clause_satisfied(ci, assignment)
                if new_sat != clause_sat[ci]:
                    if new_sat:
                        # Clause became satisfied
                        delta += 1
                        unsat_set.discard(ci)
                        # Remove from unsat_indices list
                        # Find index and swap-remove
                        if ci in unsat_set: # It should be there
                            pass
                        # Efficient removal from list:
                        # Since m is small, linear search is okay, but let's try to be smart.
                        # Actually, just rebuilding the list from set is O(m) which is fine?
                        # No, we are in a tight loop.
                        # Let's just use a list and do linear search for removal.
                        # m is ~300. 300 ops is nothing.
                        try:
                            idx = unsat_indices.index(ci)
                            unsat_indices[idx] = unsat_indices[-1]
                            unsat_indices.pop()
                        except ValueError:
                            pass
                    else:
                        # Clause became unsatisfied
                        delta -= 1
                        unsat_set.add(ci)
                        unsat_indices.append(ci)
                    clause_sat[ci] = new_sat
                    
            current_score += delta
            
            if current_score > local_best_score:
                local_best_score = current_score
                local_best = assignment.copy()
                
            noise *= cooling_rate
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_assignment = local_best.copy()
            
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    
    return [random.random() < 0.5 for _ in range(n)]
