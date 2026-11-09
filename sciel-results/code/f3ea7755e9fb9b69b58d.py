import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess
    var_clauses = [[] for _ in range(n)]
    # Store literals as (var, sign) where sign is 1 for pos, 0 for neg?
    # Or just use the original literals.
    # Let's use 0-indexed vars and boolean signs.
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_in_clause = []
        signs_in_clause = []
        for lit in clause:
            v = abs(lit) - 1
            vars_in_clause.append(v)
            signs_in_clause.append(lit > 0)
            var_clauses[v].append(len(clause_vars))
        clause_vars.append(vars_in_clause)
        clause_signs.append(signs_in_clause)

    def random_assignment():
        return [random.random() < 0.5 for _ in range(n)]
    
    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 100
    max_steps_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = random_assignment()
        
        # Initialize state
        clause_sat = [False] * m
        true_counts = [0] * m
        true_vars = [-1] * m # If true_counts[i]==1, this is the var index
        
        for i in range(m):
            count = 0
            tv = -1
            for v, s in zip(clause_vars[i], clause_signs[i]):
                if assignment[v] == s:
                    count += 1
                    tv = v
            true_counts[i] = count
            clause_sat[i] = (count > 0)
            if count == 1:
                true_vars[i] = tv
                
        unsat_count = sum(1 for s in clause_sat if not s)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        no_improve_steps = 0
        last_unsat_count = unsat_count
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
            # Find unsat clause
            # To avoid O(m) scan, we can keep a list of unsat indices, but updating it is tricky.
            # m is small, so scan is fine.
            unsat_indices = [i for i in range(m) if not clause_sat[i]]
            if not unsat_indices:
                return assignment
                
            c = random.choice(unsat_indices)
            
            # Heuristic: MinConflicts
            best_lit_var = -1
            best_score = float('inf')
            
            for idx, v in enumerate(clause_vars[c]):
                s = clause_signs[c][idx]
                
                # Calculate gain and loss
                gain = 0
                loss = 0
                
                for c_idx in var_clauses[v]:
                    if c_idx == c:
                        continue
                    if not clause_sat[c_idx]:
                        # This clause is unsat.
                        # v is in it. Since it's unsat, the literal for v is false.
                        # Flipping v will make it true, satisfying the clause.
                        gain += 1
                    else:
                        # This clause is sat.
                        # If v is the unique true literal, flipping it makes it unsat.
                        if true_counts[c_idx] == 1 and true_vars[c_idx] == v:
                            loss += 1
                            
                score = loss - gain
                if score < best_score:
                    best_score = score
                    best_lit_var = v
                    
            # Walk probability
            if random.random() < 0.2:
                # Pick random var from clause c
                best_lit_var = random.choice(clause_vars[c])
                
            v_to_flip = best_lit_var
            
            # Perform flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update state for clauses containing v_to_flip
            for c_idx in var_clauses[v_to_flip]:
                # Check if the literal for v_to_flip in c_idx was true
                # Find index of v_to_flip in clause_vars[c_idx]
                # This is O(3)
                was_true = False
                for idx, v in enumerate(clause_vars[c_idx]):
                    if v == v_to_flip:
                        if assignment[v_to_flip] == clause_signs[c_idx][idx]:
                            was_true = True
                        break
                        
                # Wait, assignment[v_to_flip] is already updated to new_val.
                # We need to know if it WAS true BEFORE the flip.
                # old_val is the value before flip.
                # The sign is clause_signs[c_idx][idx].
                # So was_true = (old_val == clause_signs[c_idx][idx])
                
                # Let's re-evaluate was_true correctly
                for idx, v in enumerate(clause_vars[c_idx]):
                    if v == v_to_flip:
                        sign = clause_signs[c_idx][idx]
                        was_true = (old_val == sign)
                        break
                        
                if was_true:
                    true_counts[c_idx] -= 1
                    if true_counts[c_idx] == 0:
                        clause_sat[c_idx] = False
                        unsat_count += 1
                        true_vars[c_idx] = -1
                    elif true_counts[c_idx] == 1:
                        # Find the remaining true var
                        for idx2, v2 in enumerate(clause_vars[c_idx]):
                            if v2 != v_to_flip and assignment[v2] == clause_signs[c_idx][idx2]:
                                true_vars[c_idx] = v2
                                break
                else:
                    true_counts[c_idx] += 1
                    if true_counts[c_idx] == 1:
                        clause_sat[c_idx] = True
                        unsat_count -= 1
                        true_vars[c_idx] = v_to_flip
                    # If >1, true_vars remains -1 or whatever, doesn't matter for loss check (which checks ==1)
                        
            if unsat_count < last_unsat_count:
                last_unsat_count = unsat_count
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            if no_improve_steps > 50:
                break
                
    if best_assignment is None:
        # Fallback to random
        return [random.random() < 0.5 for _ in range(n)]
        
    return best_assignment
