import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute literal arrays for faster access
    # For each clause, store (var_indices, signs) as arrays
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        signs_in_c = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_in_c)
        clause_signs.append(signs_in_c)
    
    # Build variable to clauses mapping with sign info
    var_to_clauses = [[] for _ in range(n)]
    var_clause_sign = {}
    for i, c in enumerate(clauses):
        for l in c:
            var_idx = abs(l) - 1
            var_to_clauses[var_idx].append(i)
            var_clause_sign[(var_idx, i)] = 1 if l > 0 else -1
    
    # Precompute: for each variable, list of (clause_idx, sign)
    var_clause_info = [[] for _ in range(n)]
    for var_idx in range(n):
        for ci in var_to_clauses[var_idx]:
            var_clause_info[var_idx].append((ci, var_clause_sign[(var_idx, ci)]))
    
    # Precompute clause satisfaction status efficiently
    def compute_clause_sat(clause_idx, assignment):
        vars_c = clause_vars[clause_idx]
        signs_c = clause_signs[clause_idx]
        for j in range(len(vars_c)):
            v = vars_c[j]
            s = signs_c[j]
            if s == 1:
                if assignment[v]:
                    return True
            else:
                if not assignment[v]:
                    return True
        return False
    
    def compute_clause_sat_batch(assignment):
        clause_sat = [False] * num_clauses
        for i in range(num_clauses):
            clause_sat[i] = compute_clause_sat(i, assignment)
        return clause_sat
    
    rng = random.Random(42)
    
    max_restarts = 80
    start_time = time.time()
    deadline = start_time + 285.0
    
    best_assign = None
    best_unsat = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial clause satisfaction
        clause_sat = compute_clause_sat_batch(assignment)
        unsat_count = sum(1 for s in clause_sat if not s)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assignment[:]
        
        # Adaptive walk probability: start low, increase over time
        base_walk_prob = 0.25
        max_steps = 8000 + restart * 1500
        step = 0
        no_improve_steps = 0
        last_unsat = unsat_count
        
        # Track clause frequencies for weighted selection
        unsat_clause_freq = [0] * num_clauses
        for i in range(num_clauses):
            if not clause_sat[i]:
                unsat_clause_freq[i] = 1
        
        while step < max_steps:
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
            
            # Adaptive walk probability: increases as we get stuck
            walk_prob = base_walk_prob + 0.15 * min(1.0, no_improve_steps / 1000.0)
            
            # Select an unsatisfied clause using weighted selection
            unsat_indices = [i for i in range(num_clauses) if not clause_sat[i]]
            if not unsat_indices:
                return assignment
            
            # Weight by frequency of being unsatisfied
            weights = [unsat_clause_freq[i] for i in unsat_indices]
            total_w = sum(weights)
            if total_w == 0:
                c_idx = rng.choice(unsat_indices)
            else:
                r = rng.random() * total_w
                cum = 0
                c_idx = unsat_indices[0]
                for idx, w in zip(unsat_indices, weights):
                    cum += w
                    if r <= cum:
                        c_idx = idx
                        break
            
            c = clause_vars[c_idx]
            signs_c = clause_signs[c_idx]
            
            # Select variable to flip
            if rng.random() < walk_prob:
                # Walk step: random variable from clause
                j = rng.randint(0, len(c) - 1)
                var_idx = c[j]
            else:
                # Greedy step: compute gain for each variable
                best_gain = float('inf')
                best_var = None
                
                for j in range(len(c)):
                    var_idx = c[j]
                    sign_c = signs_c[j]
                    current_val = assignment[var_idx]
                    new_val = not current_val
                    
                    # Compute gain: how many clauses become unsat minus how many become sat
                    gain = 0
                    
                    for ci, sign in var_clause_info[var_idx]:
                        is_sat_now = clause_sat[ci]
                        
                        # Check if this literal satisfies the clause
                        if sign == 1:
                            new_lit_sat = new_val
                            old_lit_sat = current_val
                        else:
                            new_lit_sat = not new_val
                            old_lit_sat = not current_val
                        
                        if is_sat_now:
                            # Clause is currently satisfied
                            if old_lit_sat and not new_lit_sat:
                                # This literal was the only one satisfying it?
                                # Check other literals in the clause
                                vars_cc = clause_vars[ci]
                                signs_cc = clause_signs[ci]
                                others_sat = False
                                for k in range(len(vars_cc)):
                                    if vars_cc[k] != var_idx:
                                        v2 = vars_cc[k]
                                        s2 = signs_cc[k]
                                        if s2 == 1:
                                            if assignment[v2]:
                                                others_sat = True
                                                break
                                        else:
                                            if not assignment[v2]:
                                                others_sat = True
                                                break
                                if not others_sat:
                                    gain += 1
                        else:
                            # Clause is currently unsatisfied
                            if new_lit_sat:
                                gain -= 1
                    
                    if gain < best_gain:
                        best_gain = gain
                        best_var = var_idx
                
                if best_var is not None:
                    var_idx = best_var
                else:
                    j = rng.randint(0, len(c) - 1)
                    var_idx = c[j]
            
            # Perform the flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause statuses
            for ci in var_to_clauses[var_idx]:
                old_sat = clause_sat[ci]
                new_sat = compute_clause_sat(ci, assignment)
                
                if old_sat and not new_sat:
                    unsat_count += 1
                    unsat_clause_freq[ci] += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                    unsat_clause_freq[ci] = max(1, unsat_clause_freq[ci] - 1)
                else:
                    # No change in sat status, but update frequency slightly
                    if not old_sat:
                        unsat_clause_freq[ci] += 1
                    
                clause_sat[ci] = new_sat
            
            step += 1
            
            # Track progress
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            if no_improve_steps > 3000:
                break
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment[:]
        
        # End of inner loop for this restart
        
    return best_assign
