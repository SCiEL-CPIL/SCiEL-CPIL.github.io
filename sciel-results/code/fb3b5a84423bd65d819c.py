import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Precompute: for each clause, store literals as (var_idx, sign) where sign=+1 or -1
    # Also build var_to_clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []  # list of lists of (var_idx, sign)
    for i, c in enumerate(clauses):
        var_list = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            var_list.append((v, s))
            var_to_clauses[v].append(i)
        clause_vars.append(var_list)
    
    # For fast gain computation: precompute for each variable, the list of (clause_idx, sign)
    var_clause_sigs = [[] for _ in range(n)]
    for v in range(n):
        for ci in var_to_clauses[v]:
            for (vv, ss) in clause_vars[ci]:
                if vv == v:
                    var_clause_sigs[v].append((ci, ss))
                    break
    
    rng = np.random.RandomState(42)
    
    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize random assignment as int array: 0 or 1
        assign = rng.randint(0, 2, size=n).astype(np.int8)
        
        # Compute clause satisfaction vectorized
        # For each clause, check if any literal is satisfied
        # lit_sat = (sign * (2*assign[var] - 1)) > 0
        # i.e., sign*(2*val-1) > 0 means satisfied
        
        clause_sat = np.zeros(m, dtype=np.bool_)
        for i in range(m):
            sat = False
            for (v, s) in clause_vars[i]:
                if s * (2 * int(assign[v]) - 1) > 0:
                    sat = True
                    break
            clause_sat[i] = sat
        
        unsat_count = int(np.sum(~clause_sat))
        
        if unsat_count == 0:
            return assign.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
        
        # Adaptive walk probability
        walk_prob = 0.3
        no_improve_steps = 0
        last_unsat = unsat_count
        step = 0
        max_steps = 20000
        
        while step < max_steps:
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assign.tolist()
            
            # Select random unsatisfied clause
            unsat_mask = ~clause_sat
            if not unsat_mask.any():
                return assign.tolist()
            
            unsat_indices = np.where(unsat_mask)[0]
            c_idx = int(rng.choice(unsat_indices))
            
            c_vars = clause_vars[c_idx]
            
            # Choose variable to flip
            if rng.random() < walk_prob:
                # Random walk step
                var_idx, sign = c_vars[int(rng.randint(0, len(c_vars)))]
            else:
                # Greedy: pick var in c that minimizes gain
                best_gain = float('inf')
                best_var = None
                best_sign = None
                
                for (v, s) in c_vars:
                    # Compute gain for flipping v
                    old_val = int(assign[v])
                    new_val = 1 - old_val
                    
                    gain = 0
                    for (ci, cs) in var_clause_sigs[v]:
                        if clause_sat[ci]:
                            # Currently satisfied: becomes unsat only if v's lit was the only sat lit
                            old_lit_sat = cs * (2 * old_val - 1) > 0
                            new_lit_sat = cs * (2 * new_val - 1) > 0
                            if old_lit_sat and not new_lit_sat:
                                # Check if any other lit in ci is satisfied
                                others_sat = False
                                for (ov, os) in clause_vars[ci]:
                                    if ov != v:
                                        if os * (2 * int(assign[ov]) - 1) > 0:
                                            others_sat = True
                                            break
                                if not others_sat:
                                    gain += 1
                        else:
                            # Currently unsat: becomes sat if v's new lit is satisfied
                            new_lit_sat = cs * (2 * new_val - 1) > 0
                            if new_lit_sat:
                                gain -= 1
                    
                    if gain < best_gain:
                        best_gain = gain
                        best_var = v
                        best_sign = s
                
                if best_var is not None:
                    var_idx = best_var
                else:
                    var_idx, sign = c_vars[int(rng.randint(0, len(c_vars)))]
            
            # Perform flip
            old_val = int(assign[var_idx])
            assign[var_idx] = 1 - old_val
            new_val = 1 - old_val
            
            # Update clause statuses for clauses containing var_idx
            for ci in var_to_clauses[var_idx]:
                old_sat = clause_sat[ci]
                # Recompute satisfaction
                new_sat = False
                for (v, s) in clause_vars[ci]:
                    if s * (2 * int(assign[v]) - 1) > 0:
                        new_sat = True
                        break
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                clause_sat[ci] = new_sat
            
            step += 1
            
            # Adaptive walk probability
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
                walk_prob = max(0.1, walk_prob * 0.95)
            else:
                no_improve_steps += 1
                if no_improve_steps % 500 == 0:
                    walk_prob = min(0.8, walk_prob * 1.1)
            
            if no_improve_steps > 5000:
                break
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
        
        # End of inner loop
    
    if best_assign is not None:
        return best_assign.tolist()
    return [False] * n
