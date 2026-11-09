"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Build NumPy representation
    # vars_matrix: (m, 3) array of 0-based variable indices
    # signs_matrix: (m, 3) array of +1/-1
    vars_matrix = np.empty((m, 3), dtype=np.int32)
    signs_matrix = np.empty((m, 3), dtype=np.int8)
    
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            vars_matrix[ci, li] = abs(lit) - 1
            signs_matrix[ci, li] = 1 if lit > 0 else -1
    
    # Build var_to_clauses: for each variable, list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            v = int(vars_matrix[ci, li])
            s = int(signs_matrix[ci, li])
            var_to_clauses[v].append((ci, s))
    
    # Pre-compute clause satisfaction check helper
    def is_satisfied(assign):
        """assign: numpy bool array of length n. Returns True if all clauses satisfied."""
        # For each clause, check if any literal is satisfied
        # assign[vars_matrix] gives the truth value of each variable in each clause
        # A literal (var, sign) is satisfied if assign[var] == (sign == 1)
        # sign == 1 means we want True, sign == -1 means we want False
        # So literal is satisfied if (sign == 1 and assign[var]) or (sign == -1 and not assign[var])
        # Which is: assign[var] == (sign > 0)
        # In numpy: assign[vars_matrix] == (signs_matrix > 0)
        lit_sat = (assign[vars_matrix] == (signs_matrix > 0))
        clause_sat = lit_sat.any(axis=1)
        return clause_sat.all()
    
    def count_unsatisfied(assign):
        """Return number of unsatisfied clauses."""
        lit_sat = (assign[vars_matrix] == (signs_matrix > 0))
        clause_sat = lit_sat.any(axis=1)
        return int((~clause_sat).sum())
    
    def get_unsat_clauses(assign):
        """Return list of indices of unsatisfied clauses."""
        lit_sat = (assign[vars_matrix] == (signs_matrix > 0))
        clause_sat = lit_sat.any(axis=1)
        return np.where(~clause_sat)[0]
    
    # WalkSAT with efficient bookkeeping and restarts
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    best_assign = None
    best_unsat = m + 1
    
    # Phase-biased initial assignment based on literal frequency
    # Count how many times each variable appears with each sign
    pos_count = np.zeros(n, dtype=np.int32)
    neg_count = np.zeros(n, dtype=np.int32)
    for ci in range(m):
        for li in range(3):
            v = int(vars_matrix[ci, li])
            s = int(signs_matrix[ci, li])
            if s > 0:
                pos_count[v] += 1
            else:
                neg_count[v] += 1
    
    # Phase bias: assign variable to the value that appears more often
    phase_bias = (pos_count > neg_count).astype(bool)
    
    max_restarts = 50
    restart_budget = 0.3  # fraction of time for each restart phase
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with phase-biased random assignment
        if restart == 0:
            assign = phase_bias.copy()
        else:
            # Start from best known assignment with some noise, or fresh random
            if best_assign is not None and rng.random() < 0.5:
                assign = best_assign.copy()
                # Flip 10% of variables
                flip_mask = np_rng.random(n) < 0.1
                assign[flip_mask] = ~assign[flip_mask]
            else:
                # Phase-biased random: with prob p use phase bias, else random
                p = 0.7
                assign = np.where(np_rng.random(n) < p, phase_bias, 
                                  np_rng.random(n) < 0.5)
        
        unsat_count = count_unsatisfied(assign)
        
        if unsat_count == 0:
            return [bool(x) for x in assign]
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign.copy()
        
        # WalkSAT loop for this restart
        max_steps = 10000
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            unsat_idx = get_unsat_clauses(assign)
            if len(unsat_idx) == 0:
                return [bool(x) for x in assign]
            
            unsat_count = len(unsat_idx)
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
            
            # Pick a random unsatisfied clause
            ci = int(unsat_idx[rng.randint(len(unsat_idx))])
            
            # Get the three literals in this clause
            clause_vars = vars_matrix[ci]
            clause_signs = signs_matrix[ci]
            
            # For each literal in the clause, compute the cost of flipping that variable
            # Cost = number of currently satisfied clauses that would become unsatisfied
            #       - number of currently unsatisfied clauses that would become satisfied
            best_var = -1
            best_cost = float('inf')
            
            for li in range(3):
                v = int(clause_vars[li])
                s = int(clause_signs[li])
                new_val = (s > 0)
                
                # Current value
                cur_val = bool(assign[v])
                
                if cur_val == new_val:
                    # Flipping would make this literal false, which is bad
                    # This variable is already satisfying this clause's literal
                    # But we only consider unsatisfied clauses, so this shouldn't happen
                    # Actually, in an unsatisfied clause, all literals are false
                    # So cur_val != new_val for all literals in the clause
                    pass
                
                # Compute cost of flipping variable v
                # For each clause containing v, check if flipping v changes its satisfaction
                cost = 0
                for (c_idx, c_sign) in var_to_clauses[v]:
                    # Current satisfaction of clause c_idx
                    # After flipping v to new_val, check if clause is still satisfied
                    
                    # Current state: is clause c_idx satisfied?
                    # We can check by looking at all three literals
                    cv = vars_matrix[c_idx]
                    cs = signs_matrix[c_idx]
                    
                    # Current satisfaction
                    cur_sat = False
                    for k in range(3):
                        if cv[k] == v:
                            # This is the variable we're flipping
                            # Current value is cur_val, literal sign is cs[k]
                            if cur_val == (cs[k] > 0):
                                cur_sat = True
                                break
                        else:
                            if bool(assign[int(cv[k])]) == (cs[k] > 0):
                                cur_sat = True
                                break
                    
                    # New satisfaction after flipping v to new_val
                    new_sat = False
                    for k in range(3):
                        if cv[k] == v:
                            if new_val == (cs[k] > 0):
                                new_sat = True
                                break
                        else:
                            if bool(assign[int(cv[k])]) == (cs[k] > 0):
                                new_sat = True
                                break
                    
                    if cur_sat and not new_sat:
                        cost += 1
                    elif not cur_sat and new_sat:
                        cost -= 1
                
                if cost < best_cost:
                    best_cost = cost
                    best_var = v
            
            # With some probability, pick a random variable from the clause (to escape local minima)
            if rng.random() < 0.15:
                li = rng.randint(3)
                best_var = int(clause_vars[li])
            
            # Flip the variable
            assign[best_var] = not assign[best_var]
        
        # Check if we found a solution
        if count_unsatisfied(assign) == 0:
            return [bool(x) for x in assign]
    
    # Return best assignment found
    if best_assign is not None:
        return [bool(x) for x in best_assign]
    
    # Fallback: random
    return [bool(x) for x in np_rng.random(n) < 0.5]


# EVOLVE-BLOCK-END
