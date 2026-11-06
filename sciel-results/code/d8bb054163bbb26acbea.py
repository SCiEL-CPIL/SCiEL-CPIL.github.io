"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    timeout = 315.0  # Leave margin for overhead (330s limit)
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays for fast vectorized evaluation
    # clause_signs: shape (m, 3), values are +1 or -1
    # clause_vars: shape (m, 3), values are 0-based variable indices
    clause_signs = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_signs[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause incidence for fast counting
    # For each variable, track which clauses contain it (as positive or negative)
    var_pos_clauses = [[] for _ in range(n)]  # clauses where var appears positively
    var_neg_clauses = [[] for _ in range(n)]  # clauses where var appears negatively
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                var_pos_clauses[v].append(i)
            else:
                var_neg_clauses[v].append(i)
    
    def time_left():
        return timeout - (time.time() - start_time)
    
    def evaluate(assignment):
        """Return number of satisfied clauses using vectorized ops."""
        # assignment: numpy array of bools, length n
        # For each clause, check if any literal is satisfied
        # lit_val = assignment[var] if sign==+1 else not assignment[var]
        # satisfied if any lit_val is True
        
        # Get assignment values for all clause literals
        # signs: +1 means we want assignment[var] to be True
        #         -1 means we want assignment[var] to be False
        vals = assignment[clause_vars]  # shape (m, 3)
        # A literal is satisfied if: (sign==1 AND val==True) OR (sign==-1 AND val==False)
        # = (sign * (val ? 1 : -1)) > 0  ... let's think differently
        # satisfied_literal = (clause_signs == 1) & vals | (clause_signs == -1) & ~vals
        sat_lits = ((clause_signs == 1) & vals) | ((clause_signs == -1) & (~vals))
        # A clause is satisfied if any literal in it is satisfied
        sat_clauses = sat_lits.any(axis=1)
        return sat_clauses.sum()
    
    def is_satisfied(assignment):
        return evaluate(assignment) == m
    
    # Restart-based WalkSAT with adaptive parameters
    best_assignment = None
    best_score = -1
    
    restart = 0
    while time_left() > 2.0:
        restart += 1
        
        # Random initial assignment
        rng_seed = restart * 12345 + 67890
        rng = np.random.RandomState(rng_seed)
        assignment = rng.rand(n) < 0.5
        
        score = evaluate(assignment)
        if score > best_score:
            best_score = score
            best_assignment = assignment.copy()
        
        if score == m:
            return [bool(x) for x in assignment]
        
        # WalkSAT loop
        flip_budget = min(5000 + restart * 500, 50000)
        step = 0
        while step < flip_budget and time_left() > 1.0:
            step += 1
            
            # Check if satisfied
            if score == m:
                return [bool(x) for x in assignment]
            
            # Find unsatisfied clauses
            vals = assignment[clause_vars]
            sat_lits = ((clause_signs == 1) & vals) | ((clause_signs == -1) & (~vals))
            sat_clauses = sat_lits.any(axis=1)
            unsat_idx = np.where(~sat_clauses)[0]
            
            if len(unsat_idx) == 0:
                return [bool(x) for x in assignment]
            
            # Select an unsatisfied clause (weighted: prefer those with fewer satisfied literals)
            # Count satisfied literals per unsat clause
            unsat_sat_counts = sat_lits[unsat_idx].sum(axis=1)
            # For 3-CNF unsat clauses, sat_counts can be 0 (all false) 
            # Actually if clause is unsat, sat_count must be 0 for all 3 literals
            # So all unsat clauses have 0 satisfied literals
            # Pick randomly among unsat clauses
            clause_idx = unsat_idx[rng.randint(len(unsat_idx))]
            
            # Select variable to flip from this clause
            # Use score-based heuristic: prefer variables that would satisfy most clauses
            # For each of the 3 literals in the clause, compute the impact of flipping
            clause_lits_vars = clause_vars[clause_idx]
            clause_lits_signs = clause_signs[clause_idx]
            
            best_var = -1
            best_gain = -1
            
            for j in range(3):
                v = clause_lits_vars[j]
                s = clause_lits_signs[j]
                
                # Current state of variable v
                current_val = assignment[v]
                new_val = not current_val
                
                # Compute gain: how many currently unsatisfied clauses become satisfied,
                # minus how many currently satisfied clauses become unsatisfied
                # Only need to check clauses containing v
                
                # Clauses where v appears positively
                pos_clauses = var_pos_clauses[v]
                neg_clauses = var_neg_clauses[v]
                
                gain = 0
                # If we flip v, clauses where v is positive: 
                #   if v was False -> True, those clauses may become satisfied
                #   if v was True -> False, those clauses may become unsatisfied
                # Clauses where v is negative:
                #   if v was True -> False, those clauses may become satisfied
                #   if v was False -> True, those clauses may become unsatisfied
                
                if current_val == False:  # flipping to True
                    # Positive clauses: v becomes True, may satisfy
                    for ci in pos_clauses:
                        if not sat_clauses[ci]:
                            gain += 1
                    # Negative clauses: v becomes True, may unsatisfy
                    for ci in neg_clauses:
                        if sat_clauses[ci]:
                            gain -= 1
                else:  # current_val == True, flipping to False
                    # Positive clauses: v becomes False, may unsatisfy
                    for ci in pos_clauses:
                        if sat_clauses[ci]:
                            gain -= 1
                    # Negative clauses: v becomes False, may satisfy
                    for ci in neg_clauses:
                        if not sat_clauses[ci]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
            
            # With some probability, pick randomly to escape local optima
            if rng.rand() < 0.15:
                best_var = clause_lits_vars[rng.randint(3)]
            
            if best_var < 0:
                best_var = clause_lits_vars[rng.randint(3)]
            
            # Flip
            assignment[best_var] = not assignment[best_var]
            
            # Update score incrementally
            # Recompute affected clauses
            # For simplicity with numpy, recompute full score periodically
            if step % 10 == 0:
                score = evaluate(assignment)
                if score > best_score:
                    best_score = score
                    best_assignment = assignment.copy()
            else:
                # Quick check: just update score by checking affected clauses
                # This is an approximation; full recompute every 10 steps
                # For now, just recompute
                score = evaluate(assignment)
                if score > best_score:
                    best_score = score
                    best_assignment = assignment.copy()
    
    # Return best found
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
