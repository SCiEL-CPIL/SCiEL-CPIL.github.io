"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    timeout = 325.0  # Leave margin before 330s kill
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays
    clause_signs = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j in range(3):
            lit = c[j]
            clause_signs[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause index
    var_clause_map = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clause_map[v].append((i, j))
    
    def evaluate_assignment(assignment):
        lit_assign = assignment[clause_vars]
        lit_true = ((clause_signs == 1) & lit_assign) | ((clause_signs == -1) & (~lit_assign))
        clause_sat = lit_true.any(axis=1)
        return clause_sat, lit_true
    
    # Main search loop with restarts
    best_assignment = None
    best_unsat_count = m + 1
    
    rng = np.random.RandomState(42)
    
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization
        assignment = rng.rand(n) < 0.5
        
        clause_sat, lit_true = evaluate_assignment(assignment)
        unsat_count = int((~clause_sat).sum())
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # Adaptive flip budget
        flip_budget = 3000
        
        for step in range(flip_budget):
            if time.time() - start_time > timeout:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            clause_idx = unsat_indices[rng.randint(len(unsat_indices))]
            
            # Score each of the 3 variables in this clause
            best_var = -1
            best_score = -1000000
            best_rand = -1.0
            
            for pos in range(3):
                v = clause_vars[clause_idx, pos]
                
                # Compute score: net change in satisfied clauses if we flip v
                gain = 0
                loss = 0
                
                for (ci, pi) in var_clause_map[v]:
                    current_true = lit_true[ci, pi]
                    if current_true:
                        # After flip, this literal becomes false
                        # Check if clause ci remains satisfied
                        if ci == clause_idx:
                            # This is the clause we picked; after flip this lit becomes true
                            # But wait, we're evaluating the effect of flipping v
                            # The literal at (ci, pi) currently is true, after flip becomes false
                            # But for ci == clause_idx, this literal is currently FALSE (unsat clause)
                            # Contradiction - let me re-examine
                            pass
                        # For ci != clause_idx:
                        # If current_true is True, after flip it's False
                        # Clause ci might become unsatisfied if no other lit is true
                        other_true = lit_true[ci, :].copy()
                        other_true[pi] = False
                        if not other_true.any():
                            loss += 1
                    else:
                        # current_true is False, after flip becomes True
                        # If clause ci is currently unsatisfied, it becomes satisfied
                        if not clause_sat[ci]:
                            gain += 1
                
                score = gain - loss
                rand_val = rng.random()
                
                if score > best_score or (score == best_score and rand_val > best_rand):
                    best_score = score
                    best_rand = rand_val
                    best_var = v
            
            if best_var < 0:
                break
            
            # Flip best_var
            assignment[best_var] = not assignment[best_var]
            
            # Incremental update
            for (ci, pi) in var_clause_map[best_var]:
                lit_true[ci, pi] = not lit_true[ci, pi]
                clause_sat[ci] = lit_true[ci, :].any()
            
            unsat_count = int((~clause_sat).sum())
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
    
    if best_assignment is not None:
        return best_assignment.tolist()
    
    return [True] * n
# EVOLVE-BLOCK-END
