import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess: convert to 0-based variables and signs
    m = len(clauses)
    clause_vars = []    # clause_vars[i] = list of var indices (0-based)
    clause_signs = []   # clause_signs[i] = list of signs (+1/-1)
    
    for c in clauses:
        vars_list = []
        signs_list = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            vars_list.append(var)
            signs_list.append(sign)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    # For each variable, store list of (clause_idx, sign) where it appears
    var_occurrences = [[] for _ in range(n)]
    for i in range(m):
        for j, var in enumerate(clause_vars[i]):
            var_occurrences[var].append((i, clause_signs[i][j]))
    
    # Build adjacency: for each variable, list of clause indices where it appears
    # and the sign in each clause
    var_clause_list = [[] for _ in range(n)]
    var_sign_list = [[] for _ in range(n)]
    for i in range(m):
        for j, var in enumerate(clause_vars[i]):
            var_clause_list[var].append(i)
            var_sign_list[var].append(clause_signs[i][j])
    
    rng = random.Random()
    max_time = 285.0  # Safety margin
    start_time = time.time()
    
    def compute_sat_counts(assignment):
        """Compute sat_count for each clause using vectorized operations."""
        sat_counts = np.zeros(m, dtype=np.int32)
        for i in range(m):
            for j, var in enumerate(clause_vars[i]):
                sign = clause_signs[i][j]
                if assignment[var] == (sign == 1):
                    sat_counts[i] += 1
        return sat_counts
    
    # Track best solution found
    best_assignment = None
    best_unsat = m + 1
    
    def update_best(assignment, unsat_count):
        nonlocal best_assignment, best_unsat
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
    
    # WalkSAT with restarts and optimized delta computation
    restart_limit = 500
    walk_prob = 0.25
    
    while time.time() - start_time < max_time:
        # Random restart
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        sat_counts = compute_sat_counts(assignment)
        unsat_count = int(np.sum(sat_counts == 0))
        update_best(assignment, unsat_count)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        steps = 0
        while steps < restart_limit:
            if unsat_count == 0:
                return assignment.tolist()
            
            # Check time periodically (every 100 steps to reduce overhead)
            if steps % 100 == 0 and time.time() - start_time > max_time:
                break
            
            # Find an unsatisfied clause
            unsat_indices = np.where(sat_counts == 0)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            c_idx = rng.choice(unsat_indices.tolist())
            
            # Choose which variable to flip in this clause
            if rng.random() < walk_prob:
                # Random flip
                j = rng.randrange(len(clause_vars[c_idx]))
                var = clause_vars[c_idx][j]
            else:
                # Greedy: compute delta for each variable in the clause
                # Delta = (# unsat clauses where literal becomes true) - (# sat clauses with exactly 1 true lit where literal becomes false)
                best_var = -1
                best_delta = -10**9
                
                for j, var in enumerate(clause_vars[c_idx]):
                    old_val = assignment[var]
                    # new_val = not old_val
                    
                    delta = 0
                    for ci, si in var_occurrences[var]:
                        old_lit_sat = old_val == (si == 1)
                        new_lit_sat = not old_val == (si == 1)
                        
                        if old_lit_sat and not new_lit_sat:
                            # Literal was satisfied, becomes unsatisfied
                            # Clause becomes unsat only if it had exactly 1 satisfied literal
                            if sat_counts[ci] == 1:
                                delta -= 1
                        elif not old_lit_sat and new_lit_sat:
                            # Literal was unsatisfied, becomes satisfied
                            # Clause becomes sat only if it was unsat (0 satisfied literals)
                            if sat_counts[ci] == 0:
                                delta += 1
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_var = var
                
                if best_var != -1:
                    var = best_var
                else:
                    j = rng.randrange(len(clause_vars[c_idx]))
                    var = clause_vars[c_idx][j]
            
            # Apply the flip
            assignment[var] = not assignment[var]
            
            # Update sat_counts incrementally
            for ci, si in var_occurrences[var]:
                # The literal (var, si) flipped its truth value
                # If it was true, it becomes false: sat_counts[ci] -= 1
                # If it was false, it becomes true: sat_counts[ci] += 1
                # We can determine this from the new assignment value
                if assignment[var] == (si == 1):
                    sat_counts[ci] += 1
                else:
                    sat_counts[ci] -= 1
            
            # Update unsat_count
            unsat_count = int(np.sum(sat_counts == 0))
            update_best(assignment, unsat_count)
            
            steps += 1
        
        # Increase restart limit
        restart_limit = int(restart_limit * 1.3)
        if restart_limit > 5000:
            restart_limit = 5000
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
