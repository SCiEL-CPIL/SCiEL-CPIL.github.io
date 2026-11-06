"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s kill
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays
    # clause_data: shape (m, 3), values are +/-1 (sign of literal)
    # clause_vars: shape (m, 3), values are 0-based variable indices
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause incidence for faster lookups
    # var_clauses[v] = list of clause indices containing variable v
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    # Convert to numpy arrays for faster access where possible
    # But keep as lists of arrays for flexibility
    var_clauses_arr = [np.array(vc, dtype=np.int32) for vc in var_clauses]
    
    def evaluate(assignment):
        """Return number of unsatisfied clauses and the unsatisfied clause indices."""
        # assignment: numpy bool array of length n
        # For each clause, check if any literal is satisfied
        # sign * (assignment[var] if sign>0 else not assignment[var]) == 1 means satisfied
        # Let's compute: for each clause, lit_val = clause_data * (2*assignment[clause_vars] - 1)
        # A literal is satisfied if lit_val == 1
        # Clause is satisfied if any lit_val == 1
        
        # Convert assignment to int array: True -> 1, False -> -1
        assign_int = (2 * assignment.astype(np.int8) - 1)
        
        # For each clause position, compute the literal value
        # lit_vals shape: (m, 3)
        lit_vals = clause_data * assign_int[clause_vars]
        
        # Clause is satisfied if max(lit_vals, axis=1) == 1
        # Actually, a clause is satisfied if ANY literal is true (value == 1)
        # Since lit_vals are +1 or -1, clause satisfied if np.max(lit_vals, axis=1) == 1
        clause_satisfied = np.max(lit_vals, axis=1) == 1
        
        unsat_count = int(np.sum(~clause_satisfied))
        unsat_indices = np.where(~clause_satisfied)[0]
        
        return unsat_count, unsat_indices
    
    def find_best_flip(assignment, unsat_indices, rng):
        """Find the variable to flip that minimizes unsat count (with random tie-breaking)."""
        if len(unsat_indices) == 0:
            return None
        
        # Pick a random unsatisfied clause
        clause_idx = rng.choice(unsat_indices)
        clause_vars_local = clause_vars[clause_idx]
        clause_data_local = clause_data[clause_idx]
        
        # For each variable in this clause, compute the score if we flip it
        best_var = -1
        best_score = float('inf')
        best_tiebreak = float('inf')
        
        for j in range(3):
            v = clause_vars_local[j]
            sign = clause_data_local[j]
            
            # Current value of variable v
            current_val = assignment[v]
            
            # If we flip v, what happens to all clauses containing v?
            # For each clause containing v, compute how many literals become satisfied/unsatisfied
            # This is expensive, so let's use a simpler heuristic:
            # Score = number of currently unsatisfied clauses containing v that would become satisfied
            #         - number of currently satisfied clauses containing v that would become unsatisfied
            
            # Get all clauses containing v
            v_clauses = var_clauses_arr[v]
            if len(v_clauses) == 0:
                continue
            
            # For each such clause, check if v's literal is the only satisfied literal
            # and if flipping v would satisfy or break the clause
            
            # Compute current state of clauses containing v
            # For clause c containing v at position pos:
            #   lit_val = clause_data[c, pos] * assign_int[v]
            #   If we flip v, new_lit_val = -lit_val
            #   Clause becomes satisfied if it wasn't and new_lit_val == 1
            #   Clause becomes unsatisfied if it was and lit_val == 1 and no other literal is satisfied
            
            assign_int = (2 * assignment.astype(np.int8) - 1)
            
            # Get the positions of v in each clause
            # For efficiency, let's just compute the delta
            satisfied_before = np.max(clause_data[v_clauses] * assign_int[clause_vars[v_clauses]], axis=1) == 1
            
            # After flipping v:
            # For each clause, we need to recompute. But we can approximate:
            # The change is: for clauses where v's literal was the ONLY satisfied one, they become unsat
            # For clauses where v's literal was the only way to satisfy (and it was unsat), they become sat
            
            # This is getting complex. Let's use a simpler WalkSAT-style approach:
            # Just flip a random variable from a random unsatisfied clause, with some bias.
            pass
        
        # Simpler approach: use the standard WalkSAT heuristic
        # Pick a random variable from the unsatisfied clause
        # With probability p, pick the variable that appears most in unsatisfied clauses
        # With probability 1-p, pick randomly
        
        # For now, let's use a simple weighted approach:
        # Score each variable by how many unsatisfied clauses it appears in
        var_scores = {}
        for ci in unsat_indices:
            for j in range(3):
                v = clause_vars[ci, j]
                var_scores[v] = var_scores.get(v, 0) + 1
        
        # Pick from the three variables in the chosen clause, weighted by score
        candidates = []
        weights = []
        for j in range(3):
            v = clause_vars_local[j]
            candidates.append(v)
            weights.append(var_scores.get(v, 1))
        
        # Randomly pick with weights
        total_weight = sum(weights)
        r = rng.random() * total_weight
        cumsum = 0
        chosen_var = candidates[0]
        for k in range(len(candidates)):
            cumsum += weights[k]
            if r <= cumsum:
                chosen_var = candidates[k]
                break
        
        return chosen_var
    
    # Main search loop with restarts
    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random(42)
    
    while time.time() - start_time < deadline:
        # Initialize with random assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_count, unsat_indices = evaluate(assignment)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
        
        # WalkSAT loop
        max_steps = 5000
        step = 0
        while step < max_steps and time.time() - start_time < deadline:
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Find variable to flip
            var_to_flip = find_best_flip(assignment, unsat_indices, rng)
            if var_to_flip is None:
                break
            
            # Flip the variable
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Re-evaluate
            old_unsat_count = unsat_count
            unsat_count, unsat_indices = evaluate(assignment)
            
            step += 1
        
        # Check if we solved it
        if unsat_count == 0:
            return assignment.tolist()
        
        # Restart with a new random assignment
        # Occasionally, bias towards the best found so far
        if best_assignment is not None and rng.random() < 0.1:
            assignment = best_assignment.copy()
        else:
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [True] * n
# EVOLVE-BLOCK-END
