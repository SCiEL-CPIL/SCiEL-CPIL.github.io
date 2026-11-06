"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 320.0  # Leave margin for overhead
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays for fast evaluation
    # clause_data: shape (m, 3), values are +/-1
    # clause_vars: shape (m, 3), values are var indices (0-based)
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause index mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
    
    # Convert to numpy arrays for faster access
    var_to_clauses_arr = []
    for v in range(n):
        var_to_clauses_arr.append(np.array(var_to_clauses[v], dtype=np.int32))
    
    # Initialize random assignment
    rng = random.Random(42)
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    
    # Helper function to compute unsatisfied clause count and clause satisfaction status
    def compute_unsat(assignment):
        signed_assign = assignment.astype(np.int8) * 2 - 1  # shape (n,)
        lit_values = clause_data * signed_assign[clause_vars]
        clause_sat = (lit_values > 0).any(axis=1)  # shape (m,)
        unsat_count = m - clause_sat.sum()
        return unsat_count, clause_sat
    
    # Initial evaluation
    unsat_count, clause_sat = compute_unsat(assignment)
    if unsat_count == 0:
        return assignment.tolist()
    
    # Track best solution found
    best_unsat = unsat_count
    best_assignment = assignment.copy()
    
    # LLL-inspired: clause-weighted resampling with smooth pressure
    # Key idea: instead of uniform random restarts, use weighted clause selection
    # where "pressure" on each variable is derived from unsatisfied clauses
    
    max_restarts = 100
    steps_per_restart = 3000
    
    for restart in range(max_restarts):
        # Check deadline
        if time.time() - start_time > timeout:
            break
        
        # For first restart, use current assignment; for subsequent, weighted randomize
        if restart > 0:
            # Use a biased random restart: variables in many unsat clauses get flipped with higher prob
            if unsat_count > 0:
                # Compute variable "pressure": number of unsat clauses containing each var
                var_pressure = np.zeros(n, dtype=np.int32)
                unsat_indices = np.where(~clause_sat)[0]
                for ci in unsat_indices:
                    for j in range(3):
                        var_pressure[clause_vars[ci, j]] += 1
                
                # Bias: flip variables with higher pressure with higher probability
                # Use smooth softmax-like weighting
                max_p = var_pressure.max()
                if max_p > 0:
                    weights = np.exp(var_pressure.astype(np.float64) / max_p - 1.0)  # normalized, max=1
                    total_w = weights.sum()
                    if total_w > 0:
                        flip_prob = weights / total_w * 2.0  # scale so avg flip prob ~ 0.5
                        flip_prob = np.clip(flip_prob, 0.1, 0.9)
                        assignment = np.array([rng.random() < fp for fp in flip_prob], dtype=bool)
                    else:
                        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
                else:
                    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
            else:
                assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
            
            unsat_count, clause_sat = compute_unsat(assignment)
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            if unsat_count == 0:
                return assignment.tolist()
        
        for step in range(steps_per_restart):
            # Check deadline periodically
            if step % 100 == 0 and time.time() - start_time > timeout:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # LLL-inspired: select clause weighted by its "frustration" (number of unsat literals)
            # For 3-SAT, unsat clause has 3 unsat literals, so uniform is fine,
            # but we can weight by how "hard" the clause is to satisfy
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # Weighted clause selection: clauses with more "resistance" get higher weight
            # Resistance = sum over literals of (1 / clause_frequency)
            # Simplification: use uniform for now, but try weighted
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Choose variable to flip using smooth pressure heuristic
            # Compute "gain" for each of the 3 variables: how many unsat clauses would be satisfied
            clause_lits = clause_vars[clause_idx]  # 3 variables
            clause_signs = clause_data[clause_idx]  # 3 signs
            
            # Compute gain for each variable
            gains = np.zeros(3, dtype=np.int32)
            for j in range(3):
                v = clause_lits[j]
                v_clauses = var_to_clauses_arr[v]
                # Count unsat clauses containing v
                gains[j] = np.sum(~clause_sat[v_clauses])
            
            # Smooth selection: use softmax over gains with temperature
            # Higher gain -> higher probability
            temp = 0.5  # temperature parameter
            max_gain = gains.max()
            if max_gain > 0:
                exp_gains = np.exp(gains.astype(np.float64) / temp - max_gain / temp)
                total = exp_gains.sum()
                if total > 0:
                    probs = exp_gains / total
                    # Add small random component to avoid getting stuck
                    if rng.random() < 0.3:
                        flip_var = clause_lits[rng.randint(0, 2)]
                    else:
                        r = rng.random()
                        cumsum = np.cumsum(probs)
                        flip_var = clause_lits[np.searchsorted(cumsum, r)]
                else:
                    flip_var = clause_lits[rng.randint(0, 2)]
            else:
                flip_var = clause_lits[rng.randint(0, 2)]
            
            # Flip the variable
            assignment[flip_var] = not assignment[flip_var]
            
            # Update clause satisfaction for clauses containing flip_var
            signed_assign = assignment.astype(np.int8) * 2 - 1
            v_clauses = var_to_clauses_arr[flip_var]
            
            # Recompute satisfaction for affected clauses
            for ci in v_clauses:
                lit_vals = clause_data[ci] * signed_assign[clause_vars[ci]]
                new_sat = (lit_vals > 0).any()
                old_sat = clause_sat[ci]
                if new_sat and not old_sat:
                    unsat_count -= 1
                elif not new_sat and old_sat:
                    unsat_count += 1
                clause_sat[ci] = new_sat
            
            # Track best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            if unsat_count == 0:
                return assignment.tolist()
        
        # End of restart
    
    # Return best found
    if best_unsat == 0:
        return best_assignment.tolist()
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
