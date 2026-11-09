"""Improved WalkSAT solver with vectorized numpy operations and adaptive noise."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 285 seconds to be safe (330s kill limit)
    deadline = time.time() + 285
    
    # Pre-process clauses into numpy arrays for fast evaluation
    # For each clause, store:
    #   vars: array of variable indices (0-based)
    #   signs: array of +1/-1 indicating polarity
    m = len(clauses)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    # For damage calculation, we need to know for each variable, which clauses contain it
    # and at which position
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append((i, j))
    
    # Function to evaluate all clauses given an assignment (numpy boolean array)
    def evaluate_clauses(assignment):
        """Returns boolean array of shape (m,) where True means satisfied."""
        # assignment is boolean array of shape (n,)
        # For each clause i, each literal j: value = assignment[clause_vars[i,j]]
        # If sign is +1, literal is true when value is True
        # If sign is -1, literal is true when value is False
        # Clause is satisfied if any literal is true
        
        # Get truth values of all literals: shape (m, 3)
        # assignment[clause_vars] gives shape (m, 3) boolean array
        lit_values = assignment[clause_vars]  # shape (m, 3)
        
        # Apply signs: if sign is -1, flip the value
        # signed_values = lit_values if sign==+1, else ~lit_values
        # We can do: signed = lit_values ^ (signs == -1)
        # But signs is int8, so: flip_mask = (clause_signs == -1)
        flip_mask = (clause_signs == -1)  # shape (m, 3)
        signed_values = lit_values ^ flip_mask  # shape (m, 3)
        
        # Clause satisfied if any literal is true
        return signed_values.any(axis=1)  # shape (m,)
    
    # Function to compute damage for flipping variable var_idx
    # Damage = number of currently satisfied clauses that would become unsatisfied
    def compute_damage(assignment, sat_mask, var_idx):
        """
        assignment: boolean array (n,)
        sat_mask: boolean array (m,) - which clauses are currently satisfied
        var_idx: variable to flip
        Returns: number of satisfied clauses that would be broken
        """
        # Get all clauses containing var_idx
        relevant = var_to_clauses[var_idx]
        if not relevant:
            return 0
        
        # For each relevant clause, check if it's currently satisfied
        # and if var_idx's literal is the only one making it satisfied
        damage = 0
        for (ci, cj) in relevant:
            if not sat_mask[ci]:
                continue
            # Clause ci is satisfied. Check if literal (ci, cj) is the only satisfying literal.
            # Check the other two literals in clause ci
            other_satisfied = False
            for k in range(3):
                if k == cj:
                    continue
                vi = clause_vars[ci, k]
                si = clause_signs[ci, k]
                # Evaluate literal k in clause ci
                val = assignment[vi]
                if si == 1:
                    lit_true = val
                else:
                    lit_true = not val
                if lit_true:
                    other_satisfied = True
                    break
            if not other_satisfied:
                damage += 1
        return damage
    
    # WalkSAT with adaptive noise
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        
        # Random initial assignment
        assignment = np.random.random(n) < 0.5
        
        # Evaluate
        sat_mask = evaluate_clauses(assignment)
        unsat_count = m - sat_mask.sum()
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
        
        # Adaptive noise: start higher, decrease over time within a restart
        base_noise = 0.35 + 0.15 * random.random()
        max_steps = 5000 + restart * 500
        if max_steps > 20000:
            max_steps = 20000
        
        steps = 0
        while steps < max_steps and time.time() < deadline:
            # Find an unsatisfied clause
            unsat_indices = np.where(~sat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            ci = unsat_indices[random.randint(0, len(unsat_indices) - 1)]
            
            # Compute damage for each of the 3 literals in this clause
            damages = []
            for j in range(3):
                var_idx = clause_vars[ci, j]
                d = compute_damage(assignment, sat_mask, var_idx)
                damages.append(d)
            
            # Choose flip: with probability noise, pick random; otherwise pick min damage
            noise = base_noise * (1.0 - steps / (max_steps * 1.5))
            if noise < 0:
                noise = 0.05
            
            if random.random() < noise:
                j_to_flip = random.randint(0, 2)
            else:
                # Pick literal with minimum damage; break ties randomly
                min_d = min(damages)
                candidates = [j for j in range(3) if damages[j] == min_d]
                j_to_flip = random.choice(candidates)
            
            # Flip the variable
            var_idx = clause_vars[ci, j_to_flip]
            assignment[var_idx] = not assignment[var_idx]
            
            # Re-evaluate affected clauses (only those containing var_idx)
            # For efficiency, only re-check clauses that contain var_idx
            affected_clauses = [ci2 for (ci2, cj2) in var_to_clauses[var_idx]]
            for ci2 in affected_clauses:
                # Re-evaluate clause ci2
                # Check if any literal is true
                new_sat = False
                for k in range(3):
                    vi = clause_vars[ci2, k]
                    si = clause_signs[ci2, k]
                    val = assignment[vi]
                    if si == 1:
                        lit_true = val
                    else:
                        lit_true = not val
                    if lit_true:
                        new_sat = True
                        break
                sat_mask[ci2] = new_sat
            
            steps += 1
            
            # Check if all satisfied
            if sat_mask.all():
                return assignment.tolist()
            
            # Periodically update best
            if steps % 20 == 0:
                cur_unsat = m - sat_mask.sum()
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment.copy()
        
        # End of restart
        if time.time() > deadline - 3:
            break
    
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
# EVOLVE-BLOCK-END
