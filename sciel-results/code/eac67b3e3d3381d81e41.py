"""Synthetic planted 3-SAT solver (150 variables) - High-Performance WalkSAT with NumPy."""
import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s limit, use 290s for safety
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess: convert to 0-based var indices and signs
    # lit_vars: (m, 3) array of variable indices (0-based)
    # lit_signs: (m, 3) array of signs (+1 or -1)
    lit_vars = np.zeros((m, 3), dtype=np.int32)
    lit_signs = np.zeros((m, 3), dtype=np.int8)
    
    # var_to_clauses: list of lists of clause indices for each variable
    # This is used for efficient damage calculation if we do exact WalkSAT, 
    # but for high-speed NumPy WalkSAT, we often just pick a random literal 
    # from the unsatisfied clause. However, "best flip" is better.
    # Let's stick to a hybrid: NumPy for satisfaction check, Python for flip selection 
    # to keep code simple and robust, but optimize the check.
    
    # Actually, for 150 vars and ~450-600 clauses, Python loops are fine if we don't 
    # do O(m) checks every step. But NumPy is faster for the "is it satisfied?" check.
    
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            lit_vars[ci, li] = var_idx
            lit_signs[ci, li] = sign

    # Build var_to_clauses for potential exact damage calc if needed, 
    # but primarily we will use a fast randomized WalkSAT.
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            v = lit_vars[ci, li]
            var_to_clauses[v].append(ci)

    def is_satisfied(assignment):
        """
        Vectorized check: 
        assignment: (n,) bool array
        lit_vars: (m, 3) int array
        lit_signs: (m, 3) int array
        
        A literal is satisfied if assignment[var] matches sign.
        sign +1 -> assignment[var] is True
        sign -1 -> assignment[var] is False
        
        We can compute:
        val = assignment[lit_vars]  # (m, 3) bool
        target = lit_signs == 1     # (m, 3) bool (True if we want True, False if we want False)
        sat_lit = (val == target)   # (m, 3) bool
        clause_sat = sat_lit.any(axis=1) # (m,) bool
        """
        val = assignment[lit_vars]
        target = lit_signs == 1
        sat_lit = (val == target)
        clause_sat = sat_lit.any(axis=1)
        return clause_sat

    def count_unsat(clause_sat):
        return np.sum(~clause_sat)

    # Main Search Loop
    best_unsat_count = m + 1
    best_assignment = None
    
    # Parameters
    weight = 0.55  # Probability of choosing best flip (greedy) vs random
    max_steps_per_restart = 2000
    max_restarts = 10000  # High limit, controlled by time
    
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Check initial
        clause_sat = is_satisfied(assignment)
        unsat_count = count_unsat(clause_sat)
        
        if unsat_count == 0:
            return assignment.tolist()
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # WalkSAT Loop
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_indices = np.where(~clause_sat)[0]
            
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Get literals in this clause
            vars_in_clause = lit_vars[ci]
            signs_in_clause = lit_signs[ci]
            
            # Determine which literal to flip.
            # Strategy: Calculate "damage" for each of the 3 literals.
            # Damage = number of clauses that are currently satisfied ONLY by this literal.
            # If we flip the literal, those clauses become unsatisfied.
            
            best_var = -1
            min_damage = float('inf')
            
            # For 3 literals, we check 3 options
            for li in range(3):
                v = vars_in_clause[li]
                # Calculate damage for flipping v
                # Get all clauses containing v
                clauses_with_v = var_to_clauses[v]
                
                damage = 0
                for oj in clauses_with_v:
                    # If clause oj is already unsatisfied, flipping v might satisfy it (good) 
                    # or keep it unsatisfied. It doesn't add damage.
                    # Damage only comes from clauses that are currently SATISFIED 
                    # and will become UNSATISFIED after flip.
                    if not clause_sat[oj]:
                        continue
                    
                    # Clause oj is satisfied. Check if it's satisfied ONLY by v.
                    # If so, flipping v will make it unsatisfied.
                    # Check other literals in oj
                    oj_vars = lit_vars[oj]
                    oj_signs = lit_signs[oj]
                    
                    # We need to check if any other literal in oj is satisfied.
                    # val = assignment[oj_vars]
                    # target = oj_signs == 1
                    # sat_lit = (val == target)
                    # If sat_lit has exactly one True, and it's at position where var==v, then damage.
                    
                    val = assignment[oj_vars]
                    target = oj_signs == 1
                    sat_lit = (val == target)
                    
                    # Count satisfied literals in oj
                    if np.sum(sat_lit) == 1:
                        # Only one literal is satisfied. Is it v?
                        # Find index of the satisfied literal
                        idx = np.where(sat_lit)[0][0]
                        if oj_vars[idx] == v:
                            damage += 1
                            # Early break if damage is already high? 
                            # We want min damage, so if damage > min_damage, we can break.
                            if damage >= min_damage:
                                break
            
            if damage < min_damage:
                min_damage = damage
                best_var = v
                
            # Decide flip: Greedy or Random
            if rng.random() < weight and best_var != -1:
                flip_var = best_var
            else:
                # Pick random literal from the unsatisfied clause
                li = rng.randint(0, 2)
                flip_var = vars_in_clause[li]
            
            # Perform flip
            assignment[flip_var] = not assignment[flip_var]
            
            # Update clause_sat incrementally? 
            # For 150 vars, re-computing is_satisfied is O(m*3) ~ 2000 ops.
            # 2000 steps * 1000 restarts = 2M ops. Python overhead dominates.
            # Let's re-compute vectorized. It's fast enough for 150 vars.
            clause_sat = is_satisfied(assignment)
            unsat_count = count_unsat(clause_sat)
            
            if unsat_count == 0:
                return assignment.tolist()
                
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
                
            # Optional: Adaptive weight? 
            # If unsat_count is stuck, maybe lower weight to explore more?
            # For now, keep fixed 0.55 which is standard for WalkSAT.
            
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [random.random() < 0.5 for _ in range(n)]
