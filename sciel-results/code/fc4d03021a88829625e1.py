"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Pre-process clauses: store as arrays of (var_index, is_positive)
    # var_index is 0-based
    clause_vars = []
    clause_polarity = []
    for c in clauses:
        vars_in_c = []
        pols_in_c = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            vars_in_c.append(v)
            pols_in_c.append(p)
        clause_vars.append(vars_in_c)
        clause_polarity.append(pols_in_c)
    
    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clause_vars):
        for v in c:
            var_to_clauses[v].append(i)
    
    def satisfied(clause_idx, assignment):
        cvars = clause_vars[clause_idx]
        cpols = clause_polarity[clause_idx]
        for v, p in zip(cvars, cpols):
            if assignment[v] == p:
                return True
        return False
    
    def count_unsat(assignment):
        unsat_count = 0
        for i in range(m):
            if not satisfied(i, assignment):
                unsat_count += 1
        return unsat_count
    
    def find_unsat_clause(assignment):
        for i in range(m):
            if not satisfied(i, assignment):
                return i
        return -1
    
    # Initialize assignment randomly
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Initial check
    if count_unsat(assignment) == 0:
        return assignment
    
    # WalkSAT with variable flip selection based on clause weight
    # Use a weighted scheme: prefer variables that satisfy more clauses
    # when flipped (i.e., have more satisfied clauses that would remain satisfied
    # or become satisfied)
    
    best_assignment = assignment[:]
    best_unsat = count_unsat(assignment)
    
    max_steps_per_restart = 100000
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
        
        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            unsat_idx = -1
            for i in range(m):
                if not satisfied(i, assignment):
                    unsat_idx = i
                    break
            
            if unsat_idx == -1:
                return assignment
            
            steps += 1
            
            # Get the variables in this clause
            cvars = clause_vars[unsat_idx]
            cpols = clause_polarity[unsat_idx]
            
            # For each variable in the clause, compute the score:
            # How many clauses are currently satisfied that contain this variable
            # and would become unsatisfied if we flip this variable
            # vs. how many unsatisfied clauses contain this variable and would
            # become satisfied if we flip it
            
            best_var = -1
            best_score = -10**9
            
            for v in cvars:
                # Count satisfied clauses containing v that would break
                # and unsatisfied clauses containing v that would be fixed
                gain = 0
                for ci in var_to_clauses[v]:
                    if satisfied(ci, assignment):
                        # This clause is satisfied. Check if flipping v breaks it
                        # i.e., if v is the only literal satisfying this clause
                        cvars_ci = clause_vars[ci]
                        cpols_ci = clause_polarity[ci]
                        is_only = True
                        for v2, p2 in zip(cvars_ci, cpols_ci):
                            if v2 != v and assignment[v2] == p2:
                                is_only = False
                                break
                        if is_only:
                            gain -= 1  # would break this satisfied clause
                    else:
                        # This clause is unsatisfied. Check if flipping v satisfies it
                        cvars_ci = clause_vars[ci]
                        cpols_ci = clause_polarity[ci]
                        # v must be the literal in this clause
                        # Check if flipping v makes this clause satisfied
                        # The literal for v in this clause has some polarity
                        for idx2, v2 in enumerate(cvars_ci):
                            if v2 == v:
                                p2 = cpols_ci[idx2]
                                # After flip, assignment[v] becomes not assignment[v]
                                # We want not assignment[v] == p2
                                if (not assignment[v]) == p2:
                                    gain += 1  # would satisfy this clause
                                break
                
                if gain > best_score:
                    best_score = gain
                    best_var = v
            
            # With some probability, pick a random variable instead (noise)
            if rng.random() < 0.3:
                v = rng.choice(cvars)
            else:
                v = best_var
            
            if v >= 0:
                assignment[v] = not assignment[v]
            
            # Track best
            current_unsat = count_unsat(assignment)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            if current_unsat == 0:
                return assignment
            
            # Early exit if we're stuck and time is running out
            if time.time() > deadline - 5.0:
                break
    
    # Return the best assignment found
    return best_assignment
# EVOLVE-BLOCK-END
