"""Improved 3-SAT solver using optimized WalkSAT with restarts and gain-based selection."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Preprocess: convert to 0-based, store positive/negative separately for speed
    # For each clause, store (pos_vars, neg_vars) where vars are 0-based
    # A clause is satisfied if any pos_var is True or any neg_var is False
    
    # Actually, let's keep it simple but fast:
    # Store clauses as list of (abs_var_0based, is_positive) tuples
    # Or better: for each clause, precompute the variable indices and signs
    
    # Let's use a different approach: maintain clause satisfaction counts
    # For each variable, track which clauses it appears in (as pos or neg)
    
    # Simple approach: use numpy for vectorized operations where possible
    # But for WalkSAT, we need to be careful about overhead
    
    # Let's use a clean WalkSAT implementation with:
    # 1. Random restarts
    # 2. Gain-based variable selection (flip variable that maximizes satisfied clauses)
    # 3. Time-based termination
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    rng = random.Random(42)
    
    # Preprocess clauses: convert to 0-based
    # clauses[i] = list of signed 1-based literals
    # Convert to: for each clause, list of (var_index, sign) where var_index is 0-based
    # sign = +1 if positive literal, -1 if negative
    
    processed_clauses = []
    for clause in clauses:
        processed = [(abs(lit) - 1, 1 if lit > 0 else -1) for lit in clause]
        processed_clauses.append(processed)
    
    # For speed, also precompute for each variable which clauses it appears in
    # var_to_clauses[var] = list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Check if already satisfied
    def is_satisfied(assignment):
        for clause in processed_clauses:
            sat = False
            for var, sign in clause:
                val = assignment[var]
                if sign == 1 and val:
                    sat = True
                    break
                elif sign == -1 and not val:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        count = 0
        for clause in processed_clauses:
            sat = False
            for var, sign in clause:
                val = assignment[var]
                if sign == 1 and val:
                    sat = True
                    break
                elif sign == -1 and not val:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def get_gain(assignment, var):
        """Compute gain of flipping var: how many more clauses would be satisfied."""
        gain = 0
        for ci, sign in var_to_clauses[var]:
            clause = processed_clauses[ci]
            # Current state of this clause
            currently_sat = False
            for v, s in clause:
                if v == var:
                    continue
                val = assignment[v]
                if s == 1 and val:
                    currently_sat = True
                    break
                elif s == -1 and not val:
                    currently_sat = True
                    break
            
            # What happens to this clause if we flip var
            var_val = assignment[var]
            if sign == 1:
                # Positive literal: satisfied if var is True
                if currently_sat:
                    # If currently satisfied by another literal, flipping var doesn't matter
                    # unless var is the only one satisfying
                    pass
                else:
                    # Currently unsatisfied
                    if not var_val:
                        # var is False, so positive literal is False. Flipping to True satisfies.
                        gain += 1
            else:
                # Negative literal: satisfied if var is False
                if currently_sat:
                    pass
                else:
                    # Currently unsatisfied
                    if var_val:
                        # var is True, so negative literal is False. Flipping to False satisfies.
                        gain += 1
        
        # But we also need to account for clauses that become unsatisfied
        # Actually, the gain calculation above is incomplete. Let me redo.
        
        # Better approach: for each clause containing var, check if flipping var changes satisfaction
        gain = 0
        for ci, sign in var_to_clauses[var]:
            clause = processed_clauses[ci]
            # Check if clause is currently satisfied by any literal other than var
            sat_by_other = False
            for v, s in clause:
                if v == var:
                    continue
                val = assignment[v]
                if s == 1 and val:
                    sat_by_other = True
                    break
                elif s == -1 and not val:
                    sat_by_other = True
                    break
            
            var_val = assignment[var]
            var_satisfies = (sign == 1 and var_val) or (sign == -1 and not var_val)
            
            if sat_by_other:
                # Clause is satisfied regardless of var
                continue
            else:
                # Clause satisfaction depends on var
                if var_satisfies:
                    # Currently satisfied by var, flipping makes it unsatisfied
                    gain -= 1
                else:
                    # Currently unsatisfied, flipping makes it satisfied
                    gain += 1
        
        return gain
    
    # WalkSAT with restarts
    max_restarts = 100
    steps_per_restart = 5000
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT steps
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for ci, clause in enumerate(processed_clauses):
                sat = False
                for var, sign in clause:
                    val = assignment[var]
                    if sign == 1 and val:
                        sat = True
                        break
                    elif sign == -1 and not val:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            unsat_count = len(unsat_clauses)
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = processed_clauses[ci]
            
            # Choose variable to flip: with probability p, use gain; otherwise random
            if rng.random() < 0.8:  # 80% use gain-based
                best_var = None
                best_gain = -float('inf')
                for var, sign in clause:
                    gain = get_gain(assignment, var)
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
            else:
                var, sign = rng.choice(clause)
                assignment[var] = not assignment[var]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
