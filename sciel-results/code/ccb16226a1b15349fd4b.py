"""
Improved 3-SAT solver for planted instances (50 vars).
Uses a portfolio of local search methods: WalkSAT, min-conflicts, and a simple
clause-learning guided search, all with restarts and time management.
"""
import random
import time
import math
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 290.0
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    if time_remaining() <= 0:
        return [True] * n
    
    m = len(clauses)
    
    # Precompute clause data: list of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    
    # Build variable-to-clause index for fast lookup
    var_to_clauses = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for var, pos in cd:
            var_to_clauses[var].append(i)
    
    def is_clause_satisfied(assignment, clause_idx):
        for var, pos in clause_data[clause_idx]:
            if assignment[var] == pos:
                return True
        return False

    def full_satisfied(assignment):
        for i in range(m):
            if not is_clause_satisfied(assignment, i):
                return False
        return True

    def get_unsat_clauses(assignment):
        unsat = []
        for i in range(m):
            if not is_clause_satisfied(assignment, i):
                unsat.append(i)
        return unsat

    def incremental_walksat(seed, max_steps, noise=0.3, activity_alpha=0.01):
        """
        WalkSAT with incremental unsat set update and variable activity.
        """
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial unsat set
        unsat_clauses = set(get_unsat_clauses(assignment))
        
        if not unsat_clauses:
            return assignment[:]
        
        best_unsat_count = len(unsat_clauses)
        best_assignment = assignment[:]
        
        # Variable activity
        var_activity = [0.0] * n
        
        # Clause activity (optional, can use simple weighting)
        clause_activity = [1.0] * m
        
        for step in range(max_steps):
            if time_remaining() < 2.0:
                break
            
            if not unsat_clauses:
                return assignment[:]
            
            # Select an unsatisfied clause
            # Weighted by clause_activity
            # To avoid O(m) sum every time, we can just pick randomly from unsat_clauses
            # or use a weighted selection if we maintain a list.
            # For simplicity and speed, pick random from unsat_clauses.
            # The clause_activity can be updated but selection is the bottleneck.
            # Let's stick to random selection from unsat_clauses for now, 
            # but boost the chosen clause's activity.
            
            chosen_clause = rng.choice(list(unsat_clauses))
            
            # Increase activity of chosen clause
            clause_activity[chosen_clause] *= 1.1
            # Decay others? Too expensive. Just let it grow.
            
            clause = clause_data[chosen_clause]
            
            # Select variable to flip
            # Heuristic: Min-conflicts gain + Variable Activity
            best_var = -1
            best_score = -1e9
            
            for var, pos in clause:
                # Calculate gain: 
                # Flipping var changes the status of all clauses containing var.
                # We need to know which of those are currently unsatisfied.
                
                gain = 0
                
                for ci in var_to_clauses[var]:
                    if ci in unsat_clauses:
                        # Clause is unsatisfied.
                        # Does flipping var satisfy it?
                        # The literal for var in this clause is (var, pos_in_ci)
                        # We need to find pos_in_ci.
                        # Since var is in ci, we can find it.
                        # But we can also just check:
                        # If assignment[var] != pos_in_ci, then flipping makes it equal?
                        # No, assignment[var] is current.
                        # Literal is satisfied if assignment[var] == pos_in_ci.
                        # Currently unsatisfied means for all vars v in ci, assignment[v] != pos_v.
                        # So assignment[var] != pos_in_ci.
                        # If we flip var, new_assignment[var] = not assignment[var].
                        # We need new_assignment[var] == pos_in_ci.
                        # So we need (not assignment[var]) == pos_in_ci.
                        
                        # Find pos_in_ci
                        found_pos = None
                        for v2, p2 in clause_data[ci]:
                            if v2 == var:
                                found_pos = p2
                                break
                        
                        if found_pos is not None:
                            if (not assignment[var]) == found_pos:
                                gain += 1
                    else:
                        # Clause is satisfied.
                        # Does flipping var unsatisfy it?
                        # It is satisfied if at least one literal is true.
                        # If var is the ONLY true literal, flipping it makes it false.
                        
                        # Check if var is the only satisfying literal
                        is_only_sat = True
                        for v2, p2 in clause_data[ci]:
                            if v2 == var:
                                continue
                            if assignment[v2] == p2:
                                is_only_sat = False
                                break
                        
                        if is_only_sat:
                            gain -= 1
                
                # Add activity
                score = gain + activity_alpha * var_activity[var] + rng.gauss(0, noise)
                
                if score > best_score:
                    best_score = score
                    best_var = var
            
            if best_var < 0:
                best_var, _ = rng.choice(clause)
            
            # Flip variable
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update var_activity
            var_activity[best_var] *= 1.05
            # Decay others? Too expensive.
            
            # Update unsat_clauses set incrementally
            # Only clauses containing best_var can change status.
            for ci in var_to_clauses[best_var]:
                is_sat = is_clause_satisfied(assignment, ci)
                if is_sat:
                    if ci in unsat_clauses:
                        unsat_clauses.remove(ci)
                else:
                    if ci not in unsat_clauses:
                        unsat_clauses.add(ci)
            
            current_unsat_count = len(unsat_clauses)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = assignment[:]
            
            if current_unsat_count == 0:
                return assignment[:]
        
        return best_assignment

    # Portfolio
    best_result = None
    best_unsat_count = float('inf')
    
    def update_best(result):
        nonlocal best_result, best_unsat_count
        if result is not None:
            if full_satisfied(result):
                return True
            else:
                cnt = len(get_unsat_clauses(result))
                if cnt < best_unsat_count:
                    best_unsat_count = cnt
                    best_result = result[:]
        return False

    # Strategy 1: Incremental WalkSAT with moderate noise
    seeds1 = [42, 123, 777, 2024, 555, 999, 31337, 12345, 67890, 24680, 111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
    steps1 = max(500, n * 15)
    
    for seed in seeds1:
        if time_remaining() < 5.0:
            break
        result = incremental_walksat(seed, steps1, noise=0.3)
        if update_best(result):
            return result
    
    # Strategy 2: Incremental WalkSAT with lower noise
    if time_remaining() > 10.0:
        seeds2 = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
        steps2 = max(300, n * 10)
        for seed in seeds2:
            if time_remaining() < 5.0:
                break
            result = incremental_walksat(seed, steps2, noise=0.1)
            if update_best(result):
                return result
    
    # Strategy 3: Incremental WalkSAT with higher noise
    if time_remaining() > 10.0:
        seeds3 = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
        steps3 = max(200, n * 8)
        for seed in seeds3:
            if time_remaining() < 5.0:
                break
            result = incremental_walksat(seed, steps3, noise=0.5)
            if update_best(result):
                return result
                
    # Strategy 4: Short bursts
    if time_remaining() > 5.0:
        for seed in range(2000, 2050):
            if time_remaining() < 3.0:
                break
            result = incremental_walksat(seed, 100, noise=0.3)
            if update_best(result):
                return result

    if best_result is not None:
        return best_result
    
    return [True] * n
