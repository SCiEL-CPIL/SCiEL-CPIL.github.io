# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Use a deterministic seed for reproducibility, but we can try multiple seeds if needed
    # For now, use a fixed seed based on instance characteristics for stability
    rng = random.Random(42)
    
    # Preprocess: convert clauses to lists of (var, sign) for faster access
    # var is 0-indexed, sign is +1 or -1
    parsed_clauses = []
    for clause in clauses:
        parsed = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            parsed.append((var, sign))
        parsed_clauses.append(parsed)
    
    # WalkSAT with improvements:
    # 1. Better variable selection (weighted by frequency in unsat clauses)
    # 2. Dynamic noise parameter
    # 3. Multiple restarts with different initial assignments
    # 4. Local search with clause weighting (MiniSAT-style)
    
    def evaluate(assignment):
        """Count satisfied and unsatisfied clauses."""
        sat_count = 0
        unsat_clauses = []
        for i, clause in enumerate(parsed_clauses):
            is_sat = False
            for var, sign in clause:
                if (assignment[var] and sign == 1) or (not assignment[var] and sign == -1):
                    is_sat = True
                    break
            if is_sat:
                sat_count += 1
            else:
                unsat_clauses.append(i)
        return sat_count, unsat_clauses
    
    def walk_sat_restarts(max_restarts=5, flips_per_restart=10000):
        """Run WalkSAT with multiple restarts."""
        best_assignment = None
        best_sat_count = -1
        
        for restart in range(max_restarts):
            # Random initial assignment
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Track clause weights for weighted WalkSAT
            clause_weights = [1.0] * len(parsed_clauses)
            
            sat_count, unsat_idx = evaluate(assignment)
            if sat_count == len(parsed_clauses):
                return assignment
            
            best_in_run = sat_count
            best_assignment_in_run = assignment[:]
            
            # Adaptive noise: start high, decrease over time
            base_noise = 0.5
            min_noise = 0.1
            
            for step in range(flips_per_restart):
                if not unsat_idx:
                    return assignment
                
                # Select an unsatisfied clause (weighted by clause weight)
                total_weight = sum(clause_weights[i] for i in unsat_idx)
                if total_weight <= 0:
                    # Reset weights if they've decayed too much
                    clause_weights = [1.0] * len(parsed_clauses)
                    total_weight = len(unsat_idx)
                
                r = rng.random() * total_weight
                cumulative = 0
                selected_clause_idx = unsat_idx[0]
                for i in unsat_idx:
                    cumulative += clause_weights[i]
                    if r <= cumulative:
                        selected_clause_idx = i
                        break
                
                clause = parsed_clauses[selected_clause_idx]
                
                # Choose which literal to flip
                # With probability 'noise', pick a random literal from the clause
                # Otherwise, pick the literal that minimizes the number of newly unsatisfied clauses
                noise = base_noise - (base_noise - min_noise) * (step / flips_per_restart)
                
                if rng.random() < noise:
                    # Random flip
                    var, sign = rng.choice(clause)
                else:
                    # Greedy: pick literal that causes least damage
                    best_var = None
                    best_score = float('inf')
                    for var, sign in clause:
                        # Count how many other clauses would become unsatisfied if we flip this var
                        damage = 0
                        for ci in unsat_idx:
                            if ci == selected_clause_idx:
                                continue
                            other_clause = parsed_clauses[ci]
                            for v, s in other_clause:
                                if v == var:
                                    # Check if this clause is currently satisfied only by this variable
                                    # i.e., if flipping var makes it unsatisfied
                                    is_sat_without_var = False
                                    for v2, s2 in other_clause:
                                        if v2 != var:
                                            if (assignment[v2] and s2 == 1) or (not assignment[v2] and s2 == -1):
                                                is_sat_without_var = True
                                                break
                                    if not is_sat_without_var:
                                        damage += clause_weights[ci]
                                    break
                        # Also consider that flipping might satisfy other clauses (benefit)
                        # For simplicity, just use damage
                        if damage < best_score:
                            best_score = damage
                            best_var = var
                    if best_var is None:
                        var, sign = rng.choice(clause)
                    else:
                        var = best_var
                
                # Flip the variable
                assignment[var] = not assignment[var]
                
                # Update clause weights
                for var, sign in clause:
                    # If this clause is now satisfied, reduce weight
                    pass
                
                # Re-evaluate only the affected clauses for efficiency
                # Find all clauses containing 'var'
                affected = []
                for ci, c in enumerate(parsed_clauses):
                    for v, s in c:
                        if v == var:
                            affected.append(ci)
                            break
                
                # Update satisfaction status
                for ci in affected:
                    c = parsed_clauses[ci]
                    is_sat = False
                    for v, s in c:
                        if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                            is_sat = True
                            break
                    if is_sat:
                        # Clause became satisfied: reduce weight
                        clause_weights[ci] *= 0.5
                        if ci in unsat_idx:
                            unsat_idx.remove(ci)
                    else:
                        # Clause is unsatisfied: increase weight
                        clause_weights[ci] *= 2.0
                        if ci not in unsat_idx:
                            unsat_idx.append(ci)
                
                # Recompute sat_count
                sat_count = len(parsed_clauses) - len(unsat_idx)
                
                if sat_count > best_in_run:
                    best_in_run = sat_count
                    best_assignment_in_run = assignment[:]
                
                if sat_count == len(parsed_clauses):
                    return assignment
            
            if best_in_run > best_sat_count:
                best_sat_count = best_in_run
                best_assignment = best_assignment_in_run
        
        return best_assignment
    
    # Try WalkSAT with restarts
    result = walk_sat_restarts(max_restarts=3, flips_per_restart=15000)
    
    # Verify the result
    sat_count, _ = evaluate(result)
    if sat_count == len(parsed_clauses):
        return result
    
    # If WalkSAT failed, try a different approach: local search with random restarts
    # and a different variable ordering heuristic
    
    def local_search_with_restarts(max_restarts=10, flips_per_restart=5000):
        best_assignment = None
        best_sat_count = -1
        
        for restart in range(max_restarts):
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            sat_count, unsat_idx = evaluate(assignment)
            if sat_count == len(parsed_clauses):
                return assignment
            
            best_in_run = sat_count
            best_assignment_in_run = assignment[:]
            
            for step in range(flips_per_restart):
                if not unsat_idx:
                    return assignment
                
                # Pick a random unsatisfied clause
                clause_idx = rng.choice(unsat_idx)
                clause = parsed_clauses[clause_idx]
                
                # Try flipping each variable in the clause, pick the one that maximizes satisfied clauses
                best_var = None
                best_score = -1
                
                for var, sign in clause:
                    # Temporarily flip
                    old_val = assignment[var]
                    assignment[var] = not old_val
                    
                    # Count how many clauses are now satisfied
                    # Only need to check clauses containing var
                    score = 0
                    for ci, c in enumerate(parsed_clauses):
                        has_var = any(v == var for v, s in c)
                        if has_var:
                            is_sat = False
                            for v, s in c:
                                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                                    is_sat = True
                                    break
                            if is_sat:
                                score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_var = var
                    
                    # Revert
                    assignment[var] = old_val
                
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
                    
                    # Update unsat_idx
                    new_unsat = []
                    for ci, c in enumerate(parsed_clauses):
                        is_sat = False
                        for v, s in c:
                            if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                                is_sat = True
                                break
                        if not is_sat:
                            new_unsat.append(ci)
                    unsat_idx = new_unsat
                    sat_count = len(parsed_clauses) - len(unsat_idx)
                    
                    if sat_count > best_in_run:
                        best_in_run = sat_count
                        best_assignment_in_run = assignment[:]
                    
                    if sat_count == len(parsed_clauses):
                        return assignment
                else:
                    # No improvement, break
                    break
            
            if best_in_run > best_sat_count:
                best_sat_count = best_in_run
                best_assignment = best_assignment_in_run
        
        return best_assignment
    
    result2 = local_search_with_restarts(max_restarts=5, flips_per_restart=3000)
    
    sat_count2, _ = evaluate(result2)
    if sat_count2 == len(parsed_clauses):
        return result2
    
    # Return the better of the two
    if sat_count >= sat_count2:
        return result
    else:
        return result2

# EVOLVE-BLOCK-END
