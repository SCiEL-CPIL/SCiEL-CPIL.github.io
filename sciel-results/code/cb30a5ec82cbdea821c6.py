"""Rewritten program using a multi-strategy approach: WalkSAT with restarts, clause weighting, and local search."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n
    
    rng = random.Random(42)
    
    # Precompute clause lengths
    clause_lens = [len(c) for c in clauses]
    
    # Helper to check if a clause is satisfied
    def is_satisfied(clause, assignment):
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if lit > 0:
                if val:
                    return True
            else:
                if not val:
                    return True
        return False
    
    # Helper to count true literals in a clause
    def true_count(clause, assignment):
        count = 0
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if lit > 0:
                if val:
                    count += 1
            else:
                if not val:
                    count += 1
        return count
    
    # Strategy 1: WalkSAT with clause weighting and restarts
    def walksat(max_restarts=50, max_flips_per_restart=5000):
        # Weighted WalkSAT
        weights = [1.0] * num_clauses
        for restart in range(max_restarts):
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Initialize weights based on clause length (shorter clauses get higher weight)
            for i, c in enumerate(clauses):
                weights[i] = 1.0 / (len(c) + 1)
            
            for flip in range(max_flips_per_restart):
                # Find unsatisfied clauses
                unsat_indices = []
                for i, c in enumerate(clauses):
                    if not is_satisfied(c, assignment):
                        unsat_indices.append(i)
                
                if not unsat_indices:
                    return assignment
                
                # Select a clause to work on, weighted by weights
                # Use weighted random selection
                total_weight = sum(weights[i] for i in unsat_indices)
                r = rng.random() * total_weight
                cumulative = 0
                selected_idx = unsat_indices[0]
                for idx in unsat_indices:
                    cumulative += weights[idx]
                    if r <= cumulative:
                        selected_idx = idx
                        break
                
                clause = clauses[selected_idx]
                
                # Find variables in this clause that are not forced
                # (i.e., flipping them won't break other satisfied clauses)
                # For simplicity, just pick a random variable from the clause
                # with some bias toward variables that appear in fewer unsat clauses
                
                # Count how many unsat clauses each variable appears in
                var_unsat_count = {}
                for i in unsat_indices:
                    for lit in clauses[i]:
                        var = abs(lit)
                        var_unsat_count[var] = var_unsat_count.get(var, 0) + 1
                
                # Score each literal in the selected clause
                best_literal = None
                best_score = -1
                for lit in clause:
                    var = abs(lit)
                    var_idx = var - 1
                    # Score: prefer variables with fewer unsat clauses (less disruptive)
                    # and prefer the assignment that makes this literal true
                    current_val = assignment[var_idx]
                    desired_val = (lit > 0)
                    if current_val != desired_val:
                        # Flipping would make this literal true
                        score = var_unsat_count.get(var, 0) * -1 + rng.random() * 0.1
                    else:
                        # Flipping would make this literal false (already true)
                        score = var_unsat_count.get(var, 0) * -2 + rng.random() * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_literal = lit
                
                if best_literal is None:
                    best_literal = rng.choice(clause)
                
                var_idx = abs(best_literal) - 1
                assignment[var_idx] = not assignment[var_idx]
                
                # Update weights: increase weight of clauses that became unsatisfied
                # This is a simplified version; in full WalkSAT, weights are updated
                # For now, just do periodic weight decay
                if flip % 100 == 0:
                    for i in range(num_clauses):
                        weights[i] *= 0.95
                        if not is_satisfied(clauses[i], assignment):
                            weights[i] *= 1.5
            
            # Restart with new assignment
            # Decay weights
            for i in range(num_clauses):
                weights[i] *= 0.8
        
        return None
    
    # Strategy 2: Simple WalkSAT with no weighting (baseline)
    def simple_walksat(max_flips=30000):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        for _ in range(max_flips):
            unsat = [c for c in clauses if not is_satisfied(c, assignment)]
            if not unsat:
                return assignment
            
            clause = rng.choice(unsat)
            literal = rng.choice(clause)
            var_idx = abs(literal) - 1
            assignment[var_idx] = not assignment[var_idx]
        
        return None
    
    # Strategy 3: Local search with clause learning (simplified)
    def local_search(max_restarts=20, max_flips_per_restart=10000):
        for restart in range(max_restarts):
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            for flip in range(max_flips_per_restart):
                unsat_indices = []
                for i, c in enumerate(clauses):
                    if not is_satisfied(c, assignment):
                        unsat_indices.append(i)
                
                if not unsat_indices:
                    return assignment
                
                # Pick the most constrained variable: one that appears in the most unsat clauses
                var_counts = {}
                for i in unsat_indices:
                    for lit in clauses[i]:
                        var = abs(lit)
                        var_counts[var] = var_counts.get(var, 0) + 1
                
                if not var_counts:
                    break
                    
                # Pick variable with highest count
                max_count = max(var_counts.values())
                candidates = [v for v, cnt in var_counts.items() if cnt == max_count]
                var = rng.choice(candidates)
                var_idx = var - 1
                
                # Determine which assignment reduces unsat clauses more
                # Try both values and see which is better
                current_val = assignment[var_idx]
                
                # Count unsat clauses affected by this variable
                affected_unsat = []
                for i in unsat_indices:
                    for lit in clauses[i]:
                        if abs(lit) == var:
                            affected_unsat.append(i)
                            break
                
                # Also count satisfied clauses that might become unsatisfied
                affected_sat = []
                for i, c in enumerate(clauses):
                    if is_satisfied(c, assignment):
                        for lit in c:
                            if abs(lit) == var:
                                affected_sat.append(i)
                                break
                
                # Estimate impact
                # If we set to True:
                # - Clauses with positive var lit become satisfied
                # - Clauses with negative var lit may become unsatisfied
                # If we set to False:
                # - Clauses with negative var lit become satisfied  
                # - Clauses with positive var lit may become unsatisfied
                
                pos_sat_gain = 0
                pos_sat_loss = 0
                neg_sat_gain = 0
                neg_sat_loss = 0
                
                for i in affected_unsat:
                    for lit in clauses[i]:
                        if abs(lit) == var:
                            if lit > 0:
                                pos_sat_gain += 1
                            else:
                                neg_sat_gain += 1
                            break
                
                for i in affected_sat:
                    for lit in clauses[i]:
                        if abs(lit) == var:
                            if lit > 0:
                                pos_sat_loss += 1
                            else:
                                neg_sat_loss += 1
                            break
                
                # Net change if set to True:
                set_true_net = pos_sat_gain - pos_sat_loss
                # Net change if set to False:
                set_false_net = neg_sat_gain - neg_sat_loss
                
                if set_true_net >= set_false_net:
                    new_val = True
                else:
                    new_val = False
                
                # Add some randomness
                if rng.random() < 0.1:
                    new_val = not new_val
                
                assignment[var_idx] = new_val
            
            # Restart
            continue
        
        return None
    
    # Try strategies in order, return first solution found
    result = walksat(max_restarts=30, max_flips_per_restart=8000)
    if result is not None:
        return result
    
    result = local_search(max_restarts=15, max_flips_per_restart=12000)
    if result is not None:
        return result
    
    result = simple_walksat(max_flips=50000)
    if result is not None:
        return result
    
    # Fallback: return random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
