"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: must finish well before 330s wall-clock
    deadline = time.time() + 290.0
    
    # Preprocess: convert to 0-based indices
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # Build variable-to-clause index for fast damage calculation
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, _ in cv:
            var_clauses[var_idx].append(i)
    
    def is_satisfied(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                count += 1
        return count
    
    def random_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]
    
    def walk_sat(assignment, rng, max_steps, deadline):
        """Run WalkSAT with clause weighting and damage-minimizing literal selection."""
        # Compute initial unsat count
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment, True
        
        # Track unsat clauses for efficiency
        unsat_clauses = set()
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_clauses.add(i)
        
        # Clause weights (for weighted selection)
        weights = [1.0] * m
        
        # Parameters
        p = 0.55  # probability of picking a literal that satisfies the clause
        weight_decay = 0.95
        weight_growth = 1.2
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            if not unsat_clauses:
                return assignment, True
            
            # Weighted random selection of an unsatisfied clause
            total_weight = sum(weights[i] for i in unsat_clauses)
            r = rng.random() * total_weight
            cumulative = 0
            chosen_clause = None
            for ci in unsat_clauses:
                cumulative += weights[ci]
                if r <= cumulative:
                    chosen_clause = ci
                    break
            if chosen_clause is None:
                chosen_clause = list(unsat_clauses)[-1]
            
            # Select literal to flip
            cv = clause_vars[chosen_clause]
            # Find literals that are false (would satisfy if flipped)
            false_literals = []
            true_literals = []
            for var_idx, is_pos in cv:
                if assignment[var_idx] != is_pos:
                    false_literals.append((var_idx, is_pos))
                else:
                    true_literals.append((var_idx, is_pos))
            
            if rng.random() < p and false_literals:
                # Pick a literal that satisfies the clause (minimize damage)
                # Choose the one with minimal damage
                best_var = None
                best_damage = float('inf')
                for var_idx, is_pos in false_literals:
                    damage = 0
                    for ci2 in var_clauses[var_idx]:
                        if ci2 in unsat_clauses:
                            # Check if flipping this var would make this clause still unsat
                            # After flip, this var becomes true for this clause
                            # But other clauses containing this var might become unsat
                            cv2 = clause_vars[ci2]
                            # Check if any other literal in ci2 is true
                            still_satisfied = False
                            for v2, pos2 in cv2:
                                if v2 == var_idx:
                                    continue
                                if assignment[v2] == pos2:
                                    still_satisfied = True
                                    break
                            if not still_satisfied:
                                damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_var = var_idx
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
            else:
                # Pick a random literal from the clause
                var_idx, is_pos = rng.choice(cv)
                assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat clauses affected by the flip
            # Re-evaluate all clauses containing the flipped variable
            # We need to know which variable was flipped
            # Let's recompute more carefully
            # Actually, let's just recompute the unsat set periodically or track changes
            # For simplicity and correctness, recompute affected clauses
            
            # Determine which variable was flipped
            # This is tricky - let's just recompute all unsat clauses
            # But that's O(m) per step. For 150 vars and reasonable m, this might be ok
            # Let's optimize: only recheck clauses containing the flipped var
            # But we need to know which var was flipped...
            
            # Simpler approach: recompute unsat_count and unsat_clauses
            # This is O(m * 3) per step
            new_unsat = set()
            for i in range(m):
                if not is_satisfied(i, assignment):
                    new_unsat.add(i)
            
            # Update weights
            for i in range(m):
                if i in new_unsat and i not in unsat_clauses:
                    weights[i] *= weight_growth
                elif i in unsat_clauses and i not in new_unsat:
                    weights[i] *= weight_decay
            
            unsat_clauses = new_unsat
            unsat_count = len(unsat_clauses)
        
        return assignment, (unsat_count == 0)
    
    # Multi-restart strategy
    rng = random.Random(0)
    
    # First, try a few random assignments quickly
    best_assignment = random_assignment(rng)
    best_unsat = count_unsat(best_assignment)
    if best_unsat == 0:
        return best_assignment
    
    # Run multiple restarts of WalkSAT
    num_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        assignment = random_assignment(rng)
        result, solved = walk_sat(assignment, rng, steps_per_restart, deadline)
        
        if solved:
            return result
        
        # Track best assignment found
        cur_unsat = count_unsat(result)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = result
    
    # If we still have time, do a final long run from the best assignment
    if time.time() < deadline - 10:
        result, solved = walk_sat(best_assignment, rng, 100000, deadline)
        if solved:
            return result
        cur_unsat = count_unsat(result)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = result
    
    return best_assignment
# EVOLVE-BLOCK-END
