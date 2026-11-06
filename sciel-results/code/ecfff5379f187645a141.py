"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    # Planted 3-SAT solver with WalkSAT + clause weighting + restarts.
    # Deadline guard: return best assignment before 320s.
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    deadline = time.time() + 320.0
    
    # Precompute clause structure for speed
    # clauses are lists of signed 1-based literals
    # Convert to 0-based for faster indexing
    clauses_0 = [[l - 1 if l > 0 else -l - 1 for l in clause] for clause in clauses]
    
    def is_satisfied(assignment, clause):
        for lit in clause:
            if lit >= 0:
                if assignment[lit]:
                    return True
            else:
                if not assignment[-lit - 1]:
                    return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for clause in clauses_0:
            sat = False
            for lit in clause:
                if lit >= 0:
                    if assignment[lit]:
                        sat = True
                        break
                else:
                    if not assignment[-lit - 1]:
                        sat = True
                        break
            if not sat:
                cnt += 1
        return cnt
    
    # Try multiple random restarts with WalkSAT
    # Use clause weights to bias clause selection
    best_assignment = None
    best_unsat = m + 1
    
    max_restarts = 200
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Random initial assignment
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Clause weights for biased selection
        weights = [1.0] * m
        
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT loop
        # Parameters: p = probability of flipping a literal in the chosen clause
        # that makes the clause satisfied (vs random flip)
        p = 0.5  # probability of "satisfying" flip vs random
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 2.0:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for i, clause in enumerate(clauses_0):
                sat = False
                for lit in clause:
                    if lit >= 0:
                        if assignment[lit]:
                            sat = True
                            break
                    else:
                        if not assignment[-lit - 1]:
                            sat = True
                            break
                if not sat:
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment
            
            # Weighted clause selection
            total_weight = sum(weights[i] for i in unsat_clauses)
            r = rng.random() * total_weight
            cumulative = 0.0
            chosen_clause_idx = unsat_clauses[-1]
            for idx in unsat_clauses:
                cumulative += weights[idx]
                if r <= cumulative:
                    chosen_clause_idx = idx
                    break
            
            clause = clauses_0[chosen_clause_idx]
            
            # Decide which literal to flip
            # With probability p, pick a literal that would satisfy the clause
            # Otherwise, pick a random literal from the clause
            if rng.random() < p:
                # Pick a literal that, if flipped, would satisfy the clause
                # All literals in an unsatisfied clause are currently "wrong"
                # Flipping any one of them satisfies the clause
                lit = clause[rng.randint(0, len(clause) - 1)]
            else:
                lit = clause[rng.randint(0, len(clause) - 1)]
            
            # Flip the variable
            if lit >= 0:
                assignment[lit] = not assignment[lit]
            else:
                assignment[-lit - 1] = not assignment[-lit - 1]
            
            # Update clause weights: increase weight of clauses that became unsatisfied
            # by this flip (simple heuristic)
            # Actually, let's just periodically update weights based on unsat count
            # For speed, update weights every 100 steps
            if step % 100 == 99:
                for i, clause in enumerate(clauses_0):
                    sat = False
                    for lit in clause:
                        if lit >= 0:
                            if assignment[lit]:
                                sat = True
                                break
                        else:
                            if not assignment[-lit - 1]:
                                sat = True
                                break
                    if sat:
                        weights[i] = max(0.1, weights[i] * 0.9)
                    else:
                        weights[i] = min(10.0, weights[i] * 1.1)
            
            # Check if solved
            unsat_count = count_unsat(assignment)
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
        
        # End of restart, update best
        unsat_count = count_unsat(assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
