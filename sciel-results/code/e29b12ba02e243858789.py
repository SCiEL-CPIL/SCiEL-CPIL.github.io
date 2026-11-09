"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # WalkSAT with deadline guard, min-conflicts heuristic, and restarts.
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 330s kill
    
    def walk_sat_trial(assignment, max_steps, p_flip_random):
        """Run one WalkSAT trial. Returns True if solved."""
        if time.time() > deadline:
            return False
        
        # Precompute clause literal indices for speed
        clause_indices = [[abs(l) - 1 for l in c] for c in clauses]
        clause_signs = [[(l > 0) for l in c] for c in clauses]
        
        # Track unsatisfied clause count
        def is_sat(clause_idx):
            idx = clause_indices[clause_idx]
            signs = clause_signs[clause_idx]
            for i, s in zip(idx, signs):
                if assignment[i] == s:
                    return True
            return False
        
        # Find initially unsatisfied clauses
        unsat_clauses = [i for i in range(len(clauses)) if not is_sat(i)]
        
        if not unsat_clauses:
            return True
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False
            
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = random.randint(0, len(unsat_clauses) - 1)
            clause = unsat_clauses[ci]
            idx = clause_indices[clause]
            signs = clause_signs[clause]
            
            # Score each literal in the clause: count how many other unsat clauses
            # would be satisfied if we flip this variable
            best_var = -1
            best_score = -1
            best_flip = True
            
            for var_idx, sign in zip(idx, signs):
                # If we flip this variable, it satisfies the current clause
                # Count how many other unsat clauses contain this variable
                # and would become satisfied
                score = 0
                for uc in unsat_clauses:
                    if uc == clause:
                        continue
                    uc_idx = clause_indices[uc]
                    uc_signs = clause_signs[uc]
                    for vi, us in zip(uc_idx, uc_signs):
                        if vi == var_idx:
                            # Check if flipping would satisfy this clause
                            # Current assignment has var_idx = assignment[var_idx]
                            # After flip: not assignment[var_idx]
                            # This clause is satisfied by var_idx if (not assignment[var_idx]) == us
                            if (not assignment[var_idx]) == us:
                                score += 1
                            break
                
                # Also consider: flipping might break other clauses
                # (but min-conflicts just counts new satisfactions)
                if score > best_score:
                    best_score = score
                    best_var = var_idx
                    best_flip = True
            
            if best_var == -1:
                # Fallback: random literal
                var_idx = idx[random.randint(0, len(idx) - 1)]
            else:
                var_idx = best_var
            
            # With probability p, flip randomly; otherwise use heuristic
            if random.random() < p_flip_random:
                var_idx = idx[random.randint(0, len(idx) - 1)]
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat_clauses: check which clauses changed status
            new_unsat = []
            for uc in unsat_clauses:
                if not is_sat(uc):
                    new_unsat.append(uc)
            # Check if any previously sat clause became unsat
            for uc in range(len(clauses)):
                if uc not in unsat_clauses:
                    if not is_sat(uc):
                        new_unsat.append(uc)
            unsat_clauses = new_unsat
        
        return len(unsat_clauses) == 0
    
    # Restart strategy with increasing randomness
    base_steps = 5000
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and restarts < max_restarts:
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Decrease p_flip_random over restarts (more deterministic later)
        p = max(0.05, 0.5 * (0.8 ** restarts))
        steps = base_steps + restarts * 1000
        
        if walk_sat_trial(assignment, steps, p):
            return assignment
        
        restarts += 1
    
    # Return best effort: last assignment or all-true
    if 'assignment' in locals():
        return assignment
    return [True] * n
# EVOLVE-BLOCK-END
