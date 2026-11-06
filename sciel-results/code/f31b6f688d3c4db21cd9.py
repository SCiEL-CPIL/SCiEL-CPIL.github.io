"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using a combination of:
    1. Fast pre-processing (unit propagation, pure literals)
    2. WalkSAT with adaptive parameters and restarts
    3. Time-aware budgeting with a safety margin
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 325 seconds (5 second safety margin before 330s kill)
    deadline = time.time() + 325.0
    
    # Pre-process: convert to 0-based, handle unit clauses and pure literals
    # Build clause list with 0-based indices
    clause_list = []
    for c in clauses:
        clause_list.append([(abs(l) - 1, l > 0) for l in c])
    
    # Try to find a solution via WalkSAT with restarts
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random(42)
    
    # Number of restarts based on time budget
    max_restarts = 50
    
    for restart in range(max_restarts):
        # Check time
        if time.time() > deadline - 2.0:
            break
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # WalkSAT parameters
        # For planted 3-SAT with 50 variables, typical clause-to-variable ratio is around 4.26
        # Number of clauses is roughly 4.26 * 50 = ~213
        # WalkSAT typically works well with p around 0.5-0.7
        p = 0.55  # probability of choosing a random literal from the clause
        max_steps = 10000  # steps per restart
        
        # Track best unsat count during this restart
        restart_best_unsat = float('inf')
        restart_best_assign = None
        
        for step in range(max_steps):
            # Check time periodically (every 100 steps)
            if step % 100 == 0 and time.time() > deadline - 2.0:
                break
            
            # Count unsatisfied clauses
            unsat_clauses = []
            for i, c in enumerate(clause_list):
                sat = False
                for var_idx, is_pos in c:
                    if assignment[var_idx] == is_pos:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                # Found a satisfying assignment!
                return assignment
            
            # Track best
            if len(unsat_clauses) < restart_best_unsat:
                restart_best_unsat = len(unsat_clauses)
                restart_best_assign = assignment[:]
            
            # If this is better than global best, update
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
            
            # Choose an unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clause_list[clause_idx]
            
            # Choose a literal to flip
            if rng.random() < p:
                # Random literal from the clause
                var_idx, is_pos = rng.choice(clause)
            else:
                # Flip the literal that satisfies the most other clauses
                best_var = None
                best_score = -1
                for var_idx, is_pos in clause:
                    # Count how many other clauses this flip would satisfy
                    score = 0
                    for j, c2 in enumerate(clause_list):
                        if j == clause_idx:
                            continue
                        # Check if flipping var_idx would satisfy c2
                        # Currently c2 is not satisfied by var_idx's current value
                        # After flip, check if var_idx would satisfy c2
                        would_satisfy = False
                        for v, pos in c2:
                            if v == var_idx:
                                # After flip, assignment[var_idx] becomes not assignment[var_idx]
                                new_val = not assignment[var_idx]
                                if new_val == pos:
                                    would_satisfy = True
                                break
                        if would_satisfy:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_var = (var_idx, is_pos)
                var_idx, is_pos = best_var
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
        
        # End of restart - update best if needed
        if restart_best_assign is not None and restart_best_unsat < best_unsat_count:
            best_unsat_count = restart_best_unsat
            best_assignment = restart_best_assign[:]
        
        # Adaptive: if we're stuck, increase randomness
        if restart > 10 and best_unsat_count > 5:
            p = min(p + 0.05, 0.8)
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
