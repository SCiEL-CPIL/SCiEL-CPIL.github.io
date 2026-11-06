"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Deadline guard: 330s limit, use 325s to be safe
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause indices for faster access
    # clauses are lists of signed literals
    # We need to check satisfaction efficiently
    
    def check_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def count_unsat(clauses, assignment):
        count = 0
        for c in clauses:
            if not check_satisfied(c, assignment):
                count += 1
        return count

    best_assignment = None
    min_unsat = float('inf')
    
    # WalkSAT with restarts
    max_steps_per_run = 5000
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment
        # Phase saving: start with previous best or random
        if best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb the best assignment slightly
            assignment = list(best_assignment)
            for _ in range(n // 5):
                idx = rng.randint(0, n - 1)
                assignment[idx] = not assignment[idx]
        
        # Walk
        for step in range(max_steps_per_run):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses:
                if not check_satisfied(c, assignment):
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                # Solved
                return assignment
            
            # Track best so far
            current_unsat = len(unsat_clauses)
            if current_unsat < min_unsat:
                min_unsat = current_unsat
                best_assignment = list(assignment)
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Pick a literal to flip
            # Heuristic: flip a literal that appears in the fewest unsatisfied clauses?
            # Or just random WalkSAT?
            # Let's stick to standard WalkSAT: random literal from the clause
            literal = rng.choice(clause)
            assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
            
        if time.time() > deadline:
            break

    # If we didn't find a solution, return the best assignment found (which might not be satisfying, 
    # but the function contract says "Return a satisfying assignment...". 
    # If we return a non-satisfying one, it scores 0 for that instance.
    # However, if we are out of time, we have no choice.
    # If best_assignment is None, return a random one.
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
# EVOLVE-BLOCK-END
