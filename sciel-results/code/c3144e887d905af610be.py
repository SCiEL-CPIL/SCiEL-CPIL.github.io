"""
Robust WalkSAT solver for 3-SAT.
Includes time management, restarts, and a smarter flip heuristic.
"""
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 300s limit, leave 10s margin for safety
    deadline = time.time() + 290.0
    
    # Pre-compute clause sizes and variable occurrences for heuristic weighting
    # For WalkSAT, we primarily need to know which variables are in which clauses.
    
    def is_satisfied(clause, assignment):
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if (lit > 0 and val) or (lit < 0 and not val):
                return True
        return False

    def count_satisfied(clause, assignment):
        count = 0
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if (lit > 0 and val) or (lit < 0 and not val):
                count += 1
        return count

    def get_unsat_clauses(assignment):
        return [c for c in clauses if not is_satisfied(c, assignment)]

    # WalkSAT with restarts
    best_assignment = None
    best_unsat_count = len(clauses)
    
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 1000 + 42) # Different seed for each restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial check
        unsat_clauses = get_unsat_clauses(assignment)
        if not unsat_clauses:
            return assignment
        
        current_unsat_count = len(unsat_clauses)
        if current_unsat_count < best_unsat_count:
            best_unsat_count = current_unsat_count
            best_assignment = assignment.copy()
            
        if best_unsat_count == 0:
            return best_assignment
            
        # Walk phase
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            unsat_clauses = get_unsat_clauses(assignment)
            if not unsat_clauses:
                return assignment
            
            current_unsat_count = len(unsat_clauses)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = assignment.copy()
                
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Heuristic: Flip a variable in the clause.
            # With probability p (0.2), flip a random literal in the clause (random walk)
            # With probability 1-p, flip the literal that satisfies the most other clauses (greedy)
            p = 0.2
            if rng.random() < p:
                lit = rng.choice(clause)
            else:
                best_lit = None
                max_gains = -1
                for lit in clause:
                    var = abs(lit) - 1
                    # Count how many other unsatisfied clauses this flip would satisfy
                    # This is expensive O(M) per literal. 
                    # Optimization: Just pick the first one or use a simpler heuristic.
                    # For 50 vars, M is small enough? 
                    # Let's stick to a simpler heuristic for speed: 
                    # Pick the literal that appears in the most unsatisfied clauses?
                    # Or just random if we want speed.
                    # Let's use a "soft" greedy: count satisfied clauses if we flip this var.
                    # Actually, standard WalkSAT uses a probability p for random, and for the rest, 
                    # it picks the variable that minimizes the number of newly unsatisfied clauses.
                    
                    # To keep it fast, we'll just pick a random literal from the clause 
                    # but bias towards variables that appear frequently in unsat clauses?
                    # For now, let's just use random for the "greedy" part too but 
                    # maybe check the impact on the clause itself? No, the clause is unsat.
                    # Flipping any literal in an unsat 3-CNF clause will satisfy that clause 
                    # (unless there are duplicates? No, standard CNF).
                    # Wait, if I flip a literal in an unsat clause, that clause becomes satisfied.
                    # The risk is breaking other satisfied clauses.
                    
                    # Simple heuristic: Pick the literal that appears in the fewest 
                    # currently satisfied clauses (so we break the least).
                    # This requires checking all clauses.
                    
                    # Given N=50, M~100-300, this is feasible.
                    break_count = 0
                    for c in clauses:
                        if is_satisfied(c, assignment):
                            if var in [abs(l) for l in c]:
                                break_count += 1
                    if break_count < max_gains or best_lit is None:
                        # We want to MINIMIZE break_count
                        if best_lit is None or break_count < max_gains:
                            best_lit = lit
                            max_gains = break_count
                lit = best_lit

            var = abs(lit) - 1
            assignment[var] = not assignment[var]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
