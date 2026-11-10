import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    import random
    rng = random.Random(42)
    
    # WalkSAT parameters
    max_restarts = 50
    max_steps = 20000
    
    def count_unsat(assignment):
        unsat_clauses = []
        for c in clauses:
            sat = False
            for lit in c:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(c)
        return unsat_clauses

    def walk_sat():
        for restart in range(max_restarts):
            assignment = [rng.random() < 0.5 for _ in range(n)]
            unsat_clauses = count_unsat(assignment)
            if not unsat_clauses:
                return assignment
                
            for step in range(max_steps):
                if not unsat_clauses:
                    return assignment
                    
                clause = rng.choice(unsat_clauses)
                
                best_var = -1
                best_score = float('inf')
                
                for lit in clause:
                    var = abs(lit) - 1
                    score = 0
                    for other_clause in unsat_clauses:
                        if other_clause is clause:
                            continue
                        for o_lit in other_clause:
                            if abs(o_lit) - 1 == var:
                                score += 1
                                break
                    if score < best_score:
                        best_score = score
                        best_var = var
                        
                if best_var != -1:
                    assignment[best_var] = not assignment[best_var]
                    
                unsat_clauses = count_unsat(assignment)
                
        return None

    solution = walk_sat()
    if solution is not None:
        return solution
    else:
        # Fallback: random assignment
        return [rng.random() < 0.5 for _ in range(n)]
