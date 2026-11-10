# EVOLVE-BLOCK-START
import time

def solve(instance):
    """
    Solve a planted 3-SAT instance using a systematic DPLL solver with unit propagation.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # We use -1 for unassigned, 0 for False, 1 for True
    assignment = [-1] * n
    
    def dpll(clauses, level):
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                sat = False
                unassigned_lits = []
                
                for lit in clause:
                    var = abs(lit) - 1
                    val = assignment[var]
                    if val != -1:
                        if (val == 1) == (lit > 0):
                            sat = True
                            break
                    else:
                        unassigned_lits.append(lit)
                
                if sat:
                    continue
                
                if len(unassigned_lits) == 0:
                    return None # Conflict
                
                if len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    var = abs(lit) - 1
                    val = 1 if lit > 0 else 0
                    if assignment[var] != -1:
                        if assignment[var] != val:
                            return None # Conflict
                    else:
                        assignment[var] = val
                        changed = True
                else:
                    new_clauses.append(clause)
            
            clauses = new_clauses
            
            if not clauses:
                for v in range(n):
                    if assignment[v] == -1:
                        assignment[v] = 0
                return True

        # 2. Variable Selection (Most Frequent)
        var_counts = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == -1:
                    var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            for v in range(n):
                if assignment[v] == -1:
                    assignment[v] = 0
            return True

        var = max(var_counts, key=lambda v: var_counts[v])
        
        # Try True
        assignment[var] = 1
        if dpll(clauses, level + 1):
            return True
        
        # Try False
        assignment[var] = 0
        if dpll(clauses, level + 1):
            return True
        
        # Backtrack
        assignment[var] = -1
        return False

    if dpll(clauses, 0):
        return [bool(assignment[v]) for v in range(n)]
    else:
        return [False] * n

# EVOLVE-BLOCK-END
