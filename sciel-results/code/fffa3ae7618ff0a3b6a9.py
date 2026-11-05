import random
import time
import copy

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # DPLL Solver
    # assignment[i] corresponds to variable i+1
    # None: unassigned, True: True, False: False
    
    def unit_propagate(clauses, assignment):
        """
        Returns (new_clauses, new_assignment, contradiction)
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Check for satisfied
                sat = False
                unassigned = 0
                new_clause = []
                for lit in clause:
                    var = abs(lit) - 1
                    val = assignment[var]
                    if val is None:
                        unassigned += 1
                        new_clause.append(lit)
                    else:
                        if (lit > 0 and val == True) or (lit < 0 and val == False):
                            sat = True
                            break
                if sat:
                    continue
                if unassigned == 0:
                    return None, None, True # Contradiction
                if len(clause) == 1:
                    # Unit clause
                    lit = clause[0]
                    var = abs(lit) - 1
                    if assignment[var] is not None:
                        # Already assigned, check if consistent
                        if (lit > 0 and assignment[var] == False) or (lit < 0 and assignment[var] == True):
                            return None, None, True
                    else:
                        assignment[var] = (lit > 0)
                        changed = True
                else:
                    new_clauses.append(new_clause if new_clause else clause)
            clauses = new_clauses
        return clauses, assignment, False

    def pure_literal_elimination(clauses, assignment):
        """
        Returns new_clauses, new_assignment
        """
        pos_count = [0] * n
        neg_count = [0] * n
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if lit > 0:
                    pos_count[var] += 1
                else:
                    neg_count[var] += 1
        
        new_clauses = []
        for clause in clauses:
            new_clause = []
            for lit in clause:
                var = abs(lit) - 1
                if lit > 0:
                    if neg_count[var] == 0 and assignment[var] is None:
                        assignment[var] = True
                        continue # Remove from clause, clause is satisfied
                else:
                    if pos_count[var] == 0 and assignment[var] is None:
                        assignment[var] = False
                        continue # Remove from clause, clause is satisfied
                new_clause.append(lit)
            if new_clause:
                new_clauses.append(new_clause)
            # If clause becomes empty, it was satisfied by the pure literal assignment
        return new_clauses, assignment

    def dpll(clauses, assignment):
        # 1. Unit propagation
        clauses, assignment, contradiction = unit_propagate(clauses, assignment)
        if contradiction:
            return None
        
        # 2. Pure literal elimination
        clauses, assignment = pure_literal_elimination(clauses, assignment)
        
        # 3. Check if all clauses are satisfied
        if not clauses:
            for v in range(n):
                if assignment[v] is None:
                    assignment[v] = False
            return assignment
        
        # 4. Choose variable to branch
        var_counts = [0] * n
        for clause in clauses:
            for lit in clause:
                var_counts[abs(lit) - 1] += 1
        
        best_var = -1
        max_count = 0
        for v in range(n):
            if assignment[v] is None and var_counts[v] > max_count:
                max_count = var_counts[v]
                best_var = v
                
        if best_var == -1:
            for v in range(n):
                if assignment[v] is None:
                    assignment[v] = False
            for clause in clauses:
                sat = False
                for lit in clause:
                    var = abs(lit) - 1
                    if (lit > 0 and assignment[var] == True) or (lit < 0 and assignment[var] == False):
                        sat = True
                        break
                if not sat:
                    return None
            return assignment

        # 5. Branch
        # Try True
        assignment[best_var] = True
        clauses_copy = copy.deepcopy(clauses)
        assignment_copy = copy.deepcopy(assignment)
        
        result = dpll(clauses_copy, assignment_copy)
        if result is not None:
            return result
            
        # Try False
        assignment[best_var] = False
        clauses_copy = copy.deepcopy(clauses)
        assignment_copy = copy.deepcopy(assignment)
        
        result = dpll(clauses_copy, assignment_copy)
        if result is not None:
            return result
            
        return None

    assignment = [None] * n
    
    result = dpll(clauses, assignment)
    
    if result is None:
        return [random.random() < 0.5 for _ in range(n)]
        
    return result
