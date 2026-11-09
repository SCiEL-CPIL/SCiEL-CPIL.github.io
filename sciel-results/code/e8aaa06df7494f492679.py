import time

_GLOBAL_START = None
_GLOBAL_DEADLINE = None

def solve(instance):
    global _GLOBAL_START, _GLOBAL_DEADLINE
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START + 290.0

    n = instance["n"]
    clauses = instance["clauses"]
    
    # Check if we are already out of time
    if time.time() > _GLOBAL_DEADLINE:
        return [False] * n

    init_assign = [None] * n
    
    def dpll(clauses, assign):
        if time.time() > _GLOBAL_DEADLINE:
            return False, assign
            
        if not clauses:
            return True, assign
        
        # Unit Propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                filtered = []
                satisfied = False
                for lit in clause:
                    var = abs(lit) - 1
                    if assign[var] is not None:
                        if assign[var] == (lit > 0):
                            satisfied = True
                            break
                        else:
                            continue
                    else:
                        filtered.append(lit)
                
                if satisfied:
                    continue
                
                if len(filtered) == 0:
                    return False, assign
                
                if len(filtered) == 1:
                    lit = filtered[0]
                    var = abs(lit) - 1
                    assign[var] = (lit > 0)
                    changed = True
                else:
                    new_clauses.append(filtered)
            
            clauses = new_clauses
            if not clauses:
                return True, assign
        
        # Pure Literal Elimination
        pos_count = [0] * n
        neg_count = [0] * n
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if lit > 0:
                    pos_count[var] += 1
                else:
                    neg_count[var] += 1
        
        pure_vars = []
        for var in range(n):
            if assign[var] is None:
                if pos_count[var] > 0 and neg_count[var] == 0:
                    assign[var] = True
                    pure_vars.append(var)
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    assign[var] = False
                    pure_vars.append(var)
        
        if pure_vars:
            return dpll(clauses, assign)

        # Variable Selection: Max Degree
        var_counts = [0] * n
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assign[var] is None:
                    var_counts[var] += 1
        
        max_var = -1
        max_count = 0
        for var in range(n):
            if assign[var] is None and var_counts[var] > max_count:
                max_count = var_counts[var]
                max_var = var
        
        if max_var == -1:
            return False, assign

        # Find a literal for max_var in a shortest clause
        target_lit = None
        min_len = float('inf')
        for clause in clauses:
            for lit in clause:
                if abs(lit) - 1 == max_var:
                    if len(clause) < min_len:
                        min_len = len(clause)
                        target_lit = lit
        
        if target_lit is None:
            target_lit = max_var + 1

        lit = target_lit
        var = abs(lit) - 1
        
        # Branch 1
        assign_copy = assign.copy()
        assign_copy[var] = (lit > 0)
        sat1, _ = dpll(clauses, assign_copy)
        if sat1:
            return True, assign_copy
        
        # Branch 2
        assign_copy2 = assign.copy()
        assign_copy2[var] = not (lit > 0)
        sat2, final_assign = dpll(clauses, assign_copy2)
        if sat2:
            return True, final_assign
        
        return False, assign

    sat, result_assign = dpll(clauses, init_assign)
    
    if sat:
        final_assign = []
        for v in range(n):
            if result_assign[v] is None:
                final_assign.append(False)
            else:
                final_assign.append(result_assign[v])
        return final_assign
    else:
        return [False] * n
