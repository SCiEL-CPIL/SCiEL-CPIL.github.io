import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Convert to internal representation: list of sets of (var, sign)
    # var is 0-indexed, sign is 1 or -1
    internal_clauses = []
    for c in clauses:
        ic = set()
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            ic.add((v, s))
        internal_clauses.append(ic)
        
    m = len(internal_clauses)
    
    # Helper to check if DPLL should stop
    def time_check():
        return time.time() > deadline

    # DPLL implementation
    def dpll(clauses, assignment):
        # assignment is a list of (None, True, False) for each var
        if time_check():
            return None # Time out
        
        # Check for empty clause (contradiction)
        for c in clauses:
            if len(c) == 0:
                return None
        
        # Simplify clauses based on assignment
        new_clauses = []
        for c in clauses:
            new_c = set()
            satisfied = False
            for (v, s) in c:
                if assignment[v] is None:
                    new_c.add((v, s))
                else:
                    # Check if this literal is satisfied
                    if (assignment[v] == (s == 1)):
                        satisfied = True
                        break
            if not satisfied:
                new_clauses.append(new_c)
                
        if len(new_clauses) == 0:
            return assignment # All clauses satisfied
        
        clauses = new_clauses
        
        # Unit Propagation
        unit_vars = {}
        for c in clauses:
            if len(c) == 1:
                (v, s) = next(iter(c))
                if v in unit_vars and unit_vars[v] != (s == 1):
                    return None # Conflict
                unit_vars[v] = (s == 1)
                
        if unit_vars:
            # Apply unit propagation
            for v, val in unit_vars.items():
                assignment[v] = val
            return dpll(clauses, assignment)
            
        # Pure Literal Elimination
        var_occurrences = {}
        for c in clauses:
            for (v, s) in c:
                if v not in var_occurrences:
                    var_occurrences[v] = set()
                var_occurrences[v].add(s)
                
        pure_vars = {}
        for v, signs in var_occurrences.items():
            if len(signs) == 1:
                s = next(iter(signs))
                pure_vars[v] = (s == 1)
                
        if pure_vars:
            for v, val in pure_vars.items():
                assignment[v] = val
            return dpll(clauses, assignment)
            
        # Branching
        # Pick a variable
        # Heuristic: pick variable with highest frequency
        var_freq = {}
        for c in clauses:
            for (v, s) in c:
                var_freq[v] = var_freq.get(v, 0) + 1
                
        if not var_freq:
            return assignment
            
        v = max(var_freq, key=var_freq.get)
        
        # Branch on True
        assignment[v] = True
        res = dpll(clauses, assignment)
        if res is not None:
            return res
            
        # Branch on False
        assignment[v] = False
        res = dpll(clauses, assignment)
        if res is not None:
            return res
            
        # Backtrack
        assignment[v] = None
        return None

    # Initial assignment
    assignment = [None] * n
    
    result = dpll(internal_clauses, assignment)
    
    if result is not None:
        # Convert None to False just in case, though DPLL should fill all or return None
        return [bool(x) for x in result]
    else:
        # Fallback: Random assignment (should not happen for satisfiable instances if time is enough)
        # Or return the last partial assignment? No, must be valid.
        # If DPLL fails due to timeout, we might need a fallback.
        # For now, return a random assignment.
        return [random.random() < 0.5 for _ in range(n)]
