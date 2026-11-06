import sys
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based
    cnf = []
    for c in clauses:
        cnf.append([l - 1 if l > 0 else l + 1 for l in c])
        
    num_clauses = len(cnf)
    
    # Precompute variable to clauses map (not strictly needed for this DPLL variant, but good for debug)
    
    def dpll(clause_indices, assignment, depth=0):
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            new_unsat = []
            for ci in clause_indices:
                c = cnf[ci]
                unassigned = []
                satisfied = False
                for l in c:
                    v = abs(l)
                    if assignment[v] is not None:
                        if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                            satisfied = True
                            break
                    else:
                        unassigned.append(l)
                        
                if satisfied:
                    continue
                    
                if len(unassigned) == 0:
                    return None
                    
                if len(unassigned) == 1:
                    l = unassigned[0]
                    v = abs(l)
                    val = (l > 0)
                    if assignment[v] is not None:
                        if (l > 0 and not assignment[v]) or (l < 0 and assignment[v]):
                             return None
                    else:
                        assignment[v] = val
                        changed = True
                        continue
                else:
                    new_unsat.append(ci)
                    
            clause_indices = new_unsat
            if not clause_indices:
                return assignment

        # 2. Pure Literal Elimination
        pos_count = [0] * n
        neg_count = [0] * n
        for ci in clause_indices:
            c = cnf[ci]
            for l in c:
                v = abs(l)
                if assignment[v] is None:
                    if l > 0:
                        pos_count[v] += 1
                    else:
                        neg_count[v] += 1
                        
        changed_pure = True
        while changed_pure:
            changed_pure = False
            for v in range(n):
                if assignment[v] is None:
                    if pos_count[v] > 0 and neg_count[v] == 0:
                        assignment[v] = True
                        changed_pure = True
                    elif neg_count[v] > 0 and pos_count[v] == 0:
                        assignment[v] = False
                        changed_pure = True
            
            if changed_pure:
                new_unsat = []
                for ci in clause_indices:
                    c = cnf[ci]
                    satisfied = False
                    for l in c:
                        v = abs(l)
                        if assignment[v] is not None:
                            if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                                satisfied = True
                                break
                    if not satisfied:
                        all_assigned = True
                        for l in c:
                            v = abs(l)
                            if assignment[v] is None:
                                all_assigned = False
                                break
                        if all_assigned:
                            return None
                        new_unsat.append(ci)
                clause_indices = new_unsat
                if not clause_indices:
                    return assignment

        # 3. Variable Selection
        if not clause_indices:
            return assignment

        freq = [0] * n
        for ci in clause_indices:
            c = cnf[ci]
            for l in c:
                v = abs(l)
                if assignment[v] is None:
                    freq[v] += 1
                    
        best_var = -1
        max_freq = 0
        candidates = []
        for v in range(n):
            if assignment[v] is None and freq[v] > 0:
                if freq[v] > max_freq:
                    max_freq = freq[v]
                    candidates = [v]
                elif freq[v] == max_freq:
                    candidates.append(v)
                    
        if not candidates:
            return None
            
        best_var = random.choice(candidates)
        
        # Branch True
        assignment[best_var] = True
        res = dpll(clause_indices, assignment, depth + 1)
        if res is not None:
            return res
            
        # Branch False
        assignment[best_var] = False
        res = dpll(clause_indices, assignment, depth + 1)
        if res is not None:
            return res
            
        # Backtrack
        assignment[best_var] = None
        return None

    assignment = [None] * n
    start_time = time.time()
    
    init_unsat = list(range(num_clauses))
    
    result = dpll(init_unsat, assignment)
    
    if result is not None:
        return [bool(v) for v in result]
        
    # Fallback
    return [random.random() < 0.5 for _ in range(n)]
