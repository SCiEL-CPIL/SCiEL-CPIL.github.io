import time
import random
import sys

def solve(instance):
    START = time.time()
    DEADLINE = START + 290.0  # Safety margin

    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess: Convert literals to (var_index, polarity)
    # var_index is 0-based, polarity True for positive
    clause_data = []
    for c in clauses:
        cv = []
        for lit in c:
            v = abs(lit) - 1
            pol = lit > 0
            cv.append((v, pol))
        clause_data.append(cv)
    
    m = len(clause_data)

    # Helper: Check if a clause is satisfied by an assignment
    def is_satisfied(clause, assignment):
        for v, pol in clause:
            if assignment[v] is not None and assignment[v] == pol:
                return True
        return False

    # Helper: Check if a clause is falsified (all literals False)
    def is_falsified(clause, assignment):
        for v, pol in clause:
            if assignment[v] is not None and assignment[v] != pol:
                return True
        return False

    def dpll(assignment, active_clauses):
        """
        Recursive DPLL solver.
        assignment: list of bool or None
        active_clauses: list of clauses that are not yet satisfied or falsified
        Returns: assignment if SAT, None if UNSAT or timeout
        """
        if time.time() > DEADLINE:
            return None

        # 1. Unit Propagation and Pure Literal Elimination
        # We need to update active_clauses as we assign variables.
        # It's more efficient to maintain a list of active clauses and filter it.
        
        changed = True
        while changed:
            changed = False
            
            # Unit Propagation
            new_active = []
            for clause in active_clauses:
                # Check if clause is satisfied
                sat = False
                unassigned_lits = []
                for v, pol in clause:
                    if assignment[v] is not None:
                        if assignment[v] == pol:
                            sat = True
                            break
                    else:
                        unassigned_lits.append((v, pol))
                
                if sat:
                    continue # Clause satisfied, remove from active
                
                # Check if clause is falsified (all assigned and false)
                if len(unassigned_lits) == 0:
                    return None # Conflict
                
                # If only one unassigned literal, it's a unit clause
                if len(unassigned_lits) == 1:
                    v, pol = unassigned_lits[0]
                    if assignment[v] is not None:
                        # Should not happen if we check sat first, but safety
                        continue
                    assignment[v] = pol
                    changed = True
                    # Don't add to new_active yet, it will be re-evaluated or satisfied/falsified
                    # Actually, if we assign it, the clause becomes satisfied.
                    # So we don't need to keep it.
                    continue
                
                new_active.append(clause)
            
            active_clauses = new_active
            
            if not active_clauses:
                return assignment # All clauses satisfied

            # Pure Literal Elimination
            # Count occurrences of each variable polarity in active clauses
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in active_clauses:
                for v, pol in clause:
                    if assignment[v] is None:
                        if pol:
                            pos_count[v] += 1
                        else:
                            neg_count[v] += 1
            
            pure_found = False
            for v in range(n):
                if assignment[v] is None:
                    if pos_count[v] > 0 and neg_count[v] == 0:
                        assignment[v] = True
                        pure_found = True
                    elif neg_count[v] > 0 and pos_count[v] == 0:
                        assignment[v] = False
                        pure_found = True
            
            if pure_found:
                changed = True
                # Need to filter active_clauses again because assignments changed
                # The loop above will handle it in the next iteration
                continue

        # If no more propagation, we need to branch
        # If active_clauses is empty, we are done
        if not active_clauses:
            return assignment

        # Variable Selection Heuristic: Most Constrained (highest degree in active clauses)
        var_counts = [0] * n
        for clause in active_clauses:
            for v, pol in clause:
                if assignment[v] is None:
                    var_counts[v] += 1
        
        # Find variable with max count
        max_count = -1
        branch_var = -1
        for v in range(n):
            if assignment[v] is None and var_counts[v] > max_count:
                max_count = var_counts[v]
                branch_var = v
        
        if branch_var == -1:
            return assignment # Should not happen if active_clauses not empty

        # Branch 1: Try True
        assignment[branch_var] = True
        # Filter active clauses
        new_active_1 = []
        conflict = False
        for clause in active_clauses:
            sat = False
            fals = True
            for v, pol in clause:
                if assignment[v] is not None:
                    if assignment[v] == pol:
                        sat = True
                        break
                else:
                    fals = False
            if sat:
                continue
            if fals:
                conflict = True
                break
            new_active_1.append(clause)
        
        if not conflict:
            result = dpll(assignment, new_active_1)
            if result is not None:
                return result
        
        # Branch 2: Try False
        assignment[branch_var] = False
        new_active_2 = []
        conflict = False
        for clause in active_clauses:
            sat = False
            fals = True
            for v, pol in clause:
                if assignment[v] is not None:
                    if assignment[v] == pol:
                        sat = True
                        break
                else:
                    fals = False
            if sat:
                continue
            if fals:
                conflict = True
                break
            new_active_2.append(clause)
        
        if not conflict:
            result = dpll(assignment, new_active_2)
            if result is not None:
                return result
        
        # Backtrack
        assignment[branch_var] = None
        return None

    # Initial call
    initial_assignment = [None] * n
    result = dpll(initial_assignment, clause_data)
    
    if result is not None:
        # Fill any unassigned variables with False (or True, doesn't matter as all clauses are sat)
        final_assignment = [a if a is not None else False for a in result]
        return final_assignment

    # If DPLL failed (timeout or UNSAT - though should be SAT), fallback to WalkSAT
    # Implementing a quick WalkSAT as fallback
    def walk_sat_fallback():
        rng = random.Random(42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        def count_unsat(assign):
            cnt = 0
            for cv in clause_data:
                sat = False
                for v, pol in cv:
                    if assign[v] == pol:
                        sat = True
                        break
                if not sat:
                    cnt += 1
            return cnt
        
        best_assign = assignment[:]
        best_uc = count_unsat(assignment)
        
        for _ in range(100000):
            if time.time() > DEADLINE:
                break
            if best_uc == 0:
                return best_assign
            
            # Pick random unsat clause
            unsat_clauses = [i for i, cv in enumerate(clause_data) if not any(assignment[v] == pol for v, pol in cv)]
            if not unsat_clauses:
                return assignment
            ci = rng.choice(unsat_clauses)
            cv = clause_data[ci]
            
            # Pick random literal in clause to flip
            v, pol = rng.choice(cv)
            
            # Flip
            assignment[v] = pol
            
            uc = count_unsat(assignment)
            if uc < best_uc:
                best_uc = uc
                best_assign = assignment[:]
            else:
                # With some probability, keep the flip (noise)
                if rng.random() > 0.5:
                    assignment[v] = not assignment[v]
        
        return best_assign

    return walk_sat_fallback()
