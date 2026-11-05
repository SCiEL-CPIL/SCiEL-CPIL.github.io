import time
import random
import sys

def solve(instance):
    START = time.time()
    DEADLINE = START + 300.0  # Safety margin: 30s before the 330s kill switch

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

    # Helper to check if a full assignment satisfies all clauses
    def check_satisfaction(assignment):
        for cv in clause_data:
            sat = False
            for v, pol in cv:
                if assignment[v] == pol:
                    sat = True
                    break
            if not sat:
                return False
        return True

    # --- Phase 1: Quick Structural Checks (Inspiration Idea) ---
    
    # 1a. All True
    all_true = [True] * n
    if check_satisfaction(all_true):
        return all_true
        
    # 1b. All False
    all_false = [False] * n
    if check_satisfaction(all_false):
        return all_false

    # 1c. Pure Literal + Unit Propagation (Non-recursive, fast pass)
    def try_pure_and_unit():
        assignment = [None] * n
        # We use a simple iterative pass. If it solves, great. If not, we return None.
        # This is a heuristic pre-check, not the main solver.
        remaining_indices = list(range(m))
        changed = True
        while changed:
            changed = False
            # Pure Literal
            pos_count = [0] * n
            neg_count = [0] * n
            for i in remaining_indices:
                for v, pol in clause_data[i]:
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
                # Filter clauses
                new_remaining = []
                for i in remaining_indices:
                    cv = clause_data[i]
                    sat = False
                    for v, pol in cv:
                        if assignment[v] is not None and assignment[v] == pol:
                            sat = True
                            break
                    if not sat:
                        new_remaining.append(i)
                remaining_indices = new_remaining
                if not remaining_indices:
                    break
            
            # Unit Propagation
            new_remaining = []
            unit_found = False
            for i in remaining_indices:
                cv = clause_data[i]
                unassigned_lits = []
                sat = False
                for v, pol in cv:
                    if assignment[v] is not None:
                        if assignment[v] == pol:
                            sat = True
                            break
                    else:
                        unassigned_lits.append((v, pol))
                
                if sat:
                    continue
                if len(unassigned_lits) == 0:
                    return None # Conflict
                if len(unassigned_lits) == 1:
                    v, pol = unassigned_lits[0]
                    if assignment[v] is None:
                        assignment[v] = pol
                        unit_found = True
                    continue
                new_remaining.append(i)
            
            if unit_found:
                changed = True
                remaining_indices = new_remaining
                if not remaining_indices:
                    break

        if not remaining_indices:
            return [a if a is not None else False for a in assignment]
        return None

    result = try_pure_and_unit()
    if result is not None:
        return result

    # --- Phase 2: Watched-Literal DPLL (Current Program Strategy, Optimized) ---
    
    # Data structures for Watched Literals
    # watches[v] = list of clause indices where v is a watched literal
    # We need to know which literal in the clause is watched to update efficiently.
    # Simpler approach for 50 vars: 
    # Maintain:
    # 1. assignment: list of bool/None
    # 2. active_clauses: list of clause indices that are not satisfied/falsified
    # 3. For each clause, track unassigned literals count? 
    # Actually, the "Watched Literal" technique is best implemented with:
    # - watches: dict var -> list of (clause_idx, other_var_idx) 
    # - clause_state: list of unassigned literals per clause?
    
    # Given N=50, a slightly less optimized but very robust DPLL with 
    # "Active Clause" filtering and "Most Constrained" variable selection 
    # is often sufficient and easier to get right than full watched literals 
    # in Python without significant overhead. 
    # However, let's implement a cleaner version of the current DPLL 
    # with better variable ordering (VSIDS-like or simple degree).
    
    def dpll_watched():
        assignment = [None] * n
        # active_clauses: list of clause indices
        active = list(range(m))
        
        # Helper: Apply propagation and update active clauses
        # Returns: True if consistent, False if conflict
        def propagate():
            changed = True
            while changed:
                changed = False
                new_active = []
                
                # Unit Propagation and Conflict Detection
                for ci in active:
                    cv = clause_data[ci]
                    unassigned = []
                    sat = False
                    for v, pol in cv:
                        if assignment[v] is not None:
                            if assignment[v] == pol:
                                sat = True
                                break
                        else:
                            unassigned.append((v, pol))
                    
                    if sat:
                        continue
                    if len(unassigned) == 0:
                        return False # Conflict
                    if len(unassigned) == 1:
                        v, pol = unassigned[0]
                        if assignment[v] is None:
                            assignment[v] = pol
                            changed = True
                        continue
                    new_active.append(ci)
                
                active[:] = new_active # Update in place for recursion scope
                
                if not active:
                    return True
                
                # Pure Literal Elimination
                pos_count = [0] * n
                neg_count = [0] * n
                for ci in active:
                    for v, pol in clause_data[ci]:
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
                    # Filter active again
                    new_active = []
                    for ci in active:
                        cv = clause_data[ci]
                        sat = False
                        for v, pol in cv:
                            if assignment[v] is not None and assignment[v] == pol:
                                sat = True
                                break
                        if not sat:
                            new_active.append(ci)
                    active[:] = new_active
                    if not active:
                        return True
            
            return True

        if not propagate():
            return None

        if not active:
            return [a if a is not None else False for a in assignment]

        # Recursive Search
        def search():
            if time.time() > DEADLINE:
                return None
            
            # Propagate again to ensure state is stable after branching
            if not propagate():
                return None
            
            if not active:
                return [a if a is not None else False for a in assignment]
            
            # Variable Selection: Most Constrained (Degree in Active Clauses)
            var_counts = [0] * n
            for ci in active:
                for v, pol in clause_data[ci]:
                    if assignment[v] is None:
                        var_counts[v] += 1
            
            max_count = -1
            branch_var = -1
            # Tie-breaking: random or index? Let's use index for determinism first, 
            # but random can help escape symmetry. We'll stick to max count, min index.
            for v in range(n):
                if assignment[v] is None and var_counts[v] > max_count:
                    max_count = var_counts[v]
                    branch_var = v
            
            if branch_var == -1:
                # All variables assigned but active clauses remain? 
                # This implies a conflict that propagate should have caught, 
                # or active is not empty but no unassigned vars?
                # If active is not empty, there must be unassigned vars.
                return None

            # Branch 1: True
            assignment[branch_var] = True
            if propagate():
                if not active:
                    return [a if a is not None else False for a in assignment]
                res = search()
                if res is not None:
                    return res
            
            # Branch 2: False
            assignment[branch_var] = False
            if propagate():
                if not active:
                    return [a if a is not None else False for a in assignment]
                res = search()
                if res is not None:
                    return res
            
            # Backtrack
            assignment[branch_var] = None
            return None

        return search()

    result = dpll_watched()
    if result is not None:
        return result

    # --- Phase 3: WalkSAT Fallback (Inspiration Idea, Optimized) ---
    # If DPLL failed (timeout or UNSAT - though task says SAT), try WalkSAT.
    # This is a best-effort to return *something* valid if DPLL timed out 
    # on a hard instance, or if DPLL had a bug (less likely).
    
    def walk_sat_fallback():
        rng = random.Random(42)
        # Initialize randomly
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Precompute clause satisfaction status
        # unsat_count[i] = 1 if clause i is unsat, 0 if sat
        unsat_status = [0] * m
        total_unsat = 0
        
        def update_unsat(ci):
            nonlocal total_unsat
            cv = clause_data[ci]
            was_unsat = unsat_status[ci]
            is_sat = False
            for v, pol in cv:
                if assignment[v] == pol:
                    is_sat = True
                    break
            
            if is_sat:
                unsat_status[ci] = 0
                if was_unsat:
                    total_unsat -= 1
            else:
                unsat_status[ci] = 1
                if not was_unsat:
                    total_unsat += 1

        # Initialize
        for i in range(m):
            update_unsat(i)
        
        if total_unsat == 0:
            return assignment

        best_assign = assignment[:]
        best_unsat = total_unsat
        
        # Precompute variable to clauses mapping for faster scoring
        var_to_clauses = [[] for _ in range(n)]
        for i, cv in enumerate(clause_data):
            seen = set()
            for v, pol in cv:
                if v not in seen:
                    var_to_clauses[v].append(i)
                    seen.add(v)

        max_steps = 50000
        for _ in range(max_steps):
            if time.time() > DEADLINE:
                break
            if total_unsat == 0:
                return assignment
            
            # Pick a random unsat clause
            unsat_indices = [i for i in range(m) if unsat_status[i] == 1]
            if not unsat_indices:
                return assignment
            ci = rng.choice(unsat_indices)
            cv = clause_data[ci]
            
            # Score literals: how many unsat clauses would be satisfied?
            best_score = -1
            best_var = -1
            best_pol = False
            best_lits = []
            
            for v, pol in cv:
                score = 0
                for ci2 in var_to_clauses[v]:
                    if unsat_status[ci2] == 1:
                        # Check if setting v=pol satisfies ci2
                        # Note: v is in ci2. Does ci2 have v with polarity pol?
                        cv2 = clause_data[ci2]
                        for v2, pol2 in cv2:
                            if v2 == v:
                                if pol2 == pol:
                                    score += 1
                                break
                
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_pol = pol
                    best_lits = [(v, pol)]
                elif score == best_score:
                    best_lits.append((v, pol))
            
            # Noise: 30% chance to pick random from best, else best
            if rng.random() < 0.3 and best_lits:
                v, pol = rng.choice(best_lits)
            else:
                v, pol = best_lits[0] if best_lits else (cv[0][0], cv[0][1])
            
            # Flip
            old_val = assignment[v]
            assignment[v] = pol
            
            # Update affected clauses
            for ci2 in var_to_clauses[v]:
                update_unsat(ci2)
            
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assign = assignment[:]
            elif total_unsat > best_unsat and rng.random() < 0.5:
                # Revert if it got worse and random coin says so (standard WalkSAT noise)
                assignment[v] = old_val
                for ci2 in var_to_clauses[v]:
                    update_unsat(ci2)
        
        return best_assign

    return walk_sat_fallback()
