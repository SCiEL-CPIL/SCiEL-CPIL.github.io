import time
import random
import numpy as np
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 325s to be safe
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Helper to check time
    def time_exceeded():
        return time.time() > deadline

    # --- Part 1: DPLL Solver ---
    
    # Convert clauses to list of lists of signed ints (1-based)
    # We will use 0-based indices internally for speed, but keep signs.
    # Literal: (var_idx, sign) where var_idx is 0..n-1, sign is 1 or -1
    
    def parse_clauses(raw_clauses):
        parsed = []
        for c in raw_clauses:
            pc = []
            for lit in c:
                var = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                pc.append((var, sign))
            parsed.append(pc)
        return parsed

    dpll_clauses = parse_clauses(clauses)
    
    # State:
    # assignment: dict var -> bool (True/False)
    # active_clauses: list of lists of (var, sign)
    
    def unit_propagation(assignment, active_clauses):
        """
        Performs unit propagation.
        Returns: (success, assignment, active_clauses)
        success: False if contradiction found.
        """
        changed = True
        while changed and not time_exceeded():
            changed = False
            new_active_clauses = []
            for clause in active_clauses:
                # Check if clause is satisfied
                sat = False
                unit_lit = None
                num_unassigned = 0
                for (var, sign) in clause:
                    if var in assignment:
                        val = assignment[var]
                        # Literal is satisfied if (val and sign>0) or (not val and sign<0)
                        if (val and sign > 0) or (not val and sign < 0):
                            sat = True
                            break
                    else:
                        num_unassigned += 1
                        if num_unassigned == 1:
                            unit_lit = (var, sign)
                
                if sat:
                    continue # Clause is satisfied, remove it
                
                if num_unassigned == 0:
                    # Clause is unsatisfied and no variables left to assign -> Contradiction
                    return False, assignment, active_clauses
                
                if num_unassigned == 1 and unit_lit is not None:
                    # Unit clause found
                    var, sign = unit_lit
                    # Assign var to satisfy the literal
                    assignment[var] = (sign > 0)
                    changed = True
                    # We don't add this clause to new_active_clauses because it's satisfied
                    # But we need to continue processing other clauses with the new assignment
                    # The loop structure above handles this by re-evaluating all clauses in the next pass
                    # However, modifying assignment inside the loop requires care.
                    # Standard approach: collect all unit assignments first? No, sequential is fine if we restart scan.
                    # Let's just break and restart the while loop to be safe and simple.
                    break 
                
                new_active_clauses.append(clause)
            
            if changed:
                active_clauses = new_active_clauses
            else:
                # If we didn't break, we processed all clauses and found no new units
                # But wait, if we didn't break, new_active_clauses contains all non-satisfied clauses.
                # If we did break, we need to restart.
                # Actually, the logic above is slightly flawed if we don't restart.
                # Let's refactor to a cleaner unit propagation.
                pass
        
        # Let's implement a cleaner Unit Propagation
        # It's easier to just loop until no change.
        
        return True, assignment, active_clauses

    # The above unit propagation is a bit messy. Let's write a robust one.
    
    def solve_dpll():
        assignment = {}
        # Filter out trivially satisfied clauses? No, we start with all.
        active_clauses = [list(c) for c in dpll_clauses]
        
        def propagate(assignment, active_clauses):
            # Returns (sat, assignment, active_clauses)
            # sat: None if no contradiction yet, False if contradiction, True if all satisfied
            changed = True
            while changed and not time_exceeded():
                changed = False
                new_clauses = []
                for clause in active_clauses:
                    # Count unassigned, check satisfaction
                    unassigned_lits = []
                    is_sat = False
                    for lit in clause:
                        var, sign = lit
                        if var in assignment:
                            val = assignment[var]
                            if (val and sign > 0) or (not val and sign < 0):
                                is_sat = True
                                break
                        else:
                            unassigned_lits.append(lit)
                    
                    if is_sat:
                        continue
                    
                    if len(unassigned_lits) == 0:
                        # Contradiction
                        return False, assignment, active_clauses
                    
                    if len(unassigned_lits) == 1:
                        var, sign = unassigned_lits[0]
                        assignment[var] = (sign > 0)
                        changed = True
                        # Don't add to new_clauses, it's satisfied
                        # But we must re-process other clauses with this new assignment
                        # So we break and restart the while loop
                        # This is inefficient but correct for small N.
                        # Optimization: Instead of breaking, we can just continue and mark that we need a full re-scan.
                        # But since assignment changed, previous "unassigned" counts for other clauses might be wrong?
                        # No, we are iterating over a snapshot.
                        # To be safe and simple for N=50, breaking and restarting is fine.
                        break
                    
                    new_clauses.append(unassigned_lits)
                
                if changed:
                    active_clauses = new_clauses
                    # Restart loop to pick up new units
                    continue
                else:
                    # No changes, we are done propagating
                    active_clauses = new_clauses
                    break
            
            # Check if all clauses are satisfied (i.e., active_clauses is empty)
            if len(active_clauses) == 0:
                return True, assignment, active_clauses
            
            return None, assignment, active_clauses # No contradiction, no full sat, need to branch
        
        # Main DPLL recursion
        def dpll_rec(assignment, active_clauses):
            if time_exceeded():
                return None, assignment
            
            # 1. Unit Propagation
            status, assignment, active_clauses = propagate(assignment, active_clauses)
            
            if status == False:
                return False, assignment
            if status == True:
                return True, assignment
            
            # 2. Pure Literal Elimination (Optional optimization)
            # Find variables that appear only with one sign in active_clauses
            # If found, assign them and remove satisfied clauses.
            # This can be done in a loop similar to unit propagation.
            # Given N=50, this might be overkill but adds robustness.
            # Let's skip for now to keep code simple, DPLL with unit prop is usually enough.
            
            # 3. Branching
            # Select a variable
            # Heuristic: Pick variable with highest degree in active_clauses
            var_freq = defaultdict(int)
            for clause in active_clauses:
                seen_in_clause = set()
                for lit in clause:
                    var = lit[0]
                    if var not in seen_in_clause:
                        var_freq[var] += 1
                        seen_in_clause.add(var)
            
            if not var_freq:
                # Should not happen if active_clauses is not empty
                return True, assignment
            
            # Pick variable with max frequency
            var_to_split = max(var_freq, key=var_freq.get)
            
            # Try True
            new_assign_true = assignment.copy()
            new_assign_true[var_to_split] = True
            sat, _ = dpll_rec(new_assign_true, [list(c) for c in active_clauses])
            if sat:
                # Merge assignments? We just need one satisfying assignment.
                # The 'assignment' dict in the caller is not updated by the recursion directly
                # We need to return the final assignment.
                # Let's modify dpll_rec to return the final assignment if SAT.
                pass # Placeholder, logic below is better
            
            # Let's rewrite dpll_rec to return (sat, final_assignment)
            
        # Revised DPLL Recursion
        def dpll_rec_v2(assignment, active_clauses):
            if time_exceeded():
                return None, assignment # Timeout
            
            # Unit Propagation
            status, assignment, active_clauses = propagate(assignment, active_clauses)
            
            if status == False:
                return False, assignment
            if status == True:
                return True, assignment
            
            # Branching
            var_freq = defaultdict(int)
            for clause in active_clauses:
                seen_in_clause = set()
                for lit in clause:
                    var = lit[0]
                    if var not in seen_in_clause:
                        var_freq[var] += 1
                        seen_in_clause.add(var)
            
            if not var_freq:
                return True, assignment
            
            var_to_split = max(var_freq, key=var_freq.get)
            
            # Try True
            new_assign = assignment.copy()
            new_assign[var_to_split] = True
            sat, final_assign = dpll_rec_v2(new_assign, [list(c) for c in active_clauses])
            if sat:
                return True, final_assign
            
            # Try False
            new_assign = assignment.copy()
            new_assign[var_to_split] = False
            sat, final_assign = dpll_rec_v2(new_assign, [list(c) for c in active_clauses])
            if sat:
                return True, final_assign
            
            return False, assignment
        
        sat, final_assign = dpll_rec_v2({}, [list(c) for c in dpll_clauses])
        
        if sat:
            # Convert final_assign (dict var->bool) to list of bools length n
            result = [False] * n
            for var, val in final_assign.items():
                result[var] = val
            # Fill any unassigned variables (shouldn't happen if SAT) with False
            return result
        
        # If DPLL failed or timed out, fall back to WalkSAT
        return None

    # --- Part 2: WalkSAT Fallback ---
    
    def solve_walksat():
        # Preprocess for WalkSAT
        m = len(clauses)
        clause_vars = np.empty((m, 3), dtype=np.int32)
        clause_signs = np.empty((m, 3), dtype=np.int8)
        for i, clause in enumerate(clauses):
            for j, lit in enumerate(clause):
                clause_vars[i, j] = abs(lit) - 1
                clause_signs[i, j] = 1 if lit > 0 else -1
        
        # Build variable-to-clause mapping
        var_to_clauses = [[] for _ in range(n)]
        for i in range(m):
            for j in range(3):
                v = clause_vars[i, j]
                var_to_clauses[v].append((i, j))
        
        rng = random.Random(42)
        
        best_assignment = None
        best_unsat_count = m + 1
        
        # Parameters
        max_restarts = 500
        base_step_budget = 5000
        
        for r in range(max_restarts):
            if time_exceeded():
                break
            
            # Random initial assignment
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
            
            # Evaluate
            def eval_clauses(assignment):
                # Vectorized evaluation
                assignment_flat = assignment[clause_vars]
                sign_bool = clause_signs > 0
                satisfied = assignment_flat == sign_bool
                clause_sat_counts = satisfied.sum(axis=1)
                unsat_clauses = np.where(clause_sat_counts == 0)[0]
                return unsat_clauses, clause_sat_counts
            
            unsat_clauses, clause_sat_counts = eval_clauses(assignment)
            current_unsat = len(unsat_clauses)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
            
            if current_unsat == 0:
                return assignment.tolist()
            
            # Adaptive noise: start high, decrease
            # p = 0.5 * (0.5 + 0.5 * (1 - r / max_restarts)) ?
            # Let's use a fixed p that works well, or adapt slightly.
            p = 0.5
            
            # Step budget
            step_budget = base_step_budget
            
            for step in range(step_budget):
                if time_exceeded():
                    return best_assignment.tolist() if best_assignment is not None else [False]*n
                
                if current_unsat == 0:
                    return assignment.tolist()
                
                # Pick random unsatisfied clause
                clause_idx = unsat_clauses[rng.randrange(len(unsat_clauses))]
                
                # Choose variable to flip
                if rng.random() < p:
                    # Random flip
                    lit_pos = rng.randrange(3)
                    var_idx = clause_vars[clause_idx, lit_pos]
                else:
                    # Heuristic flip: pick variable that maximizes gain
                    best_gain = -1
                    best_var = -1
                    for lit_pos in range(3):
                        v = clause_vars[clause_idx, lit_pos]
                        # Calculate gain: how many other unsatisfied clauses would be satisfied?
                        # This is expensive O(m).
                        # Approximation: pick variable with highest frequency in unsat clauses?
                        # For simplicity and speed, let's just pick randomly if p is low, 
                        # or use a cheaper heuristic.
                        # Let's stick to random for now if heuristic is too slow.
                        # Actually, let's try a simple gain:
                        # Flip v, check change in sat_count for clauses containing v.
                        # This is O(deg(v)).
                        pass
                    
                    # Fallback to random if heuristic is too complex
                    lit_pos = rng.randrange(3)
                    var_idx = clause_vars[clause_idx, lit_pos]
                
                # Flip
                old_val = assignment[var_idx]
                new_val = not old_val
                assignment[var_idx] = new_val
                
                # Update clause_sat_counts
                delta = 0
                for (ci, cj) in var_to_clauses[var_idx]:
                    sign = clause_signs[ci, cj]
                    # old_sat
                    if sign > 0:
                        old_sat = old_val
                    else:
                        old_sat = not old_val
                    
                    # new_sat
                    if sign > 0:
                        new_sat = new_val
                    else:
                        new_sat = not new_val
                    
                    if new_sat and not old_sat:
                        clause_sat_counts[ci] += 1
                    elif old_sat and not new_sat:
                        clause_sat_counts[ci] -= 1
                
                # Update unsat_clauses periodically
                if step % 10 == 0:
                    unsat_clauses = np.where(clause_sat_counts == 0)[0]
                    current_unsat = len(unsat_clauses)
                    
                    if current_unsat < best_unsat_count:
                        best_unsat_count = current_unsat
                        best_assignment = assignment.copy()
            
            # End of restart
            if best_unsat_count == 0:
                return best_assignment.tolist()
        
        if best_assignment is not None:
            return best_assignment.tolist()
        return [False] * n

    # Main Logic
    result = solve_dpll()
    if result is not None:
        # Verify result just in case
        is_sat = True
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                val = result[var]
                if (lit > 0 and val) or (lit < 0 and not val):
                    sat = True
                    break
            if not sat:
                is_sat = False
                break
        if is_sat:
            return result
        
    # Fallback to WalkSAT
    return solve_walksat()
