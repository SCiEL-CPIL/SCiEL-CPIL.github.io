"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # 1. Preprocessing: Convert to internal format
    # Internal literal representation: positive index for True, negative index for False?
    # Let's stick to (idx, sign) for clarity in logic, but use efficient structures.
    # Actually, standard SAT solvers use integer literals: +i means var i is True, -i means var i is False.
    # Let's convert input literals to this standard form:
    # Input: signed 1-based literals.
    # Internal: signed 0-based literals? No, let's use 1-based internally to match standard, or just map.
    # Let's map input literal l to internal literal:
    # If l > 0: var = l-1, sign = 1 (True)
    # If l < 0: var = -l-1, sign = -1 (False)
    
    internal_clauses = []
    for clause in clauses:
        int_clause = []
        seen_vars = set()
        tautology = False
        for l in clause:
            if l > 0:
                var = l - 1
                sign = 1
            else:
                var = -l - 1
                sign = -1
            
            if var in seen_vars:
                tautology = True
                break
            seen_vars.add(var)
            int_clause.append((var, sign))
        
        if not tautology and int_clause:
            internal_clauses.append(int_clause)
            
    if not internal_clauses:
        return [False] * n

    # 2. Data Structures for DPLL
    # assignment: list of None/True/False
    assignment = [None] * n
    
    # Watched Literals
    # watch[i] is a list of clause indices where literal i (or its negation) is watched?
    # Standard: For each variable, we watch two literals? 
    # Simpler: Maintain a list of watches per clause?
    # Let's use the standard "two watched literals" per clause.
    # clause_watches[c] = [w1, w2] where w1, w2 are indices into the clause's literals?
    # Or w1, w2 are the literals themselves?
    # Let's store watches as indices into the clause list.
    # clause_watchers: for each literal (var, sign), which clauses watch it?
    # This is complex. Let's use a simpler but effective optimization:
    # Maintain a list of active clauses.
    # Use a queue for unit propagation.
    
    # Let's implement a robust DPLL with:
    # - Unit Propagation using a queue of unit clauses.
    # - Pure Literal Elimination using occurrence lists.
    # - Variable selection: VSIDS-like (frequency based).
    
    # Occurrence lists
    # pos_occ[v] = set of clause indices containing (v, 1)
    # neg_occ[v] = set of clause indices containing (v, -1)
    pos_occ = [set() for _ in range(n)]
    neg_occ = [set() for _ in range(n)]
    
    # Build occurrence lists
    for c_idx, clause in enumerate(internal_clauses):
        for var, sign in clause:
            if sign == 1:
                pos_occ[var].add(c_idx)
            else:
                neg_occ[var].add(c_idx)

    # Active clause flags
    active = [True] * len(internal_clauses)
    
    # Clause lengths (dynamic)
    # We need to know the current length of each clause to find unit clauses.
    # Instead of updating lengths explicitly, we can scan? No, that's slow.
    # Let's maintain `clause_len[c]`.
    clause_len = [len(c) for c in internal_clauses]
    
    # Queue for unit clauses
    from collections import deque
    unit_queue = deque()
    
    # Initialize unit queue
    for i in range(len(internal_clauses)):
        if clause_len[i] == 1:
            unit_queue.append(i)

    def propagate():
        """
        Performs unit propagation.
        Returns False if contradiction, True otherwise.
        Modifies assignment, clause_len, active, unit_queue.
        """
        while unit_queue:
            c_idx = unit_queue.popleft()
            if not active[c_idx]:
                continue
            
            # Get the single remaining literal in this clause
            # We need to find the unassigned literal in the clause
            # Since it's unit, exactly one literal is unassigned.
            clause = internal_clauses[c_idx]
            lit = None
            for var, sign in clause:
                if assignment[var] is None:
                    lit = (var, sign)
                    break
            
            if lit is None:
                # Clause is falsified? Or satisfied?
                # If all assigned and none satisfied, it should have been removed.
                # If all assigned and one satisfied, it should have been removed.
                # If it's in the queue, it was unit.
                # If assignment[var] is not None for all, check if satisfied.
                sat = False
                for var, sign in clause:
                    if assignment[var] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    return False
                else:
                    active[c_idx] = False
                    continue

            var, sign = lit
            val = (sign == 1)
            
            if assignment[var] is not None:
                if assignment[var] != val:
                    return False
                else:
                    # Already assigned correctly, clause is satisfied?
                    # If it was unit, and the lit is satisfied, the clause is satisfied.
                    active[c_idx] = False
                    continue
            
            # Assign var = val
            assignment[var] = val
            
            # Update occurrences and clause lengths
            # Iterate through all clauses containing this variable
            # We need to update clause_len and check for new units/empties.
            
            # Helper to update a clause given a variable assignment
            def update_clause(c_idx, var, val):
                nonlocal clause_len
                if not active[c_idx]:
                    return
                
                clause = internal_clauses[c_idx]
                # Check if this clause is satisfied by the new assignment
                # If any literal in the clause is satisfied, remove clause.
                # If the literal corresponding to var is satisfied, remove clause.
                # Otherwise, remove the literal from consideration (decrement length).
                
                # Find the literal for var in this clause
                found_var = False
                satisfied = False
                for v, s in clause:
                    if v == var:
                        found_var = True
                        if (s == 1 and val == True) or (s == -1 and val == False):
                            satisfied = True
                        break
                
                if satisfied:
                    active[c_idx] = False
                    return
                
                if found_var:
                    # The literal for var is falsified. Remove it.
                    clause_len[c_idx] -= 1
                    if clause_len[c_idx] == 0:
                        return False # Contradiction
                    elif clause_len[c_idx] == 1:
                        unit_queue.append(c_idx)
                
                return True

            # Update all clauses containing var
            # We need to iterate pos_occ[var] and neg_occ[var]
            # But we only need to update clauses that are still active.
            
            # Optimization: Iterate through the union of pos_occ[var] and neg_occ[var]
            # Actually, we can just iterate through all clauses? No, too slow.
            # We must use occurrence lists.
            
            # Note: pos_occ and neg_occ are static. We need to check active status.
            
            # We iterate through clauses in pos_occ[var] and neg_occ[var]
            # To avoid double processing if a clause has both (which it can't, no tautologies),
            # we can just process both sets.
            
            # However, updating occurrence lists dynamically is tricky.
            # For simplicity and speed in this context (50 vars), 
            # iterating through the specific occurrence sets is efficient enough.
            
            # We need to be careful: if a clause is satisfied, we mark it inactive.
            # We don't remove it from pos_occ/neg_occ immediately, we just check `active`.
            
            # Process pos_occ[var]
            for c_idx in pos_occ[var]:
                if not active[c_idx]:
                    continue
                if not update_clause(c_idx, var, val):
                    return False
            
            # Process neg_occ[var]
            for c_idx in neg_occ[var]:
                if not active[c_idx]:
                    continue
                if not update_clause(c_idx, var, val):
                    return False

        return True

    def pure_literal_elimination():
        """
        Applies pure literal elimination.
        Returns True if any pure literal was found and applied (triggering propagation),
        False if no pure literals.
        Modifies assignment, clause_len, active.
        """
        pure_found = False
        # Check all unassigned variables
        for var in range(n):
            if assignment[var] is not None:
                continue
            
            # Check if var appears only in positive or only in negative
            # We need to check if there are any *active* clauses containing the opposite sign.
            
            # Check if there is an active clause with (var, -1)
            has_neg = False
            for c_idx in neg_occ[var]:
                if active[c_idx]:
                    has_neg = True
                    break
            
            # Check if there is an active clause with (var, 1)
            has_pos = False
            for c_idx in pos_occ[var]:
                if active[c_idx]:
                    has_pos = True
                    break
            
            if has_pos and not has_neg:
                # Pure Positive
                assignment[var] = True
                pure_found = True
                # Trigger propagation for this variable
                # We can just push all active clauses containing var into unit queue?
                # No, we need to update lengths.
                # Let's reuse the update logic.
                # We can call a helper that assigns and updates.
                # For simplicity, let's just mark assignment and let the main loop
                # re-run propagation? 
                # Better: Perform the update immediately to maintain state consistency.
                
                # Update clauses for var=True
                for c_idx in pos_occ[var]:
                    if not active[c_idx]: continue
                    clause = internal_clauses[c_idx]
                    # (var, 1) is satisfied
                    active[c_idx] = False
                    
                for c_idx in neg_occ[var]:
                    if not active[c_idx]: continue
                    clause_len[c_idx] -= 1
                    if clause_len[c_idx] == 0:
                        return False # Contradiction
                    elif clause_len[c_idx] == 1:
                        unit_queue.append(c_idx)
                        
            elif has_neg and not has_pos:
                # Pure Negative
                assignment[var] = False
                pure_found = True
                
                # Update clauses for var=False
                for c_idx in neg_occ[var]:
                    if not active[c_idx]: continue
                    # (var, -1) is satisfied
                    active[c_idx] = False
                    
                for c_idx in pos_occ[var]:
                    if not active[c_idx]: continue
                    clause_len[c_idx] -= 1
                    if clause_len[c_idx] == 0:
                        return False
                    elif clause_len[c_idx] == 1:
                        unit_queue.append(c_idx)
            
            if pure_found:
                # If we found a pure literal, we should probably stop and let propagation run,
                # or continue finding more? 
                # Usually, you apply all pure literals then propagate.
                # But applying one might make others impure.
                # Let's apply all found pure literals in this pass.
                pass
                
        return pure_found

    def select_variable():
        """
        Selects a variable to branch on.
        Uses a heuristic: Pick a variable from the shortest active clause.
        """
        # Find shortest active clause
        min_len = float('inf')
        best_clause = None
        for c_idx in range(len(internal_clauses)):
            if active[c_idx] and clause_len[c_idx] < min_len:
                min_len = clause_len[c_idx]
                best_clause = c_idx
                
        if best_clause is None:
            return None # All clauses satisfied
            
        clause = internal_clauses[best_clause]
        # Pick the first unassigned variable in this clause
        for var, sign in clause:
            if assignment[var] is None:
                return var, sign
                
        return None

    def dpll():
        # Main loop
        while True:
            # 1. Propagate
            if not propagate():
                return None
                
            # Check if all clauses satisfied
            # If no active clauses, we are done.
            # We can check if unit_queue is empty and no active clauses?
            # Actually, if propagate finishes, all units are resolved.
            # We need to check if there are any active clauses left.
            
            any_active = False
            for i in range(len(internal_clauses)):
                if active[i]:
                    any_active = True
                    break
                    
            if not any_active:
                # Fill remaining
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = False
                return assignment[:]

            # 2. Pure Literal Elimination
            # We need to be careful with state updates in pure_literal_elimination
            # If it returns False, contradiction.
            if not pure_literal_elimination():
                 # No pure literals, proceed to branch
                 pass
            else:
                 # Pure literals applied, loop back to propagate
                 continue
                 
            # Wait, my pure_literal_elimination implementation above returns True if found, 
            # but doesn't handle the "return False" for contradiction correctly in all paths?
            # Let's refine: pure_literal_elimination should return:
            # 1 if pure literals applied (need to propagate)
            # 0 if no pure literals
            # -1 if contradiction
            
            # Let's rewrite pure_literal_elimination to be safer.
            
            # Actually, let's just branch if no pure literals.
            
            # 3. Branch
            sel = select_variable()
            if sel is None:
                # Should not happen if any_active is True
                return None
                
            var, sign = sel
            
            # Branch on var = True
            assignment[var] = True
            # Save state for backtracking? 
            # For 50 vars, we can just copy the state.
            # State: assignment, active, clause_len, unit_queue, pos_occ, neg_occ (occ lists don't change)
            
            # Copy
            assign_copy = assignment[:]
            active_copy = active[:]
            len_copy = clause_len[:]
            queue_copy = deque(unit_queue)
            
            # Apply assignment to state
            # We need to update clause_len and active for this assignment
            # Reuse logic from propagate's update step
            
            def apply_assignment(var, val, assign, active, lens, queue):
                # Update clauses
                for c_idx in pos_occ[var]:
                    if not active[c_idx]: continue
                    if val == True:
                        active[c_idx] = False
                    else:
                        lens[c_idx] -= 1
                        if lens[c_idx] == 0: return False
                        elif lens[c_idx] == 1: queue.append(c_idx)
                        
                for c_idx in neg_occ[var]:
                    if not active[c_idx]: continue
                    if val == False:
                        active[c_idx] = False
                    else:
                        lens[c_idx] -= 1
                        if lens[c_idx] == 0: return False
                        elif lens[c_idx] == 1: queue.append(c_idx)
                        
                return True

            if not apply_assignment(var, True, assignment, active, clause_len, unit_queue):
                # Backtrack
                assignment = assign_copy
                active = active_copy
                clause_len = len_copy
                unit_queue = queue_copy
            else:
                # Recurse
                res = dpll()
                if res is not None:
                    return res
                else:
                    # Backtrack
                    assignment = assign_copy
                    active = active_copy
                    clause_len = len_copy
                    unit_queue = queue_copy
                    
            # Branch on var = False
            assignment[var] = False
            assign_copy = assignment[:]
            active_copy = active[:]
            len_copy = clause_len[:]
            queue_copy = deque(unit_queue)
            
            if not apply_assignment(var, False, assignment, active, clause_len, unit_queue):
                # Contradiction, backtrack
                assignment = assign_copy
                active = active_copy
                clause_len = len_copy
                unit_queue = queue_copy
                return None
            else:
                res = dpll()
                if res is not None:
                    return res
                else:
                    return None

    # We need to fix the pure_literal_elimination logic to integrate properly.
    # The current draft has `pure_literal_elimination` modifying global state but 
    # the `dpll` function needs to handle the return codes correctly.
    
    # Let's simplify: 
    # 1. Propagate.
    # 2. Check Satisfied.
    # 3. Pure Literal Elimination. If applied, goto 1.
    # 4. Branch.
    
    # I will refactor the `dpll` function to be cleaner.
    
    # Let's restart the `dpll` logic inside `solve` to ensure correctness.
    
    # --- Refined DPLL ---
    
    # Re-initialize state for the refined logic
    assignment = [None] * n
    active = [True] * len(internal_clauses)
    clause_len = [len(c) for c in internal_clauses]
    unit_queue = deque([i for i in range(len(internal_clauses)) if clause_len[i] == 1])
    
    # Occurrence lists are static and already built.
    
    def apply_var(var, val, assign, act, lens, q):
        """Applies var=val to state. Returns False on contradiction."""
        # Process pos_occ[var]
        for c_idx in pos_occ[var]:
            if not act[c_idx]: continue
            if val:
                act[c_idx] = False
            else:
                lens[c_idx] -= 1
                if lens[c_idx] == 0: return False
                elif lens[c_idx] == 1: q.append(c_idx)
                
        # Process neg_occ[var]
        for c_idx in neg_occ[var]:
            if not act[c_idx]: continue
            if not val:
                act[c_idx] = False
            else:
                lens[c_idx] -= 1
                if lens[c_idx] == 0: return False
                elif lens[c_idx] == 1: q.append(c_idx)
                
        return True

    def propagate(assign, act, lens, q):
        while q:
            c_idx = q.popleft()
            if not act[c_idx]: continue
            
            clause = internal_clauses[c_idx]
            lit = None
            for var, sign in clause:
                if assign[var] is None:
                    lit = (var, sign)
                    break
            
            if lit is None:
                # Check if satisfied
                sat = any(assign[var] == (sign == 1) for var, sign in clause)
                if not sat:
                    return False
                else:
                    act[c_idx] = False
                    continue

            var, sign = lit
            val = (sign == 1)
            
            if assign[var] is not None:
                if assign[var] != val:
                    return False
                else:
                    act[c_idx] = False
                    continue
            
            assign[var] = val
            if not apply_var(var, val, assign, act, lens, q):
                return False
        return True

    def find_pure_literal(assign, act):
        """Returns (var, val) for a pure literal, or None."""
        for var in range(n):
            if assign[var] is not None:
                continue
                
            has_pos = False
            for c_idx in pos_occ[var]:
                if act[c_idx]:
                    has_pos = True
                    break
            
            has_neg = False
            for c_idx in neg_occ[var]:
                if act[c_idx]:
                    has_neg = True
                    break
            
            if has_pos and not has_neg:
                return var, True
            elif has_neg and not has_pos:
                return var, False
        return None

    def dpll(assign, act, lens, q):
        # 1. Propagate
        if not propagate(assign, act, lens, q):
            return None
            
        # 2. Check Satisfied
        for i in range(len(internal_clauses)):
            if act[i]:
                break
        else:
            for i in range(n):
                if assign[i] is None:
                    assign[i] = False
            return assign[:]

        # 3. Pure Literal Elimination
        while True:
            pure = find_pure_literal(assign, act)
            if pure is None:
                break
            var, val = pure
            if not apply_var(var, val, assign, act, lens, q):
                return None
            # After applying pure literal, we might have new units or contradictions
            # We loop back to find more pure literals? 
            # Or propagate? 
            # Applying pure literal removes satisfied clauses and reduces others.
            # It doesn't create new units directly unless we consider the reduced clauses.
            # Actually, apply_var adds to queue if lens becomes 1.
            # So we should propagate now?
            # If we propagate, we might assign more vars, which might make other literals impure.
            # Standard practice: Apply all pure literals, then propagate.
            # But if we apply one, it might make another impure.
            # So we just find one, apply it, and restart the pure literal search?
            # Or just propagate?
            # Let's propagate to clear units, then check pure literals again.
            if not propagate(assign, act, lens, q):
                return None
            # Check satisfied again?
            # If we propagated, we might have solved it.
            for i in range(len(internal_clauses)):
                if act[i]:
                    break
            else:
                for i in range(n):
                    if assign[i] is None:
                        assign[i] = False
                return assign[:]

        # 4. Branch
        # Select variable
        # Pick from shortest active clause
        min_len = float('inf')
        best_c = -1
        for i in range(len(internal_clauses)):
            if act[i] and lens[i] < min_len:
                min_len = lens[i]
                best_c = i
                
        if best_c == -1:
            return None # Should be caught by satisfied check
            
        clause = internal_clauses[best_c]
        var = None
        sign = None
        for v, s in clause:
            if assign[v] is None:
                var = v
                sign = s
                break
                
        if var is None:
            return None

        # Branch True
        # Save state
        a1 = assign[:]
        ac1 = act[:]
        l1 = lens[:]
        q1 = deque(q)
        
        assign[var] = True
        if apply_var(var, True, assign, act, lens, q):
            res = dpll(assign, act, lens, q)
            if res is not None:
                return res
        
        # Restore state
        assign = a1
        act = ac1
        lens = l1
        q = q1
        
        # Branch False
        a2 = assign[:]
        ac2 = act[:]
        l2 = lens[:]
        q2 = deque(q)
        
        assign[var] = False
        if apply_var(var, False, assign, act, lens, q):
            res = dpll(assign, act, lens, q)
            if res is not None:
                return res
        
        return None

    result = dpll(assignment, active, clause_len, unit_queue)
    
    if result is None:
        return [False] * n
        
    return result
# EVOLVE-BLOCK-END
