import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 325.0  # 5s margin before 330s kill
    
    m = len(clauses)
    if m == 0:
        return [False] * n
    
    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    
    for clause in clauses:
        vars_list = []
        signs_list = []
        for lit in clause:
            var_idx = abs(lit) - 1
            vars_list.append(var_idx)
            signs_list.append(lit > 0)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    # Build var_to_clauses: for each variable, list of (clause_idx, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            var_to_clauses[v].append((i, clause_signs[i][j]))
    
    # Precompute for each variable: list of clause indices it appears in
    # and the sign in each clause, as numpy arrays for fast indexing
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        cidx = [ci for ci, s in var_to_clauses[v]]
        csign = [s for ci, s in var_to_clauses[v]]
        var_clause_idx.append(np.array(cidx, dtype=np.int32) if cidx else np.array([], dtype=np.int32))
        var_clause_sign.append(np.array(csign, dtype=bool) if csign else np.array([], dtype=bool))
    
    def init_state(assignment_arr):
        """Compute clause_sat_count from a numpy bool array assignment."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            vs = clause_vars[i]
            ss = clause_signs[i]
            count = 0
            for j in range(3):
                if assignment_arr[vs[j]] == ss[j]:
                    count += 1
            clause_sat_count[i] = count
        return clause_sat_count
    
    def flip_var(v, assignment_arr, clause_sat_count):
        """Flip variable v and update clause_sat_count using precomputed arrays."""
        old_val = assignment_arr[v]
        assignment_arr[v] = not old_val
        new_val = assignment_arr[v]
        cidx = var_clause_idx[v]
        csign = var_clause_sign[v]
        if len(cidx) > 0:
            # For each clause containing v, check if new_val matches the sign
            matched = (csign == new_val)
            # Where matched: clause gains a satisfied literal (+1)
            # Where not matched: clause loses a satisfied literal (-1)
            deltas = np.where(matched, 1, -1).astype(np.int32)
            np.add.at(clause_sat_count, cidx, deltas)
    
    def unit_propagate(assignment_arr, clause_sat_count):
        """Set variables in clauses with 0 satisfied literals."""
        changed = True
        max_rounds = 20
        rounds = 0
        while changed and rounds < max_rounds:
            changed = False
            rounds += 1
            unsat_mask = (clause_sat_count == 0)
            if not np.any(unsat_mask):
                break
            unsat_indices = np.where(unsat_mask)[0]
            for i in unsat_indices:
                if clause_sat_count[i] != 0:
                    continue
                # Pick variable with least damage
                best_v = clause_vars[i][0]
                best_damage = float('inf')
                for v in clause_vars[i]:
                    damage = 0
                    for other_ci, sign in var_to_clauses[v]:
                        if other_ci != i and clause_sat_count[other_ci] > 0:
                            damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_v = v
                sign = None
                for j, v in enumerate(clause_vars[i]):
                    if v == best_v:
                        sign = clause_signs[i][j]
                        break
                if assignment_arr[best_v] != sign:
                    flip_var(best_v, assignment_arr, clause_sat_count)
                    changed = True
        return True
    
    def pure_literal(assignment_arr, clause_sat_count):
        """Find and assign pure literals."""
        pos_count = np.zeros(n, dtype=np.int32)
        neg_count = np.zeros(n, dtype=np.int32)
        for i in range(m):
            for j, v in enumerate(clause_vars[i]):
                if clause_signs[i][j]:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        
        changed = False
        # Pure positive: pos > 0, neg == 0 -> set True
        pure_pos = (pos_count > 0) & (neg_count == 0)
        for v in np.where(pure_pos)[0]:
            if not assignment_arr[v]:
                flip_var(int(v), assignment_arr, clause_sat_count)
                changed = True
        # Pure negative: neg > 0, pos == 0 -> set False
        pure_neg = (neg_count > 0) & (pos_count == 0)
        for v in np.where(pure_neg)[0]:
            if assignment_arr[v]:
                flip_var(int(v), assignment_arr, clause_sat_count)
                changed = True
        return changed
    
    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1
    
    # Vectorized WalkSAT with score-based clause selection
    # Key idea: precompute the "damage" for each variable in each clause
    # and use numpy to evaluate all 3 candidates at once
    
    while time.time() - start_time < timeout:
        # Random restart
        assignment_arr = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        clause_sat_count = init_state(assignment_arr)
        
        if np.all(clause_sat_count > 0):
            return assignment_arr.tolist()
        
        # Pure literal elimination
        pure_literal(assignment_arr, clause_sat_count)
        
        if np.all(clause_sat_count > 0):
            return assignment_arr.tolist()
        
        # Unit propagation
        unit_propagate(assignment_arr, clause_sat_count)
        
        if np.all(clause_sat_count > 0):
            return assignment_arr.tolist()
        
        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment_arr.copy()
        
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time < 5.0:
            break
        
        # Adaptive parameters
        if current_unsat <= 3:
            max_flips = 15000
            noise = 0.15
        elif current_unsat <= 10:
            max_flips = 10000
            noise = 0.25
        elif current_unsat <= 30:
            max_flips = 7000
            noise = 0.35
        else:
            max_flips = 5000
            noise = 0.45
        
        for flip_num in range(max_flips):
            if time.time() - start_time > timeout - 3.0:
                break
            
            # Find unsatisfied clauses
            unsat_mask = (clause_sat_count == 0)
            unsat_count = int(np.sum(unsat_mask))
            if unsat_count == 0:
                return assignment_arr.tolist()
            
            # Pick a random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            
            # Get the 3 variables in this clause
            vs = clause_vars[ci]
            ss = clause_signs[ci]
            
            if rng.random() < noise:
                # Random flip
                v = vs[rng.randrange(3)]
            else:
                # Score-based: for each of the 3 variables, compute the net change
                # in unsatisfied clause count if we flip it
                # delta_unsat = (clauses that become unsat) - (clauses that become sat)
                # For variable v in clause ci (which is currently unsat):
                #   Flipping v: 
                #     - ci: if new_val matches sign, ci becomes sat (good, -1 unsat)
                #     - other clauses containing v: 
                #       if they were sat and new_val doesn't match sign -> become unsat (bad, +1)
                #       if they were unsat and new_val matches sign -> become sat (good, -1)
                
                best_v = vs[0]
                best_score = float('inf')
                
                for v in vs:
                    new_val = not assignment_arr[v]
                    cidx = var_clause_idx[v]
                    csign = var_clause_sign[v]
                    
                    if len(cidx) == 0:
                        score = 0
                    else:
                        # Determine what happens to each clause containing v
                        matched = (csign == new_val)  # True if new assignment satisfies the literal
                        current_sat = clause_sat_count[cidx] > 0  # Is the clause currently satisfied?
                        
                        # Clauses that will become unsatisfied: currently sat, new_val doesn't match
                        newly_unsat = np.sum(current_sat & ~matched)
                        # Clauses that will become satisfied: currently unsat, new_val matches
                        newly_sat = np.sum(~current_sat & matched)
                        
                        score = newly_unsat - newly_sat
                    
                    if score < best_score:
                        best_score = score
                        best_v = v
                
                v = best_v
            
            flip_var(v, assignment_arr, clause_sat_count)
            
            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat == 0:
                return assignment_arr.tolist()
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment_arr.copy()
    
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
