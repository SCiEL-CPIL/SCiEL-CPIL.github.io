<<<<<<< SEARCH
    def walksat_comp(assign_list, var_map, clause_indices, time_limit, rng):
        """
        WalkSAT for a subset of clauses with incremental delta computation.
        assign_list: list of bools for the component variables
        var_map: dict local_idx -> global_var
        clause_indices: list of global clause indices
        Returns: list of bools or None
        """
        local_n = len(var_map)
        
        # Precompute local clause structure with efficient lookup
        var_to_local = {v: i for i, v in var_map.items()}
        
        local_clauses = []
        for ci in clause_indices:
            local_clause = []
            for var, sign in clauses_internal[ci]:
                if var in var_to_local:
                    local_clause.append((var_to_local[var], sign))
            local_clauses.append(local_clause)
        
        # Build local variable-to-clause index for delta computation
        local_var_to_clauses = [[] for _ in range(local_n)]
        for ci, lc in enumerate(local_clauses):
            for li, sign in lc:
                local_var_to_clauses[li].append((ci, sign))
        
        def local_unsat_count(assign):
            count = 0
            for lc in local_clauses:
                sat = False
                for li, sign in lc:
                    if assign[li] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        def compute_delta(li, new_val, assign):
            """Compute delta (change in unsatisfied count) when flipping var li to new_val."""
            delta = 0
            for ci, sign in local_var_to_clauses[li]:
                lc = local_clauses[ci]
                # Check if clause is currently satisfied
                was_sat = False
                for lli, s in lc:
                    if assign[lli] == (s == 1):
                        was_sat = True
                        break
                
                # Check if clause would be satisfied after flip
                would_be_sat = False
                for lli, s in lc:
                    if lli == li:
                        if new_val == (s == 1):
                            would_be_sat = True
                            break
                    else:
                        if assign[lli] == (s == 1):
                            would_be_sat = True
                            break
                
                if was_sat and not would_be_sat:
                    delta += 1
                elif not was_sat and would_be_sat:
                    delta -= 1
            return delta
        
        best_assign = None
        best_unsat = float('inf')
        noise_start = 0.45
        patience = 5000
        
        while time.time() < time_limit:
            # Initialize
            current = [rng.random() < 0.5 for _ in range(local_n)]
            current_unsat = local_unsat_count(current)
            run_best = current_unsat
            no_improve = 0
            
            for step in range(100000):
                if time.time() > time_limit:
                    break
                
                if current_unsat == 0:
                    return current
                
                if current_unsat < run_best:
                    run_best = current_unsat
                    no_improve = 0
                    if current_unsat < best_unsat:
                        best_unsat = current_unsat
                        best_assign = list(current)
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                # Find unsatisfied clauses
                unsat_clauses = []
                for ci, lc in enumerate(local_clauses):
                    sat = False
                    for li, sign in lc:
                        if current[li] == (sign == 1):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                
                if not unsat_clauses:
                    return current
                
                chosen = rng.choice(unsat_clauses)
                lc = local_clauses[chosen]
                
                # Pick variable to flip
                if rng.random() < noise_start:
                    candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                    if not candidates:
                        candidates = [li for li, sign in lc]
                    best_var = rng.choice(candidates)
                else:
                    # Min-delta heuristic with incremental computation
                    best_var = None
                    best_delta = float('inf')
                    for li, sign in lc:
                        if current[li] == (sign == 1):
                            continue
                        new_val = not current[li]
                        delta = compute_delta(li, new_val, current)
                        if delta < best_delta:
                            best_delta = delta
                            best_var = li
                    if best_var is None:
                        candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                        if not candidates:
                            candidates = [li for li, sign in lc]
                        best_var = rng.choice(candidates)
                
                current[best_var] = not current[best_var]
                current_unsat = local_unsat_count(current)
            
            noise_start = max(0.15, noise_start * 0.95)
        
        return best_assign
=======
    def walksat_comp(assign_list, var_map, clause_indices, time_limit, rng):
        """
        WalkSAT for a subset of clauses with fast incremental clause-state tracking.
        assign_list: list of bools for the component variables
        var_map: dict local_idx -> global_var
        clause_indices: list of global clause indices
        Returns: list of bools or None
        """
        local_n = len(var_map)
        
        # Precompute local clause structure
        var_to_local = {v: i for i, v in var_map.items()}
        
        local_clauses = []
        for ci in clause_indices:
            local_clause = []
            for var, sign in clauses_internal[ci]:
                if var in var_to_local:
                    local_clause.append((var_to_local[var], sign))
            local_clauses.append(local_clause)
        
        # Build local variable-to-clause index
        local_var_to_clauses = [[] for _ in range(local_n)]
        for ci, lc in enumerate(local_clauses):
            for li, sign in lc:
                local_var_to_clauses[li].append((ci, sign))
        
        num_clauses = len(local_clauses)
        
        def initial_state(assign):
            """Compute clause_sat_count and total_unsat for initial assignment."""
            clause_sat_count = np.zeros(num_clauses, dtype=np.int32)
            for ci, lc in enumerate(local_clauses):
                cnt = 0
                for li, sign in lc:
                    if assign[li] == (sign == 1):
                        cnt += 1
                clause_sat_count[ci] = cnt
            total_unsat = int(np.sum(clause_sat_count == 0))
            return clause_sat_count, total_unsat
        
        def flip_var(li, assign, clause_sat_count, total_unsat):
            """
            Flip variable li. Returns new total_unsat.
            Updates clause_sat_count in place.
            """
            new_val = not assign[li]
            old_unsat = total_unsat
            
            for ci, sign in local_var_to_clauses[li]:
                was_sat = clause_sat_count[ci] > 0
                # Count how many other literals in this clause are satisfied
                other_sat = 0
                for lli, s in local_clauses[ci]:
                    if lli != li and assign[lli] == (s == 1):
                        other_sat += 1
                
                if new_val == (sign == 1):
                    # This literal becomes satisfied
                    if was_sat:
                        # Already satisfied, just increment count
                        clause_sat_count[ci] += 1
                    else:
                        # Becomes satisfied
                        clause_sat_count[ci] = other_sat + 1
                        old_unsat -= 1
                else:
                    # This literal becomes unsatisfied
                    if was_sat:
                        if other_sat > 0:
                            # Still satisfied by other literals
                            clause_sat_count[ci] = other_sat
                        else:
                            # No longer satisfied
                            clause_sat_count[ci] = 0
                            old_unsat += 1
                    else:
                        # Still unsatisfied
                        clause_sat_count[ci] = other_sat  # should be 0
        
            assign[li] = new_val
            return old_unsat
        
        best_assign = None
        best_unsat = float('inf')
        noise_start = 0.45
        patience = 8000
        
        while time.time() < time_limit:
            # Initialize
            current = [rng.random() < 0.5 for _ in range(local_n)]
            clause_sat_count, current_unsat = initial_state(current)
            run_best = current_unsat
            no_improve = 0
            
            # Track unsatisfied clause indices for quick selection
            unsat_set = set()
            for ci in range(num_clauses):
                if clause_sat_count[ci] == 0:
                    unsat_set.add(ci)
            
            for step in range(100000):
                if time.time() > time_limit:
                    break
                
                if current_unsat == 0:
                    return current
                
                if current_unsat < run_best:
                    run_best = current_unsat
                    no_improve = 0
                    if current_unsat < best_unsat:
                        best_unsat = current_unsat
                        best_assign = list(current)
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                if not unsat_set:
                    return current
                
                chosen = rng.choice(list(unsat_set))
                lc = local_clauses[chosen]
                
                # Pick variable to flip
                if rng.random() < noise_start:
                    candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                    if not candidates:
                        candidates = [li for li, sign in lc]
                    best_var = rng.choice(candidates)
                else:
                    # Min-delta: prefer flipping a variable that satisfies this clause
                    # and doesn't break too many others
                    best_var = None
                    best_score = float('inf')
                    for li, sign in lc:
                        if current[li] == (sign == 1):
                            continue
                        # Count how many other clauses this var satisfies
                        sat_count = 0
                        for ci2, s2 in local_var_to_clauses[li]:
                            if clause_sat_count[ci2] == 1:
                                # Only this literal satisfies the clause
                                sat_count += 1
                        # We want to minimize sat_count (fewest clauses we'd break)
                        # and prefer making this clause satisfied
                        score = sat_count
                        if score < best_score:
                            best_score = score
                            best_var = li
                    if best_var is None:
                        candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                        if not candidates:
                            candidates = [li for li, sign in lc]
                        best_var = rng.choice(candidates)
                
                # Flip and update state
                old_unsat = current_unsat
                current_unsat = flip_var(best_var, current, clause_sat_count, current_unsat)
                
                # Update unsat_set
                # Check clauses affected by best_var
                for ci, sign in local_var_to_clauses[best_var]:
                    if clause_sat_count[ci] == 0:
                        unsat_set.add(ci)
                    elif ci in unsat_set:
                        unsat_set.discard(ci)
            
            noise_start = max(0.15, noise_start * 0.95)
        
        # Final polish: try to reduce remaining unsat with aggressive local search
        if best_assign is not None and best_unsat > 0:
            polish_time = min(2.0, time_limit - time.time())
            if polish_time > 0.5:
                current = list(best_assign)
                clause_sat_count, current_unsat = initial_state(current)
                unsat_set = set()
                for ci in range(num_clauses):
                    if clause_sat_count[ci] == 0:
                        unsat_set.add(ci)
                
                polish_deadline = time.time() + polish_time
                while time.time() < polish_deadline and current_unsat > 0:
                    if not unsat_set:
                        break
                    chosen = rng.choice(list(unsat_set))
                    lc = local_clauses[chosen]
                    # Try each variable in the clause, pick best delta
                    best_var = None
                    best_delta = float('inf')
                    for li, sign in lc:
                        if current[li] == (sign == 1):
                            continue
                        # Simulate flip
                        delta = 0
                        for ci2, s2 in local_var_to_clauses[li]:
                            was_sat = clause_sat_count[ci2] > 0
                            other_sat = 0
                            for lli, s in local_clauses[ci2]:
                                if lli != li and current[lli] == (s == 1):
                                    other_sat += 1
                            if (not current[li]) == (s2 == 1):
                                if was_sat:
                                    pass  # still sat
                                else:
                                    if other_sat > 0:
                                        delta -= 1
                            else:
                                if was_sat and other_sat == 0:
                                    delta += 1
                        if delta < best_delta:
                            best_delta = delta
                            best_var = li
                    if best_var is not None and best_delta < 0:
                        current_unsat = flip_var(best_var, current, clause_sat_count, current_unsat)
                        for ci, sign in local_var_to_clauses[best_var]:
                            if clause_sat_count[ci] == 0:
                                unsat_set.add(ci)
                            elif ci in unsat_set:
                                unsat_set.discard(ci)
                        if current_unsat < best_unsat:
                            best_unsat = current_unsat
                            best_assign = list(current)
                    else:
                        break
        
        return best_assign
>>>>>>> REPLACE

Now let me also improve the DPLL to be more efficient with proper unit propagation using a worklist, and add a final verification/polish step at the top level:

