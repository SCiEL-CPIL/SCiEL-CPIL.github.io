"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Safety margin: 330s limit, keep 300s for logic
    deadline = time.time() + 290.0

    # --- Preprocessing ---
    # Convert to 0-based literals for internal use? 
    # Let's keep 1-based for clarity in preprocessing, then convert for search.
    # Actually, 0-based is faster for array access.
    # Let's convert immediately.
    # Clause: list of signed 1-based ints -> list of (var_idx, is_pos)
    
    def preprocess(cnf_raw):
        """
        Applies unit propagation and pure literal elimination.
        Returns (solved_assignment, reduced_cnf, partial_assignment)
        solved_assignment: list of bools if fully solved, else None
        reduced_cnf: list of lists of (var_idx, is_pos)
        partial_assignment: list of bools/None
        """
        # Work with 0-based indices internally
        # cnf_raw is list of lists of 1-based signed ints
        # Convert to 0-based (var, sign) where sign is True for positive
        cnf = [[(abs(l)-1, l>0) for l in c] for c in cnf_raw]
        
        assignment = [None] * n
        
        changed = True
        while changed:
            changed = False
            
            # Remove satisfied clauses
            new_cnf = []
            for c in cnf:
                sat = False
                for v, s in c:
                    if assignment[v] is not None and assignment[v] == s:
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf = new_cnf
            
            # Check for empty clause (contradiction)
            for c in cnf:
                if len(c) == 0:
                    return None, None, assignment
            
            # Unit propagation
            for i, c in enumerate(cnf):
                if len(c) == 1:
                    v, s = c[0]
                    if assignment[v] is None:
                        assignment[v] = s
                        changed = True
                    elif assignment[v] != s:
                        return None, None, assignment # Contradiction
                    else:
                        cnf[i] = [] # Satisfied
                        changed = True
            
            # Pure literal elimination
            # Count occurrences of each literal sign
            lit_counts = [[0, 0] for _ in range(n)] # [pos_count, neg_count]
            for c in cnf:
                seen_vars = set()
                for v, s in c:
                    if v in seen_vars:
                        continue
                    seen_vars.add(v)
                    if s:
                        lit_counts[v][0] += 1
                    else:
                        lit_counts[v][1] += 1
            
            for v in range(n):
                if assignment[v] is not None:
                    continue
                pos_c, neg_c = lit_counts[v]
                if pos_c == 0 and neg_c > 0:
                    assignment[v] = False
                    changed = True
                elif neg_c == 0 and pos_c > 0:
                    assignment[v] = True
                    changed = True
        
        # Check if all assigned
        if all(a is not None for a in assignment):
            # Verify
            final_assign = [bool(a) for a in assignment]
            # Verify against original cnf_raw? Or reduced?
            # If no contradiction and all assigned, it should be satisfied.
            # But let's be safe and check against original.
            is_sat = True
            for c in cnf_raw:
                sat = False
                for l in c:
                    v = abs(l)-1
                    s = l > 0
                    if final_assign[v] == s:
                        sat = True
                        break
                if not sat:
                    is_sat = False
                    break
            if is_sat:
                return final_assign, None, final_assign
            else:
                return None, None, assignment
        
        return None, cnf, assignment

    solved, reduced_cnf, partial_assign = preprocess(clauses)
    if solved is not None:
        return solved

    # Prepare search space
    if reduced_cnf is not None:
        search_cnf = reduced_cnf
    else:
        # If preprocessing failed or didn't reduce, use original (converted)
        search_cnf = [[(abs(l)-1, l>0) for l in c] for c in clauses]
    
    # Filter out empty clauses (shouldn't be any if no contradiction, but just in case)
    search_cnf = [c for c in search_cnf if len(c) > 0]
    m = len(search_cnf)
    
    if m == 0:
        # All clauses satisfied by preprocessing?
        # If partial_assign is full and valid, return it.
        # If partial_assign has Nones, it means we didn't solve it but no clauses left?
        # That implies the remaining variables don't appear in any clause.
        # So any assignment to them works.
        res = []
        for i in range(n):
            if partial_assign[i] is not None:
                res.append(bool(partial_assign[i]))
            else:
                res.append(False)
        return res

    # Build variable to clauses index for search_cnf
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(search_cnf):
        for v, s in c:
            var_to_clauses[v].append(ci)
            
    # Initialize assignment
    # Use partial assignment if available, else random
    def make_full_assignment(partial):
        rng = random.Random()
        assign = []
        for i in range(n):
            if partial[i] is not None:
                assign.append(bool(partial[i]))
            else:
                assign.append(rng.random() < 0.5)
        return assign

    # Helper: check if clause is satisfied
    def is_clause_sat(assign, c):
        for v, s in c:
            if assign[v] == s:
                return True
        return False

    # Helper: count unsat
    def count_unsat(assign):
        count = 0
        for c in search_cnf:
            if not is_clause_sat(assign, c):
                count += 1
        return count

    # Initial assignment
    start_assign = make_full_assignment(partial_assign)
    if count_unsat(start_assign) == 0:
        return start_assign

    best_assign = start_assign[:]
    best_unsat = count_unsat(start_assign)

    # Fast WalkSAT with incremental updates
    def fast_walksat(start, max_steps, seed):
        rng = random.Random(seed)
        cur = start[:]
        
        # Initialize unsat set
        unsat = set()
        for ci in range(m):
            if not is_clause_sat(cur, search_cnf[ci]):
                unsat.add(ci)
        
        cur_unsat = len(unsat)
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        if cur_unsat == 0:
            return True, cur, 0

        noise = 0.2
        stuck_counter = 0
        last_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat:
                return True, cur, 0
            
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
                last_improve = step
                stuck_counter = 0
                # Reset noise if we improved significantly?
                # Or keep it adaptive based on recent history
            else:
                stuck_counter += 1
                # Adaptive noise: increase if stuck for a while
                if stuck_counter > 100 and stuck_counter % 50 == 0:
                    noise = min(0.8, noise + 0.1)
                
            # Select an unsatisfied clause
            ci = rng.choice(list(unsat))
            clause = search_cnf[ci]
            
            # Select a literal to flip
            # With probability (1 - noise), pick a literal that is currently false in this clause
            # (flipping it will satisfy the clause)
            # With probability noise, pick any literal in the clause
            false_lits = []
            for v, s in clause:
                if cur[v] != s:
                    false_lits.append((v, s))
            
            if false_lits and rng.random() > noise:
                v, s = rng.choice(false_lits)
            else:
                v, s = rng.choice(clause)
            
            # Flip
            old_val = cur[v]
            cur[v] = not old_val
            
            # Incremental update
            # Clauses containing v might change status
            affected = var_to_clauses[v]
            
            for aci in affected:
                # Check if clause aci was unsat
                was_unsat = aci in unsat
                # Check if it is now sat
                is_now_sat = is_clause_sat(cur, search_cnf[aci])
                
                if was_unsat:
                    if is_now_sat:
                        unsat.discard(aci)
                        cur_unsat -= 1
                else:
                    if not is_now_sat:
                        unsat.add(aci)
                        cur_unsat += 1
            
        return (min_unsat == 0), (cur if min_unsat == 0 else min_assign), min_unsat

    # Multi-restart strategy
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765, 1, 2, 3, 4, 5]
    
    # We can afford more steps now due to incremental updates
    # 50 variables, ~200-300 clauses. 
    # 10k steps is very fast.
    base_steps = 20000
    
    while time.time() < deadline - 2:
        seed = seeds[attempt % len(seeds)]
        
        # Vary starting assignment sometimes
        if attempt % 3 == 0:
            start = make_full_assignment(partial_assign)
        else:
            start = best_assign[:]
            
        success, result, min_unsat = fast_walksat(start, base_steps, seed)
        
        if success:
            return result
        
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]
            
        attempt += 1
        
        # If we've tried a few times with no improvement, try a completely fresh random start
        if attempt % 5 == 0:
            fresh = [random.Random(attempt).random() < 0.5 for _ in range(n)]
            fu = count_unsat(fresh)
            if fu < best_unsat:
                best_assign = fresh
                best_unsat = fu

    # Final check
    if count_unsat(best_assign) == 0:
        return best_assign
        
    return best_assign

# EVOLVE-BLOCK-END
