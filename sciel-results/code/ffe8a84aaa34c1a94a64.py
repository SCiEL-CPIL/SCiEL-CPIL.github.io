"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: return best result before 330s wall-clock limit
    start_time = time.time()
    DEADLINE = 325.0  # 5s safety margin
    
    # Convert clauses to 0-based indices with sign
    # literal > 0 means variable must be True, literal < 0 means False
    # Store as (var_index, is_positive)
    clauses_tup = []
    for c in clauses:
        clauses_tup.append([(abs(l) - 1, l > 0) for l in c])
    
    m = len(clauses_tup)
    
    # --- DPLL with watched literals, unit propagation, pure literal elimination ---
    def dpll_solve():
        """Complete DPLL solver. Returns assignment list or None if timeout/failed."""
        # Build watched literals structure
        # For each clause, maintain two watched literals
        # watches[literal] = set of clause indices that watch this literal
        # literal is represented as (var_idx, is_pos)
        
        # Build initial watches
        # watches: dict mapping (var_idx, is_pos) -> list of clause indices
        watches = {}
        # For each clause, pick two literals to watch
        clause_watches = []  # clause_watches[i] = [lit1, lit2]
        
        for i, clause in enumerate(clauses_tup):
            if len(clause) <= 2:
                clause_watches.append(list(clause))
            else:
                clause_watches.append([clause[0], clause[1]])
            for lit in clause_watches[i]:
                watches.setdefault(lit, []).append(i)
        
        # Assignment: None = unassigned, True/False = assigned
        assignment = [None] * n
        # Track number of satisfied/unsatisfied for each clause
        # clause_state: 0 = unsat, 1 = exactly one true, 2 = two or more true
        clause_state = [0] * m
        
        # Helper: is literal satisfied?
        def lit_satisfied(lit):
            var, is_pos = lit
            if assignment[var] is None:
                return None
            return assignment[var] == is_pos
        
        # Helper: is literal false?
        def lit_false(lit):
            var, is_pos = lit
            if assignment[var] is None:
                return None
            return assignment[var] != is_pos
        
        # Initialize clause states
        for i, clause in enumerate(clauses_tup):
            count_true = 0
            for lit in clause:
                if lit_satisfied(lit):
                    count_true += 1
            clause_state[i] = min(count_true, 2)
        
        # Unit propagation queue
        # We'll use a worklist of literals that became true
        # When a literal becomes true, we need to check clauses watching its negation
        
        def propagate():
            """Unit propagation using watched literals. Returns False on conflict."""
            # Use a stack for propagation
            # When we assign a variable, we check clauses watching the negated literal
            stack = []
            
            # We need to process assignments. Let's use a different approach:
            # Keep a queue of newly assigned variables
            # Actually, let's restructure: do propagation iteratively
            
            # We'll use a list as a queue of (var, value) that need processing
            # But we need to track which clauses to update
            
            # Simpler approach: iterative unit propagation
            # Find all unit clauses, assign, repeat
            
            while True:
                # Find unit clauses
                unit_found = False
                for i in range(m):
                    if clause_state[i] == 2:
                        continue
                    if clause_state[i] == 1:
                        # Exactly one literal is true, find the unassigned one
                        for lit in clauses_tup[i]:
                            s = lit_satisfied(lit)
                            if s is None:
                                # This is the unit literal
                                var, is_pos = lit
                                if assignment[var] is not None:
                                    continue
                                assignment[var] = is_pos
                                unit_found = True
                                # Now propagate: check clauses watching (var, not is_pos)
                                neg_lit = (var, not is_pos)
                                if neg_lit in watches:
                                    for j in list(watches[neg_lit]):
                                        if j == i:
                                            continue
                                        # Check if clause j still has this literal
                                        # and if it can be satisfied
                                        cj = clauses_tup[j]
                                        # Check if the negated literal is in clause j
                                        if neg_lit not in cj:
                                            continue
                                        # Check if clause j is already satisfied
                                        if clause_state[j] >= 1:
                                            # Find a new watch
                                            new_watch = None
                                            for lit2 in cj:
                                                if lit2 == neg_lit:
                                                    continue
                                                s2 = lit_satisfied(lit2)
                                                if s2 is None:
                                                    new_watch = lit2
                                                    break
                                                elif s2:
                                                    # Clause is satisfied, update state
                                                    clause_state[j] = 2
                                                    break
                                            if new_watch is not None:
                                                # Replace watch
                                                watches[neg_lit].remove(j)
                                                watches.setdefault(new_watch, []).append(j)
                                                clause_watches[j] = [l for l in clause_watches[j] if l != neg_lit] + [new_watch]
                                                # Remove neg_lit from watches
                                                if not watches.get(neg_lit):
                                                    del watches[neg_lit]
                                            else:
                                                # No new watch found, clause might be unit or empty
                                                # Check if clause is now unit or empty
                                                unsat_count = 0
                                                for lit2 in cj:
                                                    s2 = lit_satisfied(lit2)
                                                    if s2 is False:
                                                        unsat_count += 1
                                                if unsat_count == len(cj):
                                                    return False  # Conflict
                                                elif unsat_count == len(cj) - 1:
                                                    # It's a unit clause, will be picked up next iteration
                                                    pass
                                        else:
                                            # clause_state[j] == 0, this literal was the only possibility
                                            # Check if clause is now empty
                                            unsat_count = 0
                                            for lit2 in cj:
                                                s2 = lit_satisfied(lit2)
                                                if s2 is False:
                                                    unsat_count += 1
                                            if unsat_count == len(cj):
                                                return False  # Conflict
                                            elif unsat_count == len(cj) - 1:
                                                # Unit clause, will be picked up
                                                pass
                                break
                if not unit_found:
                    break
            
            return True
        
        # Pure literal elimination
        def pure_literal_elimination():
            """Assign pure literals. Returns True if any assigned."""
            # Count for each variable: how many times it appears positive, negative
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clauses_tup:
                for lit in clause:
                    var, is_pos = lit
                    if assignment[var] is not None:
                        continue
                    if is_pos:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
            
            assigned_any = False
            for var in range(n):
                if assignment[var] is not None:
                    continue
                if pos_count[var] == 0 and neg_count[var] > 0:
                    assignment[var] = False
                    assigned_any = True
                elif neg_count[var] == 0 and pos_count[var] > 0:
                    assignment[var] = True
                    assigned_any = True
            
            return assigned_any
        
        # Variable selection: pick variable with highest degree in unsatisfied clauses
        def select_variable():
            # Count occurrences in unsatisfied clauses
            var_count = [0] * n
            for i in range(m):
                if clause_state[i] >= 1:
                    continue
                for lit in clauses_tup[i]:
                    var, _ = lit
                    if assignment[var] is None:
                        var_count[var] += 1
            
            best_var = -1
            best_count = -1
            for var in range(n):
                if assignment[var] is None and var_count[var] > best_count:
                    best_count = var_count[var]
                    best_var = var
            
            if best_var == -1:
                return None
            
            # Choose value: pick the one that satisfies more clauses
            pos_sat = 0
            neg_sat = 0
            for i in range(m):
                if clause_state[i] >= 1:
                    continue
                has_pos = False
                has_neg = False
                for lit in clauses_tup[i]:
                    var, is_pos = lit
                    if var == best_var:
                        if is_pos:
                            has_pos = True
                        else:
                            has_neg = True
                if has_pos:
                    pos_sat += 1
                if has_neg:
                    neg_sat += 1
            
            return best_var, (pos_sat >= neg_sat)
        
        # Recursive DPLL
        def dpll():
            # Check deadline
            if time.time() - start_time > DEADLINE:
                return None
            
            # Check if all clauses satisfied
            all_sat = True
            for i in range(m):
                if clause_state[i] < 1:
                    all_sat = False
                    break
            if all_sat:
                return assignment[:]
            
            # Check for conflict
            for i in range(m):
                # Check if clause is empty (all literals false)
                all_false = True
                for lit in clauses_tup[i]:
                    s = lit_satisfied(lit)
                    if s is None or s:
                        all_false = False
                        break
                if all_false:
                    return None  # Conflict
            
            # Pure literal elimination
            if pure_literal_elimination():
                # Re-run propagation
                if not propagate():
                    return None
                return dpll()
            
            # Unit propagation
            if not propagate():
                return None
            
            # Check if all clauses satisfied again
            all_sat = True
            for i in range(m):
                if clause_state[i] < 1:
                    all_sat = False
                    break
            if all_sat:
                return assignment[:]
            
            # Select variable
            result = select_variable()
            if result is None:
                # All variables assigned, check satisfaction
                all_sat = True
                for i in range(m):
                    if clause_state[i] < 1:
                        all_sat = False
                        break
                return assignment[:] if all_sat else None
            
            var, value = result
            
            # Try value first
            assignment[var] = value
            # Save state for backtracking
            saved_state = assignment[:]
            
            if propagate():
                result = dpll()
                if result is not None:
                    return result
            
            # Backtrack
            assignment[:] = saved_state
            assignment[var] = not value
            
            if propagate():
                result = dpll()
                if result is not None:
                    return result
            
            return None
        
        return dpll()
    
    # Try DPLL first
    result = dpll_solve()
    if result is not None:
        return result
    
    # --- Fallback: WalkSAT with adaptive noise ---
    if time.time() - start_time > DEADLINE:
        # Return random assignment as fallback
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]
    
    rng = random.Random(42)
    
    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def count_unsat(assignment):
        count = 0
        for clause in clauses_tup:
            sat = False
            for var, is_pos in clause:
                if assignment[var] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    best_assignment = assignment[:]
    best_unsat = count_unsat(assignment)
    
    # WalkSAT with adaptive temperature
    max_steps = 50000
    restarts = 20
    base_noise = 0.25
    noise_decay = 0.95
    
    for restart in range(restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # New random assignment on restart (after first)
        if restart > 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            best_unsat = count_unsat(assignment)
        
        noise = base_noise * (noise_decay ** restart)
        
        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
            
            unsat_count = count_unsat(assignment)
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            # Find unsatisfied clauses
            unsat_clauses = []
            for i, clause in enumerate(clauses_tup):
                sat = False
                for var, is_pos in clause:
                    if assignment[var] == is_pos:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment
            
            clause_idx = rng.choice(unsat_clauses)
            clause = clauses_tup[clause_idx]
            
            # Count how many literals in this clause would be satisfied if we flip each variable
            best_var = -1
            best_gain = -1
            candidates = []
            
            for var, is_pos in clause:
                # If we flip var, does it satisfy this clause?
                # Currently assignment[var] != is_pos (since clause is unsat)
                # After flip: assignment[var] == is_pos, so yes, it satisfies
                # Count how many other unsatisfied clauses this flip would break
                gain = 0  # This flip satisfies the chosen clause
                # Count clauses that would become unsatisfied
                for j in range(m):
                    if j == clause_idx:
                        continue
                    cj = clauses_tup[j]
                    # Check if clause j is currently unsatisfied
                    cj_sat = False
                    for var2, is_pos2 in cj:
                        if assignment[var2] == is_pos2:
                            cj_sat = True
                            break
                    if cj_sat:
                        continue
                    # Clause j is unsatisfied. Check if flipping var would make it more unsat
                    # It's already unsat, so flipping var can only keep it unsat or... 
                    # Wait, if clause j is unsat, all its literals are false.
                    # If var is in clause j, and we flip it, one literal becomes true, so clause j becomes sat.
                    # So flipping var in an unsat clause that contains var actually helps.
                    # We need to count clauses that are currently SAT and would become UNSAT.
                    pass
                
                # Simpler: count net gain
                # Gain = (# clauses that become sat) - (# clauses that become unsat)
                # The chosen clause becomes sat: +1
                # Other clauses containing var: if they were sat and only var was true, they become unsat
                # If they were unsat and var was in them, they become sat: +1 each
                
                # Let's compute properly
                clauses_gained = 1  # The chosen clause
                clauses_lost = 0
                
                for j in range(m):
                    if j == clause_idx:
                        continue
                    cj = clauses_tup[j]
                    if (var, is_pos) not in cj and (var, not is_pos) not in cj:
                        continue
                    
                    # Check if var is in this clause
                    var_in_cj = False
                    var_is_pos_in_cj = False
                    for var2, is_pos2 in cj:
                        if var2 == var:
                            var_in_cj = True
                            var_is_pos_in_cj = is_pos2
                            break
                    
                    if not var_in_cj:
                        continue
                    
                    # Current state of clause j
                    cj_sat = False
                    cj_sat_count = 0
                    for var2, is_pos2 in cj:
                        if assignment[var2] == is_pos2:
                            cj_sat = True
                            cj_sat_count += 1
                    
                    # After flipping var
                    new_cj_sat_count = cj_sat_count
                    if assignment[var] == var_is_pos_in_cj:
                        # var was satisfying this clause, after flip it won't
                        new_cj_sat_count -= 1
                    else:
                        # var was not satisfying, after flip it will
                        new_cj_sat_count += 1
                    
                    if cj_sat and new_cj_sat_count == 0:
                        clauses_lost += 1
                    elif not cj_sat and new_cj_sat_count > 0:
                        clauses_gained += 1
                
                net_gain = clauses_gained - clauses_lost
                candidates.append((var, net_gain))
                
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_var = var
            
            # Choose variable with probability based on gain and noise
            # With probability noise, pick random; otherwise pick best
            if rng.random() < noise:
                var_to_flip = rng.choice(candidates)[0]
            else:
                # Pick from top candidates with some randomness
                max_gain = max(g for _, g in candidates)
                top = [v for v, g in candidates if g == max_gain]
                var_to_flip = rng.choice(top)
            
            # Flip
            assignment[var_to_flip] = not assignment[var_to_flip]
    
    return best_assignment
# EVOLVE-BLOCK-END
