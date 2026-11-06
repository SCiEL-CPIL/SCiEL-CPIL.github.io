"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # 1. Preprocessing: Simplify clauses, remove tautologies, build indices
    simplified_clauses = []
    for clause in clauses:
        seen_vars = set()
        new_clause = []
        for lit in clause:
            v = abs(lit)
            if v not in seen_vars:
                seen_vars.add(v)
                new_clause.append(lit)
        if len(new_clause) > 0:
            simplified_clauses.append(new_clause)
    
    # If no clauses, any assignment works
    if not simplified_clauses:
        return [False] * n
        
    # Check for empty clause (UNSAT) - though problem says satisfiable
    for c in simplified_clauses:
        if len(c) == 0:
             # Should not happen per contract, but return something
             return [False] * n

    m = len(simplified_clauses)
    
    # Build indices:
    # clause_vars[i] = list of (var_idx, is_pos) for clause i
    # var_clauses[v] = list of (clause_idx, is_pos) for variable v
    clause_vars = []
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(simplified_clauses):
        cv = []
        for lit in c:
            v_idx = abs(lit) - 1
            is_pos = lit > 0
            cv.append((v_idx, is_pos))
            var_clauses[v_idx].append((i, is_pos))
        clause_vars.append(cv)
    
    # 2. Fast DPLL with Unit Propagation (Limited Depth)
    # This handles easy instances instantly.
    start_time = time.time()
    time_limit = 290.0 # Safety margin
    
    # Convert to 0-based for internal use in DPLL
    # assignment: -1 unassigned, 0 False, 1 True
    assignment = [-1] * n
    # Trail for backtracking
    trail = []
    
    def dpll_unit_prop(assignment, trail, depth):
        """
        Attempts to solve via DPLL with unit propagation.
        Returns True if solved, False if conflict or timeout/depth exceeded.
        Modifies assignment and trail in place.
        """
        if time.time() - start_time > time_limit:
            return False
        if depth > 100: # Prevent deep recursion in Python
            return False
            
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            for i, c in enumerate(clause_vars):
                sat = False
                false_count = 0
                unassigned = []
                for v_idx, is_pos in c:
                    val = assignment[v_idx]
                    if val == 1 and is_pos:
                        sat = True
                        break
                    elif val == 0 and not is_pos:
                        sat = True
                        break
                    elif val == -1:
                        unassigned.append((v_idx, is_pos))
                    else:
                        false_count += 1
                
                if sat:
                    continue
                    
                if len(unassigned) == 0:
                    # Conflict
                    return False
                
                if len(unassigned) == 1:
                    # Unit clause
                    v_idx, is_pos = unassigned[0]
                    new_val = 1 if is_pos else 0
                    if assignment[v_idx] != -1:
                        if assignment[v_idx] != new_val:
                            return False # Conflict
                    else:
                        assignment[v_idx] = new_val
                        trail.append((v_idx, new_val))
                        changed = True
                        # Optimization: break and restart loop to catch cascading units
                        break
            
        if time.time() - start_time > time_limit:
            return False
            
        # Check if all assigned
        if all(v != -1 for v in assignment):
            # Verify solution (should be satisfied if no conflict)
            for i, c in enumerate(clause_vars):
                sat = False
                for v_idx, is_pos in c:
                    if assignment[v_idx] == (1 if is_pos else 0):
                        sat = True
                        break
                if not sat:
                    return False
            return True
        
        # 2. Pick a variable
        # Simple heuristic: pick first unassigned or highest degree
        # For speed, just pick first unassigned to avoid O(n) scan every time
        # A better heuristic would be VSIDS, but let's keep it simple for the fast path
        var_to_decide = -1
        for v in range(n):
            if assignment[v] == -1:
                var_to_decide = v
                break
        
        if var_to_decide == -1:
            return True
            
        # Try True
        assignment[var_to_decide] = 1
        trail.append((var_to_decide, 1))
        
        if dpll_unit_prop(assignment, trail, depth + 1):
            return True
            
        # Backtrack
        while trail and trail[-1][0] != var_to_decide:
            v, val = trail.pop()
            assignment[v] = -1
        if trail:
            v, val = trail.pop()
            assignment[v] = -1
            
        # Try False
        assignment[var_to_decide] = 0
        trail.append((var_to_decide, 0))
        
        if dpll_unit_prop(assignment, trail, depth + 1):
            return True
            
        # Backtrack
        while trail and trail[-1][0] != var_to_decide:
            v, val = trail.pop()
            assignment[v] = -1
        if trail:
            v, val = trail.pop()
            assignment[v] = -1
            
        return False

    # Run DPLL
    # We need to clear assignment for DPLL
    dpll_assignment = [-1] * n
    dpll_trail = []
    if dpll_unit_prop(dpll_assignment, dpll_trail, 0):
        # Convert to list of booleans
        return [bool(v) for v in dpll_assignment]

    # 3. WalkSAT with Damage Minimization
    # If DPLL fails, use WalkSAT
    rng = random.Random(42)
    
    # Initialize assignment randomly
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Calculate initial satisfaction counts
    counts = [0] * m
    for i in range(m):
        cnt = 0
        for v_idx, is_pos in clause_vars[i]:
            if assignment[v_idx] == is_pos:
                cnt += 1
        counts[i] = cnt
        
    unsat_clauses = [i for i in range(m) if counts[i] == 0]
    
    if not unsat_clauses:
        return assignment

    best_assignment = assignment[:]
    best_unsat_count = len(unsat_clauses)
    
    # WalkSAT Parameters
    # Noise: probability to pick a random literal instead of min-damage
    # For planted 3-SAT, typical noise is around 0.25-0.5
    noise = 0.25
    steps_per_restart = 5000
    max_restarts = 1000 # Will be limited by time
    
    current_restart = 0
    
    while time.time() - start_time < time_limit:
        # Generate new assignment
        # Mix of random and best-so-far to escape local minima
        if current_restart > 0 and rng.random() < 0.3:
            assignment = best_assignment[:]
            # Add some noise to best assignment
            for _ in range(n // 10):
                v = rng.randint(0, n-1)
                assignment[v] = not assignment[v]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
        # Recalculate counts
        counts = [0] * m
        for i in range(m):
            cnt = 0
            for v_idx, is_pos in clause_vars[i]:
                if assignment[v_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
            
        unsat_clauses = [i for i in range(m) if counts[i] == 0]
        
        if not unsat_clauses:
            return assignment
            
        if len(unsat_clauses) < best_unsat_count:
            best_unsat_count = len(unsat_clauses)
            best_assignment = assignment[:]
            
        # WalkSAT Loop
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            if not unsat_clauses:
                return assignment
                
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clause_vars[clause_idx]
            
            # Heuristic: Pick variable with minimum damage
            best_var = -1
            best_flip_val = False
            best_damage = float('inf')
            
            for v_idx, is_pos in clause:
                # We want to flip v_idx to is_pos to satisfy this clause.
                # Current value: assignment[v_idx] != is_pos (since clause is unsat)
                
                # Calculate damage:
                # How many satisfied clauses containing (v_idx, NOT is_pos) will become unsatisfied?
                damage = 0
                for c_idx, c_is_pos in var_clauses[v_idx]:
                    if c_idx == clause_idx:
                        continue
                    # If this other clause has the opposite literal (v_idx, NOT is_pos)
                    if c_is_pos != is_pos:
                        # Check if this other clause is currently satisfied
                        if counts[c_idx] > 0:
                            # It is satisfied. If we flip v_idx, this literal becomes unsatisfied.
                            # Does the clause become unsatisfied?
                            # It becomes unsatisfied only if this was the ONLY satisfied literal.
                            if counts[c_idx] == 1:
                                damage += 1
                
                # Minimize damage
                if damage < best_damage or (damage == best_damage and rng.random() < 0.5):
                    best_damage = damage
                    best_var = v_idx
                    best_flip_val = is_pos
            
            # With some probability, pick a random variable from the clause to escape local minima
            if rng.random() < noise:
                v_idx, is_pos = rng.choice(clause)
                best_var = v_idx
                best_flip_val = is_pos
            
            # Perform flip
            if best_var != -1:
                old_val = assignment[best_var]
                new_val = best_flip_val
                
                # Update counts incrementally
                for c_idx, c_is_pos in var_clauses[best_var]:
                    was_sat = (old_val == c_is_pos)
                    is_sat = (new_val == c_is_pos)
                    
                    if was_sat and not is_sat:
                        # Lost satisfaction
                        counts[c_idx] -= 1
                        if counts[c_idx] == 0:
                            # Newly unsatisfied
                            # We don't maintain a dynamic set for simplicity, 
                            # but we can update unsat_clauses list if we were tracking it dynamically.
                            # Here we just recalculate unsat_clauses periodically or at end of step?
                            # Actually, updating the list is O(m). 
                            # Let's just recalculate unsat_clauses every step? No, too slow.
                            # Let's maintain a count of unsat clauses and update it.
                            # But we need the list to pick from.
                            # Strategy: Keep unsat_clauses list, but it might get stale.
                            # Better: Just recalculate unsat_clauses every N steps or if we want strict correctness.
                            # Given m~600, recalculating is O(m) = 600 ops. 
                            # 5000 steps * 600 = 3M ops per restart. 
                            # If we do 100 restarts, 300M ops. Might be tight in Python.
                            # Let's try to update the list incrementally.
                            pass
                    elif not was_sat and is_sat:
                        # Gained satisfaction
                        counts[c_idx] += 1
                        pass
                
                # Efficiently update unsat_clauses list is hard.
                # Let's just recalculate it. It's a list comprehension, which is fast in C.
                unsat_clauses = [i for i in range(m) if counts[i] == 0]
                
                if not unsat_clauses:
                    return assignment
                    
                if len(unsat_clauses) < best_unsat_count:
                    best_unsat_count = len(unsat_clauses)
                    best_assignment = assignment[:]
                    
                assignment[best_var] = new_val
        
        current_restart += 1

    # Return best found
    return best_assignment

# EVOLVE-BLOCK-END
