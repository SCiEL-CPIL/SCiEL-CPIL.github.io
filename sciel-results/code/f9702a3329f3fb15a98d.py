"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import heapq


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0

    # Pre-compute clause structure for fast updates
    # For each clause, store (var_idx, is_positive) pairs
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)

    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))

    # State: assignment array, clause satisfaction tracking
    # For each clause, track:
    #   sat_count[ci] = number of literals satisfied in clause ci
    #   false_vars[ci] = set of var indices whose literal is False in clause ci

    def init_state(assignment):
        sat_count = [0] * m
        false_vars = [set() for _ in range(m)]
        for ci, cv in enumerate(clause_vars):
            for vi, pos in cv:
                if assignment[vi] == pos:
                    sat_count[ci] += 1
                else:
                    false_vars[ci].add(vi)
        return sat_count, false_vars

    def count_unsat(sat_count):
        cnt = 0
        for sc in sat_count:
            if sc == 0:
                cnt += 1
        return cnt

    def find_unsat_clause(sat_count):
        for ci in range(m):
            if sat_count[ci] == 0:
                return ci
        return -1

    def flip_variable(vi, assignment, sat_count, false_vars):
        """Flip variable vi and update state. Returns number of newly unsatisfied clauses."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val

        delta = 0  # net change in unsat count
        for (ci, pos) in var_clauses[vi]:
            if old_val == pos:
                # Was satisfied, now unsatisfied
                sat_count[ci] -= 1
                false_vars[ci].add(vi)
                if sat_count[ci] == 0:
                    delta += 1
            else:
                # Was unsatisfied, now satisfied
                sat_count[ci] += 1
                false_vars[ci].discard(vi)
                if sat_count[ci] == 1:
                    delta -= 1
        return delta

    # ==================== CDCL SOLVER ====================
    def cdcl_solve():
        """CDCL solver with watched literals and clause learning."""
        # Variable values: 0 = unassigned, 1 = True, -1 = False
        # Using 0-based indexing internally
        values = [0] * n
        trail = []  # Stack of assigned variables in order
        trail_lim = [0]  # Decision levels (indices into trail)
        level = [0] * n  # Decision level at which variable was assigned
        reason = [None] * n  # Clause that forced the assignment (None for decisions)
        
        # Clause representation: list of (var, is_positive) tuples
        # For watched literals, we store clauses as lists of literals
        # literal = (var_idx, polarity) where polarity is True/False
        
        # Watched literals: for each variable, list of (clause_idx, polarity)
        # polarity indicates which literal of the clause is watched
        watches = [[] for _ in range(n)]
        
        # All clauses (original + learned)
        all_clauses = []
        for cv in clause_vars:
            all_clauses.append(cv)
        
        # Initialize watches
        for ci, cv in enumerate(all_clauses):
            if len(cv) >= 2:
                # Watch first two literals
                v1, p1 = cv[0]
                v2, p2 = cv[1]
                watches[v1].append((ci, p1))
                watches[v2].append((ci, p2))
            elif len(cv) == 1:
                v1, p1 = cv[0]
                watches[v1].append((ci, p1))
        
        # Unit clause propagation
        unit_queue = []
        
        def is_satisfied(ci):
            """Check if clause ci is satisfied."""
            cv = all_clauses[ci]
            for vi, pos in cv:
                if values[vi] != 0 and values[vi] == (1 if pos else -1):
                    return True
            return False
        
        def is_false(ci):
            """Check if all literals in clause ci are false."""
            cv = all_clauses[ci]
            for vi, pos in cv:
                if values[vi] == 0:
                    return False
                if values[vi] == (1 if pos else -1):
                    return False
            return True
        
        def propagate():
            """Perform unit propagation. Returns conflict clause index or -1 if no conflict."""
            while unit_queue:
                vi = unit_queue.pop()
                if values[vi] == 0:
                    continue  # Already assigned
                
                # Process watches for variable vi
                new_watches = []
                for (ci, polarity) in watches[vi]:
                    cv = all_clauses[ci]
                    # Check if the watched literal is satisfied
                    if values[vi] == (1 if polarity else -1):
                        new_watches.append((ci, polarity))
                        continue
                    
                    # The watched literal is false, look for another literal to watch
                    found_new_watch = False
                    for j, (vj, pj) in enumerate(cv):
                        if vj == vi:
                            continue
                        if values[vj] == 0:
                            # Found an unassigned literal, swap watch
                            watches[vj].append((ci, pj))
                            found_new_watch = True
                            break
                        elif values[vj] == (1 if pj else -1):
                            # Clause is satisfied, no need to watch
                            found_new_watch = True
                            break
                    
                    if not found_new_watch:
                        # All literals are false -> conflict
                        return ci
                    # Don't re-add to new_watches since we found a new watch or clause is satisfied
                
                watches[vi] = new_watches
            
            return -1
        
        def analyze_conflict(conflict_ci):
            """Analyze conflict and return learned clause and backjump level."""
            # Build implication graph
            learned = []  # List of (var, polarity) in learned clause
            seen = set()
            counter = 0
            backjump_level = 0
            current_level = len(trail_lim) - 1
            
            # Start with conflict clause
            ci = conflict_ci
            cv = all_clauses[ci]
            
            # Find the variable with highest level in the clause
            # We'll do this iteratively
            while True:
                # Find the literal in this clause that was assigned at the current highest level
                max_level = -1
                max_var = -1
                max_pol = True
                for vi, pos in cv:
                    if level[vi] > max_level:
                        max_level = level[vi]
                        max_var = vi
                        max_pol = pos
                
                if max_var == -1 or max_level < current_level:
                    break
                
                # Add negation of this literal to learned clause
                learned.append((max_var, not max_pol))
                seen.add(max_var)
                
                # If this is a decision variable, we're done with this level
                if reason[max_var] is None:
                    if max_level > backjump_level:
                        backjump_level = max_level
                    break
                
                # Get the reason clause
                reason_ci = reason[max_var]
                reason_cv = all_clauses[reason_ci]
                
                # Process reason clause: add unassigned or lower-level literals
                new_cv = []
                for vi, pos in reason_cv:
                    if vi in seen:
                        continue
                    if values[vi] == 0:
                        new_cv.append((vi, pos))
                        seen.add(vi)
                    elif level[vi] < current_level:
                        new_cv.append((vi, pos))
                        seen.add(vi)
                
                cv = new_cv
                if not cv:
                    break
                # Continue loop to find next highest level variable
            
            # Determine backjump level
            # Find the second highest level in learned clause
            levels = []
            for vi, pos in learned:
                if level[vi] > 0:
                    levels.append(level[vi])
            levels.sort(reverse=True)
            if len(levels) >= 2:
                backjump_level = levels[1]
            elif len(levels) == 1:
                backjump_level = levels[0]
            
            # Create learned clause
            learned_ci = len(all_clauses)
            all_clauses.append(learned)
            
            # Set up watches for learned clause
            if len(learned) >= 2:
                v1, p1 = learned[0]
                v2, p2 = learned[1]
                watches[v1].append((learned_ci, p1))
                watches[v2].append((learned_ci, p2))
            elif len(learned) == 1:
                v1, p1 = learned[0]
                watches[v1].append((learned_ci, p1))
                unit_queue.append(v1)
            
            return learned_ci, backjump_level
        
        def backtrack(to_level):
            """Backtrack to the specified decision level."""
            while len(trail_lim) > to_level + 1:
                trail_lim.pop()
                # Unassign variables assigned at this level
                while len(trail) > trail_lim[-1] if trail_lim else len(trail) > 0:
                    if not trail_lim or len(trail) > trail_lim[-1]:
                        vi = trail.pop()
                        values[vi] = 0
                        level[vi] = 0
                        reason[vi] = None
                    else:
                        break
            # More robust backtrack
            while len(trail) > (trail_lim[-1] if trail_lim else 0):
                vi = trail.pop()
                values[vi] = 0
                level[vi] = 0
                reason[vi] = None
        
        # Main CDCL loop
        decisions = 0
        max_decisions = 500000  # Limit to avoid infinite loops
        
        while decisions < max_decisions:
            if time.time() > deadline - 5.0:
                return None  # Timeout, fall back to WalkSAT
            
            # Propagate
            conflict = propagate()
            
            if conflict != -1:
                # Conflict detected
                if len(trail_lim) <= 1:
                    # Conflict at decision level 0 -> UNSAT
                    return None
                
                learned_ci, backjump_level = analyze_conflict(conflict)
                backtrack(backjump_level)
                decisions += 100  # Count conflict as work
                continue
            
            # Check if all variables assigned
            if len(trail) == n:
                # Check if solution is valid
                for ci, cv in enumerate(all_clauses):
                    if not is_satisfied(ci):
                        return None  # Should not happen if propagation is correct
                return [values[vi] == 1 for vi in range(n)]
            
            # Pick next decision variable
            # Use VSIDS-like heuristic: pick variable with highest activity
            # For simplicity, pick unassigned variable that appears in most unsatisfied clauses
            best_var = -1
            best_score = -1
            for vi in range(n):
                if values[vi] != 0:
                    continue
                # Count how many clauses this variable appears in that are not yet satisfied
                score = 0
                for (ci, pos) in var_clauses[vi]:
                    # Check if clause is not satisfied
                    cv = all_clauses[ci]
                    sat = False
                    for vj, pj in cv:
                        if values[vj] != 0 and values[vj] == (1 if pj else -1):
                            sat = True
                            break
                    if not sat:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_var = vi
            
            if best_var == -1:
                break  # No unassigned variables
            
            # Assign the variable
            # Try both polarities, start with False
            values[best_var] = -1
            level[best_var] = len(trail_lim)
            reason[best_var] = None
            trail.append(best_var)
            trail_lim.append(len(trail))
            decisions += 1
        
        return None
    
    # Try CDCL first
    cdcl_result = cdcl_solve()
    if cdcl_result is not None:
        # Verify the solution
        valid = True
        for ci, cv in enumerate(clause_vars):
            sat = False
            for vi, pos in cv:
                if cdcl_result[vi] == pos:
                    sat = True
                    break
            if not sat:
                valid = False
                break
        if valid:
            return cdcl_result
    
    # ==================== WALKSAT FALLBACK ====================
    # WalkSAT with multiple restarts and adaptive strategy
    rng = random.Random(0)

    # Weight parameter: probability of flipping a literal in the chosen unsat clause
    # vs. flipping a random variable that appears in unsat clauses
    w = 0.4
    w_min = 0.0
    w_max = 1.0
    w_decay = 0.98

    best_unsat = m + 1
    best_assignment = [False] * n

    max_restarts = 200
    base_steps = 5000

    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # Initialize with random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_count, false_vars = init_state(assignment)
        cur_unsat = count_unsat(sat_count)

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Adaptive step budget: more steps if we're making progress
        # Scale steps based on restart number and current difficulty
        steps = base_steps + restart * 2000

        for step in range(steps):
            if time.time() > deadline - 2.0:
                break

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Find an unsatisfied clause
            ci = find_unsat_clause(sat_count)
            if ci == -1:
                return assignment

            # Choose which variable to flip
            if rng.random() < w:
                # Flip a random literal in the chosen unsat clause
                cv = clause_vars[ci]
                idx = rng.randrange(len(cv))
                vi = cv[idx][0]
            else:
                # Flip a variable that appears in some unsat clause
                # Weight by how many unsat clauses it appears in
                # Pick from false_vars of the chosen clause, or a random unsat clause
                # Use the chosen clause's false variables
                fv = false_vars[ci]
                if fv:
                    vi = rng.choice(list(fv))
                else:
                    cv = clause_vars[ci]
                    idx = rng.randrange(len(cv))
                    vi = cv[idx][0]

            # Try the flip
            old_cur_unsat = cur_unsat
            new_delta = flip_variable(vi, assignment, sat_count, false_vars)
            cur_unsat += new_delta

            # WalkSAT-style: if we increased unsat, consider reverting
            # but with some probability keep it to escape local minima
            if new_delta > 0:
                # With some probability, revert
                if rng.random() > 0.5:
                    flip_variable(vi, assignment, sat_count, false_vars)
                    cur_unsat = old_cur_unsat

        # Decay w to explore more broadly as we continue
        w = max(w_min, min(w_max, w * w_decay))

    # Return best assignment found
    return best_assignment
# EVOLVE-BLOCK-END
