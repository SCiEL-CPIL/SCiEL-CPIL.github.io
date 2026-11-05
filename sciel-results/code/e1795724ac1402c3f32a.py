"""Rewritten 3-SAT solver combining DPLL with watched literals and efficient local search fallback."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def solve(instance):
    # Deadline guard: kill before 330s, return best found with safety margin.
    START = time.time()
    DEADLINE = START + 325.0  # 5s safety margin

    n, clauses = instance["n"], instance["clauses"]

    # Precompute clause data: for each clause, list of (var_index, polarity_bool)
    # var_index is 0-based, polarity True means positive literal
    clause_vars = []
    for c in clauses:
        cv = []
        for lit in c:
            v = abs(lit) - 1
            pol = lit > 0
            cv.append((v, pol))
        clause_vars.append(cv)

    m = len(clause_vars)

    # Precompute variable-to-clauses incidence for faster updates
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        seen = set()
        for v, pol in cv:
            if v not in seen:
                var_to_clauses[v].append(i)
                seen.add(v)

    # Helper: Check if a clause is satisfied by an assignment
    def check_all(assignment):
        """Check if assignment satisfies all clauses. Returns True/False."""
        for cv in clause_vars:
            sat = False
            for v, pol in cv:
                if assignment[v] == pol:
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Try all-True
    all_true = [True] * n
    if check_all(all_true):
        return all_true

    # Try all-False
    all_false = [False] * n
    if check_all(all_false):
        return all_false

    # --- DPLL with watched literals ---
    def dpll_watched():
        """
        DPLL solver with watched literals and unit propagation.
        Returns assignment if SAT, None if UNSAT or timeout.
        """
        # State:
        # assignment[v] = True/False/None
        assignment = [None] * n
        
        # Watched literals: for each clause, we watch 2 literals
        # watch[i] = [lit1, lit2] where lit is signed 1-based
        # We need to track which literals are watched
        # For efficiency, we'll maintain:
        # - active_clauses: set of clause indices that are not yet satisfied or falsified
        # - For each clause, we track its literals and which ones are assigned
        
        # Convert to signed literals for easier manipulation
        def to_signed(v, pol):
            return v + 1 if pol else -(v + 1)
        
        def from_signed(lit):
            return abs(lit) - 1, lit > 0
        
        # Initialize watched literals
        # For each clause, pick two literals to watch
        watches = []  # watches[i] = [lit1, lit2]
        for i, cv in enumerate(clause_vars):
            if len(cv) >= 2:
                watches.append([to_signed(cv[0][0], cv[0][1]), to_signed(cv[1][0], cv[1][1])])
            elif len(cv) == 1:
                watches.append([to_signed(cv[0][0], cv[0][1]), None])
            else:
                watches.append([None, None])
        
        # Active clauses: initially all
        active = set(range(m))
        
        # Queue for unit propagation
        unit_queue = []
        
        # Process initial unit clauses
        for i in range(m):
            cv = clause_vars[i]
            if len(cv) == 1:
                v, pol = cv[0]
                unit_queue.append((v, pol, i))
        
        # Propagate units
        def propagate():
            """Process unit propagation. Returns False on conflict."""
            while unit_queue:
                if time.time() > DEADLINE:
                    return "timeout"
                
                v, pol, source_clause = unit_queue.pop()
                
                # If already assigned, check for conflict
                if assignment[v] is not None:
                    if assignment[v] != pol:
                        return False  # Conflict
                    continue
                
                # Assign v = pol
                assignment[v] = pol
                
                # For each clause containing v, update watches
                for ci in var_to_clauses[v]:
                    if ci not in active:
                        continue
                    
                    # Check if clause is satisfied
                    cv = clause_vars[ci]
                    sat = False
                    for v2, pol2 in cv:
                        if assignment[v2] is not None and assignment[v2] == pol2:
                            sat = True
                            break
                    
                    if sat:
                        active.discard(ci)
                        continue
                    
                    # Check if clause is falsified
                    falsified = True
                    for v2, pol2 in cv:
                        if assignment[v2] is None or assignment[v2] != pol2:
                            falsified = False
                            break
                    
                    if falsified:
                        return False  # Conflict
                    
                    # Update watches: if a watched literal is falsified, find a new one
                    w = watches[ci]
                    new_queue = []
                    for idx, lit in enumerate(w):
                        if lit is None:
                            continue
                        lv, lpol = from_signed(lit)
                        if assignment[lv] is not None and assignment[lv] != lpol:
                            # This watched literal is falsified, need to find a replacement
                            found = False
                            for v2, pol2 in cv:
                                if assignment[v2] is None and to_signed(v2, pol2) != w[0] and to_signed(v2, pol2) != w[1]:
                                    # Found a new literal to watch
                                    if idx == 0:
                                        w[0] = to_signed(v2, pol2)
                                    else:
                                        w[1] = to_signed(v2, pol2)
                                    found = True
                                    break
                            if not found:
                                # No more unassigned literals: this is a unit clause
                                for v2, pol2 in cv:
                                    if assignment[v2] is None:
                                        unit_queue.append((v2, pol2, ci))
                                        break
                                return "continue"
                    
                    # Check if clause has only one unassigned literal (unit)
                    unassigned = []
                    for v2, pol2 in cv:
                        if assignment[v2] is None:
                            unassigned.append((v2, pol2))
                    
                    if len(unassigned) == 1:
                        v2, pol2 = unassigned[0]
                        unit_queue.append((v2, pol2, ci))
            
            return True
        
        # Initial propagation
        result = propagate()
        if result is False:
            return None
        if result == "timeout":
            return None
        
        # Check if all clauses are satisfied
        if not active:
            return [a if a is not None else False for a in assignment]
        
        # Branching: select variable with highest degree in active clauses
        def select_variable():
            var_counts = [0] * n
            for ci in active:
                cv = clause_vars[ci]
                for v, pol in cv:
                    if assignment[v] is None:
                        var_counts[v] += 1
            
            max_count = -1
            best_var = -1
            for v in range(n):
                if assignment[v] is None and var_counts[v] > max_count:
                    max_count = var_counts[v]
                    best_var = v
            
            return best_var
        
        # Recursive DPLL with backtracking
        def dpll_recursive(depth=0):
            if time.time() > DEADLINE:
                return None
            
            # Check if all clauses satisfied
            if not active:
                return [a if a is not None else False for a in assignment]
            
            # Select variable to branch on
            var = select_variable()
            if var == -1:
                # No unassigned variables but active clauses remain: UNSAT
                return None
            
            # Try True
            saved_assignment = assignment[var]
            assignment[var] = True
            
            # Save state for backtracking
            saved_active = active.copy()
            saved_watches = [w[:] for w in watches]
            saved_queue = unit_queue[:]
            
            result = propagate()
            if result is True:
                # Check if all clauses satisfied
                if not active:
                    return [a if a is not None else False for a in assignment]
                else:
                    sub_result = dpll_recursive(depth + 1)
                    if sub_result is not None:
                        return sub_result
            
            # Restore state
            assignment[var] = saved_assignment
            active = saved_active
            watches = saved_watches
            unit_queue = saved_queue
            
            # Try False
            assignment[var] = False
            saved_active2 = active.copy()
            saved_watches2 = [w[:] for w in watches]
            saved_queue2 = unit_queue[:]
            
            result = propagate()
            if result is True:
                # Check if all clauses satisfied
                if not active:
                    return [a if a is not None else False for a in assignment]
                else:
                    sub_result = dpll_recursive(depth + 1)
                    if sub_result is not None:
                        return sub_result
            
            # Restore state
            assignment[var] = saved_assignment
            active = saved_active2
            watches = saved_watches2
            unit_queue = saved_queue2
            
            return None
        
        result = dpll_recursive()
        return result
    
    # Try DPLL first
    dpll_result = dpll_watched()
    if dpll_result is not None:
        return dpll_result

    # --- Fallback: Efficient WalkSAT with good heuristics ---
    # Use multiple restarts with different random seeds and increasing effort
    best_assignment = [False] * n
    best_unsat_count = m

    def count_unsat(assignment):
        cnt = 0
        for cv in clause_vars:
            sat = False
            for v, pol in cv:
                if assignment[v] == pol:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    # Track unsat counts per clause for efficient updates
    def init_unsat_counts(assignment):
        counts = [0] * m
        for i, cv in enumerate(clause_vars):
            sat = False
            for v, pol in cv:
                if assignment[v] == pol:
                    sat = True
                    break
            if not sat:
                counts[i] = 1
        return counts

    def walk_sat(assignment, max_steps, rng, deadline):
        """WalkSAT with clause-based scoring. Returns (assignment, unsat_count, found_sat)."""
        n_local = len(assignment)
        
        unsat = init_unsat_counts(assignment)
        total_unsat = sum(unsat)
        
        if total_unsat == 0:
            return assignment, 0, True
        
        # Track unsat clause indices for faster selection
        unsat_indices = [i for i in range(m) if unsat[i] > 0]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if total_unsat == 0:
                return assignment, 0, True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            cv = clause_vars[ci]
            
            # Score each literal in the clause: count how many other unsat clauses it would satisfy
            best_var = -1
            best_score = -1
            best_flip = True
            best_lits = []
            
            for v, pol in cv:
                # If we flip v to pol, how many currently unsat clauses become satisfied?
                score = 0
                for ci2 in var_to_clauses[v]:
                    if unsat[ci2] > 0:
                        # Check if setting v=pol satisfies clause ci2
                        cv2 = clause_vars[ci2]
                        for v2, pol2 in cv2:
                            if v2 == v:
                                if pol2 == pol:
                                    score += 1
                                break
                
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_flip = pol
                    best_lits = [(v, pol, score)]
                elif score == best_score:
                    best_lits.append((v, pol, score))
            
            # With some probability, pick randomly (noise) to escape local minima
            if best_lits:
                if rng.random() < 0.3:
                    v, pol, _ = rng.choice(best_lits)
                else:
                    v, pol, _ = best_lits[0]
            else:
                v, pol = cv[0]
            
            # Flip v to pol
            if assignment[v] != pol:
                assignment[v] = pol
                # Update unsat counts for all clauses containing v
                for ci2 in var_to_clauses[v]:
                    old_unsat = unsat[ci2]
                    # Check if clause ci2 is now satisfied
                    cv2 = clause_vars[ci2]
                    sat = False
                    for v2, pol2 in cv2:
                        if assignment[v2] == pol2:
                            sat = True
                            break
                    new_unsat = 0 if sat else 1
                    if new_unsat != old_unsat:
                        total_unsat += new_unsat - old_unsat
                        unsat[ci2] = new_unsat
                        if new_unsat == 1 and old_unsat == 0:
                            unsat_indices.append(ci2)
                        elif new_unsat == 0 and old_unsat == 1:
                            # Remove from unsat_indices (lazy deletion, we'll rebuild if needed)
                            pass
                # Periodically clean up unsat_indices
                if step % 1000 == 0:
                    unsat_indices = [i for i in range(m) if unsat[i] > 0]
        
        return assignment, total_unsat, total_unsat == 0

    # Multiple restarts with different initial assignments
    rng = random.Random(42)
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE:
            break
        
        # Different initialization strategies
        if attempt == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif attempt == 1:
            assignment = [True] * n
            # Perturb randomly
            for _ in range(n // 2):
                assignment[rng.randint(0, n-1)] = not assignment[rng.randint(0, n-1)]
        elif attempt % 2 == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Try biased initialization
            bias = 0.3 + 0.4 * rng.random()
            assignment = [rng.random() < bias for _ in range(n)]
        
        # Check if already satisfied
        uc = count_unsat(assignment)
        if uc == 0:
            return assignment
        
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]
        
        # Run WalkSAT with step budget based on remaining time
        elapsed = time.time() - START
        remaining_time = DEADLINE - elapsed
        if remaining_time < 1:
            break
        
        # Allocate time per attempt
        time_per_attempt = remaining_time / (max_attempts - attempt)
        max_steps = int(time_per_attempt * 100000)  # Estimate ~100k steps per second
        max_steps = max(1000, min(max_steps, 500000))
        
        result_assignment, result_uc, found = walk_sat(
            assignment[:], max_steps, rng, DEADLINE
        )
        
        if found:
            return result_assignment
        
        if result_uc < best_unsat_count:
            best_unsat_count = result_uc
            best_assignment = result_assignment[:]

    return best_assignment

# EVOLVE-BLOCK-END
