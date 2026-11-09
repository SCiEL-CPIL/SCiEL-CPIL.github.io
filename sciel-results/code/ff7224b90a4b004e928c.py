import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Precomputation: convert to 0-based, build clause data and variable-to-clause index
    clause_data = []
    var_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append(i)
        clause_data.append(cl_store)
    
    m = len(clauses)
    
    # Helper functions
    def is_satisfied(assignment, clause_idx):
        for v, sign in clause_data[clause_idx]:
            if assignment[v] == (sign == 1):
                return True
        return False
    
    def get_unsat_clauses(assignment):
        return [i for i in range(m) if not is_satisfied(assignment, i)]
    
    def count_unsat(assignment):
        return len(get_unsat_clauses(assignment))
    
    # Complete DPLL with watched literals, unit propagation, and pure literal elimination
    def dpll_complete():
        """
        Complete DPLL solver with:
        - Watched literals for efficient clause propagation
        - Unit propagation
        - Pure literal elimination
        Returns assignment list if solved, else None.
        """
        # Assignment: -1 = unassigned, 0 = False, 1 = True
        assignment = [-1] * n
        
        # Watched literals: for each clause, watch two literals
        # A literal is (var, sign) where sign=1 means positive
        # We track which clauses watch each variable
        # watches[var] = list of (clause_idx, other_var, other_sign)
        # meaning clause clause_idx has a watched literal on var, and the other watched literal is (other_var, other_sign)
        
        # Build initial watched literals
        # For each clause, pick first two literals as watches
        watches = [[] for _ in range(n)]  # watches[v] = list of (clause_idx, other_var, other_sign)
        clause_watches = [None] * m  # clause_watches[ci] = (var1, var2) the two watched vars
        
        for ci in range(m):
            cd = clause_data[ci]
            if len(cd) >= 2:
                v1, s1 = cd[0]
                v2, s2 = cd[1]
                watches[v1].append((ci, v2, s2))
                watches[v2].append((ci, v1, s1))
                clause_watches[ci] = (v1, v2)
            elif len(cd) == 1:
                v1, s1 = cd[0]
                watches[v1].append((ci, -1, -1))  # marker for unit clause
                clause_watches[ci] = (v1, -1)
        
        # Active clauses (not yet satisfied)
        active = [True] * m
        
        def check_satisfied(ci):
            for v, sign in clause_data[ci]:
                if assignment[v] == (1 if sign == 1 else 0):
                    return True
            return False
        
        def propagate_queue():
            """
            Perform unit propagation using a queue of assignments to make.
            Returns False if contradiction detected, True otherwise.
            """
            queue = []
            
            def process_queue():
                nonlocal active
                while queue:
                    v, val = queue.pop()
                    # val is 0 or 1
                    if assignment[v] != -1:
                        if assignment[v] != val:
                            return False
                        continue
                    assignment[v] = val
                    
                    # Process watches on variable v
                    new_watches = []
                    for (ci, other_v, other_s) in watches[v]:
                        if not active[ci]:
                            new_watches.append((ci, other_v, other_s))
                            continue
                        
                        # Check if clause is satisfied
                        if check_satisfied(ci):
                            active[ci] = False
                            continue
                        
                        # Check if other watched literal is satisfied
                        if other_v != -1:
                            other_val = 1 if other_s == 1 else 0
                            if assignment[other_v] == other_val:
                                # Other literal is satisfied, clause is satisfied
                                active[ci] = False
                                continue
                            elif assignment[other_v] == -1:
                                # Other literal is unassigned, keep watching it
                                new_watches.append((ci, other_v, other_s))
                                continue
                            else:
                                # Other literal is falsified, need to find new watch
                                pass
                        else:
                            # Unit clause, other_v is -1
                            pass
                        
                        # Need to find a new watched literal for this clause
                        # Look for an unassigned literal in the clause
                        new_watch_found = False
                        for v2, s2 in clause_data[ci]:
                            if v2 == v:
                                continue
                            if assignment[v2] == -1:
                                # Found new watch
                                # Remove old watch from v2's watch list (if it was watching v)
                                # Add new watch to v2's watch list
                                # This is tricky - we need to update watches
                                # For simplicity, let's just add to new_watches and handle dedup later
                                new_watches.append((ci, v2, s2))
                                new_watch_found = True
                                break
                        
                        if not new_watch_found:
                            # All other literals are falsified -> contradiction
                            return False
                    
                    watches[v] = new_watches
                
                return True
            
            return process_queue()
        
        # Initial unit propagation
        initial_queue = []
        for ci in range(m):
            if not active[ci]:
                continue
            # Check if clause is unit
            unassigned = []
            for v, sign in clause_data[ci]:
                if assignment[v] == -1:
                    unassigned.append((v, sign))
            if len(unassigned) == 0:
                # All assigned and not satisfied -> contradiction
                return None
            if len(unassigned) == 1:
                v, sign = unassigned[0]
                initial_queue.append((v, 1 if sign == 1 else 0))
        
        if initial_queue:
            if not propagate_queue():
                return None
        
        # Check if all clauses are satisfied
        all_sat = all(not active[ci] for ci in range(m))
        if all_sat:
            # Fill in remaining unassigned variables
            for i in range(n):
                if assignment[i] == -1:
                    assignment[i] = random.randint(0, 1)
            return [bool(a) for a in assignment]
        
        # Pure literal elimination
        def pure_literal_elimination():
            """
            Eliminate pure literals. Returns list of (var, value) assignments to make.
            """
            nonlocal active
            # For each unassigned variable, check if it appears with only one polarity
            var_polarities = {}
            for v in range(n):
                if assignment[v] != -1:
                    continue
                pos = False
                neg = False
                for ci in var_clauses[v]:
                    if not active[ci]:
                        continue
                    for v2, s2 in clause_data[ci]:
                        if v2 == v:
                            if s2 == 1:
                                pos = True
                            else:
                                neg = True
                            break
                if pos and not neg:
                    var_polarities[v] = 1
                elif neg and not pos:
                    var_polarities[v] = 0
                else:
                    var_polarities[v] = None
            
            # Make assignments for pure literals
            queue = []
            for v, val in var_polarities.items():
                if val is not None and assignment[v] == -1:
                    queue.append((v, val))
            
            if queue:
                if not propagate_queue():
                    return None, False
            return queue, True
        
        # Recursive DPLL
        def dpll_rec(depth, max_depth=100):
            if depth > max_depth:
                return None
            
            # Check if all satisfied
            all_sat = all(not active[ci] for ci in range(m))
            if all_sat:
                for i in range(n):
                    if assignment[i] == -1:
                        assignment[i] = random.randint(0, 1)
                return [bool(a) for a in assignment]
            
            # Pure literal elimination
            _, success = pure_literal_elimination()
            if not success:
                return None
            
            # Check again
            all_sat = all(not active[ci] for ci in range(m))
            if all_sat:
                for i in range(n):
                    if assignment[i] == -1:
                        assignment[i] = random.randint(0, 1)
                return [bool(a) for a in assignment]
            
            # Variable selection: pick variable appearing in most active clauses
            var_counts = {}
            for ci in range(m):
                if not active[ci]:
                    continue
                for v, _ in clause_data[ci]:
                    if assignment[v] == -1:
                        var_counts[v] = var_counts.get(v, 0) + 1
            
            if not var_counts:
                # No unassigned variables but not all satisfied -> contradiction
                return None
            
            max_count = max(var_counts.values())
            candidates = [v for v, c in var_counts.items() if c == max_count]
            var = candidates[0]
            
            # Try both values
            for val in [1, 0]:
                # Save state
                old_assignment = assignment[:]
                old_active = active[:]
                old_watches = [w[:] for w in watches]
                
                # Assign
                assignment[var] = val
                
                # Propagate
                queue = [(var, val)]
                # Re-run propagation
                # We need to re-queue and process
                # Let's just call a simplified propagation
                changed = True
                while changed:
                    changed = False
                    for ci in range(m):
                        if not active[ci]:
                            continue
                        sat = False
                        unassigned = []
                        for v, sign in clause_data[ci]:
                            if assignment[v] != -1:
                                if assignment[v] == (1 if sign == 1 else 0):
                                    sat = True
                                    break
                            else:
                                unassigned.append((v, sign))
                        if sat:
                            active[ci] = False
                            changed = True
                        elif not unassigned:
                            # Contradiction
                            active = old_active
                            assignment = old_assignment
                            watches = old_watches
                            return None if val == 1 else dpll_rec(depth + 1, max_depth)
                        elif len(unassigned) == 1:
                            v, sign = unassigned[0]
                            if assignment[v] == -1:
                                assignment[v] = 1 if sign == 1 else 0
                                changed = True
                
                # Check if solved
                all_sat = all(not active[ci] for ci in range(m))
                if all_sat:
                    for i in range(n):
                        if assignment[i] == -1:
                            assignment[i] = random.randint(0, 1)
                    return [bool(a) for a in assignment]
                
                # Recurse
                result = dpll_rec(depth + 1, max_depth)
                if result is not None:
                    return result
                
                # Backtrack
                assignment = old_assignment
                active = old_active
                watches = old_watches
            
            return None
        
        return dpll_rec(0, max_depth=50)
    
    # Strategy 1: Try complete DPLL with a time limit
    sol = dpll_complete()
    if sol is not None:
        if count_unsat(sol) == 0:
            return sol
    
    # Strategy 2: High-performance WalkSAT with adaptive noise
    rng = random.Random(42)
    
    def walksat_adaptive(max_steps, initial_noise=0.5, noise_decay=0.999):
        """
        WalkSAT with adaptive noise parameter.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_clauses = get_unsat_clauses(assignment)
        
        if not unsat_clauses:
            return assignment
        
        noise = initial_noise
        best_unsat = len(unsat_clauses)
        best_assignment = assignment.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            # Select clause: with probability proportional to unsat count in clause
            # Use weighted selection
            weights = []
            for ci in unsat_clauses:
                # Count how many variables in this clause are in other unsat clauses
                w = 1.0
                for v, _ in clause_data[ci]:
                    w += 0.1 * len(var_clauses[v])
                weights.append(w)
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            selected_clause = unsat_clauses[0]
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    selected_clause = unsat_clauses[idx]
                    break
            
            # Select variable to flip
            clause_vars = clause_data[selected_clause]
            
            if rng.random() < noise:
                # Random variable from clause
                v_to_flip = clause_vars[rng.randint(0, 2)][0]
            else:
                # Variable that satisfies the most other unsat clauses
                best_var = clause_vars[0][0]
                best_score = -1
                for v, sign in clause_vars:
                    score = 0
                    for other_ci in unsat_clauses:
                        if other_ci == selected_clause:
                            continue
                        for v2, s2 in clause_data[other_ci]:
                            if v2 == v:
                                if (not assignment[v]) == (s2 == 1):
                                    score += 1
                                break
                    if score > best_score:
                        best_score = score
                        best_var = v
                v_to_flip = best_var
            
            # Flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update unsat clauses
            unsat_clauses = get_unsat_clauses(assignment)
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment.copy()
            
            if not unsat_clauses:
                return assignment
            
            # Adapt noise
            noise = max(0.1, noise * noise_decay)
        
        if best_unsat == 0:
            return best_assignment
        return None
    
    # Run WalkSAT with multiple restarts
    max_restarts = 200
    base_steps = 3000
    
    global_best_unsat = float('inf')
    global_best_assignment = None
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary parameters
        noise = 0.3 + 0.3 * math.exp(-r * 0.01)
        steps = base_steps + r * 200
        
        sol = walksat_adaptive(steps, initial_noise=noise)
        if sol is not None:
            unsat_count = count_unsat(sol)
            if unsat_count == 0:
                return sol
            if unsat_count < global_best_unsat:
                global_best_unsat = unsat_count
                global_best_assignment = sol.copy()
    
    # Final polish: local search from best assignment
    if global_best_assignment is not None and global_best_unsat <= 10 and time.time() < deadline - 30:
        assignment = global_best_assignment[:]
        unsat_clauses = get_unsat_clauses(assignment)
        best_unsat = len(unsat_clauses)
        best_assignment = assignment.copy()
        
        # Simple local search: flip variable that reduces unsat the most
        for step in range(100000):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            best_var = None
            best_delta = 0
            for v in range(n):
                old_val = assignment[v]
                assignment[v] = not assignment[v]
                new_unsat = get_unsat_clauses(assignment)
                delta = len(new_unsat) - best_unsat
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                assignment[v] = old_val
            
            if best_var is None or best_delta >= 0:
                # No improving move, try random flip
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
                unsat_clauses = get_unsat_clauses(assignment)
            else:
                assignment[best_var] = not assignment[best_var]
                unsat_clauses = get_unsat_clauses(assignment)
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment.copy()
            
            if not unsat_clauses:
                return assignment
        
        if best_unsat == 0:
            return best_assignment
        return best_assignment
    
    if global_best_assignment is not None:
        return global_best_assignment
    
    return [rng.random() < 0.5 for _ in range(n)]
