import time
import random
import math

def solve(instance):
    """
    Solves a planted 3-SAT instance using a hybrid MinConflicts and WalkSAT strategy.
    
    Args:
        instance: A dictionary with keys 'n' (number of variables) and 'clauses' (list of lists of signed literals).
        
    Returns:
        A list of booleans representing the assignment.
    """
    n = instance["n"]
    raw_clauses = instance["clauses"]
    
    # Deadline management: 330s total, use 325s as safety margin
    start_time = time.time()
    deadline = start_time + 325.0
    
    # --- Preprocessing ---
    # Convert clauses to 0-indexed variables and store sign
    # clause: list of (var_idx, sign) where sign is +1 or -1
    # For speed, we pre-compute for each variable which clauses contain it and with what sign.
    # var_clauses[v] = list of (clause_idx, sign)
    
    var_clauses = [[] for _ in range(n)]
    processed_clauses = []
    
    for ci, clause in enumerate(raw_clauses):
        p_clause = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            p_clause.append((v, sign))
            var_clauses[v].append((ci, sign))
        processed_clauses.append(p_clause)
        
    num_clauses = len(processed_clauses)
    
    # Helper to check if a clause is satisfied given an assignment
    def is_clause_satisfied(ci, assignment):
        # Check if any literal in the clause is true
        for v, sign in processed_clauses[ci]:
            if assignment[v] == (sign == 1):
                return True
        return False

    # Helper to count unsatisfied clauses
    def count_unsat(assignment):
        count = 0
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                count += 1
        return count

    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    # We will run multiple restarts
    max_restarts = 1000
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialize assignment
        # For planted instances, random initialization is often good.
        # We can also try a heuristic: assign variables to satisfy as many unit clauses as possible? 
        # But 3-SAT doesn't have units initially. Random is fine.
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat count
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        # 2. Local Search Loop
        # We alternate between MinConflicts and WalkSAT to escape local minima.
        # MinConflicts is greedy, WalkSAT is random.
        
        max_steps_per_restart = 2000
        stuck_counter = 0
        best_local_unsat = current_unsat
        best_local_assignment = assignment[:]
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            if current_unsat == 0:
                return assignment
                
            if current_unsat < best_local_unsat:
                best_local_unsat = current_unsat
                best_local_assignment = assignment[:]
                stuck_counter = 0
            else:
                stuck_counter += 1
                
            # Choose strategy:
            # If stuck for a while, use WalkSAT (random flips) to escape.
            # Otherwise, use MinConflicts (greedy flips).
            
            if stuck_counter > 50 and rng.random() < 0.5:
                # WalkSAT Phase
                # Pick a random unsatisfied clause
                unsat_clause_idx = -1
                for _ in range(10): # Try random sampling first
                    ci = rng.randrange(num_clauses)
                    if not is_clause_satisfied(ci, assignment):
                        unsat_clause_idx = ci
                        break
                if unsat_clause_idx == -1:
                    # Fallback: scan
                    for ci in range(num_clauses):
                        if not is_clause_satisfied(ci, assignment):
                            unsat_clause_idx = ci
                            break
                            
                if unsat_clause_idx == -1:
                    break # Should not happen if current_unsat > 0
                    
                # Pick a random variable in the clause to flip
                clause_vars = processed_clauses[unsat_clause_idx]
                v_idx = rng.randrange(len(clause_vars))
                v, _ = clause_vars[v_idx]
                
                # Flip
                assignment[v] = not assignment[v]
                
                # Update unsat count efficiently? 
                # For n=50, full recount is ~50*3 = 150 ops. Very fast.
                # But we can do incremental.
                # Let's stick to full recount for simplicity and correctness, it's fast enough for 50 vars.
                # 150 ops * 2000 steps * 1000 restarts = 300M ops. Might be tight in Python.
                # Let's optimize: only recount clauses affected by v.
                
                # Actually, let's implement an incremental update for speed.
                # We need to know which clauses are currently unsatisfied.
                # Maintaining a set of unsatisfied clauses is better.
                
                # Let's refactor to maintain a set of unsat clauses.
                pass # Placeholder, see below for optimized version
                
            else:
                # MinConflicts Phase
                # Pick a random unsatisfied clause
                unsat_clause_idx = -1
                for _ in range(10):
                    ci = rng.randrange(num_clauses)
                    if not is_clause_satisfied(ci, assignment):
                        unsat_clause_idx = ci
                        break
                if unsat_clause_idx == -1:
                    for ci in range(num_clauses):
                        if not is_clause_satisfied(ci, assignment):
                            unsat_clause_idx = ci
                            break
                            
                if unsat_clause_idx == -1:
                    break
                    
                clause_vars = processed_clauses[unsat_clause_idx]
                
                # Evaluate impact of flipping each variable in this clause
                best_v = -1
                best_delta = 100 # High number
                best_new_unsat = current_unsat
                
                for v, sign in clause_vars:
                    # If we flip v, which clauses change status?
                    # Only clauses in var_clauses[v]
                    
                    # To calculate delta, we need to know current status of those clauses.
                    # This is expensive if we don't track status.
                    
                    # Alternative: Just try the flip, count new unsat, if worse, undo?
                    # That's expensive (O(C) per flip).
                    
                    # Better: Pre-compute "witness" or just use a simpler heuristic for MinConflicts.
                    # Standard MinConflicts: Flip variable that minimizes number of unsatisfied clauses 
                    # among the clauses it appears in.
                    
                    # Let's calculate the "cost" of flipping v.
                    # Cost = (# clauses currently satisfied that would become unsatisfied) 
                    #       - (# clauses currently unsatisfied that would become satisfied)
                    
                    # This requires knowing the status of each clause in var_clauses[v].
                    # Since n=50, var_clauses[v] is small (maybe 5-10 clauses).
                    # Checking status of a clause is O(3).
                    # So checking impact of one v is O(30).
                    # Checking impact of 3 variables in clause is O(90).
                    # This is fast.
                    
                    new_unsat_if_flip = 0
                    for ci2, sign2 in var_clauses[v]:
                        # Check if clause ci2 is satisfied with current assignment
                        # But wait, if we flip v, the status of ci2 changes only if v is the 
                        # "witness" or the only false literal? No.
                        
                        # Simpler: 
                        # 1. Save current assignment of v
                        # 2. Flip v
                        # 3. Count unsat in affected clauses
                        # 4. Restore v
                        # This is O(k) where k is clauses containing v.
                        
                        pass # Too complex to inline efficiently without state.
                        
                    # Let's use a simpler heuristic for MinConflicts:
                    # Pick the variable in the unsatisfied clause that appears in the 
                    # fewest unsatisfied clauses.
                    
                    # Count unsatisfied clauses for each v in the clause
                    min_unsat_count = num_clauses + 1
                    chosen_v = -1
                    
                    for v, sign in clause_vars:
                        # Count how many clauses in var_clauses[v] are currently unsatisfied
                        unsat_count_for_v = 0
                        for ci2, _ in var_clauses[v]:
                            if not is_clause_satisfied(ci2, assignment):
                                unsat_count_for_v += 1
                                if unsat_count_for_v >= min_unsat_count:
                                    break
                        if unsat_count_for_v < min_unsat_count:
                            min_unsat_count = unsat_count_for_v
                            chosen_v = v
                            
                    # Flip chosen_v
                    assignment[chosen_v] = not assignment[chosen_v]
                    
                    # Update current_unsat
                    # Recount affected clauses
                    # We need to know which clauses were unsatisfied before and after.
                    # For n=50, let's just recount all unsat clauses.
                    # 50 vars * ~5 clauses/var * 3 literals/clause = 750 ops.
                    # 2000 steps * 1000 restarts = 2M steps. 2M * 750 = 1.5B ops.
                    # This might be too slow in Python (1.5B ops ~ 10-30s? No, Python is slower, ~100M ops/s).
                    # 1.5B ops in Python could take 15-30 seconds just for the counting.
                    # We need to be smarter or limit restarts/steps.
                    
                    # Optimization: Maintain a set of unsatisfied clause indices.
                    # When flipping v, only clauses in var_clauses[v] can change status.
                    # For each such clause:
                    #   - If it was unsatisfied and now satisfied: remove from set, decrement count
                    #   - If it was satisfied and now unsatisfied: add to set, increment count
                    #   - If it was unsatisfied and still unsatisfied: no change
                    #   - If it was satisfied and still satisfied: no change
                    
                    # This is O(k) per flip, where k is clauses containing v.
                    # k is small. This is efficient.
                    
                    # Let's implement this properly in the main loop.
                    pass # See final implementation below

        # End of local search for this restart
        
        if best_local_unsat < best_unsat_count:
            best_unsat_count = best_local_unsat
            best_assignment = best_local_assignment[:]
            
        # If we found a solution, return immediately
        if best_unsat_count == 0:
            return best_assignment
            
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n

# --- Optimized Implementation with Incremental Updates ---

def solve(instance):
    n = instance["n"]
    raw_clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess
    var_clauses = [[] for _ in range(n)]
    processed_clauses = []
    for ci, clause in enumerate(raw_clauses):
        p_clause = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            p_clause.append((v, sign))
            var_clauses[v].append((ci, sign))
        processed_clauses.append(p_clause)
        
    num_clauses = len(processed_clauses)
    
    def is_clause_satisfied(ci, assignment):
        for v, sign in processed_clauses[ci]:
            if assignment[v] == (sign == 1):
                return True
        return False

    best_assignment = None
    best_unsat_count = num_clauses + 1
    rng = random.Random(12345)
    
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize unsat set
        unsat_set = set()
        current_unsat = 0
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat_set.add(ci)
                current_unsat += 1
                
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        # Local Search
        max_steps = 1000
        stuck = 0
        best_local_unsat = current_unsat
        best_local_assign = assignment[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if current_unsat == 0:
                return assignment
                
            if current_unsat < best_local_unsat:
                best_local_unsat = current_unsat
                best_local_assign = assignment[:]
                stuck = 0
            else:
                stuck += 1
                
            # Strategy: MinConflicts with random tie-breaking
            # Pick a random unsatisfied clause
            if unsat_set:
                # Random choice from set
                unsat_ci = next(iter(unsat_set)) # O(1) if we just pick one, but set iteration is not random.
                # For true random, we need a list or index.
                # Since unsat_set can be large, converting to list is O(U).
                # Let's just pick a random index and check if it's unsat.
                # Or maintain a list of unsat clauses.
                
                # Let's use a list for unsat clauses for faster random access.
                # But updating a list (removing/adding) is O(U).
                # For n=50, U is small. Let's use a list.
                
                pass # Refactor to use list
                
            else:
                break
                
        if best_local_unsat < best_unsat_count:
            best_unsat_count = best_local_unsat
            best_assignment = best_local_assign[:]
            
        if best_unsat_count == 0:
            return best_assignment
            
    return best_assignment if best_assignment else [False] * n

# --- Final Optimized Implementation ---

def solve(instance):
    n = instance["n"]
    raw_clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess
    var_clauses = [[] for _ in range(n)]
    processed_clauses = []
    for ci, clause in enumerate(raw_clauses):
        p_clause = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            p_clause.append((v, sign))
            var_clauses[v].append((ci, sign))
        processed_clauses.append(p_clause)
        
    num_clauses = len(processed_clauses)
    
    def is_clause_satisfied(ci, assignment):
        for v, sign in processed_clauses[ci]:
            if assignment[v] == (sign == 1):
                return True
        return False

    best_assignment = None
    best_unsat_count = num_clauses + 1
    rng = random.Random(42)
    
    max_restarts = 2000 # Increase restarts, steps will be shorter
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize list of unsatisfied clauses
        unsat_list = []
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat_list.append(ci)
                
        current_unsat = len(unsat_list)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        # Local Search
        max_steps = 500 # Shorter steps, more restarts
        stuck = 0
        best_local_unsat = current_unsat
        best_local_assign = assignment[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if current_unsat == 0:
                return assignment
                
            if current_unsat < best_local_unsat:
                best_local_unsat = current_unsat
                best_local_assign = assignment[:]
                stuck = 0
            else:
                stuck += 1
                
            if not unsat_list:
                break
                
            # Pick a random unsatisfied clause
            unsat_ci = unsat_list[rng.randrange(len(unsat_list))]
            
            # MinConflicts: Find variable in this clause that minimizes unsat count in its clauses
            clause_vars = processed_clauses[unsat_ci]
            
            best_v = -1
            best_score = num_clauses + 1 # We want to minimize this score
            
            for v, sign in clause_vars:
                # Count unsat clauses in var_clauses[v]
                score = 0
                for ci2, _ in var_clauses[v]:
                    if not is_clause_satisfied(ci2, assignment):
                        score += 1
                        if score >= best_score:
                            break
                if score < best_score:
                    best_score = score
                    best_v = v
                    
            if best_v == -1:
                continue
                
            # Flip best_v
            assignment[best_v] = not assignment[best_v]
            
            # Update unsat_list
            # Only clauses in var_clauses[best_v] can change status
            # We need to rebuild unsat_list or update it.
            # Rebuilding is O(C). C is ~50-100. This is fast.
            # 500 steps * 2000 restarts = 1M steps. 1M * 100 = 100M ops. Acceptable.
            
            new_unsat_list = []
            for ci in range(num_clauses):
                if not is_clause_satisfied(ci, assignment):
                    new_unsat_list.append(ci)
                    
            unsat_list = new_unsat_list
            current_unsat = len(unsat_list)
            
        if best_local_unsat < best_unsat_count:
            best_unsat_count = best_local_unsat
            best_assignment = best_local_assign[:]
            
        if best_unsat_count == 0:
            return best_assignment
            
    return best_assignment if best_assignment else [False] * n

