"""Portfolio-based 3-SAT solver for planted satisfiable instances (50 vars)."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _preprocess(clauses, n):
    """
    Preprocess clauses for fast access.
    Returns:
    - clause_vars: list of lists of variable indices (0-based)
    - clause_polarity: list of lists of booleans (True for positive, False for negative)
    - var_to_clauses: list of lists of clause indices for each variable
    """
    clause_vars = []
    clause_polarity = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        vars_list = []
        pol_list = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            vars_list.append(v)
            pol_list.append(p)
            var_to_clauses[v].append(i)
        clause_vars.append(vars_list)
        clause_polarity.append(pol_list)
        
    return clause_vars, clause_polarity, var_to_clauses

def _count_unsat_fast(clause_vars, clause_polarity, assignment):
    """Count unsatisfied clauses using preprocessed data."""
    count = 0
    for cv, cp in zip(clause_vars, clause_polarity):
        sat = False
        for v, p in zip(cv, cp):
            if assignment[v] == p:
                sat = True
                break
        if not sat:
            count += 1
    return count

def _heuristic_seed(n, clauses, rng):
    """
    Create an initial assignment biased by literal frequency (Phase Heuristic).
    """
    pos_count = [0] * n
    neg_count = [0] * n
    for c in clauses:
        for l in c:
            var = abs(l) - 1
            if l > 0:
                pos_count[var] += 1
            else:
                neg_count[var] += 1

    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(True)
        elif neg_count[i] > pos_count[i]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)
    return assignment

def _walksat_optimized(clause_vars, clause_polarity, var_to_clauses, n, initial_assignment, rng, max_flips, noise_schedule, deadline):
    """
    Optimized WalkSAT with incremental state updates.
    """
    assignment = initial_assignment[:]
    
    # Initialize state
    unsat_count = _count_unsat_fast(clause_vars, clause_polarity, assignment)
    if unsat_count == 0:
        return assignment, 0
        
    # var_unsat_count[v] = number of unsatisfied clauses containing v
    var_unsat_count = [0] * n
    # We need to know which clauses are currently unsatisfied to update counts correctly
    # Let's maintain a boolean array for clause satisfaction status
    clause_sat = [False] * len(clause_vars)
    
    for i, (cv, cp) in enumerate(zip(clause_vars, clause_polarity)):
        sat = False
        for v, p in zip(cv, cp):
            if assignment[v] == p:
                sat = True
                break
        clause_sat[i] = sat
        if not sat:
            for v in cv:
                var_unsat_count[v] += 1
                
    for step in range(max_flips):
        if time.time() > deadline:
            break
            
        if unsat_count == 0:
            return assignment, 0

        # Find unsat clauses
        # Optimization: Instead of scanning all clauses, we could pick a random unsat clause
        # but scanning is O(C). With C ~ 500-1000, it's fast enough.
        unsat_indices = [i for i in range(len(clause_vars)) if not clause_sat[i]]
        if not unsat_indices:
            return assignment, 0
            
        clause_idx = rng.choice(unsat_indices)
        cv = clause_vars[clause_idx]
        cp = clause_polarity[clause_idx]
        
        # Score variables in the clause
        # We want to flip a variable. 
        # Heuristic: Pick variable with highest var_unsat_count (fixes most clauses)
        # But also consider noise.
        
        noise = noise_schedule(step, max_flips)
        
        if rng.random() < noise:
            # Random flip
            var_idx = rng.randrange(len(cv))
            v = cv[var_idx]
        else:
            # Greedy: highest var_unsat_count
            best_v = cv[0]
            best_score = var_unsat_count[best_v]
            for v in cv[1:]:
                score = var_unsat_count[v]
                if score > best_score:
                    best_score = score
                    best_v = v
            # Tie breaking randomly among max
            max_score = best_score
            candidates = [v for v in cv if var_unsat_count[v] == max_score]
            v = rng.choice(candidates)
            
        # Flip v
        assignment[v] = not assignment[v]
        
        # Update state incrementally
        # Check all clauses containing v
        for ci in var_to_clauses[v]:
            cv_i = clause_vars[ci]
            cp_i = clause_polarity[ci]
            
            was_sat = clause_sat[ci]
            
            # Check if it is satisfied now
            is_sat = False
            for var, pol in zip(cv_i, cp_i):
                if assignment[var] == pol:
                    is_sat = True
                    break
            
            if was_sat and not is_sat:
                # Became unsat
                unsat_count += 1
                clause_sat[ci] = False
                # Update var_unsat_count for all vars in this clause
                for var in cv_i:
                    var_unsat_count[var] += 1
            elif not was_sat and is_sat:
                # Became sat
                unsat_count -= 1
                clause_sat[ci] = True
                for var in cv_i:
                    var_unsat_count[var] -= 1
            # else: status unchanged (sat->sat or unsat->unsat)
            # Note: if unsat->unsat, counts don't change.
            # if sat->sat, counts don't change.

    return assignment, unsat_count

def _min_conflicts_optimized(clause_vars, clause_polarity, var_to_clauses, n, initial_assignment, rng, max_flits, deadline):
    """
    Optimized Min-Conflicts: Flip variable that reduces unsat count the most.
    """
    assignment = initial_assignment[:]
    
    unsat_count = _count_unsat_fast(clause_vars, clause_polarity, assignment)
    if unsat_count == 0:
        return assignment, 0
        
    var_unsat_count = [0] * n
    clause_sat = [False] * len(clause_vars)
    
    for i, (cv, cp) in enumerate(zip(clause_vars, clause_polarity)):
        sat = False
        for v, p in zip(cv, cp):
            if assignment[v] == p:
                sat = True
                break
        clause_sat[i] = sat
        if not sat:
            for v in cv:
                var_unsat_count[v] += 1
                
    for step in range(max_flits):
        if time.time() > deadline:
            break
            
        if unsat_count == 0:
            return assignment, 0
            
        # Find variable to flip
        # Candidate: variable in an unsat clause
        # We can pick a random unsat clause, then pick best var in it
        unsat_indices = [i for i in range(len(clause_vars)) if not clause_sat[i]]
        if not unsat_indices:
            return assignment, 0
            
        # To be more global, we might sample a few unsat clauses
        # But standard MinConflicts picks one. Let's stick to one for speed, 
        # but maybe pick the clause with highest "potential" (sum of var_unsat_count)
        best_clause_idx = -1
        best_clause_score = -1
        # Sample up to 10 unsat clauses to find a good one, or just random if few
        sample_size = min(10, len(unsat_indices))
        sample = rng.sample(unsat_indices, sample_size)
        
        for ci in sample:
            score = sum(var_unsat_count[v] for v in clause_vars[ci])
            if score > best_clause_score:
                best_clause_score = score
                best_clause_idx = ci
                
        cv = clause_vars[best_clause_idx]
        
        # Find best var to flip in this clause
        # Delta for flipping v:
        # For each clause containing v:
        # If clause was sat: it becomes unsat if v was the ONLY satisfying literal.
        # If clause was unsat: it becomes sat if v becomes satisfying.
        
        best_var = cv[0]
        best_delta = 0
        
        for v in cv:
            delta = 0
            for ci in var_to_clauses[v]:
                cv_i = clause_vars[ci]
                cp_i = clause_polarity[ci]
                was_sat = clause_sat[ci]
                
                # Check if v is currently satisfying this clause
                v_pol = cp_i[cv_i.index(v)] # O(L) lookup, L=3
                v_satisfies = (assignment[v] == v_pol)
                
                if was_sat:
                    # If we flip v, does it break satisfaction?
                    # It breaks if v was satisfying AND no other literal satisfies it.
                    if v_satisfies:
                        # Check if any other literal satisfies
                        others_satisfy = False
                        for idx, var in enumerate(cv_i):
                            if var == v:
                                continue
                            if assignment[var] == cp_i[idx]:
                                others_satisfy = True
                                break
                        if not others_satisfy:
                            delta += 1 # Becomes unsat
                else: # not was_sat
                    # If we flip v, does it satisfy?
                    # It satisfies if the new value of v matches its polarity.
                    # new_val = not assignment[v]
                    if not v_satisfies:
                        # v was not satisfying, so flipping it makes it satisfying
                        delta -= 1 # Becomes sat
            
            if delta < best_delta:
                best_delta = delta
                best_var = v
                
        # Flip best_var
        v = best_var
        assignment[v] = not assignment[v]
        
        # Update state incrementally
        for ci in var_to_clauses[v]:
            cv_i = clause_vars[ci]
            cp_i = clause_polarity[ci]
            
            was_sat = clause_sat[ci]
            
            is_sat = False
            for var, pol in zip(cv_i, cp_i):
                if assignment[var] == pol:
                    is_sat = True
                    break
            
            if was_sat and not is_sat:
                unsat_count += 1
                clause_sat[ci] = False
                for var in cv_i:
                    var_unsat_count[var] += 1
            elif not was_sat and is_sat:
                unsat_count -= 1
                clause_sat[ci] = True
                for var in cv_i:
                    var_unsat_count[var] -= 1

    return assignment, unsat_count

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # safety margin

    rng = random.Random(42)

    # Preprocess
    clause_vars, clause_polarity, var_to_clauses = _preprocess(clauses, n)
    
    # Best assignment found so far
    best_assignment = None
    best_unsat = float('inf')

    def update_best(assignment, unsat_count):
        nonlocal best_assignment, best_unsat
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        if unsat_count == 0:
            return True
        return False

    # Strategy 1: Phase Heuristic Seed + Optimized WalkSAT
    # Use multiple restarts with varying noise
    num_restarts = 100
    flips_per_restart = 10000
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break

        if r == 0:
            assignment = _heuristic_seed(n, clauses, rng)
        elif r % 5 == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Mutate best
            if best_assignment is not None:
                assignment = best_assignment[:]
                # Flip 1-3 random variables
                for _ in range(rng.randint(1, 3)):
                    idx = rng.randrange(n)
                    assignment[idx] = not assignment[idx]
            else:
                assignment = _heuristic_seed(n, clauses, rng)

        # Adaptive noise: start low (greedy), end high (random)
        def noise_schedule(step, max_steps):
            progress = step / max_steps
            return 0.1 + 0.4 * progress

        assignment, unsat = _walksat_optimized(
            clause_vars, clause_polarity, var_to_clauses, n, 
            assignment, rng, flips_per_restart, noise_schedule, deadline
        )
        if update_best(assignment, unsat):
            return assignment

    # Strategy 2: Optimized Min-Conflicts
    # Start from best found
    if best_assignment is not None:
        if time.time() < deadline - 10:
            assignment = best_assignment[:]
            assignment, unsat = _min_conflicts_optimized(
                clause_vars, clause_polarity, var_to_clauses, n,
                assignment, rng, 20000, deadline
            )
            if update_best(assignment, unsat):
                return assignment

    # Strategy 3: More aggressive WalkSAT with higher noise
    for r in range(50):
        if time.time() > deadline - 5:
            break
        assignment = _heuristic_seed(n, clauses, rng)
        
        def noise_schedule2(step, max_steps):
            return 0.3 + 0.3 * (step / max_steps)
            
        assignment, unsat = _walksat_optimized(
            clause_vars, clause_polarity, var_to_clauses, n,
            assignment, rng, 15000, noise_schedule2, deadline
        )
        if update_best(assignment, unsat):
            return assignment

    # Strategy 4: Min-Conflicts from random seeds
    for r in range(20):
        if time.time() > deadline - 5:
            break
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment, unsat = _min_conflicts_optimized(
            clause_vars, clause_polarity, var_to_clauses, n,
            assignment, rng, 5000, deadline
        )
        if update_best(assignment, unsat):
            return assignment

    # If no full solution, return the best assignment
    if best_assignment is not None:
        return best_assignment
    else:
        return _heuristic_seed(n, clauses, rng)

# EVOLVE-BLOCK-END
