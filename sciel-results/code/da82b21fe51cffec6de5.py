"""Synthetic planted 3-SAT solver (150 variables) with LP relaxation init + WalkSAT."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np


def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance.
    
    Strategy:
    1. Use a clause-counting LP relaxation to get a continuous probability vector.
    2. Round to binary to initialize local search.
    3. Run WalkSAT with multiple restarts and a time deadline guard.
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: leave 10s safety margin before 330s kill
    start_time = time.time()
    deadline = start_time + 320.0  # 320s out of 330s
    
    # --- Step 1: LP relaxation via clause counting ---
    # For each variable, compute a "pressure" from clauses.
    # The idea: if variable x appears positively in many unsatisfied clauses (under a uniform random assignment),
    # it should be biased toward True, and vice versa.
    # We do a few iterations of a simple averaging to get a continuous vector.
    
    # Initialize with uniform 0.5
    x = np.full(n, 0.5)
    
    # Build literal index arrays for speed
    # clauses is a list of lists of signed 1-based integers
    # Convert to 0-based: literal l -> index abs(l)-1, sign = 1 if l>0 else -1
    
    # Precompute for each clause the variable indices and signs
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_ = [abs(l) - 1 for l in c]
        signs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Simple LP-like relaxation: iterate a few times
    # For each clause, the "soft satisfaction" is the sum of x_i for positive literals and (1-x_i) for negative.
    # We want all clauses to have soft satisfaction >= 1.
    # Use a gradient-like update: for each clause that's "under-satisfied", push its literals toward satisfying it.
    
    for it in range(50):
        if time.time() > deadline - 300:
            break
        # Compute soft satisfaction for each clause
        # soft_sat[c] = sum over literals of (x_i if positive, 1-x_i if negative)
        # We want soft_sat >= 1 for all clauses.
        
        # For efficiency, compute contributions
        # delta_x[i] accumulates the gradient
        delta_x = np.zeros(n)
        
        for c_idx, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
            # Compute soft satisfaction
            soft_sat = 0.0
            for v, s in zip(vars_, signs):
                if s == 1:
                    soft_sat += x[v]
                else:
                    soft_sat += 1.0 - x[v]
            
            # If under-satisfied, push literals toward satisfying
            if soft_sat < 1.0:
                deficit = 1.0 - soft_sat
                for v, s in zip(vars_, signs):
                    if s == 1:
                        delta_x[v] += deficit / 3.0
                    else:
                        delta_x[v] -= deficit / 3.0
        
        # Update x with a learning rate
        lr = 0.1
        x += lr * delta_x
        # Clip to [0, 1]
        x = np.clip(x, 0.0, 1.0)
    
    # Round to binary assignment
    assignment = [bool(v > 0.5) for v in x]
    
    # --- Step 2: Check if already satisfied ---
    def is_satisfied(assignment, clauses):
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    if is_satisfied(assignment, clauses):
        return assignment
    
    # --- Step 3: WalkSAT with restarts ---
    # Precompute clause data for fast evaluation
    m = len(clauses)
    
    # For each clause, store the list of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        data = []
        for l in c:
            data.append((abs(l) - 1, l > 0))
        clause_data.append(data)
    
    def count_unsat(assignment, clause_data):
        """Count number of unsatisfied clauses."""
        count = 0
        for c in clause_data:
            sat = False
            for v, pos in c:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def walk_sat(assignment, clause_data, rng, max_steps):
        """Run WalkSAT with a given number of steps. Returns (assignment, satisfied)."""
        n_local = len(assignment)
        m_local = len(clause_data)
        
        # Precompute unsatisfied clause indices
        unsat_clauses = []
        for ci in range(m_local):
            c = clause_data[ci]
            sat = False
            for v, pos in c:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(ci)
        
        if not unsat_clauses:
            return assignment, True
        
        # Track clause satisfaction status for fast updates
        # sat_count[ci] = number of satisfied literals in clause ci
        sat_count = np.zeros(m_local, dtype=np.int32)
        for ci, c in enumerate(clause_data):
            for v, pos in c:
                if assignment[v] == pos:
                    sat_count[ci] += 1
        
        # unsat_clauses is a set for O(1) membership
        unsat_set = set(unsat_clauses)
        
        for step in range(max_steps):
            if not unsat_set:
                return assignment, True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            c = clause_data[ci]
            
            # Count how many literals are false in this clause (for weighting)
            false_lits = []
            for v, pos in c:
                if assignment[v] != pos:
                    false_lits.append(v)
            
            if not false_lits:
                # Shouldn't happen if clause is unsatisfied
                v = rng.choice(c)[0]
            else:
                # With probability p, flip a random false literal; else flip any literal
                if rng.random() < 0.5:
                    v = rng.choice(false_lits)
                else:
                    v = rng.choice(c)[0]
            
            # Flip variable v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update sat_count for all clauses containing v
            # We need to know which clauses contain v. Let's build a var_to_clauses map.
            # For efficiency, we'll do this outside the loop.
            
        return assignment, False
    
    # Build variable-to-clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clause_data):
        for v, pos in c:
            var_to_clauses[v].append(ci)
    
    def walk_sat_fast(assignment, clause_data, var_to_clauses, rng, max_steps, deadline):
        """Run WalkSAT with fast clause updates."""
        n_local = len(assignment)
        m_local = len(clause_data)
        
        # sat_count[ci] = number of satisfied literals in clause ci
        sat_count = np.zeros(m_local, dtype=np.int32)
        for ci, c in enumerate(clause_data):
            for v, pos in c:
                if assignment[v] == pos:
                    sat_count[ci] += 1
        
        unsat_count = 0
        for ci in range(m_local):
            if sat_count[ci] == 0:
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment, True
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            if unsat_count == 0:
                return assignment, True
            
            # Pick a random unsatisfied clause
            # We need to efficiently pick from unsatisfied clauses.
            # Use a list of unsatisfied clause indices, updated dynamically.
            # For simplicity, scan or maintain a list.
            
            # Maintain a list of unsat clauses
            # To avoid rebuilding, we'll use a simple approach: pick random clauses until we find an unsat one
            # But this is slow. Instead, maintain a list.
            
            # Let's use a different approach: keep a list of unsat clause indices
            # This is getting complex. Let's use a simpler but still effective approach.
            
            # Pick a random clause index
            ci = rng.randint(0, m_local - 1)
            if sat_count[ci] > 0:
                # Retry a few times
                for _ in range(5):
                    ci = rng.randint(0, m_local - 1)
                    if sat_count[ci] == 0:
                        break
                if sat_count[ci] > 0:
                    # Still satisfied, skip this step
                    continue
            
            c = clause_data[ci]
            
            # Count false literals
            false_lits = []
            for v, pos in c:
                if assignment[v] != pos:
                    false_lits.append(v)
            
            if not false_lits:
                continue
            
            # Weighted selection: prefer literals that appear in many unsatisfied clauses
            # For simplicity, use random selection with bias
            if rng.random() < 0.5:
                v = rng.choice(false_lits)
            else:
                v = rng.choice(c)[0]
            
            # Flip variable v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update sat_count for all clauses containing v
            for ci2 in var_to_clauses[v]:
                pos2 = clause_data[ci2][0][1] if len(clause_data[ci2]) == 1 else None
                # Need to find the sign for v in clause ci2
                # This is inefficient. Let's precompute.
            
            # This is getting too complex. Let's use a cleaner approach.
        
        return assignment, False
    
    # Let's use a cleaner WalkSAT implementation
    def walk_sat_clean(assignment, clause_data, var_to_clauses, rng, max_steps, deadline):
        """Clean WalkSAT with proper clause tracking."""
        n_local = len(assignment)
        m_local = len(clause_data)
        
        # Precompute: for each clause, for each variable in it, store the sign
        # clause_signs[ci][v] = 1 if v is positive in clause ci, -1 if negative, 0 if not in clause
        # But this is a big matrix. Instead, store for each clause a dict or use the list.
        
        # sat_count[ci] = number of satisfied literals
        sat_count = [0] * m_local
        for ci, c in enumerate(clause_data):
            for v, pos in c:
                if assignment[v] == pos:
                    sat_count[ci] += 1
        
        unsat_count = sum(1 for sc in sat_count if sc == 0)
        
        if unsat_count == 0:
            return assignment, True
        
        # For each variable, store list of (clause_idx, is_positive)
        # var_to_clauses already has clause indices, but we need the sign too.
        # Let's build var_clause_info[v] = list of (ci, pos)
        var_clause_info = [[] for _ in range(n_local)]
        for ci, c in enumerate(clause_data):
            for v, pos in c:
                var_clause_info[v].append((ci, pos))
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            if unsat_count == 0:
                return assignment, True
            
            # Pick a random unsatisfied clause
            # Use rejection sampling
            found = False
            for _ in range(20):
                ci = rng.randint(0, m_local - 1)
                if sat_count[ci] == 0:
                    found = True
                    break
            if not found:
                continue
            
            c = clause_data[ci]
            
            # Find false literals
            false_lits = []
            for v, pos in c:
                if assignment[v] != pos:
                    false_lits.append(v)
            
            if not false_lits:
                continue
            
            # Select variable to flip
            if rng.random() < 0.5:
                v = rng.choice(false_lits)
            else:
                v = rng.choice(c)[0]
            
            # Flip variable v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update sat_count for all clauses containing v
            for ci2, pos2 in var_clause_info[v]:
                if pos2:
                    # v is positive in clause ci2
                    if assignment[v]:
                        # Now satisfied
                        sat_count[ci2] += 1
                    else:
                        # Now not satisfied
                        sat_count[ci2] -= 1
                else:
                    # v is negative in clause ci2
                    if not assignment[v]:
                        # Now satisfied
                        sat_count[ci2] += 1
                    else:
                        # Now not satisfied
                        sat_count[ci2] -= 1
                
                # Update unsat_count
                if sat_count[ci2] == 0:
                    unsat_count += 1
                elif sat_count[ci2] == 1 and (pos2 == assignment[v] if pos2 else not assignment[v]):
                    # This logic is getting messy. Let's just recompute unsat_count periodically.
                    pass
            
            # Recompute unsat_count every 100 steps to avoid drift
            if step % 100 == 0:
                unsat_count = sum(1 for sc in sat_count if sc == 0)
        
        return assignment, False
    
    # Run WalkSAT with multiple restarts
    rng = random.Random(42)
    best_assignment = assignment[:]
    best_unsat = count_unsat(best_assignment, clause_data)
    
    max_steps_per_run = 50000
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # For restarts after the first, randomize the assignment
        if restart == 0:
            current = best_assignment[:]
        else:
            current = [rng.random() < 0.5 for _ in range(n)]
        
        current, sat = walk_sat_clean(current, clause_data, var_to_clauses, rng, max_steps_per_run, deadline)
        
        if sat:
            return current
        
        # Check if this is better than best
        cur_unsat = count_unsat(current, clause_data)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = current[:]
        
        if time.time() > deadline - 5:
            break
    
    return best_assignment
# EVOLVE-BLOCK-END
