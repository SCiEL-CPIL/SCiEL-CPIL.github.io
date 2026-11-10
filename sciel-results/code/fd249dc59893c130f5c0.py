"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    DEADLINE = 295.0

    # Preprocess: convert to 0-based variable indices and signs
    # literals are (var, sign) where sign is 1 for positive, -1 for negative
    # We represent literals as integers: +v for positive, -v for negative (1-based)
    # Internally we use 0-based vars and store sign separately or use signed ints.
    # Let's use signed ints: + (v+1) for positive, - (v+1) for negative.
    
    # Parse clauses into list of lists of signed ints
    parsed_clauses = []
    for c in clauses:
        parsed_clauses.append(list(c))
        
    # Helper to check if a literal is satisfied by an assignment
    # assignment: list of bools (True/False) for 0-based vars
    def is_sat(lit, assignment):
        v = abs(lit) - 1
        if lit > 0:
            return assignment[v]
        else:
            return not assignment[v]

    def count_unsat(assignment):
        cnt = 0
        for c in parsed_clauses:
            sat = False
            for lit in c:
                if is_sat(lit, assignment):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    # 1. Try DPLL with Conflict-Driven Clause Learning (CDCL)
    # This is often very fast for n=50.
    
    # State for DPLL
    # assignment: dict or list. Let's use a list of None/True/False.
    # stack: list of (var, value) for backtracking
    # learned_clauses: list of clauses learned from conflicts
    
    def dpll_cdcl():
        # Simple CDCL implementation
        # We maintain a trail of decisions.
        # We use a queue of clauses to process for unit propagation.
        
        # To keep it simple and robust within time limits, we'll implement a 
        # standard DPLL with some restarts and random variable selection,
        # but with a limit on nodes. If it fails, we return None.
        
        # Actually, for n=50, a simple DPLL with good heuristics (VSIDS) 
        # is very strong. But implementing full VSIDS + Learning is complex.
        # Let's use a simpler approach: 
        # 1. Unit Propagation
        # 2. Pure Literal Elimination
        # 3. Branch on variable with highest degree (VSIDS-like)
        # 4. Backtrack on conflict.
        
        # Given the small size, even a basic DPLL might work, but WalkSAT is often faster 
        # on average for random 3-SAT near the phase transition. 
        # However, "planted" instances might be easier.
        
        # Let's stick to a robust WalkSAT with multiple restarts and 
        # a specialized "polish" phase, as the current program does, 
        # but optimize the WalkSAT loop significantly.
        
        # The current program's WalkSAT is decent. 
        # Let's improve it by:
        # 1. Using a better clause selection heuristic (Phase Saving + Random).
        # 2. Using a better variable selection heuristic (Min-Conflicts + Random).
        # 3. Adding a simple local search improvement (Polish) that is more aggressive.
        
        pass

    # Let's refine the WalkSAT approach from the seed, as it's already at 1.0.
    # The goal is to ensure it's robust and fast.
    
    # Precompute for WalkSAT
    clause_vars = []
    clause_sigs = []
    for c in parsed_clauses:
        cv = [abs(l) - 1 for l in c]
        cs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(cv)
        clause_sigs.append(cs)

    # Build var -> list of (clause_idx, pos_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append((i, j))

    # Precompute variable degrees
    var_degree = [0] * n
    for i in range(m):
        for j in range(3):
            var_degree[clause_vars[i][j]] += 1

    # Precompute clause weights
    clause_weights = [0] * m
    for i in range(m):
        w = 0
        for j in range(3):
            w += var_degree[clause_vars[i][j]]
        clause_weights[i] = w

    # Numpy arrays for fast counting
    cv_np = np.array(clause_vars, dtype=np.int32)
    cs_np = np.array(clause_sigs, dtype=np.int8)

    rng = random.Random(42)

    def walk_sat_run(assignment, max_flips, p=0.55):
        """Optimized WalkSAT."""
        # Initialize clause satisfaction status
        clause_sat_count = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            cnt = 0
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    cnt += 1
            clause_sat_count[i] = cnt

        # Unsat list management
        unsat_list = []
        unsat_pos = [-1] * m
        for i in range(m):
            if clause_sat_count[i] == 0:
                unsat_pos[i] = len(unsat_list)
                unsat_list.append(i)

        # Track best unsat count for early termination or tracking
        min_unsat = len(unsat_list)
        
        for flip in range(max_flips):
            if time.time() - start_time > DEADLINE:
                return None, assignment

            if len(unsat_list) == 0:
                return assignment[:], assignment

            # Select clause
            # Heuristic: Pick clause with min number of literals that are currently false? 
            # No, unsat clauses have 0 true literals.
            # Pick clause with highest weight or random.
            if rng.random() < 0.7:
                # Pick random unsat clause
                idx = rng.randint(0, len(unsat_list) - 1)
                ci = unsat_list[idx]
            else:
                # Pick unsat clause with max weight
                best_ci = -1
                best_w = -1
                # Sample a few to speed up? For m~200-500, full scan is ok.
                for idx in unsat_list:
                    if clause_weights[idx] > best_w:
                        best_w = clause_weights[idx]
                        best_ci = idx
                ci = best_ci

            # Select variable to flip
            # Heuristic: Flip variable that minimizes the number of newly unsatisfied clauses
            # (Min-Conflicts), with probability (1-p).
            # With probability p, flip random variable in clause.
            
            if rng.random() < p:
                li = rng.randint(0, 2)
            else:
                best_li = 0
                best_cost = float('inf')
                for li in range(3):
                    var = clause_vars[ci][li]
                    # Calculate cost: number of clauses containing var that are currently SAT 
                    # and would become UNSAT if we flip var.
                    cost = 0
                    for (oi, oj) in var_to_clauses[var]:
                        # If clause oi is currently satisfied (clause_sat_count[oi] > 0)
                        # and the literal (oi, oj) is currently true, 
                        # then flipping var will make this literal false.
                        # If clause_sat_count[oi] was 1, it becomes 0 (unsat).
                        if clause_sat_count[oi] > 0:
                            # Check if this literal is the only true one?
                            # We need to know if the literal is true.
                            osig = clause_sigs[oi][oj]
                            if assignment[var] == (osig > 0):
                                # This literal is currently true.
                                # If clause_sat_count[oi] == 1, flipping makes it unsat.
                                if clause_sat_count[oi] == 1:
                                    cost += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_li = li
                li = best_li

            var = clause_vars[ci][li]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val

            # Update clause_sat_count and unsat_list
            for (oi, oj) in var_to_clauses[var]:
                osig = clause_sigs[oi][oj]
                was_sat = (old_val == (osig > 0))
                now_sat = (new_val == (osig > 0))
                
                if was_sat and not now_sat:
                    # Clause lost a true literal
                    clause_sat_count[oi] -= 1
                    if clause_sat_count[oi] == 0:
                        # Became unsat
                        unsat_pos[oi] = len(unsat_list)
                        unsat_list.append(oi)
                elif not was_sat and now_sat:
                    # Clause gained a true literal
                    clause_sat_count[oi] += 1
                    if clause_sat_count[oi] == 1:
                        # Was unsat, now sat. Remove from unsat_list.
                        if unsat_pos[oi] != -1:
                            pos = unsat_pos[oi]
                            last_idx = len(unsat_list) - 1
                            last_clause = unsat_list[last_idx]
                            unsat_list[pos] = last_clause
                            unsat_pos[last_clause] = pos
                            unsat_list.pop()
                            unsat_pos[oi] = -1

            if len(unsat_list) < min_unsat:
                min_unsat = len(unsat_list)
                # Optional: save best assignment if we want, but we return on 0.
        
        if len(unsat_list) == 0:
            return assignment[:], assignment
        return None, assignment

    def count_unsat_fast(assignment):
        a_np = np.array(assignment, dtype=bool)
        lit_sat = a_np[cv_np] == (cs_np > 0)
        clause_sat = lit_sat.any(axis=1)
        return int((~clause_sat).sum())

    def polish(assignment):
        """Aggressive local search to fix remaining unsat clauses."""
        current_unsat = count_unsat_fast(assignment)
        if current_unsat == 0:
            return assignment
        
        # Identify unsat clauses
        unsat_clauses = []
        for i in range(m):
            if not any(assignment[clause_vars[i][j]] == (clause_sigs[i][j] > 0) for j in range(3)):
                unsat_clauses.append(i)
        
        # Try flipping each variable involved in unsat clauses
        # and keep the flip if it reduces unsat count.
        # Repeat until no improvement.
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 5):
            improved = False
            current_unsat = count_unsat_fast(assignment)
            if current_unsat == 0:
                return assignment
            
            # Get variables from unsat clauses
            vars_to_try = set()
            for ci in unsat_clauses:
                for j in range(3):
                    vars_to_try.add(clause_vars[ci][j])
            
            for var in vars_to_try:
                old_val = assignment[var]
                assignment[var] = not old_val
                new_unsat = count_unsat_fast(assignment)
                if new_unsat < current_unsat:
                    current_unsat = new_unsat
                    improved = True
                    # Update unsat_clauses list? 
                    # For simplicity, recompute at end of loop or just continue.
                    # Recomputing unsat_clauses is O(m), which is fine.
                else:
                    assignment[var] = old_val
            
            # If we made progress, recompute unsat_clauses for next iteration
            if improved:
                unsat_clauses = []
                for i in range(m):
                    if not any(assignment[clause_vars[i][j]] == (clause_sigs[i][j] > 0) for j in range(3)):
                        unsat_clauses.append(i)

        return assignment

    best_assignment = None
    best_unsat = m + 1

    # Luby restarts
    multipliers = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    base_flips = 200
    max_restarts = 2000

    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        mult = multipliers[restart % len(multipliers)]
        flips = base_flips * mult
        if flips > 5000:
            flips = 5000

        # Randomize assignment on restart
        if restart > 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        result, final_assignment = walk_sat_run(assignment, max_flips=flips, p=0.55)
        
        if result is not None:
            return result

        unsat_count = count_unsat_fast(final_assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = final_assignment[:]
            
            # If close to solution, try polishing
            if unsat_count <= 5:
                polished = polish(best_assignment)
                polished_unsat = count_unsat_fast(polished)
                if polished_unsat == 0:
                    return polished
                if polished_unsat < best_unsat:
                    best_unsat = polished_unsat
                    best_assignment = polished[:]

    if best_assignment is not None:
        return best_assignment

    return [True] * n

# EVOLVE-BLOCK-END
