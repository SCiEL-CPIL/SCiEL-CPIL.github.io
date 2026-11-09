"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 300s limit

    # Pre-process clauses into list of (var, is_positive) for speed
    # var is 0-indexed
    pre_clauses = []
    # Also build var_clauses: var -> list of (clause_index, is_positive)
    var_clauses = [[] for _ in range(n)]
    
    for ci, c in enumerate(clauses):
        pc = []
        for lit in c:
            if lit > 0:
                v = lit - 1
                pc.append((v, True))
                var_clauses[v].append((ci, True))
            else:
                v = -lit - 1
                pc.append((v, False))
                var_clauses[v].append((ci, False))
        pre_clauses.append(pc)

    m = len(pre_clauses)

    # Initial state tracking
    # clause_sat_count[i] = number of satisfied literals in clause i
    # unsat_count = number of clauses with clause_sat_count == 0
    
    best_assignment = None
    best_unsat = m

    rng = random.Random(42)

    # Parameters for WalkSAT
    # Noise parameter: probability of flipping a random variable from the clause
    # Standard WalkSAT uses a noise parameter p, where with prob p we pick a random var from the clause,
    # otherwise we pick the var that minimizes the number of newly unsatisfied clauses.
    noise = 0.3
    max_steps_per_restart = 20000
    max_restarts = 1000 # High number, but we break on time

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause_sat_count
        clause_sat_count = [0] * m
        for c in pre_clauses:
            for v, is_pos in c:
                if assignment[v] == is_pos:
                    clause_sat_count[pre_clauses.index(c)] += 1 
        # The above initialization is O(m^2) if we use index, let's do it properly in the loop above or a separate pass.
        # Let's re-initialize properly to avoid O(m^2) or index errors.
        clause_sat_count = [0] * m
        for ci, c in enumerate(pre_clauses):
            cnt = 0
            for v, is_pos in c:
                if assignment[v] == is_pos:
                    cnt += 1
            clause_sat_count[ci] = cnt
            
        unsat_count = 0
        for cnt in clause_sat_count:
            if cnt == 0:
                unsat_count += 1

        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        # WalkSAT loop
        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break

            if unsat_count == 0:
                return assignment

            # Pick a random unsatisfied clause
            # Instead of scanning all m, we can maintain a list of unsat clauses, 
            # but updating it is tricky. 
            # For m ~ 1000-5000, scanning is okay if we do it smartly.
            # Better: pick random clause until we find an unsat one.
            # Expected trials = m / unsat_count. If unsat_count is small, this is bad.
            # Let's maintain a list of unsat clause indices.
            # To keep it simple and fast, let's just scan a few random ones.
            # Or, since we have clause_sat_count, we can pick a random index and check.
            
            # Optimization: If unsat_count is small, we should find it quickly.
            # If unsat_count is large, we find it quickly.
            # The worst case is when unsat_count is ~ m/2.
            # Let's just pick a random clause index.
            ci = rng.randrange(m)
            if clause_sat_count[ci] != 0:
                # Retry a few times, then fallback to scanning
                found = False
                for _ in range(10):
                    ci = rng.randrange(m)
                    if clause_sat_count[ci] == 0:
                        found = True
                        break
                if not found:
                    # Fallback: scan to find one
                    for i in range(m):
                        if clause_sat_count[i] == 0:
                            ci = i
                            found = True
                            break
                    if not found:
                        # Should not happen if unsat_count > 0
                        break

            clause = pre_clauses[ci]
            
            # Variable selection heuristic
            # 1. Calculate the "delta" for each variable in the clause
            # Delta = (number of unsat clauses this var appears in that would become sat) 
            #         - (number of sat clauses this var appears in that would become unsat)
            # Actually, standard WalkSAT minimizes the number of *newly* unsatisfied clauses.
            # Let's compute cost for each var in the clause.
            
            best_var = -1
            best_cost = float('inf')
            
            for v, is_pos in clause:
                # If we flip v, what happens to clauses containing v?
                cost = 0
                for cj, is_pos_j in var_clauses[v]:
                    if cj == ci:
                        continue # Skip the clause we are trying to satisfy
                    # Check current status of clause cj
                    # We need to know if literal (v, is_pos_j) is currently satisfied
                    current_val = assignment[v]
                    is_satisfied_now = (current_val == is_pos_j)
                    
                    # After flip, assignment[v] becomes not current_val
                    # New status: (not current_val) == is_pos_j
                    # This is equivalent to: is_satisfied_now is False
                    # So if is_satisfied_now is True, it becomes False (bad, +1 cost)
                    # If is_satisfied_now is False, it becomes True (good, -1 cost? No, we count newly unsat)
                    
                    # Let's stick to: Cost = number of clauses that become unsatisfied
                    # A clause becomes unsatisfied if:
                    # 1. It was satisfied (count > 0)
                    # 2. The literal v in it was satisfied
                    # 3. After flip, the literal v is not satisfied
                    # 4. AND no other literal in the clause is satisfied.
                    # This is hard to check without scanning the clause.
                    
                    # Alternative Heuristic: MinCon (Minimize Conflicts)
                    # Count how many unsatisfied clauses contain v.
                    # Flip the variable that appears in the most unsatisfied clauses.
                    pass
                
            # Simpler and effective heuristic: 
            # Pick the variable in the unsat clause that appears in the most other unsat clauses.
            # We can pre-compute "var_unsat_count" dynamically? 
            # Or just estimate: 
            # For each v in clause, count how many clauses in var_clauses[v] have clause_sat_count == 0.
            
            best_v = -1
            best_unsat_occ = -1
            
            for v, is_pos in clause:
                unsat_occ = 0
                for cj, is_pos_j in var_clauses[v]:
                    if clause_sat_count[cj] == 0:
                        unsat_occ += 1
                if unsat_occ > best_unsat_occ:
                    best_unsat_occ = unsat_occ
                    best_v = v
            
            # Apply noise
            if rng.random() < noise:
                v, is_pos = rng.choice(clause)
            else:
                v = best_v
                # We don't need is_pos for the flip, just the var index
                # But we need to update the state.

            # Flip variable v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause_sat_count and unsat_count
            for cj, is_pos_j in var_clauses[v]:
                # Was the literal satisfied before?
                was_sat = (old_val == is_pos_j)
                # Is the literal satisfied now?
                is_sat_now = (assignment[v] == is_pos_j)
                
                if was_sat and not is_sat_now:
                    # Literal went from Sat to Unsat
                    clause_sat_count[cj] -= 1
                    if clause_sat_count[cj] == 0:
                        unsat_count += 1
                elif not was_sat and is_sat_now:
                    # Literal went from Unsat to Sat
                    clause_sat_count[cj] += 1
                    if clause_sat_count[cj] == 1:
                        unsat_count -= 1

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            steps += 1

    return best_assignment
# EVOLVE-BLOCK-END
