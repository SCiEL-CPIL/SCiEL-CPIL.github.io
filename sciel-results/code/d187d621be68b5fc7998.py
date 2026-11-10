"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]
    
    # Convert to 0-based: store literal as signed int (positive = var+1, negative = -(var+1))
    # But we need var_idx and polarity for fast lookup
    clause_literals = []  # list of lists of (var_idx, is_pos)
    for c in clauses_raw:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_literals.append(cv)
        
    m = len(clause_literals)
    
    # Build variable to clause index with polarity
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_literals):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((i, is_pos))
    
    rng = random.Random(42)
    deadline = time.monotonic() + 290.0
    
    # Precompute clause satisfaction counts for each variable assignment
    # For each clause, track how many literals are satisfied
    # clause_sat_count[i] = number of literals in clause i that are currently true
    
    def init_clause_counts(assignment):
        counts = [0] * m
        for i, cv in enumerate(clause_literals):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts
    
    def get_unsat_from_counts(counts):
        return [i for i in range(m) if counts[i] == 0]
    
    # Weighted WalkSAT with SA escape
    def weighted_walksat(assignment, max_steps, sa_temp=1.0):
        counts = init_clause_counts(assignment)
        unsat_set = set(get_unsat_from_counts(counts))
        
        # Clause weights: increase weight of frequently violated clauses
        weights = [1.0] * m
        weight_alpha = 0.05
        weight_max = 10.0
        
        for step in range(max_steps):
            if time.monotonic() > deadline - 2.0:
                return False, assignment, unsat_set, counts, weights
                
            if not unsat_set:
                return True, assignment, unsat_set, counts, weights
            
            # Select unsatisfied clause with probability proportional to weight
            # Use weighted random selection
            unsat_list = list(unsat_set)
            # For speed, just pick random from unsat (approximation)
            c_idx = rng.choice(unsat_list)
            
            # Decay weights slowly
            # Instead of updating all, just decay the selected clause's weight influence
            # Actually, let's use a simpler approach: track hit counts
            
            cv = clause_literals[c_idx]
            
            # Compute break count for each variable in the clause
            # break_count(v) = number of currently satisfied clauses containing v that would become unsat
            best_var = None
            best_break = float('inf')
            best_score = float('inf')
            
            for var_idx, is_pos in cv:
                # Count how many satisfied clauses containing var_idx would break
                break_count = 0
                for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                    if counts[c_aff] > 0:  # currently satisfied
                        # Would it still be satisfied after flip?
                        # Check if any other literal in c_aff is satisfied
                        cv_aff = clause_literals[c_aff]
                        still_sat = False
                        for v2, p2 in cv_aff:
                            if v2 == var_idx:
                                continue
                            if assignment[v2] == p2:
                                still_sat = True
                                break
                        if not still_sat:
                            break_count += 1
                
                # Score: primarily break count, secondarily random tiebreak
                score = break_count + rng.random() * 0.5
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
                    best_break = break_count
            
            if best_var is None:
                best_var = cv[0][0]
            
            var_idx = best_var
            
            # Simulated annealing: sometimes accept worse moves
            # Estimate delta in unsat count
            old_unsat = len(unsat_set)
            
            # Flip variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update counts incrementally
            for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                if assignment[var_idx] == is_pos_aff:
                    # This literal is now satisfied
                    if counts[c_aff] == 0:
                        unsat_set.discard(c_aff)
                    counts[c_aff] += 1
                else:
                    # This literal is now unsatisfied
                    counts[c_aff] -= 1
                    if counts[c_aff] == 0:
                        unsat_set.add(c_aff)
            
            new_unsat = len(unsat_set)
            
            # SA: if we made things worse, accept with probability exp(-delta/T)
            if new_unsat > old_unsat:
                delta = new_unsat - old_unsat
                # Temperature decreases over time
                T = sa_temp * max(0.01, 1.0 - step / max_steps)
                if rng.random() > math.exp(-delta / max(T, 0.001)):
                    # Reject: flip back
                    assignment[var_idx] = not assignment[var_idx]
                    for c_aff, is_pos_aff in var_to_clauses[var_idx]:
                        if assignment[var_idx] == is_pos_aff:
                            if counts[c_aff] == 0:
                                unsat_set.discard(c_aff)
                            counts[c_aff] += 1
                        else:
                            counts[c_aff] -= 1
                            if counts[c_aff] == 0:
                                unsat_set.add(c_aff)
                    # Revert unsat_set to previous state (approximate)
                    # This is tricky; for simplicity, just continue
                    pass
        
        return False, assignment, unsat_set, counts, weights
    
    best_assignment = [False] * n
    best_unsat_count = m + 1
    
    # Multi-restart with varying SA temperatures
    restart_budget = 500
    base_steps = 3000
    
    for restart in range(restart_budget):
        if time.monotonic() > deadline - 5.0:
            break
        
        # Initialize: random with slight bias
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Vary temperature for diversity
        sa_temp = 0.5 + rng.random() * 1.5
        
        remaining = deadline - time.monotonic()
        if remaining < 10.0:
            break
        elif remaining < 30.0:
            steps = 1500
        elif remaining < 60.0:
            steps = 2500
        else:
            steps = base_steps
        
        solved, assignment, unsat_set, counts, weights = weighted_walksat(assignment, steps, sa_temp)
        
        if solved:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
