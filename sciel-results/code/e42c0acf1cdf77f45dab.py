"""Synthetic planted 3-SAT solver (150 variables) - WalkSAT with restarts and deadline guard."""
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: return best result before 330s limit with safety margin
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin
    
    # Preprocess: convert clauses to 0-based var indices and signs
    # clause_data[i] = list of (var_idx, sign) where sign is +1 or -1
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        cdata = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cdata.append((var_idx, sign))
            var_to_clauses[var_idx].append(ci)
        clause_data.append(cdata)
    
    def check_assignment(assignment):
        """Check if assignment satisfies all clauses."""
        for ci, cdata in enumerate(clause_data):
            sat = False
            for var_idx, sign in cdata:
                if assignment[var_idx] == (sign == 1):
                    sat = True
                    break
            if not sat:
                return False, ci
        return True, -1
    
    def count_unsatisfied(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for ci, cdata in enumerate(clause_data):
            sat = False
            for var_idx, sign in cdata:
                if assignment[var_idx] == (sign == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def walksat_step(assignment, rng, weight=0.55):
        """One step of WalkSAT: pick random unsat clause, flip a literal in it."""
        # Find all unsatisfied clauses
        unsat_clauses = []
        for ci, cdata in enumerate(clause_data):
            sat = False
            for var_idx, sign in cdata:
                if assignment[var_idx] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(ci)
        
        if not unsat_clauses:
            return assignment, True
        
        # Pick a random unsatisfied clause
        ci = rng.choice(unsat_clauses)
        cdata = clause_data[ci]
        
        # For each literal in the clause, compute the impact of flipping
        # We want to flip a literal that makes this clause satisfied
        # and minimizes the damage to other clauses
        best_var = -1
        best_score = float('inf')
        
        for var_idx, sign in cdata:
            # Flipping var_idx will satisfy this clause
            # Count how many other clauses become unsatisfied
            damage = 0
            for oj in var_to_clauses[var_idx]:
                if oj == ci:
                    continue
                # Check if clause oj is currently satisfied only by var_idx
                # i.e., if we flip var_idx, does clause oj become unsatisfied?
                oj_data = clause_data[oj]
                # Current state of var_idx in clause oj
                current_val = assignment[var_idx]
                # Find the sign of var_idx in clause oj
                oj_sign = None
                for v, s in oj_data:
                    if v == var_idx:
                        oj_sign = s
                        break
                if oj_sign is None:
                    continue
                # Currently, this literal in clause oj is satisfied if:
                # assignment[var_idx] == (oj_sign == 1)
                currently_sat = (current_val == (oj_sign == 1))
                if currently_sat:
                    # Check if any other literal in clause oj is satisfied
                    other_sat = False
                    for v2, s2 in oj_data:
                        if v2 == var_idx:
                            continue
                        if assignment[v2] == (s2 == 1):
                            other_sat = True
                            break
                    if not other_sat:
                        damage += 1
            
            score = damage
            if score < best_score:
                best_score = score
                best_var = var_idx
        
        # With probability weight, flip the best variable; otherwise flip random
        if best_var >= 0 and rng.random() < weight:
            assignment[best_var] = not assignment[best_var]
        else:
            var_idx, sign = rng.choice(cdata)
            assignment[var_idx] = not assignment[var_idx]
        
        return assignment, False
    
    # Try multiple random restarts
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        rng = random.Random(restart * 12345 + 67890)
        
        # Quick check
        sat, _ = check_assignment(assignment)
        if sat:
            return assignment
        
        unsat_count = count_unsatisfied(assignment)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT steps
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            assignment, solved = walksat_step(assignment, rng, weight=0.55)
            
            if solved:
                return assignment
            
            # Periodically check unsat count
            if step % 100 == 0:
                unsat_count = count_unsatisfied(assignment)
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
        
        # Final check for this restart
        sat, _ = check_assignment(assignment)
        if sat:
            return assignment
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: random assignment
        return [random.random() < 0.5 for _ in range(n)]
