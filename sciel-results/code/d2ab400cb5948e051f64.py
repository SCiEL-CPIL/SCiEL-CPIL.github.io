"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock limit, use 325s to be safe
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess: convert clauses to 0-based indices and signs
    # clause_vars[c] = array of variable indices (0-based)
    # clause_sigs[c] = array of signs (+1 or -1)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_sigs = np.empty((m, 3), dtype=np.int8)
    
    for c_idx, clause in enumerate(clauses):
        for l_idx, lit in enumerate(clause):
            clause_vars[c_idx, l_idx] = abs(lit) - 1
            clause_sigs[c_idx, l_idx] = 1 if lit > 0 else -1
    
    # Variable-to-clauses mapping for DLS
    var_clauses = [[] for _ in range(n)]
    for c_idx in range(m):
        for v in clause_vars[c_idx]:
            var_clauses[v].append(c_idx)
    
    def time_left():
        return deadline - time.time()
    
    def check_satisfied(assign):
        """Check if all clauses are satisfied. Returns True if SAT."""
        # For each clause, check if any literal is true
        for c_idx in range(m):
            vars_c = clause_vars[c_idx]
            sigs_c = clause_sigs[c_idx]
            sat = False
            for i in range(3):
                if assign[vars_c[i]] == (sigs_c[i] == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assign):
        """Count number of unsatisfied clauses."""
        cnt = 0
        for c_idx in range(m):
            vars_c = clause_vars[c_idx]
            sigs_c = clause_sigs[c_idx]
            sat = False
            for i in range(3):
                if assign[vars_c[i]] == (sigs_c[i] == 1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    # WalkSAT with DLS (Dynamic Large Selection) and restarts
    best_assign = None
    best_unsat = m + 1
    
    rng = random.Random(42)
    attempt = 0
    
    while time_left() > 5.0:
        attempt += 1
        
        # Random initial assignment
        assign = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Adaptive parameters based on attempt
        # p is probability to flip a literal that satisfies the clause
        # Use a value around 0.5-0.7 for planted 3-SAT
        p = 0.55 + 0.05 * min(attempt, 10)  # gradually increase p
        p = min(p, 0.85)
        
        # Max steps per attempt, scaled with n
        max_steps = 500 * n  # generous budget per attempt
        
        # DLS tracking: number of times each variable was flipped
        flip_count = np.zeros(n, dtype=np.int32)
        
        steps = 0
        found = False
        while steps < max_steps:
            if time_left() < 2.0:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for c_idx in range(m):
                vars_c = clause_vars[c_idx]
                sigs_c = clause_sigs[c_idx]
                sat = False
                for i in range(3):
                    if assign[vars_c[i]] == (sigs_c[i] == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(c_idx)
            
            if not unsat_clauses:
                found = True
                break
            
            # DLS: pick the unsatisfied clause with the variable that has been flipped most
            # Compute for each unsat clause the max flip count among its variables
            best_c = unsat_clauses[0]
            best_max_flip = -1
            # Sample up to 20 clauses for efficiency if many are unsat
            if len(unsat_clauses) > 20:
                sampled = rng.sample(unsat_clauses, 20)
            else:
                sampled = unsat_clauses
            
            for c_idx in sampled:
                max_flip = max(flip_count[clause_vars[c_idx][i]] for i in range(3))
                if max_flip > best_max_flip:
                    best_max_flip = max_flip
                    best_c = c_idx
            
            # Pick a literal from the chosen clause
            vars_c = clause_vars[best_c]
            sigs_c = clause_sigs[best_c]
            
            # With probability p, pick a literal that would satisfy the clause
            if rng.random() < p:
                # Pick among literals that are currently false (flipping them satisfies)
                candidates = []
                for i in range(3):
                    var = vars_c[i]
                    sig = sigs_c[i]
                    # Literal is false if assign[var] != (sig == 1)
                    if assign[var] != (sig == 1):
                        candidates.append(i)
                if candidates:
                    i = rng.choice(candidates)
                else:
                    i = rng.randint(0, 2)
            else:
                # Random literal
                i = rng.randint(0, 2)
            
            var = vars_c[i]
            assign[var] = not assign[var]
            flip_count[var] += 1
            steps += 1
        
        if found:
            return assign.tolist()
        
        # Track best assignment seen
        unsat_cnt = count_unsat(assign)
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assign = assign.copy()
    
    # If we didn't find a solution, return the best assignment we had
    # But we need to verify it's actually satisfying (it shouldn't be if we didn't find one)
    # Return the last assignment or best_assign
    if best_assign is not None:
        # Verify
        if check_satisfied(best_assign):
            return best_assign.tolist()
        return best_assign.tolist()
    
    # Fallback: random assignment
    assign = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    return assign.tolist()
# EVOLVE-BLOCK-END
