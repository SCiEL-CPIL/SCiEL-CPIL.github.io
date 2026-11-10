"""Stochastic local search 3-SAT solver with conflict-driven variable selection and simulated annealing."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Pre-compute variable to clause mapping
    # var_to_clauses[v] = list of (clause_idx, sign) where sign=1 means +v, -1 means -v
    var_to_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[v].append((c_idx, sign))
    
    # Pre-compute clause literal info for fast satisfaction checks
    # clause_vars[c] = list of (var_idx, sign)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((v, sign))
        clause_vars.append(cv)
    
    # Random initial assignment
    assignment = [rng.randint(0, 1) for _ in range(n)]
    
    # Helper: is clause c satisfied?
    def is_sat(c_idx):
        for v, s in clause_vars[c_idx]:
            if assignment[v] == (1 if s == 1 else 0):
                return True
        return False
    
    # Build initial clause satisfaction status
    clause_sat = [False] * m
    unsat_clauses = []
    for c_idx in range(m):
        if is_sat(c_idx):
            clause_sat[c_idx] = True
        else:
            unsat_clauses.append(c_idx)
    
    current_energy = len(unsat_clauses)
    
    if current_energy == 0:
        return [bool(x) for x in assignment]
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    max_iterations = 5000000
    cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
    T = T_start
    
    # WalkSAT-style variable selection parameters
    walk_prob = 0.2  # probability of picking a random variable from the clause
    
    for iteration in range(max_iterations):
        if current_energy == 0:
            break
        
        # Select a random unsatisfied clause
        c_idx = rng.choice(unsat_clauses)
        clause = clause_vars[c_idx]
        
        # For each variable in this clause, compute the cost of flipping it
        # Cost = number of clauses that would become unsatisfied minus those that would become satisfied
        best_var = -1
        best_cost = float('inf')
        
        # With some probability, pick a random variable from the clause (WalkSAT step)
        if rng.random() < walk_prob:
            var = rng.randint(0, len(clause) - 1)
            v_idx = clause[var][0]
            # Compute delta for this variable
            delta = 0
            for cc, sign in var_to_clauses[v_idx]:
                if clause_sat[cc]:
                    # If flipping makes it unsatisfied, that's bad
                    # Check if this clause has another satisfied literal besides v_idx
                    has_other = False
                    for vv, ss in clause_vars[cc]:
                        if vv == v_idx:
                            continue
                        if assignment[vv] == (1 if ss == 1 else 0):
                            has_other = True
                            break
                    if not has_other:
                        delta += 1
                else:
                    # If flipping makes it satisfied, that's good
                    # Check if after flipping, this clause would be satisfied
                    # The clause is currently unsatisfied, so no literal is satisfied
                    # After flipping v_idx, check if v_idx satisfies it
                    s = sign
                    new_val = 1 - assignment[v_idx]
                    if (new_val == 1 and s == 1) or (new_val == 0 and s == -1):
                        delta -= 1
            
            v_idx_to_use = v_idx
            delta_to_use = delta
        else:
            # Min-conflicts: pick variable with minimum cost
            for var_pos in range(len(clause)):
                v_idx = clause[var_pos][0]
                # Compute delta for flipping v_idx
                delta = 0
                for cc, sign in var_to_clauses[v_idx]:
                    if clause_sat[cc]:
                        # Check if v_idx is the only satisfied literal in clause cc
                        has_other = False
                        for vv, ss in clause_vars[cc]:
                            if vv == v_idx:
                                continue
                            if assignment[vv] == (1 if ss == 1 else 0):
                                has_other = True
                                break
                        if not has_other:
                            delta += 1
                    else:
                        # Check if flipping v_idx would satisfy clause cc
                        s = sign
                        new_val = 1 - assignment[v_idx]
                        if (new_val == 1 and s == 1) or (new_val == 0 and s == -1):
                            delta -= 1
                
                if delta < best_cost:
                    best_cost = delta
                    best_var = v_idx
            
            v_idx_to_use = best_var
            delta_to_use = best_cost
        
        # Simulated Annealing acceptance
        if delta_to_use <= 0:
            accept = True
        else:
            if T > 0:
                accept = rng.random() < math.exp(-delta_to_use / T)
            else:
                accept = False
        
        if accept:
            # Apply the flip
            old_val = assignment[v_idx_to_use]
            assignment[v_idx_to_use] = 1 - old_val
            
            # Update clause satisfaction for all clauses containing v_idx_to_use
            for cc, sign in var_to_clauses[v_idx_to_use]:
                old_sat = clause_sat[cc]
                new_sat = False
                for vv, ss in clause_vars[cc]:
                    if assignment[vv] == (1 if ss == 1 else 0):
                        new_sat = True
                        break
                clause_sat[cc] = new_sat
                
                if old_sat and not new_sat:
                    # Clause became unsatisfied
                    unsat_clauses.append(cc)
                    current_energy += 1
                elif not old_sat and new_sat:
                    # Clause became satisfied
                    # Remove cc from unsat_clauses efficiently
                    # Find its position
                    try:
                        pos = unsat_clauses.index(cc)
                        unsat_clauses[pos] = unsat_clauses[-1]
                        unsat_clauses.pop()
                    except ValueError:
                        pass
                    current_energy -= 1
        
        # Cool down
        T *= cooling_rate
    
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
