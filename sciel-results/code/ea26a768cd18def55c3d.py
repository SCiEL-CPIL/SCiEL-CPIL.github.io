# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285.0  # safety margin before 300s kill

    if m == 0:
        return [False] * n

    # Convert to 0-based, store polarity as +1/-1
    # lit = var_idx * polarity where polarity is +1 for positive, -1 for negative
    # Actually, let's store as (var, sign) where sign is +1 or -1
    # For a literal l: var = abs(l)-1, sign = 1 if l>0 else -1
    # A literal is satisfied if assignment[var] == (sign == 1)
    
    # Preprocess
    var_idx = np.zeros((m, 3), dtype=np.int32)
    signs = np.zeros((m, 3), dtype=np.int8)  # +1 or -1
    
    for ci, c in enumerate(clauses):
        for i, l in enumerate(c):
            var_idx[ci, i] = abs(l) - 1
            signs[ci, i] = 1 if l > 0 else -1

    # Build variable to clause incidence
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for i in range(3):
            v = var_idx[ci, i]
            var_to_clauses[v].append(ci)

    # Count occurrences of each literal for weighting
    lit_count = defaultdict(int)
    for ci in range(m):
        for i in range(3):
            lit = (var_idx[ci, i], signs[ci, i])
            lit_count[lit] += 1

    def count_unsat(assignment):
        """Count unsatisfied clauses. assignment is a numpy bool array."""
        # For each clause, check if any literal is satisfied
        # assignment[var] should equal (sign == 1)
        # satisfied = assignment[var_idx] == (signs == 1)
        target = (signs == 1)  # bool array, True if positive literal
        sat = (assignment[var_idx] == target).any(axis=1)
        return int((~sat).sum())

    def find_unsat(assignment):
        target = (signs == 1)
        sat = (assignment[var_idx] == target).any(axis=1)
        return np.where(~sat)[0]

    def walksat_step(assignment, unsat_indices, rng, p=0.55):
        """
        WalkSAT step. p is the probability of greedy move.
        Returns modified assignment.
        """
        if len(unsat_indices) == 0:
            return assignment
        
        # Choose a random unsatisfied clause
        ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
        
        # For each variable in this clause, compute the delta if we flip it
        # delta = (newly satisfied) - (newly unsatisfied)
        # We want to maximize delta, or with probability (1-p), choose randomly
        
        best_var = -1
        best_delta = -999
        deltas = [0] * 3
        
        for i in range(3):
            v = var_idx[ci, i]
            s = signs[ci, i]
            
            # If we flip v, does it satisfy this clause?
            # Currently unsatisfied, so flipping v will satisfy this clause if the literal becomes true
            # After flip, assignment[v] becomes not assignment[v]
            # Literal is satisfied if new_assignment[v] == (s == 1)
            new_val = not assignment[v]
            will_satisfy = (new_val == (s == 1))
            
            delta = 1 if will_satisfy else 0  # This clause will be satisfied
            
            # Check impact on other clauses containing v
            for oc in var_to_clauses[v]:
                if oc == ci:
                    continue
                
                # Check if oc is currently satisfied
                oc_sat = False
                oc_only_this = True
                for j in range(3):
                    ov = var_idx[oc, j]
                    os = signs[oc, j]
                    if assignment[ov] == (os == 1):
                        oc_sat = True
                        if ov != v:
                            oc_only_this = False
                
                if oc_sat:
                    # If we flip v, and v was the only satisfied literal, oc becomes unsatisfied
                    if oc_only_this:
                        delta -= 1
                else:
                    # oc is already unsatisfied. If flipping v satisfies it, delta += 1
                    # But we need to check if after flip, this literal would be satisfied
                    # and if other literals are still unsatisfied
                    # Actually, for delta calculation, we only care about net change
                    # If oc is unsatisfied, and flipping v makes its literal true,
                    # but other literals are false, oc remains unsatisfied (no change)
                    # If flipping v makes its literal true AND other literals were already true... 
                    # but oc is unsatisfied, so at least one other literal is false.
                    # So flipping v alone can't satisfy oc unless all other literals are true.
                    # Let's check:
                    others_all_true = True
                    for j in range(3):
                        ov = var_idx[oc, j]
                        os = signs[oc, j]
                        if ov == v:
                            continue
                        if not (assignment[ov] == (os == 1)):
                            others_all_true = False
                            break
                    if others_all_true:
                        # Flipping v will satisfy oc
                        delta += 1
            
            deltas[i] = delta
            if delta > best_delta:
                best_delta = delta
                best_var = v
        
        # Choose variable: greedy with probability p, random with probability (1-p)
        if rng.random() < p:
            # Choose best (or random among tied best)
            max_d = max(deltas)
            candidates = [var_idx[ci, i] for i in range(3) if deltas[i] == max_d]
            v_to_flip = candidates[rng.randint(0, len(candidates) - 1)]
        else:
            v_to_flip = var_idx[ci, rng.randint(0, 2)]
        
        assignment[v_to_flip] = not assignment[v_to_flip]
        return assignment

    # Main search
    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')
    
    # Try a few different p values
    p_values = [0.55, 0.5, 0.6, 0.45, 0.65]
    
    max_restarts = 200
    max_steps = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        p = p_values[restart % len(p_values)]
        
        # Random initialization
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment.tolist()
        
        local_best_unsat = unsat_count
        local_best_assign = assignment.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            unsat_count = count_unsat(assignment)
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                local_best_assign = assignment.copy()
                if local_best_unsat < best_unsat:
                    best_unsat = local_best_unsat
                    best_assignment = local_best_assign.copy()
            
            unsat_indices = find_unsat(assignment)
            assignment = walksat_step(assignment, unsat_indices, rng, p)
        
        if local_best_unsat < best_unsat:
            best_unsat = local_best_unsat
            best_assignment = local_best_assign.copy()
        
        if best_unsat == 0:
            return best_assignment.tolist()
    
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
