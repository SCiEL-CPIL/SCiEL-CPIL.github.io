"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import defaultdict


def _compute_clause_sat(assignment, clause_lits):
    """Compute clause satisfaction counts."""
    m = len(clause_lits)
    clause_sat = [0] * m
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            if assignment[var] == (sign == 1):
                clause_sat[ci] += 1
    return clause_sat


def _belief_propagation_init(clause_lits, n, m, rng, num_iters=50):
    """
    Use belief propagation (Gibbs sampling on factor graph) to get a high-quality initial assignment.
    This exploits the planted structure to find likely true values.
    """
    # Initialize beliefs as 0.5 (uniform)
    beliefs = [0.5] * n
    
    # Precompute for each variable, the list of (clause_idx, sign) it appears in
    var_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            var_clauses[var].append((ci, sign))
    
    # Precompute for each clause, the list of (var_idx, sign)
    clause_vars = clause_lits  # already in this format
    
    for it in range(num_iters):
        # Update variable beliefs based on clause messages
        new_beliefs = [0.5] * n
        for v in range(n):
            if not var_clauses[v]:
                continue
            # Compute the probability that the variable should be True
            # based on satisfying its clauses
            log_true = 0.0
            log_false = 0.0
            for ci, sign in var_clauses[v]:
                # If sign == 1, literal is x_v; if sign == -1, literal is not x_v
                # We want to estimate P(x_v = True | other variables in clause)
                # Simplified: use the current beliefs of other variables in the clause
                other_sat_true = 0.0  # prob clause sat if v=True
                other_sat_false = 0.0  # prob clause sat if v=False
                for v2, sign2 in clause_vars[ci]:
                    if v2 == v:
                        continue
                    b2 = beliefs[v2]
                    if sign2 == 1:
                        # literal is x_v2
                        other_sat_true += b2
                        other_sat_false += b2
                    else:
                        # literal is not x_v2
                        other_sat_true += (1 - b2)
                        other_sat_false += (1 - b2)
                # If v is True: clause is sat if (sign==1) OR any other literal sat
                # If v is False: clause is sat if (sign==-1) OR any other literal sat
                # Approximate: probability clause is sat given v's value
                if sign == 1:
                    p_sat_true = 1.0  # v=True satisfies this literal directly
                    p_sat_false = 1.0 - math.prod(1.0 - (beliefs[v2] if sign2 == 1 else (1 - beliefs[v2])) for v2, sign2 in clause_vars[ci] if v2 != v)
                else:
                    p_sat_true = 1.0 - math.prod(1.0 - (beliefs[v2] if sign2 == 1 else (1 - beliefs[v2])) for v2, sign2 in clause_vars[ci] if v2 != v)
                    p_sat_false = 1.0  # v=False satisfies this literal directly
                
                log_true += math.log(max(p_sat_true, 1e-10))
                log_false += math.log(max(p_sat_false, 1e-10))
            
            # Convert to probability
            exp_true = math.exp(log_true)
            exp_false = math.exp(log_false)
            total = exp_true + exp_false
            new_beliefs[v] = exp_true / total
        
        beliefs = new_beliefs
    
    # Generate assignment from beliefs
    assignment = [rng.random() < beliefs[v] for v in range(n)]
    return assignment, beliefs


def _local_search(assignment, clause_lits, pos_clauses, neg_clauses, n, m, rng, deadline, max_steps=50000):
    """
    WalkSAT-style local search with incremental clause tracking.
    Returns (assignment, unsat_count).
    """
    clause_sat = _compute_clause_sat(assignment, clause_lits)
    unsat_count = sum(1 for s in clause_sat if s == 0)
    
    if unsat_count == 0:
        return assignment, 0
    
    best_unsat = unsat_count
    best_assignment = assignment[:]
    
    # Precompute clause lengths for weighted variable selection
    clause_len = [len(lits) for lits in clause_lits]
    
    steps = 0
    while steps < max_steps:
        if time.time() >= deadline - 2:
            break
        
        if unsat_count == 0:
            return assignment, 0
        
        # Select a random unsat clause
        unsat_ci = -1
        for _ in range(10):  # try a few times to find unsat clause
            ci = rng.randint(0, m - 1)
            if clause_sat[ci] == 0:
                unsat_ci = ci
                break
        if unsat_ci == -1:
            # Fallback: scan
            for ci in range(m):
                if clause_sat[ci] == 0:
                    unsat_ci = ci
                    break
        if unsat_ci == -1:
            break
        
        # Pick a variable in this clause to flip with weighting
        lits = clause_lits[unsat_ci]
        # Weight by inverse clause length (prefer variables in shorter clauses)
        weights = [1.0 / clause_len[unsat_ci]] * len(lits)
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0.0
        v = lits[0][0]
        for i, lit in enumerate(lits):
            cumulative += weights[i]
            if r <= cumulative:
                v = lit[0]
                break
        
        # Compute delta efficiently
        delta = 0
        if assignment[v]:
            for ci in pos_clauses[v]:
                if clause_sat[ci] == 1:
                    delta += 1
        else:
            for ci in pos_clauses[v]:
                if clause_sat[ci] == 0:
                    delta -= 1
        if not assignment[v]:
            for ci in neg_clauses[v]:
                if clause_sat[ci] == 1:
                    delta += 1
        else:
            for ci in neg_clauses[v]:
                if clause_sat[ci] == 0:
                    delta -= 1
        
        # Accept flip with WalkSAT probability
        if delta <= 0 or rng.random() < 0.3:
            assignment[v] = not assignment[v]
            if assignment[v]:
                for ci in pos_clauses[v]:
                    clause_sat[ci] += 1
                for ci in neg_clauses[v]:
                    clause_sat[ci] -= 1
            else:
                for ci in neg_clauses[v]:
                    clause_sat[ci] += 1
                for ci in pos_clauses[v]:
                    clause_sat[ci] -= 1
            unsat_count += delta
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
        
        steps += 1
    
    return best_assignment, best_unsat


def solve(instance):
    """
    Hybrid solver: Belief Propagation initialization + WalkSAT local search with restarts.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 330s kill

    # Precompute literal indices (0-based) and sign for each clause
    clause_lits = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_lits.append(lits)

    # For each variable, track which clauses contain it (pos and neg)
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            if sign == 1:
                pos_clauses[var].append(ci)
            else:
                neg_clauses[var].append(ci)

    rng = random.Random(42)
    best_unsat = float('inf')
    best_assignment = None

    # Phase 1: Belief Propagation to get a good initial assignment
    bp_assignment, beliefs = _belief_propagation_init(clause_lits, n, m, rng, num_iters=50)
    
    # Phase 2: Local search from BP assignment
    assignment, unsat = _local_search(bp_assignment, clause_lits, pos_clauses, neg_clauses, n, m, rng, deadline, max_steps=40000)
    if unsat == 0:
        return assignment
    if unsat < best_unsat:
        best_unsat = unsat
        best_assignment = assignment[:]

    # Phase 3: Multiple restarts with random initializations
    num_restarts = 15
    restart_time = (deadline - time.time()) / max(num_restarts, 1)
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5:
            break
        
        # Vary initialization strategy
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif restart == 1:
            assignment = [False] * n
        elif restart == 2:
            assignment = [True] * n
        elif restart == 3:
            # Use BP beliefs with added noise
            assignment = [rng.random() < (beliefs[v] + (rng.random() - 0.5) * 0.3) for v in range(n)]
        elif restart == 4:
            # Use BP beliefs with different noise
            assignment = [rng.random() < (beliefs[v] + (rng.random() - 0.5) * 0.5) for v in range(n)]
        else:
            bias = 0.2 + 0.6 * rng.random()
            assignment = [rng.random() < bias for _ in range(n)]
        
        # Local search
        assignment, unsat = _local_search(assignment, clause_lits, pos_clauses, neg_clauses, n, m, rng, deadline, max_steps=25000)
        
        if unsat == 0:
            return assignment
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
    
    # Phase 4: Final polish pass on best assignment
    if best_assignment is not None and best_unsat > 0 and time.time() < deadline - 10:
        # Try targeted flips on remaining unsat clauses
        clause_sat = _compute_clause_sat(best_assignment, clause_lits)
        unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]
        
        for ci in unsat_clauses[:20]:  # Try up to 20 unsat clauses
            if time.time() >= deadline - 5:
                break
            lits = clause_lits[ci]
            for v, sign in lits:
                # Try flipping this variable
                old_val = best_assignment[v]
                best_assignment[v] = not old_val
                # Recompute satisfaction for affected clauses
                new_unsat = best_unsat
                if old_val:
                    for cj in pos_clauses[v]:
                        if clause_sat[cj] == 1:
                            new_unsat += 1
                    for cj in neg_clauses[v]:
                        if clause_sat[cj] == 0:
                            new_unsat -= 1
                else:
                    for cj in pos_clauses[v]:
                        if clause_sat[cj] == 0:
                            new_unsat -= 1
                    for cj in neg_clauses[v]:
                        if clause_sat[cj] == 1:
                            new_unsat += 1
                
                if new_unsat < best_unsat:
                    # Accept this flip
                    best_unsat = new_unsat
                    # Update clause_sat
                    if old_val:
                        for cj in pos_clauses[v]:
                            clause_sat[cj] -= 1
                        for cj in neg_clauses[v]:
                            clause_sat[cj] += 1
                    else:
                        for cj in pos_clauses[v]:
                            clause_sat[cj] += 1
                        for cj in neg_clauses[v]:
                            clause_sat[cj] -= 1
                    break
                else:
                    # Revert
                    best_assignment[v] = old_val
        
        if best_unsat == 0:
            return best_assignment
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
