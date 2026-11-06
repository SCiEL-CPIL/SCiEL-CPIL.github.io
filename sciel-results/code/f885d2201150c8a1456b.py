import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    if m == 0:
        return [True] * n

    # Build clause structure
    clause_vars = []   # clause_vars[ci] = list of var indices (0-based)
    clause_signs = []  # clause_signs[ci] = list of +1/-1
    for clause in clauses:
        vars_in = []
        signs_in = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_in.append(v)
            signs_in.append(s)
        clause_vars.append(vars_in)
        clause_signs.append(signs_in)

    # Variable occurrence: var_occ[v] = list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for ci in range(m):
        for k in range(len(clause_vars[ci])):
            v = clause_vars[ci][k]
            s = clause_signs[ci][k]
            var_occ[v].append((ci, s))

    # --- Preprocessing: Unit Propagation ---
    assignment = [None] * n
    clause_sat = [0] * m
    
    def compute_clause_sat():
        for ci in range(m):
            sc = 0
            for k in range(len(clause_vars[ci])):
                v = clause_vars[ci][k]
                s = clause_signs[ci][k]
                if assignment[v] is not None:
                    if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                        sc += 1
            clause_sat[ci] = sc

    def propagate():
        changed = True
        while changed:
            changed = False
            # Unit propagation
            for ci in range(m):
                if clause_sat[ci] > 0:
                    continue
                unassigned = []
                for k in range(len(clause_vars[ci])):
                    v = clause_vars[ci][k]
                    if assignment[v] is None:
                        unassigned.append((v, clause_signs[ci][k]))
                if len(unassigned) == 1:
                    v, s = unassigned[0]
                    assignment[v] = (s == 1)
                    changed = True
            if changed:
                compute_clause_sat()
            # Check for contradiction (empty clause)
            # If a clause has 0 unassigned and 0 sat, it's unsat. 
            # In this simple propagation, if we have a unit clause, we assign.
            # If we have an empty clause (all assigned and unsat), we fail.
            # But we are doing local search, so we don't fail, we just restart or handle later.
            # Actually, if we detect a contradiction during preprocessing, we can just return a random assignment 
            # or let the local search handle it. Since it's planted SAT, it should be solvable.

    compute_clause_sat()
    propagate()

    # Check if solved
    if all(clause_sat[ci] > 0 for ci in range(m)):
        return [a if a is not None else True for a in assignment]

    # Initialize unassigned variables randomly
    rng = random.Random(42)
    for v in range(n):
        if assignment[v] is None:
            assignment[v] = rng.random() < 0.5

    # Phase saving array: phase[v] = last value assigned to v
    phase = [bool(a) for a in assignment]

    # Convert to numpy for fast operations? 
    # Python lists are fine for 150 vars.
    
    def recompute_clause_sat():
        for ci in range(m):
            sc = 0
            for k in range(len(clause_vars[ci])):
                v = clause_vars[ci][k]
                s = clause_signs[ci][k]
                if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                    sc += 1
            clause_sat[ci] = sc

    recompute_clause_sat()
    unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]

    if not unsat_clauses:
        return [bool(a) for a in assignment]

    best_unsat = len(unsat_clauses)
    best_assignment = assignment[:]

    # Clause weights for WMinOnes
    clause_weights = [1.0] * m

    max_restarts = 5000

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        # Adaptive step budget
        remaining = deadline - time.time()
        if remaining < 10:
            break
        steps_per_restart = max(100, int(remaining * 100))

        # Initialize or perturb
        if restart == 0:
            pass
        elif restart % 10 == 0:
            # Perturb from best
            assignment = best_assignment[:]
            phase = best_assignment[:]
            # Flip a few random variables
            for _ in range(min(n // 5, 5)):
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
            recompute_clause_sat()
        else:
            # Random restart with phase saving bias
            for v in range(n):
                # Use phase saving: 80% chance to use last phase, 20% random
                if rng.random() < 0.8:
                    assignment[v] = phase[v]
                else:
                    assignment[v] = rng.random() < 0.5
            recompute_clause_sat()

        unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]
        unsat_count = len(unsat_clauses)

        if unsat_count == 0:
            return [bool(a) for a in assignment]

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            phase = assignment[:]

        # Decay weights
        for i in range(m):
            clause_weights[i] *= 0.95

        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break

            if not unsat_clauses:
                return [bool(a) for a in assignment]

            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]
                phase = assignment[:]

            # Pick a random unsat clause
            c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            cd_vars = clause_vars[c_idx]
            cd_signs = clause_signs[c_idx]
            
            # Increase weight of this clause
            clause_weights[c_idx] += 1.0

            # WMinOnes Heuristic
            best_gain = -1000
            best_var = -1
            
            for k in range(len(cd_vars)):
                v = cd_vars[k]
                s = cd_signs[k]
                gain = 0.0
                # Iterate over all clauses containing v
                for (ci2, s2) in var_occ[v]:
                    w = clause_weights[ci2]
                    # Check current satisfaction of ci2
                    # If flipping v satisfies ci2, add w
                    # If flipping v falsifies ci2, subtract w
                    
                    # Current state of v
                    v_val = assignment[v]
                    
                    # Does current v satisfy the literal in ci2?
                    # Literal is s2 * v_val (conceptually)
                    # If s2 == 1, literal is true if v_val is True
                    # If s2 == -1, literal is true if v_val is False
                    cur_lit_true = (s2 == 1 and v_val) or (s2 == -1 and not v_val)
                    
                    # If we flip v, the literal becomes !cur_lit_true
                    new_lit_true = not cur_lit_true
                    
                    # How does this affect clause_sat[ci2]?
                    if cur_lit_true and not new_lit_true:
                        # We are falsifying a currently true literal
                        # If this was the ONLY true literal, the clause becomes unsat
                        if clause_sat[ci2] == 1:
                            gain -= w
                    elif not cur_lit_true and new_lit_true:
                        # We are satisfying a currently false literal
                        # If this makes the clause satisfied (it was unsat), gain +w
                        if clause_sat[ci2] == 0:
                            gain += w
                            
                if gain > best_gain:
                    best_gain = gain
                    best_var = v

            # Flip the best variable
            v_to_flip = best_var
            assignment[v_to_flip] = not assignment[v_to_flip]
            phase[v_to_flip] = assignment[v_to_flip]

            # Update clause_sat for all clauses containing v_to_flip
            newly_unsat = []
            for (ci2, s2) in var_occ[v_to_flip]:
                old_sat = clause_sat[ci2]
                # Recompute
                sc = 0
                for k in range(len(clause_vars[ci2])):
                    v2 = clause_vars[ci2][k]
                    s3 = clause_signs[ci2][k]
                    if (s3 == 1 and assignment[v2]) or (s3 == -1 and not assignment[v2]):
                        sc += 1
                clause_sat[ci2] = sc
                if sc == 0 and old_sat > 0:
                    newly_unsat.append(ci2)

            # Update unsat_clauses list
            # Remove satisfied, add newly unsat
            unsat_clauses = [ci for ci in unsat_clauses if clause_sat[ci] == 0]
            unsat_clauses.extend(newly_unsat)

    return best_assignment
