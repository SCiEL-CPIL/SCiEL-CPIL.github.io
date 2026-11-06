import time
import random
import sys
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin (330s kill - 5s buffer)

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
            var_occ[clause_vars[ci][k]].append((ci, clause_signs[ci][k]))

    # Build numpy arrays for fast access
    # For each variable, store clause indices and signs as numpy arrays
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        idxs = [ci for (ci, s) in var_occ[v]]
        signs = [s for (ci, s) in var_occ[v]]
        var_clause_idx.append(np.array(idxs, dtype=np.int32) if idxs else np.array([], dtype=np.int32))
        var_clause_sign.append(np.array(signs, dtype=np.int8) if signs else np.array([], dtype=np.int8))

    # --- Preprocessing: unit propagation + pure literal elimination ---
    assignment = [None] * n
    clause_sat = [0] * m  # number of satisfied literals per clause

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
            # Pure literal elimination
            for v in range(n):
                if assignment[v] is not None:
                    continue
                has_pos = False
                has_neg = False
                for (ci, s) in var_occ[v]:
                    if clause_sat[ci] > 0:
                        continue
                    if s == 1:
                        has_pos = True
                    else:
                        has_neg = True
                if has_pos and not has_neg:
                    assignment[v] = True
                    changed = True
                elif has_neg and not has_pos:
                    assignment[v] = False
                    changed = True
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

    # Convert to numpy for fast operations
    assignment_np = np.array([1 if a else 0 for a in assignment], dtype=np.int8)

    # Build vectorized structures
    # clause_sat_np: satisfied literal count per clause
    clause_sat_np = np.zeros(m, dtype=np.int32)
    for ci in range(m):
        sc = 0
        for k in range(len(clause_vars[ci])):
            v = clause_vars[ci][k]
            s = clause_signs[ci][k]
            if (s == 1 and assignment_np[v] == 1) or (s == -1 and assignment_np[v] == 0):
                sc += 1
        clause_sat_np[ci] = sc

    # For each variable, precompute clause indices and signs as numpy arrays
    var_clause_idx = []
    var_clause_sign = []
    for v in range(n):
        idxs = [ci for (ci, s) in var_occ[v]]
        signs = [s for (ci, s) in var_occ[v]]
        var_clause_idx.append(np.array(idxs, dtype=np.int32) if idxs else np.array([], dtype=np.int32))
        var_clause_sign.append(np.array(signs, dtype=np.int8) if signs else np.array([], dtype=np.int8))

    # Track unsat clauses
    unsat_set = set(ci for ci in range(m) if clause_sat_np[ci] == 0)
    unsat_list = list(unsat_set)
    
    if not unsat_list:
        return assignment.tolist()

    best_unsat = len(unsat_list)
    best_assignment = assignment[:]

    def full_recompute():
        """Recompute clause_sat_np and unsat_list from scratch."""
        nonlocal clause_sat_np, unsat_list, unsat_set
        clause_sat_np[:] = 0
        for ci in range(m):
            sc = 0
            for k in range(len(clause_vars[ci])):
                v = clause_vars[ci][k]
                s = clause_signs[ci][k]
                if (s == 1 and assignment_np[v] == 1) or (s == -1 and assignment_np[v] == 0):
                    sc += 1
            clause_sat_np[ci] = sc
        unsat_set = set(ci for ci in range(m) if clause_sat_np[ci] == 0)
        unsat_list = list(unsat_set)

    def compute_gain(v):
        """Compute the gain of flipping variable v using vectorized numpy ops."""
        idxs_v = var_clause_idx[v]
        signs_v = var_clause_sign[v]
        if len(idxs_v) == 0:
            return 0
        var_val = assignment_np[v]
        # Literal is satisfied if (sign==1 and var==1) or (sign==-1 and var==0)
        # = (sign == 1) == (var_val == 1)
        lit_true = (signs_v == 1) == (var_val == 1)
        cs_vals = clause_sat_np[idxs_v]
        # gain = +1 for each clause where lit is false and clause is unsat
        #        -1 for each clause where lit is true and clause has exactly 1 satisfied
        gain = int(np.sum((~lit_true) & (cs_vals == 0))) - int(np.sum(lit_true & (cs_vals == 1)))
        return gain

    def compute_gains_batch(vars_to_check):
        """Compute gains for multiple variables at once."""
        gains = []
        for v in vars_to_check:
            idxs_v = var_clause_idx[v]
            signs_v = var_clause_sign[v]
            if len(idxs_v) == 0:
                gains.append(0)
                continue
            var_val = assignment_np[v]
            lit_true = (signs_v == 1) == (var_val == 1)
            cs_vals = clause_sat_np[idxs_v]
            gain = int(np.sum((~lit_true) & (cs_vals == 0))) - int(np.sum(lit_true & (cs_vals == 1)))
            gains.append(gain)
        return gains

    max_restarts = 10000
    restart = 0

    while time.time() < deadline - 2.0:
        restart += 1
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Vectorized steps are ~10x faster than original, so allow many more
        steps_per_restart = max(1000, int(remaining * 5000))

        # Initialize or perturb
        if restart == 1:
            pass  # already initialized
        elif restart % 10 == 0:
            # Perturb from best
            assignment_np = np.array([1 if a else 0 for a in best_assignment], dtype=np.int8)
            num_perturb = min(n // 5, 15)
            for _ in range(num_perturb):
                v = rng.randint(0, n - 1)
                assignment_np[v] = 1 - assignment_np[v]
            full_recompute()
        else:
            # Random restart
            assignment_np = (rng.random(n) < 0.5).astype(np.int8)
            full_recompute()

        unsat_count = len(unsat_list)

        if unsat_count == 0:
            return assignment_np.astype(bool).tolist()

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment_np.astype(bool).tolist()

        # Adaptive greedy probability
        p_greedy = 0.5 + 0.3 * (1.0 - unsat_count / max(m, 1))

        for step in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break

            if not unsat_list:
                return assignment_np.astype(bool).tolist()

            unsat_count = len(unsat_list)
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment_np.astype(bool).tolist()

            # Pick a random unsat clause
            c_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            cd_vars = clause_vars[c_idx]

            # Compute gains for the 3 variables in this clause
            gains = [compute_gain(v) for v in cd_vars]
            
            # Find best gain
            best_gain = max(gains)
            best_vars = [cd_vars[i] for i in range(len(cd_vars)) if gains[i] == best_gain]

            # Choose variable to flip
            if rng.random() < p_greedy:
                v_to_flip = best_vars[rng.randint(0, len(best_vars) - 1)]
            else:
                v_to_flip = cd_vars[rng.randint(0, len(cd_vars) - 1)]

            # Flip the variable
            old_val = assignment_np[v_to_flip]
            new_val = 1 - old_val
            assignment_np[v_to_flip] = new_val

            # Update clause_sat_np for all clauses containing v_to_flip
            idxs_v = var_clause_idx[v_to_flip]
            signs_v = var_clause_sign[v_to_flip]
            
            if len(idxs_v) > 0:
                # New lit_true
                new_lit_true = (signs_v == 1) == (new_val == 1)
                # Old lit_true
                old_lit_true = (signs_v == 1) == (old_val == 1)
                
                # Clauses where literal went from unsatisfied to satisfied: gain +1
                became_sat = (~old_lit_true) & new_lit_true
                # Clauses where literal went from satisfied to unsatisfied: gain -1
                became_unsat = old_lit_true & (~new_lit_true)
                
                for i in range(len(idxs_v)):
                    ci = int(idxs_v[i])
                    if became_sat[i]:
                        clause_sat_np[ci] += 1
                        if clause_sat_np[ci] == 1:
                            # Was unsat, now sat
                            unsat_set.discard(ci)
                    elif became_unsat[i]:
                        clause_sat_np[ci] -= 1
                        if clause_sat_np[ci] == 0:
                            # Was sat (exactly 1), now unsat
                            unsat_set.add(ci)
                
                # Rebuild unsat_list periodically or when size changes significantly
                if len(unsat_set) != len(unsat_list):
                    unsat_list = list(unsat_set)

    # Final polish: try to reduce unsat further with more aggressive search
    if time.time() < deadline - 5.0:
        assignment_np = np.array([1 if a else 0 for a in best_assignment], dtype=np.int8)
        full_recompute()
        
        polish_steps = 5000
        for step in range(polish_steps):
            if time.time() > deadline - 3.0:
                break
            if not unsat_list:
                return assignment_np.astype(bool).tolist()
            
            unsat_count = len(unsat_list)
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment_np.astype(bool).tolist()
            
            # Pick random unsat clause
            c_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            cd_vars = clause_vars[c_idx]
            
            # Compute gains
            gains = compute_gains_batch(cd_vars)
            best_gain = max(gains)
            best_vars = [cd_vars[i] for i in range(len(cd_vars)) if gains[i] == best_gain]
            
            # Always pick best in polish
            v_to_flip = best_vars[rng.randint(0, len(best_vars) - 1)]
            
            old_val = assignment_np[v_to_flip]
            new_val = 1 - old_val
            assignment_np[v_to_flip] = new_val
            
            idxs_v = var_clause_idx[v_to_flip]
            signs_v = var_clause_sign[v_to_flip]
            
            if len(idxs_v) > 0:
                new_lit_true = (signs_v == 1) == (new_val == 1)
                old_lit_true = (signs_v == 1) == (old_val == 1)
                became_sat = (~old_lit_true) & new_lit_true
                became_unsat = old_lit_true & (~new_lit_true)
                
                for i in range(len(idxs_v)):
                    ci = int(idxs_v[i])
                    if became_sat[i]:
                        clause_sat_np[ci] += 1
                        if clause_sat_np[ci] == 1:
                            unsat_set.discard(ci)
                    elif became_unsat[i]:
                        clause_sat_np[ci] -= 1
                        if clause_sat_np[ci] == 0:
                            unsat_set.add(ci)
                
                if len(unsat_set) != len(unsat_list):
                    unsat_list = list(unsat_set)
    
    return best_assignment
