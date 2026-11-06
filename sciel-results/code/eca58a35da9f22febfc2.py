import numpy as np
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Build data structures
    # clause_vars[i] = list of (var_idx, sign)
    # var_to_clauses[v] = list of clause indices
    # lit_var[i][j] = variable index for literal j in clause i
    # lit_sign[i][j] = sign (1 or -1)
    
    lit_var = np.zeros((C, 3), dtype=np.int32)
    lit_sign = np.zeros((C, 3), dtype=np.int8)
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            lit_var[i, j] = v
            lit_sign[i, j] = s
            var_to_clauses[v].append(i)
    
    # Convert to numpy arrays for vectorized ops
    lit_var_arr = lit_var  # shape (C, 3)
    lit_sign_arr = lit_sign  # shape (C, 3)
    
    # For each variable, precompute which (clause, position) it appears in
    # var_clause_pos[v] = list of (clause_idx, pos)
    var_clause_pos = [[] for _ in range(n)]
    for v in range(n):
        for ci in var_to_clauses[v]:
            # Find position of v in clause ci
            for p in range(3):
                if lit_var[ci, p] == v:
                    var_clause_pos[v].append((ci, p))
                    break
    
    start_time = time.time()
    deadline = start_time + 295.0
    
    rng = random.Random(12345)
    
    def evaluate_vec(assignment):
        """Vectorized evaluation. Returns (num_unsat, unsat_clauses_array)"""
        # assignment is numpy bool array of length n
        # lit_sat[c, j] = whether literal j in clause c is satisfied
        # A literal (v, s) is sat if (s==1 and assignment[v]) or (s==-1 and not assignment[v])
        # = (s==1 and a[v]) or (s==-1 and not a[v])
        # = (s * a[v] == 1) if we encode True as 1, False as 0... 
        # Actually: lit_sat = (sign == 1) * a[v] + (sign == -1) * (1 - a[v])
        # In numpy: lit_sat = np.where(lit_sign == 1, assignment[lit_var], 1 - assignment[lit_var])
        
        a = assignment  # numpy bool array
        # Get values for all literals
        lit_vals = a[lit_var_arr]  # shape (C, 3), bool
        # A literal is satisfied if sign==1 and val==True, or sign==-1 and val==False
        # sat = (sign == 1) & val  OR  (sign == -1) & ~val
        sat = (lit_sign_arr == 1) & lit_vals | (lit_sign_arr == -1) & ~lit_vals
        # clause_sat[c] = any literal in clause c is sat
        clause_sat = sat.any(axis=1)  # shape (C,)
        unsat_clauses = np.where(~clause_sat)[0]
        return len(unsat_clauses), unsat_clauses
    
    def compute_deltas(v, assignment, unsat_clauses_set):
        """Compute delta for each variable in the clause, return array of deltas"""
        # For variable v, delta = (# clauses that become sat) - (# clauses that become unsat)
        # A clause containing v:
        #   - If currently unsat: flipping v makes it sat iff the other two lits are unsat
        #     (because after flip, v's lit becomes sat, so clause becomes sat regardless)
        #     Wait: if clause is unsat, ALL three lits are unsat. After flipping v, v's lit becomes sat,
        #     so clause becomes sat. So delta += 1 for each unsat clause containing v.
        #   - If currently sat: flipping v makes it unsat iff v's lit was the ONLY sat literal
        #     (i.e., the other two are unsat)
        # 
        # So for each clause ci containing v:
        #   if ci in unsat: delta += 1
        #   else: check if other two lits are unsat; if so, delta -= 1
        
        deltas = {}
        for (ci, pos) in var_clause_pos[v]:
            # Get other two positions
            others = [p for p in range(3) if p != pos]
            o1, o2 = others[0], others[1]
            
            # Check if other two literals are satisfied
            # lit_sat for (o1): sign==1 ? a[lit_var[ci,o1]] : not a[lit_var[ci,o1]]
            v1 = lit_var[ci, o1]
            v2 = lit_var[ci, o2]
            s1 = lit_sign[ci, o1]
            s2 = lit_sign[ci, o2]
            
            o1_sat = (s1 == 1 and assignment[v1]) or (s1 == -1 and not assignment[v1])
            o2_sat = (s2 == 1 and assignment[v2]) or (s2 == -1 and not assignment[v2])
            
            if ci in unsat_clauses_set:
                # Clause is unsat, flipping v will satisfy it
                deltas[ci] = 1
            else:
                # Clause is sat. After flipping v, v's lit becomes unsat.
                # Clause becomes unsat iff both other lits are also unsat
                if not o1_sat and not o2_sat:
                    deltas[ci] = -1
                else:
                    deltas[ci] = 0
        
        return deltas
    
    def walk_sat(assignment_np, num_unsat, unsat_clauses_arr, rng, max_steps, deadline, clause_weights):
        """
        WalkSAT with clause weighting.
        assignment_np: numpy bool array
        unsat_clauses_arr: numpy array of unsatisfied clause indices
        clause_weights: numpy array of weights for each clause
        """
        unsat_set = set(unsat_clauses_arr.tolist()) if len(unsat_clauses_arr) > 0 else set()
        
        # Track best
        best_num_unsat = num_unsat
        best_assignment = assignment_np.copy()
        
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return True, assignment_np
            
            # Pick a random unsatisfied clause, weighted by clause weight
            if len(unsat_clauses_arr) > 0:
                weights = clause_weights[unsat_clauses_arr]
                # Normalize for random choice
                total_w = weights.sum()
                if total_w <= 0:
                    c_idx = rng.randrange(len(unsat_clauses_arr))
                else:
                    probs = weights / total_w
                    c_idx = rng.choices(range(len(unsat_clauses_arr)), weights=probs, k=1)[0]
                c = unsat_clauses_arr[c_idx]
            else:
                return True, assignment_np
            
            cv = lit_var[c]  # 3 variables in this clause
            cs = lit_sign[c]  # 3 signs
            
            # Compute delta for each variable in this clause
            best_v = -1
            best_score = -1e9
            
            for j in range(3):
                v = cv[j]
                # Compute delta
                delta = 0
                for (ci, pos) in var_clause_pos[v]:
                    others = [p for p in range(3) if p != pos]
                    o1, o2 = others[0], others[1]
                    
                    v1 = lit_var[ci, o1]
                    v2 = lit_var[ci, o2]
                    s1 = lit_sign[ci, o1]
                    s2 = lit_sign[ci, o2]
                    
                    o1_sat = (s1 == 1 and assignment_np[v1]) or (s1 == -1 and not assignment_np[v1])
                    o2_sat = (s2 == 1 and assignment_np[v2]) or (s2 == -1 and not assignment_np[v2])
                    
                    if ci in unsat_set:
                        delta += 1
                    else:
                        if not o1_sat and not o2_sat:
                            delta -= 1
                
                # Add noise - adaptive noise
                noise = rng.gauss(0, 1.0) * 0.8
                score = delta + noise
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            # Flip best_v
            assignment_np[best_v] = not assignment_np[best_v]
            
            # Update clause weights: increase weight of unsat clauses, decay others
            # Simple scheme: weight[ci] += 1 if unsat, weight[ci] *= 0.95 otherwise
            # But doing this every step is expensive. Do it every 100 steps.
            if step % 100 == 0:
                for ci in range(C):
                    if ci in unsat_set:
                        clause_weights[ci] += 1.0
                    else:
                        clause_weights[ci] *= 0.95
                # Normalize to prevent overflow
                max_w = clause_weights.max()
                if max_w > 0:
                    clause_weights /= max_w
            
            # Incremental update of unsat_clauses
            # Find affected clauses
            affected = set()
            for (ci, pos) in var_clause_pos[best_v]:
                affected.add(ci)
            
            # Recompute satisfaction for affected clauses
            new_unsat_list = []
            
            # First, process clauses that were unsat
            for ci in unsat_clauses_arr:
                if ci in affected:
                    # Check if now satisfied
                    sat = False
                    for p in range(3):
                        v = lit_var[ci, p]
                        s = lit_sign[ci, p]
                        if (s == 1 and assignment_np[v]) or (s == -1 and not assignment_np[v]):
                            sat = True
                            break
                    if not sat:
                        new_unsat_list.append(ci)
                else:
                    new_unsat_list.append(ci)
            
            # Check affected clauses that were sat - they might become unsat
            for ci in affected:
                if ci not in set(unsat_clauses_arr.tolist()):
                    sat = False
                    for p in range(3):
                        v = lit_var[ci, p]
                        s = lit_sign[ci, p]
                        if (s == 1 and assignment_np[v]) or (s == -1 and not assignment_np[v]):
                            sat = True
                            break
                    if not sat:
                        new_unsat_list.append(ci)
            
            unsat_clauses_arr = np.array(new_unsat_list, dtype=np.int32)
            num_unsat = len(unsat_clauses_arr)
            unsat_set = set(unsat_clauses_arr.tolist())
            
            if num_unsat < best_num_unsat:
                best_num_unsat = num_unsat
                best_assignment = assignment_np.copy()
            
            if num_unsat == 0:
                return True, assignment_np
        
        return False, best_assignment
    
    # Main loop with multiple restarts
    best_num_unsat = C + 1
    best_assignment_list = None
    
    restart = 0
    while time.time() < deadline:
        # Random initial assignment
        assignment_np = np.random.random(n) < 0.5
        
        # Evaluate
        num_unsat, unsat_clauses_arr = evaluate_vec(assignment_np)
        
        if num_unsat == 0:
            return assignment_np.tolist()
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment_list = assignment_np.tolist()
        
        # Clause weights - start uniform, vary per restart
        if restart == 0:
            clause_weights = np.ones(C, dtype=np.float64)
        else:
            # Add some randomness to weights
            clause_weights = 1.0 + rng.random() * 0.5
        
        # Adaptive step limit
        base_steps = 3000 + restart * 500
        max_steps = min(base_steps, 80000)
        
        # Check time before starting walk_sat
        if time.time() >= deadline - 5:
            break
        
        solved, result = walk_sat(
            assignment_np.copy(), num_unsat, unsat_clauses_arr,
            rng, max_steps, deadline, clause_weights.copy()
        )
        
        if solved:
            return result.tolist()
        
        # Update best
        if best_assignment_list is None:
            best_assignment_list = result.tolist()
        
        restart += 1
        
        # If we've done many restarts, try a different strategy
        if restart % 10 == 0 and time.time() < deadline - 10:
            # Try starting from best assignment found so far
            if best_assignment_list is not None:
                assignment_np = np.array(best_assignment_list, dtype=bool)
                num_unsat, unsat_clauses_arr = evaluate_vec(assignment_np)
                if num_unsat < best_num_unsat:
                    best_num_unsat = num_unsat
                    best_assignment_list = assignment_np.tolist()
    
    if best_assignment_list is not None:
        return best_assignment_list
    return [False] * n
