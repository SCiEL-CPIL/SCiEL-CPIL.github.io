"""WalkSAT + DLS hybrid with vectorized heuristics and optimized unsat tracking."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 300s kill

    # Precompute: for each variable, which clauses contain it (and with what sign)
    # var_clauses[v] = list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_clauses[abs(lit) - 1].append((ci, lit > 0))

    # Precompute clause lengths for DLS
    clause_lens = np.array([len(c) for c in clauses])

    # Build a lookup: for each clause, list of (var_idx, is_positive)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            cv.append((abs(lit) - 1, lit > 0))
        clause_vars.append(cv)

    def greedy_initialize(n, clauses, clause_vars, var_clauses):
        """
        Greedily assign variables to satisfy as many clauses as possible.
        Returns an initial assignment list.
        """
        assignment = [False] * n
        m = len(clauses)
        
        # Calculate initial satisfaction counts
        sat_count = [0] * m
        for ci, cv in enumerate(clause_vars):
            for v, is_pos in cv:
                if assignment[v] == is_pos:
                    sat_count[ci] += 1
                    
        unsat_clauses = [ci for ci in range(m) if sat_count[ci] == 0]
        
        max_iterations = n * 2 # Safety
        for _ in range(max_iterations):
            if not unsat_clauses:
                break
                
            # Count occurrences of each variable in unsat clauses
            var_counts = {}
            for ci in unsat_clauses:
                for v, _ in clause_vars[ci]:
                    var_counts[v] = var_counts.get(v, 0) + 1
                    
            if not var_counts:
                break
                
            # Find variable with max count
            best_var = max(var_counts, key=var_counts.get)
            
            # Determine best value for best_var
            gain_true = 0
            gain_false = 0
            for ci, is_pos in var_clauses[best_var]:
                if is_pos:
                    # Literal is +v
                    if assignment[best_var] == False:
                        # Currently False, so +v is False.
                        # If we set to True, +v becomes True.
                        if sat_count[ci] == 0:
                            gain_true += 1
                    else:
                        # Currently True, so +v is True.
                        # If we set to False, +v becomes False.
                        if sat_count[ci] == 1:
                            gain_false -= 1
                else:
                    # Literal is -v
                    if assignment[best_var] == True:
                        # Currently True, so -v is False.
                        # If we set to False, -v becomes True.
                        if sat_count[ci] == 0:
                            gain_false += 1
                    else:
                        # Currently False, so -v is True.
                        # If we set to True, -v becomes False.
                        if sat_count[ci] == 1:
                            gain_true -= 1
                            
            if gain_true >= gain_false:
                new_val = True
            else:
                new_val = False
                
            if assignment[best_var] == new_val:
                if gain_true <= 0 and gain_false <= 0:
                    break
            else:
                assignment[best_var] = new_val
                # Update sat_count
                for ci, is_pos in var_clauses[best_var]:
                    if is_pos == new_val:
                        sat_count[ci] += 1
                    else:
                        sat_count[ci] -= 1
                # Rebuild unsat_clauses
                unsat_clauses = [ci for ci in range(m) if sat_count[ci] == 0]
                
        return assignment

    def make_satisfied_array(assignment):
        """Return (satisfied bool array, satisfaction count array, unsat list)."""
        sat = np.zeros(m, dtype=bool)
        sat_count = np.zeros(m, dtype=np.int32)
        unsat_list = []
        for ci in range(m):
            cnt = 0
            for v, is_pos in clause_vars[ci]:
                if assignment[v] == is_pos:
                    cnt += 1
            sat_count[ci] = cnt
            if cnt > 0:
                sat[ci] = True
            else:
                unsat_list.append(ci)
        return sat, sat_count, unsat_list

    def flip_var(assignment, var_idx, sat, sat_count, unsat_list):
        """Flip variable var_idx and update sat/sat_count/unsat_list incrementally."""
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        
        for ci, is_pos in var_clauses[var_idx]:
            if is_pos == new_val:
                # This literal now satisfies the clause
                if sat_count[ci] == 0:
                    # Clause was unsat, now sat
                    sat[ci] = True
                    try:
                        idx = unsat_list.index(ci)
                        unsat_list.pop(idx)
                    except ValueError:
                        pass
                sat_count[ci] += 1
            else:
                # This literal no longer satisfies; decrement count
                sat_count[ci] -= 1
                if sat_count[ci] == 0:
                    sat[ci] = False
                    unsat_list.append(ci)

    def get_best_flip_var(assignment, sat_count, ci, rng):
        """Vectorized calculation of best variable to flip in clause ci."""
        clause = clause_vars[ci]
        best_var = -1
        best_score = -1000
        
        for v, is_pos in clause:
            gains = 0
            losses = 0
            for c2, is_pos2 in var_clauses[v]:
                if c2 == ci:
                    continue
                if is_pos2 == assignment[v]:
                    if sat_count[c2] == 1:
                        losses += 1
                else:
                    gains += 1
            score = gains - losses
            if score > best_score:
                best_score = score
                best_var = v
                
        if best_var >= 0:
            return best_var
        else:
            v, _ = rng.choice(clause)
            return v

    def dls_step(assignment, sat, sat_count, unsat_list, rng):
        """DLS step: pick the shortest unsatisfied clause, flip a random var in it."""
        if not unsat_list:
            return
        min_len = 10
        best_ci = unsat_list[0]
        for ci in unsat_list:
            l = clause_lens[ci]
            if l < min_len:
                min_len = l
                best_ci = ci
                if l <= 1:
                    break
        clause = clause_vars[best_ci]
        v, _ = rng.choice(clause)
        flip_var(assignment, v, sat, sat_count, unsat_list)

    best_assignment = None
    best_unsat = m + 1

    # Use greedy initialization for the first restart, then random for others
    # Or use greedy for all? Greedy is fast.
    
    for restart in range(500):
        if time.time() > deadline:
            break
        rng = random.Random(restart * 7919 + 13)
        
        # Use greedy initialization for the first few restarts to find a good starting point
        # Then switch to random to explore different basins
        if restart < 5:
            assignment = greedy_initialize(n, clauses, clause_vars, var_clauses)
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
        sat, sat_count, unsat_list = make_satisfied_array(assignment)
        cur_unsat = len(unsat_list)
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        steps_in_restart = 0
        max_steps = 15000
        
        dls_interval = 200
        stall_count = 0
        prev_unsat = cur_unsat
        p_walk = 0.65

        for step in range(max_steps):
            if time.time() > deadline:
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment

            if not unsat_list:
                return assignment

            # WalkSAT step
            if rng.random() < p_walk:
                ci = rng.choice(unsat_list)
                v = get_best_flip_var(assignment, sat_count, ci, rng)
            else:
                ci = rng.choice(unsat_list)
                clause = clause_vars[ci]
                v, _ = rng.choice(clause)
            
            flip_var(assignment, v, sat, sat_count, unsat_list)
            cur_unsat = len(unsat_list)
            steps_in_restart += 1

            if steps_in_restart % dls_interval == 0:
                dls_step(assignment, sat, sat_count, unsat_list, rng)
                cur_unsat = len(unsat_list)

            if step % 20 == 0:
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                if cur_unsat < prev_unsat:
                    stall_count = 0
                    p_walk = max(0.30, p_walk - 0.02)
                else:
                    stall_count += 1
                prev_unsat = cur_unsat

            if stall_count > 15:
                break

        cur_unsat = len(unsat_list)
        if cur_unsat <= 10:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                if not unsat_list:
                    break
                ci = rng.choice(unsat_list)
                v = get_best_flip_var(assignment, sat_count, ci, rng)
                flip_var(assignment, v, sat, sat_count, unsat_list)
            cur_unsat = len(unsat_list)
            if cur_unsat == 0:
                return assignment
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    if best_assignment is not None and best_unsat <= 15:
        final_rng = random.Random(999999)
        best_sat, best_sat_count, best_unsat_list = make_satisfied_array(best_assignment)
        for _ in range(5000):
            if time.time() > deadline:
                break
            if not best_unsat_list:
                break
            ci = final_rng.choice(best_unsat_list)
            v = get_best_flip_var(best_assignment, best_sat_count, ci, final_rng)
            flip_var(best_assignment, v, best_sat, best_sat_count, best_unsat_list)
        
        if not best_unsat_list:
            return best_assignment

    return best_assignment if best_assignment else [False] * n
# EVOLVE-BLOCK-END
