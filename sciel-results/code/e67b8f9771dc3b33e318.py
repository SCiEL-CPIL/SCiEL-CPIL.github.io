"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, use 290s to be safe
    deadline = time.time() + 290.0
    rng = random.Random(42)

    if m == 0:
        return [False] * n

    # Preprocessing: Convert to 0-based indices and separate signs
    # var_clauses[v] = list of clause indices containing v
    # clause_vars[c] = list of variables in clause c
    # clause_signs[c] = list of signs (+1 or -1) corresponding to clause_vars[c]
    
    clause_vars = []
    clause_signs = []
    var_clauses = [[] for _ in range(n)]
    
    for c_idx, cl in enumerate(clauses):
        vars_in_cl = []
        signs_in_cl = []
        for lit in cl:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_in_cl.append(v)
            signs_in_cl.append(s)
            var_clauses[v].append(c_idx)
        clause_vars.append(vars_in_cl)
        clause_signs.append(signs_in_cl)

    def is_satisfied(assignment):
        for c in range(m):
            vs = clause_vars[c]
            ss = clause_signs[c]
            sat = False
            for i in range(len(vs)):
                v = vs[i]
                s = ss[i]
                if s == 1:
                    if assignment[v]:
                        sat = True
                        break
                else:
                    if not assignment[v]:
                        sat = True
                        break
            if not sat:
                return False
        return True

    best_assignment = None
    best_unsat = m + 1

    # WalkSAT with Heavy Clause Selection and Adaptive Restarts
    # We use a simple restart strategy: run for a fixed number of steps or time slice, 
    # then restart if no solution found.
    
    # Parameters
    # p_heavy: probability of choosing the variable that maximizes the number of 
    #          other unsatisfied clauses satisfied.
    # restart_steps: number of flips per restart before forcing a new random start.
    
    p_heavy = 0.8
    # For 150 vars, 4.26 ratio, instances are usually easy. 
    # A short restart helps escape local minima.
    # Let's try dynamic restarts: if we don't improve unsat count for X steps, restart.
    
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat counts
        unsat_count = [0] * m
        total_unsat = 0
        for c in range(m):
            vs = clause_vars[c]
            ss = clause_signs[c]
            sat = False
            for i in range(len(vs)):
                v = vs[i]
                s = ss[i]
                if s == 1:
                    if assignment[v]:
                        sat = True
                        break
                else:
                    if not assignment[v]:
                        sat = True
                        break
            if not sat:
                unsat_count[c] = 1
                total_unsat += 1
        
        if total_unsat == 0:
            return assignment
            
        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment[:]

        # Local Search Loop
        # We maintain a list of unsat clause indices to pick from quickly?
        # Rebuilding is O(m). m ~ 630. 
        # Picking random index and checking unsat_count is O(1) average if density is high.
        # But if we get stuck with few unsat clauses, we might waste time.
        # Let's maintain a list of unsat clauses.
        
        unsat_clauses = [c for c in range(m) if unsat_count[c] > 0]
        
        steps = 0
        max_steps_per_restart = 1000 # Heuristic: short restarts are good for 3-SAT
        # Or time-based: 0.1s per restart?
        # Let's use step-based with a time check.
        
        while steps < max_steps_per_restart:
            if time.time() > deadline:
                break
            
            if total_unsat == 0:
                return assignment
                
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]
                
            # Pick a random unsatisfied clause
            if not unsat_clauses:
                break
            c_idx = rng.choice(unsat_clauses)
            vs = clause_vars[c_idx]
            ss = clause_signs[c_idx]
            
            # Calculate scores for each variable in this clause
            # Score: number of OTHER unsat clauses that would be satisfied if we flip this var
            scores = [0] * len(vs)
            
            for i, v in enumerate(vs):
                # Check all clauses containing v
                for cj in var_clauses[v]:
                    if cj == c_idx:
                        continue
                    if unsat_count[cj] == 0:
                        continue
                    
                    # Check if flipping v satisfies cj
                    # Find v in cj
                    vs_cj = clause_vars[cj]
                    ss_cj = clause_signs[cj]
                    
                    # Optimization: Pre-compute index of v in cj? 
                    # var_clauses[v] gives clause indices. 
                    # We need to know the sign of v in cj.
                    # We can just scan vs_cj (length 3) to find v.
                    
                    # Current value of v: assignment[v]
                    # If we flip v, it becomes not assignment[v]
                    # For cj to be satisfied by the flip, the literal of v in cj must be True AFTER flip.
                    # Literal is True if (s==1 and val==True) or (s==-1 and val==False).
                    # After flip, val is not assignment[v].
                    
                    # So we need:
                    # If s_cj == 1: we need not assignment[v] to be True => assignment[v] is False
                    # If s_cj == -1: we need not assignment[v] to be False => assignment[v] is True
                    
                    # So, if (s_cj == 1 and not assignment[v]) or (s_cj == -1 and assignment[v]):
                    #   Then flipping v satisfies cj.
                    
                    # We need s_cj.
                    found = False
                    for k in range(len(vs_cj)):
                        if vs_cj[k] == v:
                            s_cj = ss_cj[k]
                            if (s_cj == 1 and not assignment[v]) or (s_cj == -1 and assignment[v]):
                                scores[i] += 1
                            found = True
                            break
                    # If not found, something is wrong with data structure, but shouldn't happen.
            
            # Choose variable to flip
            # With prob p_heavy, pick max score. Else random.
            if rng.random() < p_heavy:
                max_score = max(scores)
                # Pick random among those with max score
                candidates = [i for i, s in enumerate(scores) if s == max_score]
                idx = rng.choice(candidates)
            else:
                idx = rng.randint(0, len(vs) - 1)
                
            v_to_flip = vs[idx]
            
            # Flip the variable
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update unsat counts for clauses containing v_to_flip
            for cj in var_clauses[v_to_flip]:
                vs_cj = clause_vars[cj]
                ss_cj = clause_signs[cj]
                
                was_unsat = unsat_count[cj] > 0
                
                # Determine if the literal of v_to_flip in cj is now true
                # Find index of v_to_flip in cj
                idx_v = -1
                for k in range(len(vs_cj)):
                    if vs_cj[k] == v_to_flip:
                        idx_v = k
                        break
                
                if idx_v == -1:
                    continue
                
                s_v = ss_cj[idx_v]
                
                # Check if literal is true with new_val
                # True if (s_v == 1 and new_val) or (s_v == -1 and not new_val)
                lit_now_true = (s_v == 1 and new_val) or (s_v == -1 and not new_val)
                
                # Check if literal was true with old_val
                lit_was_true = (s_v == 1 and old_val) or (s_v == -1 and not old_val)
                
                if was_unsat:
                    if lit_now_true:
                        # Clause becomes satisfied
                        unsat_count[cj] = 0
                        total_unsat -= 1
                        # Remove from unsat_clauses list? 
                        # Lists are expensive to remove from. 
                        # We can just let unsat_clauses have stale entries and filter?
                        # Or rebuild periodically.
                        # For now, we'll rebuild unsat_clauses if it gets too large or every X steps.
                        # Actually, `rng.choice(unsat_clauses)` might pick a satisfied clause.
                        # We must ensure we only pick unsat ones.
                        # Strategy: Rebuild unsat_clauses every 100 steps or if empty.
                        pass
                    else:
                        # Still unsat
                        pass
                else:
                    # Was satisfied
                    if lit_was_true and not lit_now_true:
                        # We flipped the only true literal?
                        # Check if any other literal is true
                        any_other_true = False
                        for k in range(len(vs_cj)):
                            if k == idx_v:
                                continue
                            v_k = vs_cj[k]
                            s_k = ss_cj[k]
                            if (assignment[v_k] and s_k == 1) or (not assignment[v_k] and s_k == -1):
                                any_other_true = True
                                break
                        if not any_other_true:
                            # Clause becomes unsatisfied
                            unsat_count[cj] = 1
                            total_unsat += 1
                            pass
                    # Else still satisfied
                
            steps += 1
            
            # Rebuild unsat_clauses list periodically to avoid stale entries
            if steps % 50 == 0:
                unsat_clauses = [c for c in range(m) if unsat_count[c] > 0]
                if not unsat_clauses:
                    break

        # End of restart

    # If we exhausted time, return best found if it's valid, else best effort
    if best_assignment is not None:
        if is_satisfied(best_assignment):
            return best_assignment
        # If best_assignment is not satisfied, we can't return it as a solution.
        # But the task says "Return a satisfying assignment".
        # If we fail to find one, we return the best we have? 
        # "Score: fraction of satisfiable instances solved"
        # If we return an unsat assignment, it's invalid (score 0 for that instance).
        # It's better to return something than crash, but it will be 0.
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
