"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, use 295s to be safe
    deadline = time.time() + 295.0

    # Preprocessing
    # Convert 1-based signed literals to 0-based indices and signs
    cvars = []
    csigns = []
    for cl in clauses:
        vs = []
        ss = []
        for l in cl:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            vs.append(v)
            ss.append(s)
        cvars.append(vs)
        csigns.append(ss)
        
    # Precompute variable-clause incidence for fast scoring
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for v in cvars[ci]:
            var_clauses[v].append(ci)
            
    # We will use a "Reasoning" step:
    # If a clause is unit (only one literal can be true given current assignment of others), 
    # we should force that variable.
    
    def is_satisfied(cl_idx, assignment):
        vs = cvars[cl_idx]
        ss = csigns[cl_idx]
        for i in range(len(vs)):
            v = vs[i]
            s = ss[i]
            # assignment[v] is bool. s is 1 or -1.
            # If s==1, we want assignment[v]==True.
            # If s==-1, we want assignment[v]==False.
            if s == 1:
                if assignment[v]: return True
            else:
                if not assignment[v]: return True
        return False

    def check_all(assignment):
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                return False
        return True

    best_assignment = None
    best_unsat = m + 1
    
    rng = random.Random(42)
    
    # GLARE-like parameters
    # We maintain a "trail" of decisions? No, standard WalkSAT doesn't use a trail.
    # Let's use a "Conflict-Driven" approach:
    # 1. Start random.
    # 2. Loop:
    #    a. Find all unsat clauses.
    #    b. If none, return.
    #    c. Pick a random unsat clause.
    #    d. For each var in clause, calculate "score" (how many other unsat clauses it satisfies if flipped).
    #    e. With prob p (heavy), pick var with max score. Else pick random var in clause.
    #    f. Flip var.
    #    g. If unsat count increases significantly or no improvement for X steps, restart.
    #    
    # To make it "Structurally Different" and stronger:
    # Let's add **Unit Propagation** as a pre-step in each restart and after each flip.
    # If a clause has 2 literals false and 1 true, it's satisfied.
    # If a clause has 2 literals false and 1 unassigned? In complete assignment, all are assigned.
    # So "Unit" in this context means: A clause is unsatisfied, and only ONE of its literals 
    # could potentially be made true by flipping its variable (i.e., the other two are already 
    # "fixed" to their false values by other constraints? No, in WalkSAT we flip one at a time).
    
    # A better structural change: **Simulated Annealing** on the number of unsatisfied clauses.
    # This allows escaping local minima by accepting worse states with a probability that decreases over time.
    
    def run_simulated_annealing():
        nonlocal best_assignment, best_unsat, deadline
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Count unsat
        unsat_count = [0] * m
        total_unsat = 0
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            sat = False
            for k in range(len(vs)):
                v = vs[k]
                s = ss[k]
                if s == 1:
                    if assignment[v]: sat = True; break
                else:
                    if not assignment[v]: sat = True; break
            if not sat:
                unsat_count[ci] = 1
                total_unsat += 1
                
        if total_unsat == 0:
            return assignment
            
        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment[:]
            
        # Simulated Annealing Parameters
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.9995
        T = T_start
        
        # We run for a certain number of steps or time
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random unsat clause
            # Optimization: Keep a list of unsat clause indices? 
            # Rebuilding every step is O(m). m is likely ~420 (4.26 * 150). 
            # 420 * 50000 = 21M ops. Python might be slow.
            # Let's sample. If we sample a sat clause, we just retry or pick next.
            # Better: Maintain a list of unsat clauses. Update it incrementally.
            
            # For simplicity and robustness in Python, let's stick to a slightly optimized WalkSAT
            # but with **Adaptive Restarts** and **Heavy Clause Selection**.
            
            # Find unsat clauses
            unsat_clauses = [ci for ci in range(m) if unsat_count[ci] > 0]
            if not unsat_clauses:
                return assignment
                
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]
                
            if not unsat_clauses:
                return assignment
                
            ci = rng.choice(unsat_clauses)
            vs = cvars[ci]
            ss = csigns[ci]
            
            # Calculate scores for each variable in the clause
            # Score: number of OTHER unsat clauses that would be satisfied if we flip this var
            scores = []
            for v in vs:
                score = 0
                for cj in var_clauses[v]:
                    if cj == ci: continue
                    if unsat_count[cj] == 0: continue
                    # Check if flipping v satisfies cj
                    # Find index of v in cj
                    vs_cj = cvars[cj]
                    ss_cj = csigns[cj]
                    for k in range(len(vs_cj)):
                        if vs_cj[k] == v:
                            s_cj = ss_cj[k]
                            # Current value: assignment[v]
                            # If s_cj == 1, we want True. If assignment[v] is False, flipping to True satisfies.
                            # If s_cj == -1, we want False. If assignment[v] is True, flipping to False satisfies.
                            if s_cj == 1:
                                if not assignment[v]: score += 1
                            else:
                                if assignment[v]: score += 1
                            break
                scores.append(score)
                
            # Pick variable to flip
            # With probability 0.85, pick max score. Else random.
            if rng.random() < 0.85:
                max_score = max(scores)
                # Pick random among those with max score
                candidates = [i for i, s in enumerate(scores) if s == max_score]
                idx = rng.choice(candidates)
            else:
                idx = rng.randint(0, len(vs)-1)
                
            v_to_flip = vs[idx]
            
            # Flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update unsat counts for clauses containing v_to_flip
            for cj in var_clauses[v_to_flip]:
                vs_cj = cvars[cj]
                ss_cj = csigns[cj]
                
                # Determine if cj was sat/unsat before
                was_unsat = unsat_count[cj] > 0
                
                # Determine if cj is sat/unsat after
                # We need to check if any literal in cj is true
                # Optimization: We know v_to_flip changed.
                # If it was unsat, and v_to_flip's literal is now true -> becomes sat.
                # If it was sat, and v_to_flip's literal was the ONLY true one and is now false -> becomes unsat.
                
                # Find index of v_to_flip in cj
                idx_v = -1
                for k in range(len(vs_cj)):
                    if vs_cj[k] == v_to_flip:
                        idx_v = k
                        break
                if idx_v == -1: continue # Should not happen
                
                s_v = ss_cj[idx_v]
                lit_was_true = (old_val and s_v == 1) or (not old_val and s_v == -1)
                lit_now_true = (new_val and s_v == 1) or (not new_val and s_v == -1)
                
                if was_unsat:
                    if lit_now_true:
                        # Now satisfied
                        unsat_count[cj] = 0
                        total_unsat -= 1
                    else:
                        # Still unsat (since it was unsat, no other lit was true, and this one is not true)
                        pass
                else:
                    # Was sat
                    if lit_was_true and not lit_now_true:
                        # We flipped the only true literal?
                        # Check if any other literal is true
                        any_other_true = False
                        for k in range(len(vs_cj)):
                            if k == idx_v: continue
                            v_k = vs_cj[k]
                            s_k = ss_cj[k]
                            if (assignment[v_k] and s_k == 1) or (not assignment[v_k] and s_k == -1):
                                any_other_true = True
                                break
                        if not any_other_true:
                            # Now unsat
                            unsat_count[cj] = 1
                            total_unsat += 1
                    # Else still sat (either this lit was not the only true one, or it's still true, or was false but others true)
            
            if total_unsat == 0:
                return assignment
                
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]
                
        return assignment

    # Main loop: Restarts
    # We use adaptive restarts: if we don't improve, we restart sooner.
    restarts = 0
    while time.time() < deadline:
        if best_unsat == 0:
            break
            
        result = run_simulated_annealing()
        if result is not None:
            # Verify
            if check_all(result):
                return result
        
        restarts += 1
        # If we've done many restarts, maybe increase randomness?
        # For now, just keep going.
        
    if best_assignment is not None:
        # Final check
        if check_all(best_assignment):
            return best_assignment
        # If not satisfied, return it anyway? No, task requires satisfying assignment.
        # If we return an unsat assignment, it's invalid? 
        # "invalid result... scores 0"
        # So we must return a satisfying one or nothing?
        # Usually, if we can't find one, we return the best effort? 
        # But if it's not satisfying, it's 0.
        # Let's try one last desperate random guess? No.
        # Return best_assignment (might be unsat, resulting in 0 for that instance)
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
