import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 330s limit, leave 50s margin
    
    # Pre-process: store (var_idx, sign) where sign is 1 for pos, -1 for neg?
    # Or just boolean is_pos.
    # Let's use var_idx and is_pos.
    proc_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            pc.append((v, p))
        proc_clauses.append(pc)
        
    m = len(proc_clauses)
    
    best_assignment = [False] * n
    best_score = m # max possible unsat
    
    # Heuristic parameters
    p_heuristic = 0.5
    steps_per_restart = 2000
    max_restarts = 100 # 100 * 2000 = 200k steps. 
    # 200k steps * O(M) check. If M=100, 20M ops. ~1-2 seconds.
    # We have 280s. We can afford much more.
    # Let's use time-based loop.
    
    while time.time() < deadline:
        # Random init
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # WalkSAT
        for _ in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Find unsat clauses
            unsat_clauses = []
            for i, c in enumerate(proc_clauses):
                sat = False
                for v, p in c:
                    if assignment[v] == p:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
                    
            if not unsat_clauses:
                return assignment
                
            # Pick random unsat clause
            ci = random.choice(unsat_clauses)
            clause = proc_clauses[ci]
            
            # Heuristic: Min-conflicts
            # For each literal in this clause, count how many OTHER unsat clauses are falsified by flipping this literal.
            # Flipping (v, p) means setting assignment[v] to p.
            # A clause j (j != ci) is falsified by this flip if:
            #   1. Clause j is currently unsat.
            #   2. Clause j contains the literal (v, not p). 
            #      Because if it contains (v, p), setting v to p would satisfy it (or keep it satisfied if other lit was true, but it's unsat so no other lit is true).
            #      Wait, if clause j is unsat, all its literals are false.
            #      If we set v to p, and clause j has (v, p), then that literal becomes true, so clause j becomes satisfied.
            #      So flipping (v, p) SATISFIES clause j if j contains (v, p).
            #      It FALSIFIES clause j if j contains (v, not p)?
            #      No. If j contains (v, not p), and we set v to p, then (v, not p) becomes false. 
            #      Since j was already unsat (all false), it remains unsat.
            #      So the "badness" is the number of unsat clauses that contain the literal (v, not p).
            #      Because those clauses *stay* unsat, while clauses containing (v, p) *become* satisfied.
            #      Actually, we want to minimize the number of clauses that *become* unsat.
            #      Currently, all unsat_clauses are unsat.
            #      After flip:
            #      - Clause ci becomes sat.
            #      - Any clause j in unsat_clauses that contains (v, p) becomes sat.
            #      - Any clause j in unsat_clauses that contains (v, not p) remains unsat.
            #      - Any clause j NOT in unsat_clauses (i.e., sat) that contains (v, not p) becomes unsat!
            
            # So the cost is:
            # (Number of currently SAT clauses that contain (v, not p)) 
            # - (Number of currently UNSAT clauses that contain (v, p) and are not ci)
            
            # This is expensive to calculate from scratch.
            # Approximation: Just count how many unsat clauses contain (v, not p).
            # This is the "min-conflicts" heuristic often used in simplified implementations.
            
            best_var = -1
            best_badness = float('inf')
            
            for v, p in clause:
                badness = 0
                not_p = not p
                for j in unsat_clauses:
                    if j == ci: continue
                    cj = proc_clauses[j]
                    # Check if cj contains (v, not_p)
                    for vv, pp in cj:
                        if vv == v and pp == not_p:
                            badness += 1
                            break
                            
                if badness < best_badness:
                    best_badness = badness
                    best_var = v
                    best_p = p
                    
            # With prob p_heuristic, use random flip instead
            if random.random() < p_heuristic:
                v, p = random.choice(clause)
            else:
                v, p = best_var, best_p
                
            assignment[v] = p
            
        # End of restart
        # Update best
        score = 0
        for c in proc_clauses:
            if not any(assignment[v] == p for v, p in c):
                score += 1
        if score < best_score:
            best_score = score
            best_assignment = assignment[:]
            
    return best_assignment
