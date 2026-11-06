import random
import time
from collections import deque

def solve(instance):
    """
    Solves a planted 3-SAT instance using an optimized WalkSAT algorithm.
    Combines the robust restart strategy with efficient delta evaluation.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocessing: Build efficient data structures
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cv = []
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cv.append((v, sign))
            var_to_clauses[v].append((i, j))
        clause_vars.append(cv)
    
    # Precompute for each variable and each clause it appears in,
    # the indices of the other two literals in that clause.
    var_other_lits = [[] for _ in range(n)]
    
    for v in range(n):
        for (ci, pos) in var_to_clauses[v]:
            others = []
            for p in range(3):
                if p != pos:
                    others.append(p)
            var_other_lits[v].append((ci, others[0], others[1]))
    
    def is_lit_sat(v, s, assignment):
        val = assignment[v]
        return (s == 1 and val) or (s == -1 and not val)
    
    def evaluate(assignment):
        num_unsat = 0
        clause_sat = [False] * C
        for c in range(C):
            cv = clause_vars[c]
            sat = False
            for (v, s) in cv:
                if is_lit_sat(v, s, assignment):
                    sat = True
                    break
            clause_sat[c] = sat
            if not sat:
                num_unsat += 1
        return num_unsat, clause_sat
    
    def get_best_flip(unsat_clause_idx, clause_sat, assignment, rng, w=0.5):
        clause = clause_vars[unsat_clause_idx]
        best_v = -1
        best_score = -float('inf')
        
        for (v, s) in clause:
            delta = 0
            for (ci, pos1, pos2) in var_other_lits[v]:
                v1, s1 = clause_vars[ci][pos1]
                v2, s2 = clause_vars[ci][pos2]
                
                lit1_sat = is_lit_sat(v1, s1, assignment)
                lit2_sat = is_lit_sat(v2, s2, assignment)
                
                if not clause_sat[ci]:
                    if lit1_sat and lit2_sat:
                        delta += 1
                else:
                    if not lit1_sat and not lit2_sat:
                        delta -= 1
            
            # Weighted heuristic: prefer higher delta, but allow random exploration
            # This is a soft version of pure greedy
            if w < 1.0 and rng.random() < w:
                score = rng.random()  # Pure random
            else:
                score = delta + rng.random() * 0.5  # Greedy with small noise
            
            if score > best_score:
                best_score = score
                best_v = v
                
        return best_v
    
    rng = random.Random()
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Try different weights for the heuristic
    weights = [0.5, 0.3, 0.7, 0.1, 0.9]
    rng.shuffle(weights)
    
    restart_count = 0
    max_steps_base = 20000
    
    while time.time() < deadline:
        w = weights[restart_count % len(weights)]
        max_steps = max_steps_base + restart_count * 1000
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        num_unsat, clause_sat = evaluate(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment[:]
        
        steps = 0
        while steps < max_steps and time.time() < deadline:
            unsat_clause_idx = -1
            for c in range(C):
                if not clause_sat[c]:
                    unsat_clause_idx = c
                    break
            
            if unsat_clause_idx == -1:
                return assignment
            
            v = get_best_flip(unsat_clause_idx, clause_sat, assignment, rng, w)
            
            if v == -1:
                break
            
            assignment[v] = not assignment[v]
            num_unsat, clause_sat = evaluate(assignment)
            
            if num_unsat == 0:
                return assignment
            
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment[:]
                
            steps += 1
        
        restart_count += 1
    
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
