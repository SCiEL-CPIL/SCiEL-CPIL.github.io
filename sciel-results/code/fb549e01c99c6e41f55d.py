"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s limit, use 295s for safety
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Preprocess: convert to 0-based, build index structures
    # clause_vars[c] = list of 0-based var indices
    # clause_sigs[c] = list of +1/-1 signs
    clause_vars = []
    clause_sigs = []
    var_to_clauses = [[] for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        vars_list = []
        sigs_list = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_list.append(v)
            sigs_list.append(s)
            var_to_clauses[v].append(c_idx)
        clause_vars.append(vars_list)
        clause_sigs.append(sigs_list)
    
    # WalkSAT with gain heuristic and restarts
    rng = random.Random(42)
    
    def satisfied_clauses(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for c_idx in range(m):
            vs = clause_vars[c_idx]
            ss = clause_sigs[c_idx]
            sat = False
            for i in range(len(vs)):
                if assignment[vs[i]] == (ss[i] > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(c_idx)
        return unsat
    
    def compute_gain(assignment, unsat_clauses):
        """Compute gain for each variable in unsat clauses."""
        gain = [0] * n
        for c_idx in unsat_clauses:
            vs = clause_vars[c_idx]
            ss = clause_sigs[c_idx]
            for i in range(len(vs)):
                v = vs[i]
                s = ss[i]
                # Flipping v would satisfy this clause if it's currently false
                if assignment[v] != (s > 0):
                    # This literal is false, so flipping would satisfy
                    # Count how many other unsat clauses this var appears in
                    gain[v] += 1
        return gain
    
    def walksat_attempt(assignment, max_steps):
        """Run WalkSAT with gain heuristic. Returns (solved, assignment)."""
        for step in range(max_steps):
            if time.time() >= deadline:
                return False, assignment
            
            unsat = satisfied_clauses(assignment)
            if not unsat:
                return True, assignment
            
            # Compute gain
            gain = compute_gain(assignment, unsat)
            
            # Pick a random unsatisfied clause
            c_idx = rng.choice(unsat)
            vs = clause_vars[c_idx]
            
            # Pick variable to flip from this clause
            # Weight by gain
            weights = []
            for v in vs:
                weights.append(max(1, gain[v]))
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_v = vs[0]
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_v = vs[i]
                    break
            
            assignment[chosen_v] = not assignment[chosen_v]
        
        # Final check
        unsat = satisfied_clauses(assignment)
        return (len(unsat) == 0), assignment
    
    # Restart strategy
    best_assignment = None
    best_unsat_count = m + 1
    
    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    max_restarts = 50
    base_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        # Reset to random assignment with some bias toward previous best
        if best_assignment is not None and rng.random() < 0.3:
            assignment = best_assignment[:]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        solved, assignment = walksat_attempt(assignment, base_steps)
        
        if solved:
            return assignment
        
        # Track best
        unsat_count = len(satisfied_clauses(assignment))
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Increase steps slightly each restart
        base_steps = int(base_steps * 1.1)
    
    # If we have a best assignment, return it
    if best_assignment is not None:
        return best_assignment
    
    return assignment
# EVOLVE-BLOCK-END
