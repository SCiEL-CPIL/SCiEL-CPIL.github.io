import random
import time
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses
    # clause_lits[i] = list of (var, is_positive) for clause i
    # var_to_clauses[var] = list of (clause_idx, is_positive) for each occurrence of var
    clause_lits = []
    var_to_clauses = defaultdict(list)
    
    for i, clause in enumerate(clauses):
        cl = []
        for lit in clause:
            var = abs(lit) - 1
            is_pos = lit > 0
            cl.append((var, is_pos))
            var_to_clauses[var].append((i, is_pos))
        clause_lits.append(cl)
        
    m = len(clauses)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    max_steps_per_run = 10000
    noise_prob = 0.3
    
    def is_clause_sat(ci, assign):
        for var, is_pos in clause_lits[ci]:
            if assign[var] == is_pos:
                return True
        return False
    
    def compute_unsat(assign):
        return sum(0 if is_clause_sat(i, assign) else 1 for i in range(m))
    
    def init_conflict_counts(assign):
        """
        conflict_counts[var] = number of unsatisfied clauses that contain var
        """
        cc = [0] * n
        for i in range(m):
            if not is_clause_sat(i, assign):
                for var, _ in clause_lits[i]:
                    cc[var] += 1
        return cc
    
    def update_conflict_after_flip(var, old_val, assign, conflict_counts, unsat_set):
        """
        After flipping var from old_val to new_val, update conflict_counts and unsat_set.
        Returns delta_unsat (change in number of unsatisfied clauses).
        """
        new_val = not old_val
        delta = 0
        
        for ci, is_pos in var_to_clauses[var]:
            # Determine old status of clause ci
            # A clause is unsat if no literal is satisfied
            # With old_val, this literal contributed: old_val == is_pos
            
            # Check if clause was satisfied by other literals (before flip)
            # and will be satisfied by other literals (after flip)
            
            # Old: was this clause sat?
            # If old_val == is_pos, this literal satisfied it.
            # Otherwise, check other literals.
            
            # Let's determine: was clause ci unsat before?
            was_unsat = True
            for v, s in clause_lits[ci]:
                if v == var:
                    if old_val == s:
                        was_unsat = False
                        break
                else:
                    if assign[v] == s:
                        was_unsat = False
                        break
            
            # After flip: will clause ci be sat?
            now_sat = False
            for v, s in clause_lits[ci]:
                if v == var:
                    if new_val == s:
                        now_sat = True
                        break
                else:
                    if assign[v] == s:
                        now_sat = True
                        break
            
            if was_unsat and now_sat:
                # Became satisfied
                delta -= 1
                # Remove from unsat_set
                unsat_set.discard(ci)
                # Update conflict counts for all vars in this clause
                for v, s in clause_lits[ci]:
                    conflict_counts[v] -= 1
            elif not was_unsat and not now_sat:
                # Became unsatisfied
                delta += 1
                unsat_set.add(ci)
                for v, s in clause_lits[ci]:
                    conflict_counts[v] += 1
            # else: no change in status, conflict counts unchanged
        
        return delta
    
    while time.time() < deadline:
        # Random initialization
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        current_unsat = compute_unsat(assignment)
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Initialize tracking
        conflict_counts = init_conflict_counts(assignment)
        unsat_set = set()
        for i in range(m):
            if not is_clause_sat(i, assignment):
                unsat_set.add(i)
        
        steps = 0
        last_improvement = 0
        min_unsat_in_run = current_unsat
        
        while steps < max_steps_per_run and time.time() < deadline:
            if current_unsat == 0:
                return assignment
            
            if current_unsat < min_unsat_in_run:
                min_unsat_in_run = current_unsat
                last_improvement = steps
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
            
            if steps - last_improvement > 2000:
                break
            
            if not unsat_set:
                return assignment
            
            # Pick a random unsatisfied clause
            target_clause = random.choice(list(unsat_set))
            
            # Min-conflicts heuristic: pick variable with minimum conflict count
            best_var = None
            best_conflicts = float('inf')
            
            for var, is_pos in clause_lits[target_clause]:
                if conflict_counts[var] < best_conflicts:
                    best_conflicts = conflict_counts[var]
                    best_var = var
            
            # Decide which variable to flip
            if random.random() < noise_prob:
                var_to_flip = best_var
            else:
                var_to_flip = random.choice([v for v, s in clause_lits[target_clause]])
            
            # Flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Update incrementally
            delta = update_conflict_after_flip(var_to_flip, old_val, assignment, conflict_counts, unsat_set)
            current_unsat += delta
            steps += 1
        
    return best_assignment
