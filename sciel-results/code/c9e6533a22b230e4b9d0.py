# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based representation for internal use
    # Each clause: list of (var, sign) where sign=1 means positive literal
    m = len(clauses)
    
    # Build internal representation: clauses as lists of (var, sign)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            cv.append((var, sign))
        clause_vars.append(cv)
    
    # Build variable-to-clause index for fast lookup
    var_clauses = defaultdict(list)
    for ci, cv in enumerate(clause_vars):
        for var, sign in cv:
            var_clauses[var].append(ci)
    
    # === PHASE 1: Vectorized Preprocessing ===
    # Track active clauses and assignments
    assignment = [None] * n  # None = unassigned
    clause_active = [True] * m
    clause_satisfied = [False] * m
    
    # Pure literal elimination with unit propagation
    # Use a queue-based approach for efficiency
    from collections import deque
    queue = deque()
    
    # Initial: check for pure literals
    var_polarity = {}  # var -> 'pos', 'neg', or 'both'
    for var in range(n):
        pos = False
        neg = False
        for ci in var_clauses[var]:
            if clause_active[ci]:
                # Find the sign in this clause
                for v, s in clause_vars[ci]:
                    if v == var:
                        if s == 1:
                            pos = True
                        else:
                            neg = True
                        break
        if pos and not neg:
            var_polarity[var] = 'pos'
        elif neg and not pos:
            var_polarity[var] = 'neg'
        else:
            var_polarity[var] = 'both'
    
    # Apply pure literals
    changed = True
    while changed:
        changed = False
        for var in range(n):
            if assignment[var] is not None:
                continue
            pol = var_polarity.get(var, 'both')
            if pol == 'pos':
                # Only appears positive, set to False to NOT satisfy... wait
                # If only appears positive, setting to False makes all those literals false
                # We want to set to satisfy: set to True
                assignment[var] = True
                changed = True
                _propagate_var(var, True, n, m, clause_vars, clause_active, clause_satisfied, assignment, var_clauses, var_polarity, queue)
            elif pol == 'neg':
                # Only appears negative, set to True to satisfy (negative literal satisfied when var is True)
                assignment[var] = True
                changed = True
                _propagate_var(var, True, n, m, clause_vars, clause_active, clause_satisfied, assignment, var_clauses, var_polarity, queue)
    
    # Process unit clauses from queue
    while queue:
        ci = queue.popleft()
        if not clause_active[ci] or clause_satisfied[ci]:
            continue
        # Find unit clause
        unassigned_lits = []
        for var, sign in clause_vars[ci]:
            if assignment[var] is None:
                unassigned_lits.append((var, sign))
            elif assignment[var] == (sign == 1):
                clause_satisfied[ci] = True
                clause_active[ci] = False
                break
        if not clause_active[ci]:
            continue
        if len(unassigned_lits) == 1:
            var, sign = unassigned_lits[0]
            if assignment[var] is None:
                assignment[var] = (sign == 1)
                _propagate_var(var, assignment[var], n, m, clause_vars, clause_active, clause_satisfied, assignment, var_clauses, var_polarity, queue)
    
    # Check if all clauses are satisfied
    all_sat = True
    for ci in range(m):
        if clause_active[ci] and not clause_satisfied[ci]:
            all_sat = False
            break
    
    if all_sat:
        final_assignment = []
        rng = random.Random(0)
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment
    
    # === PHASE 2: WalkSAT on reduced problem ===
    # Build reduced clause list (only active, unsatisfied clauses)
    reduced_clauses = []
    reduced_indices = []
    for ci in range(m):
        if clause_active[ci] and not clause_satisfied[ci]:
            reduced_clauses.append(clause_vars[ci])
            reduced_indices.append(ci)
    
    # Initialize WalkSAT assignment
    rng = random.Random(42)
    ws_assignment = [None] * n
    for var in range(n):
        if assignment[var] is not None:
            ws_assignment[var] = assignment[var]
        else:
            ws_assignment[var] = rng.random() < 0.5
    
    # Build efficient clause structure for WalkSAT
    # For each reduced clause, store vars and signs
    m_red = len(reduced_clauses)
    red_clause_vars = []
    for cv in reduced_clauses:
        red_clause_vars.append(cv)
    
    # Build var-to-reduced-clause index
    red_var_clauses = defaultdict(list)
    for rci, cv in enumerate(red_clause_vars):
        for var, sign in cv:
            red_var_clauses[var].append(rci)
    
    # Count unsatisfied
    def is_satisfied_fast(rci):
        cv = red_clause_vars[rci]
        for var, sign in cv:
            if ws_assignment[var] == (sign == 1):
                return True
        return False
    
    unsat_count = 0
    unsat_clauses = []
    for rci in range(m_red):
        if not is_satisfied_fast(rci):
            unsat_count += 1
            unsat_clauses.append(rci)
    
    # WalkSAT with dynamic noise
    max_flips = 80000
    noise_start = 0.30
    noise_end = 0.03
    
    for flip in range(max_flips):
        if unsat_count == 0:
            break
        
        # Dynamic noise
        progress = flip / max_flips
        noise = noise_start + (noise_end - noise_start) * progress
        
        # Select unsatisfied clause (prefer shorter ones)
        if len(unsat_clauses) > 0:
            # Weight by inverse length
            weights = [1.0 / len(red_clause_vars[rci]) for rci in unsat_clauses]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_rci = unsat_clauses[0]
            for idx, rci in enumerate(unsat_clauses):
                cumulative += weights[idx]
                if r <= cumulative:
                    chosen_rci = rci
                    break
        else:
            break
        
        clause = red_clause_vars[chosen_rci]
        
        # Choose variable to flip
        if rng.random() < noise:
            # Random flip
            var, sign = clause[rng.randrange(len(clause))]
        else:
            # Greedy: maximize gain
            best_gain = -1
            best_var = clause[0][0]
            for var, sign in clause:
                current_val = ws_assignment[var]
                new_val = not current_val
                gain = 0
                # Count how many unsatisfied clauses this flip would satisfy
                for rci2 in red_var_clauses[var]:
                    if is_satisfied_fast(rci2):
                        continue
                    cv2 = red_clause_vars[rci2]
                    for v2, s2 in cv2:
                        if v2 == var and new_val == (s2 == 1):
                            gain += 1
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            var = best_var
        
        # Flip
        old_val = ws_assignment[var]
        ws_assignment[var] = not old_val
        
        # Update unsat count
        new_unsat = 0
        new_unsat_clauses = []
        for rci in range(m_red):
            if not is_satisfied_fast(rci):
                new_unsat += 1
                new_unsat_clauses.append(rci)
        unsat_count = new_unsat
        unsat_clauses = new_unsat_clauses
    
    return ws_assignment


def _propagate_var(var, value, n, m, clause_vars, clause_active, clause_satisfied, assignment, var_clauses, var_polarity, queue):
    """Propagate the effect of assigning var=value"""
    for ci in var_clauses[var]:
        if not clause_active[ci]:
            continue
        if clause_satisfied[ci]:
            continue
        # Check if this clause is now satisfied
        for v, s in clause_vars[ci]:
            if assignment[v] is not None:
                if assignment[v] == (s == 1):
                    clause_satisfied[ci] = True
                    clause_active[ci] = False
                    break
        if clause_satisfied[ci]:
            continue
        # Check if this becomes a unit clause
        unassigned = []
        for v, s in clause_vars[ci]:
            if assignment[v] is None:
                unassigned.append((v, s))
        if len(unassigned) == 1:
            queue.append(ci)
    
    # Update polarity for other variables
    for ci in var_clauses[var]:
        if not clause_active[ci]:
            continue
        for v, s in clause_vars[ci]:
            if v == var:
                continue
            # This clause still has v, so v is not pure in this polarity
            # We need to recheck, but for simplicity we skip full recheck
            pass


# EVOLVE-BLOCK-END
