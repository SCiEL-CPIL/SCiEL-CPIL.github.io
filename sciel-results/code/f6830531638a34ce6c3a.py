"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DEADLINE_SECONDS = 280.0  # safety margin before 330s kill


def _satisfied_fast(assignment, clause):
    """Check if clause is satisfied using numpy vectorized ops."""
    idx = np.abs(clause) - 1
    vals = assignment[idx]
    signs = (clause > 0)
    return np.any(vals == signs)


def _walksat_with_tracking(n, clauses, rng, max_flips, deadline, init_assignment=None):
    """
    WalkSAT with clause-count tracking for O(degree) updates per flip.
    Returns (satisfied, assignment).
    """
    if init_assignment is not None:
        assignment = init_assignment.copy()
    else:
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    
    m = len(clauses)
    
    # Precompute for each clause: indices and signs
    clause_idx = np.array([np.abs(c) - 1 for c in clauses])  # shape (m, 3)
    clause_signs = np.array([c > 0 for c in clauses])  # shape (m, 3)
    
    # For each variable, track which clauses it appears in and with what sign
    # var_clauses[var] = list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = clause_idx[ci, j]
            var_clauses[v].append((ci, clause_signs[ci, j]))
    
    # Count satisfied literals per clause
    # sat_count[ci] = number of satisfied literals in clause ci
    sat_count = np.zeros(m, dtype=np.int32)
    for ci in range(m):
        for j in range(3):
            v = clause_idx[ci, j]
            if assignment[v] == clause_signs[ci, j]:
                sat_count[ci] += 1
    
    # unsat set: clauses with sat_count == 0
    # We'll maintain a list/set of unsat clause indices
    
    def get_unsat():
        return np.where(sat_count == 0)[0]
    
    # Track unsat clauses
    unsat_set = set(get_unsat())
    
    if not unsat_set:
        return True, assignment
    
    start_time = time.time()
    
    for flip_i in range(max_flips):
        if time.time() - start_time > deadline:
            break
        
        if not unsat_set:
            break
        
        # Pick a random unsatisfied clause
        ci = rng.choice(list(unsat_set))
        
        # Choose which literal to flip in this clause
        # Use a weighted scheme: prefer literals that are "close" to being satisfied
        # Simple: pick random literal from the clause
        j = rng.randint(0, 2)
        v = clause_idx[ci, j]
        
        # Flip variable v
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        # Update sat_count for all clauses containing v
        for (ci2, sign2) in var_clauses[v]:
            # Before flip: was this literal satisfied?
            was_sat = (old_val == sign2)
            # After flip: is this literal satisfied?
            is_sat = (new_val == sign2)
            
            if was_sat and not is_sat:
                # Lost a satisfied literal
                old_count = sat_count[ci2]
                sat_count[ci2] = old_count - 1
                if old_count == 1:
                    # Now unsat
                    unsat_set.add(ci2)
            elif not was_sat and is_sat:
                # Gained a satisfied literal
                old_count = sat_count[ci2]
                sat_count[ci2] = old_count + 1
                if old_count == 0:
                    # Now sat
                    unsat_set.discard(ci2)
            # If both sat or both unsat, no change
    
    if not unsat_set:
        return True, assignment
    
    # Check if solved
    remaining = get_unsat()
    if len(remaining) == 0:
        return True, assignment
    
    return False, assignment


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = DEADLINE_SECONDS
    
    # Multiple restarts with different seeds
    # Adaptive: use more restarts if instances are easier
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Number of flips per restart - adaptive based on time
    # Start with a moderate number, can do multiple restarts
    base_flips = 50000
    
    restart = 0
    while time.time() - start_time < deadline:
        restart += 1
        seed = restart * 12345 + 7
        rng = random.Random(seed)
        
        # For each restart, use a portion of the remaining time
        remaining_time = deadline - (time.time() - start_time)
        # Reserve a small buffer
        if remaining_time < 2.0:
            break
        
        # Scale flips with remaining time (roughly 10k flips/sec in Python)
        # But with tracking, it's faster. Let's be conservative.
        max_flips = int(min(base_flips, remaining_time * 20000))
        if max_flips < 1000:
            max_flips = 1000
        
        satisfied, assignment = _walksat_with_tracking(
            n, clauses, rng, max_flips, remaining_time - 1.0
        )
        
        if satisfied:
            return assignment.tolist()
        
        # Track best
        # Count unsat clauses
        unsat_count = sum(
            1 for c in clauses
            if not any(assignment[abs(l) - 1] == (l > 0) for l in c)
        )
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # If we're getting close to deadline, do a final quick check
        if time.time() - start_time > deadline - 5.0:
            break
    
    # Return best found (should always be a list of booleans)
    if best_assignment is not None:
        return best_assignment.tolist()
    
    # Fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
