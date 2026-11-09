"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def dpll_with_unit_propagation(n, clauses):
    """
    Try DPLL with unit propagation. Returns assignment if solved, None if failed.
    """
    # Build index
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for lit in c:
            var_to_clauses[abs(lit) - 1].append(i)
    
    # Work with mutable structures
    active_clauses = list(clauses)  # list of lists of literals
    assignment = {}  # var_idx -> bool
    
    def propagate():
        """Unit propagation. Returns False if conflict, True if consistent."""
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in active_clauses:
                # Check if clause is satisfied
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    if v in assignment:
                        if (assignment[v] and lit > 0) or (not assignment[v] and lit < 0):
                            sat = True
                            break
                if sat:
                    continue
                
                # Check if clause has a unit literal
                unit_lit = None
                unassigned = []
                for lit in c:
                    v = abs(lit) - 1
                    if v in assignment:
                        # Already assigned and not satisfying (checked above)
                        return False
                    else:
                        unassigned.append(lit)
                        if len(unassigned) == 1:
                            unit_lit = lit
                
                if len(unassigned) == 0:
                    return False  # Empty clause, conflict
                
                if len(unassigned) == 1:
                    # Unit clause
                    lit = unassigned[0]
                    v = abs(lit) - 1
                    val = (lit > 0)
                    if v in assignment:
                        if assignment[v] != val:
                            return False
                    else:
                        assignment[v] = val
                        changed = True
                        # Don't add to new_clauses, it's now satisfied
                    continue
                
                new_clauses.append(unassigned)
            
            active_clauses[:] = new_clauses
            return True
    
    def dpll(depth=0):
        if not propagate():
            return None
        
        if not active_clauses:
            # All clauses satisfied
            result = [assignment.get(i, False) for i in range(n)]
            return result
        
        # Choose a variable to branch on
        # Pick variable with most occurrences in remaining clauses
        var_counts = [0] * n
        for c in active_clauses:
            for lit in c:
                var_counts[abs(lit) - 1] += 1
        
        # Pick unassigned variable with highest count
        best_var = -1
        best_count = -1
        for v in range(n):
            if v not in assignment and var_counts[v] > best_count:
                best_count = var_counts[v]
                best_var = v
        
        if best_var == -1:
            # All variables assigned, check if all clauses satisfied
            for c in active_clauses:
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    if (assignment.get(v, False) and lit > 0) or (not assignment.get(v, True) and lit < 0):
                        sat = True
                        break
                if not sat:
                    return None
            return [assignment.get(i, False) for i in range(n)]
        
        # Try True
        assignment[best_var] = True
        saved_clauses = [c[:] for c in active_clauses]
        saved_assignment = dict(assignment)
        
        result = dpll(depth + 1)
        if result is not None:
            return result
        
        # Backtrack
        assignment.clear()
        assignment.update(saved_assignment)
        del assignment[best_var]
        active_clauses[:] = [c[:] for c in saved_clauses]
        
        # Try False
        assignment[best_var] = False
        saved_clauses2 = [c[:] for c in active_clauses]
        saved_assignment2 = dict(assignment)
        
        result = dpll(depth + 1)
        if result is not None:
            return result
        
        # Backtrack
        assignment.clear()
        assignment.update(saved_assignment2)
        del assignment[best_var]
        active_clauses[:] = [c[:] for c in saved_clauses2]
        
        return None
    
    return dpll()

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # 1. Pre-process
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_signs = []
    
    for i, c in enumerate(clauses):
        cv = []
        cs = []
        for lit in c:
            v = abs(lit) - 1
            s = (lit > 0)
            cv.append(v)
            cs.append(s)
            var_to_clauses[v].append(i)
        clause_vars.append(cv)
        clause_signs.append(cs)
        
    num_clauses = len(clauses)
    
    # Phase initialization
    pos_counts = np.zeros(n, dtype=np.int32)
    neg_counts = np.zeros(n, dtype=np.int32)
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    phase = pos_counts > neg_counts
    phase_list = phase.tolist()
    
    # Try DPLL first for easy instances
    start_time = time.time()
    dpll_deadline = start_time + 10.0  # Give DPLL 10 seconds
    
    # Simple DPLL with time limit
    def dpll_limited(n, clauses, deadline):
        var_to_clauses_local = [[] for _ in range(n)]
        for i, c in enumerate(clauses):
            for lit in c:
                var_to_clauses_local[abs(lit) - 1].append(i)
        
        active_clauses = [list(c) for c in clauses]
        assignment = {}
        
        def propagate():
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for c in active_clauses:
                    sat = False
                    for lit in c:
                        v = abs(lit) - 1
                        if v in assignment:
                            if (assignment[v] and lit > 0) or (not assignment[v] and lit < 0):
                                sat = True
                                break
                    if sat:
                        continue
                    
                    unassigned = []
                    for lit in c:
                        v = abs(lit) - 1
                        if v not in assignment:
                            unassigned.append(lit)
                    
                    if len(unassigned) == 0:
                        return False
                    
                    if len(unassigned) == 1:
                        lit = unassigned[0]
                        v = abs(lit) - 1
                        val = (lit > 0)
                        if v in assignment:
                            if assignment[v] != val:
                                return False
                        else:
                            assignment[v] = val
                            changed = True
                        continue
                    
                    new_clauses.append(unassigned)
                
                active_clauses[:] = new_clauses
                return True
        
        def dpll(depth=0):
            if time.time() > deadline:
                return None
            
            if not propagate():
                return None
            
            if not active_clauses:
                return [assignment.get(i, False) for i in range(n)]
            
            var_counts = [0] * n
            for c in active_clauses:
                for lit in c:
                    var_counts[abs(lit) - 1] += 1
            
            best_var = -1
            best_count = -1
            for v in range(n):
                if v not in assignment and var_counts[v] > best_count:
                    best_count = var_counts[v]
                    best_var = v
            
            if best_var == -1:
                for c in active_clauses:
                    sat = False
                    for lit in c:
                        v = abs(lit) - 1
                        if (assignment.get(v, False) and lit > 0) or (not assignment.get(v, True) and lit < 0):
                            sat = True
                            break
                    if not sat:
                        return None
                return [assignment.get(i, False) for i in range(n)]
            
            assignment[best_var] = True
            saved_clauses = [c[:] for c in active_clauses]
            saved_assignment = dict(assignment)
            
            result = dpll(depth + 1)
            if result is not None:
                return result
            
            assignment.clear()
            assignment.update(saved_assignment)
            del assignment[best_var]
            active_clauses[:] = [c[:] for c in saved_clauses]
            
            assignment[best_var] = False
            saved_clauses2 = [c[:] for c in active_clauses]
            saved_assignment2 = dict(assignment)
            
            result = dpll(depth + 1)
            if result is not None:
                return result
            
            assignment.clear()
            assignment.update(saved_assignment2)
            del assignment[best_var]
            active_clauses[:] = [c[:] for c in saved_clauses2]
            
            return None
        
        return dpll(0)
    
    dpll_result = dpll_limited(n, clauses, dpll_deadline)
    if dpll_result is not None:
        # Verify
        all_sat = True
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (dpll_result[v] and lit > 0) or (not dpll_result[v] and lit < 0):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return dpll_result
    
    # Fall back to WalkSAT with clause weights and adaptive noise
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Initialize clause weights
    clause_weights = np.ones(num_clauses, dtype=np.float64)
    
    best_assignment = phase_list.copy()
    best_unsat_count = float('inf')
    
    max_restarts = 200
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = phase_list.copy()
        
        # Compute initial unsat clauses and count
        unsat_clauses = []
        unsat_count = 0
        for i in range(num_clauses):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = False
            for v, s in zip(cv, cs):
                if assignment[v] == s:
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(i)
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # Adaptive noise: start high, decrease over restarts
        noise_prob = 0.25 * (0.9 ** restart) + 0.05
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assignment
                
            if unsat_count == 0:
                return assignment
                
            # Pick an unsat clause with probability proportional to weight
            # Use weighted random selection
            weights = np.array([clause_weights[i] for i in unsat_clauses], dtype=np.float64)
            total_weight = weights.sum()
            if total_weight <= 0:
                clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            else:
                probs = weights / total_weight
                r = rng.random()
                cumulative = 0.0
                clause_idx = unsat_clauses[-1]
                for i, p in enumerate(probs):
                    cumulative += p
                    if r <= cumulative:
                        clause_idx = unsat_clauses[i]
                        break
            
            cv = clause_vars[clause_idx]
            cs = clause_signs[clause_idx]
            
            # Evaluate flip for each literal in this clause
            best_var = -1
            best_delta = float('inf')
            
            for k in range(3):
                v = cv[k]
                delta = 0
                for c_idx in var_to_clauses[v]:
                    c_cv = clause_vars[c_idx]
                    c_cs = clause_signs[c_idx]
                    
                    v_sign = c_cs[c_cv.index(v)]
                    v_current_val = assignment[v]
                    v_satisfies_now = (v_current_val == v_sign)
                    
                    other_sat = False
                    for j in range(3):
                        if c_cv[j] != v:
                            if assignment[c_cv[j]] == c_cs[j]:
                                other_sat = True
                                break
                                
                    if not other_sat:
                        if v_satisfies_now:
                            delta += clause_weights[c_idx]
                        else:
                            delta -= clause_weights[c_idx]
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                elif delta == best_delta and rng.random() < 0.5:
                    best_var = v
            
            # Adaptive noise: with probability noise_prob, pick random var from clause
            if rng.random() < noise_prob:
                best_var = cv[rng.randint(0, 2)]
            
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update clause weights and unsat count
            # Clauses affected by flipping best_var
            for c_idx in var_to_clauses[best_var]:
                c_cv = clause_vars[c_idx]
                c_cs = clause_signs[c_idx]
                
                # Check old status
                old_sat = False
                for v, s in zip(c_cv, c_cs):
                    if (old_val and s) or (not old_val and not s) if v == best_var else (assignment[v] and s) or (not assignment[v] and not s):
                        old_sat = True
                        break
                # Actually, let's just recompute properly
                old_sat = False
                for v, s in zip(c_cv, c_cs):
                    if v == best_var:
                        if old_val == s:
                            old_sat = True
                            break
                    else:
                        if assignment[v] == s:
                            old_sat = True
                            break
                
                # Check new status
                new_sat = False
                for v, s in zip(c_cv, c_cs):
                    if v == best_var:
                        if (not old_val) == s:
                            new_sat = True
                            break
                    else:
                        if assignment[v] == s:
                            new_sat = True
                            break
                
                if old_sat and not new_sat:
                    # Clause became unsat
                    unsat_count += 1
                    clause_weights[c_idx] *= 1.5
                elif not old_sat and new_sat:
                    # Clause became sat
                    unsat_count -= 1
                    clause_weights[c_idx] *= 0.7
                elif not old_sat and not new_sat:
                    # Still unsat, increase weight
                    clause_weights[c_idx] *= 1.1
                else:
                    # Still sat, decrease weight
                    clause_weights[c_idx] *= 0.9
            
            # Rebuild unsat_clauses list
            unsat_clauses = []
            for i in range(num_clauses):
                cv_i = clause_vars[i]
                cs_i = clause_signs[i]
                sat = False
                for v, s in zip(cv_i, cs_i):
                    if assignment[v] == s:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
        
        # Decay clause weights after each restart
        clause_weights *= 0.8
    
    return best_assignment
# EVOLVE-BLOCK-END
