# EVOLVE-BLOCK-START
import time
import random
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Deadline: 325 seconds to be safe against the 330s kill
    deadline = time.time() + 325.0
    
    # Preprocess clauses to 0-based indexing
    # Clause: list of (var, sign) where sign is 1 for +, -1 for -
    # var is 0 to n-1
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            pc.append((v, s))
        processed_clauses.append(pc)
    
    m = len(processed_clauses)
    
    # Data structures for DPLL
    # assignment: -1 (unassigned), 0 (False), 1 (True)
    assignment = [-1] * n
    
    # Map variable to list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(processed_clauses):
        for v, s in clause:
            var_to_clauses[v].append((c_idx, s))
    
    # State: list of active clause indices
    # We will use a stack-based approach or recursive with backtracking.
    # Given N=50, recursion depth is safe.
    
    # To speed up, we'll use a "watched literals" style or simple unit propagation loop.
    # Simple Unit Propagation:
    # 1. Find unit clauses (clauses with exactly 1 unassigned literal).
    # 2. Assign that literal.
    # 3. Repeat.
    # 4. If conflict (clause with 0 unassigned and all False), backtrack.
    
    # Pure Literal Elimination:
    # If var appears only as + in all active clauses, assign True.
    # If var appears only as - in all active clauses, assign False.
    
    # We will implement a DPLL that modifies the clause list (or uses a state stack).
    # For simplicity and speed in Python, we will maintain a list of "active" clauses 
    # and their current satisfaction status dynamically.
    
    # However, modifying the clause list in-place during recursion is messy.
    # Instead, we will use a "trail" for backtracking and re-evaluate unit clauses.
    
    # Optimized DPLL with Unit Propagation
    
    def dpll(active_clauses):
        """
        active_clauses: list of lists of (var, sign) that are not yet satisfied.
        Returns: (success, assignment)
        """
        # 1. Check for empty clause (conflict)
        for clause in active_clauses:
            if len(clause) == 0:
                return False, None
        
        # 2. Check if all clauses are satisfied (i.e., all are empty? No, satisfied clauses are removed)
        # If active_clauses is empty, we found a solution.
        if not active_clauses:
            return True, assignment[:]
        
        # 3. Unit Propagation
        # Find all unit clauses (length 1)
        # We need to be careful: if multiple unit clauses assign different values to the same var, conflict.
        # Also, if a unit clause assigns a var, it might satisfy other clauses.
        
        # Loop until no more unit clauses
        unit_clauses_found = True
        while unit_clauses_found:
            unit_clauses_found = False
            # Collect unit assignments
            # Note: If we have a clause [ (v, s) ], we must assign assignment[v] = (s==1)
            
            # We'll iterate through active_clauses to find units
            # To avoid modifying list while iterating, we'll identify units first
            # But assignments change the status of other clauses.
            
            # Efficient approach:
            # 1. Identify all unit clauses.
            # 2. Check for conflicts among them (same var, different sign).
            # 3. Apply all assignments.
            # 4. Remove satisfied clauses and update remaining clauses.
            
            units = []
            for clause in active_clauses:
                if len(clause) == 1:
                    units.append(clause[0])
            
            if not units:
                break
                
            unit_clauses_found = True
            
            # Check for conflicts in units
            # If var v appears in units with sign s1 and s2 where s1 != s2, conflict
            var_assignments = {}
            conflict = False
            for v, s in units:
                if v in var_assignments:
                    if var_assignments[v] != s:
                        conflict = True
                        break
                else:
                    var_assignments[v] = s
            
            if conflict:
                return False, None
            
            # Apply assignments
            for v, s in var_assignments.items():
                assignment[v] = 1 if s == 1 else 0
                
            # Update active_clauses
            # Remove clauses satisfied by these assignments
            # Update remaining clauses by removing the assigned literals
            new_active_clauses = []
            for clause in active_clauses:
                new_clause = []
                satisfied = False
                for v, s in clause:
                    if assignment[v] != -1:
                        # If assigned, check if satisfied
                        if (assignment[v] == 1 and s == 1) or (assignment[v] == 0 and s == -1):
                            satisfied = True
                            break
                        else:
                            # Assigned but not satisfied (False literal), remove from clause
                            continue
                    else:
                        # Unassigned, keep
                        new_clause.append((v, s))
                
                if not satisfied:
                    new_active_clauses.append(new_clause)
            
            active_clauses = new_active_clauses
            
            # If any clause became empty during update, it's a conflict
            for clause in active_clauses:
                if len(clause) == 0:
                    return False, None

        # 4. Pure Literal Elimination (Optional but helpful)
        # If a variable appears only with one sign in all active clauses, assign it.
        # This is safe and reduces problem size.
        pure_vars = {}
        for v in range(n):
            if assignment[v] != -1:
                continue
            
            signs = set()
            for c_idx, s in var_to_clauses[v]:
                # We need to know if this occurrence is in an active clause
                # This is expensive to check dynamically. 
                # Alternative: Just skip pure literal elimination for simplicity 
                # and rely on unit propagation + heuristic variable selection.
                # Given N=50, unit propagation is usually strong enough.
                pass
        
        # 5. Variable Selection Heuristic
        # Pick a variable that appears in the most active clauses
        var_counts = [0] * n
        for clause in active_clauses:
            for v, s in clause:
                var_counts[v] += 1
        
        max_count = 0
        selected_var = -1
        selected_sign = 1
        # Tie-breaker: random
        for v in range(n):
            if assignment[v] == -1 and var_counts[v] > max_count:
                max_count = var_counts[v]
                selected_var = v
                # Pick a sign, let's say +1 (True)
                selected_sign = 1
                # To balance, we could pick sign randomly or based on clause context
                # Let's pick the sign that appears more often in the clauses containing v
                pos_count = 0
                neg_count = 0
                for c_idx, s in var_to_clauses[v]:
                    # Check if this clause is active and contains v
                    # This is O(M) check.
                    # Optimization: Just pick random sign for now, or +1.
                    # Let's pick +1.
                    pass
                
        if selected_var == -1:
            # No unassigned variables? If active_clauses is not empty, this is a conflict?
            # No, if all vars assigned and active_clauses not empty, they must be unsatisfied.
            # But unit propagation should have caught empty clauses.
            # If we reach here with active clauses, they must have unassigned vars.
            # So selected_var should not be -1 unless active_clauses is empty.
            if not active_clauses:
                return True, assignment[:]
            else:
                return False, None

        # 6. Branching
        # Branch 1: Assign selected_var to True
        # Branch 2: Assign selected_var to False
        
        # We need to create copies of active_clauses for the two branches
        # Or, we can use a trail and undo.
        # Given the depth is small (<=50), copying lists is acceptable.
        
        # Branch 1: True
        assignment[selected_var] = 1
        new_clauses_1 = []
        conflict_1 = False
        for clause in active_clauses:
            new_clause = []
            satisfied = False
            for v, s in clause:
                if v == selected_var:
                    if s == 1: # +var, and var=True => Satisfied
                        satisfied = True
                        break
                    else: # -var, and var=True => False literal, remove
                        continue
                else:
                    new_clause.append((v, s))
            if not satisfied:
                if len(new_clause) == 0:
                    conflict_1 = True
                    break
                new_clauses_1.append(new_clause)
        
        if not conflict_1:
            success, sol = dpll(new_clauses_1)
            if success:
                return True, sol
        
        # Branch 2: False
        assignment[selected_var] = 0
        new_clauses_2 = []
        conflict_2 = False
        for clause in active_clauses:
            new_clause = []
            satisfied = False
            for v, s in clause:
                if v == selected_var:
                    if s == -1: # -var, and var=False => Satisfied
                        satisfied = True
                        break
                    else: # +var, and var=False => False literal, remove
                        continue
                else:
                    new_clause.append((v, s))
            if not satisfied:
                if len(new_clause) == 0:
                    conflict_2 = True
                    break
                new_clauses_2.append(new_clause)
        
        if not conflict_2:
            success, sol = dpll(new_clauses_2)
            if success:
                return True, sol
        
        # Backtrack
        assignment[selected_var] = -1
        return False, None

    # Run DPLL
    # Initial active clauses are all processed clauses
    # Note: We must ensure we don't modify the original processed_clauses
    # We pass a deep copy or structure that allows mutation safely.
    # Since we are creating new lists in dpll, we can pass the original list.
    
    success, sol = dpll(processed_clauses)
    
    if success:
        # Convert internal assignment (0/1) to boolean list
        return [bool(a) for a in sol]
    
    # Fallback: WalkSAT if DPLL fails (should be rare for planted instances)
    # This handles cases where DPLL might be too slow or if my implementation has edge cases
    # that WalkSAT can navigate.
    
    # Re-initialize for WalkSAT
    deadline = time.time() + 325.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Precompute for WalkSAT
    var_to_clauses_ws = [[] for _ in range(n)]
    clause_var_info = []
    var_sign_in_clause = []
    
    for c_idx, clause in enumerate(processed_clauses):
        info = []
        sign_map = {}
        for v, s in clause:
            info.append((v, s))
            sign_map[v] = s
            var_to_clauses_ws[v].append(c_idx)
        clause_var_info.append(info)
        var_sign_in_clause.append(sign_map)

    m = len(processed_clauses)
    
    while time.time() < deadline:
        assignment_ws = [rng.random() < 0.5 for _ in range(n)]
        
        clause_weights = [0] * m
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment_ws[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
        
        total_unsat = sum(1 for w in clause_weights if w == 0)
        
        if total_unsat == 0:
            return assignment_ws
        
        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        
        for step in range(100000):
            if time.time() >= deadline:
                break
            
            if total_unsat == 0:
                return assignment_ws
            
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment_ws[:]
            
            if total_unsat < min_unsat_seen:
                min_unsat_seen = total_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > 5000:
                break
            
            clause_idx = -1
            if total_unsat > 0:
                for _ in range(min(m, 200)):
                    probe = rng.randrange(m)
                    if clause_weights[probe] == 0:
                        clause_idx = probe
                        break
                if clause_idx == -1:
                    for i in range(m):
                        if clause_weights[i] == 0:
                            clause_idx = i
                            break
                    if clause_idx == -1:
                        return assignment_ws
            
            clause_info = clause_var_info[clause_idx]
            
            if rng.random() < 0.5:
                v, s = rng.choice(clause_info)
                var_idx = v
            else:
                best_gain = float('inf')
                best_var_idx = -1
                for v, s in clause_info:
                    var_idx = v
                    delta = 0
                    for c_idx in var_to_clauses_ws[var_idx]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_sign_in_clause[c_idx][var_idx]
                        is_sat = (assignment_ws[var_idx] == (c_sign == 1))
                        
                        if is_sat:
                            if old_w == 1:
                                delta += 1
                        else:
                            if old_w == 0:
                                delta -= 1
                    
                    if delta < best_gain:
                        best_gain = delta
                        best_var_idx = var_idx
                        if best_gain == -1:
                            break
                var_idx = best_var_idx
            
            assignment_ws[var_idx] = not assignment_ws[var_idx]
            
            for c_idx in var_to_clauses_ws[var_idx]:
                old_w = clause_weights[c_idx]
                c_sign = var_sign_in_clause[c_idx][var_idx]
                is_sat = (assignment_ws[var_idx] == (c_sign == 1))
                
                if is_sat:
                    clause_weights[c_idx] = old_w + 1
                else:
                    clause_weights[c_idx] = old_w - 1
                    
                if old_w == 0 and clause_weights[c_idx] > 0:
                    total_unsat -= 1
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    total_unsat += 1

    return best_assignment

# EVOLVE-BLOCK-END
