"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import heapq
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Data structures
    assignment = [0] * (n + 1)
    level = [0] * (n + 1)
    trail = []
    trail_lim = []
    
    # VSIDS
    activity = [0.0] * (n + 1)
    var_inc = 1.0
    decay_factor = 0.95
    
    # Clauses
    all_clauses = [list(c) for c in clauses]
    num_clauses = len(all_clauses)
    
    watches = defaultdict(list)
    clause_watches = [None] * num_clauses
    clause_len = [0] * num_clauses
    
    for i, clause in enumerate(all_clauses):
        if not clause:
            return [False] * n
        if len(clause) == 1:
            clause_watches[i] = [clause[0], 0]
            clause_len[i] = 1
            watches[clause[0]].append(i)
        else:
            w1, w2 = clause[0], clause[1]
            clause_watches[i] = [w1, w2]
            clause_len[i] = len(clause)
            watches[w1].append(i)
            watches[w2].append(i)
            
    prop_queue = []
    
    def assign(var, val):
        assignment[var] = val
        level[var] = len(trail_lim)
        trail.append(var)
        lit = var if val == 1 else -var
        prop_queue.append(lit)
        
    def cancel_until(lvl):
        while trail and trail_lim and trail_lim[-1] > lvl:
            lim = trail_lim.pop()
            while len(trail) > lim:
                v = trail.pop()
                assignment[v] = 0
                level[v] = 0
                
    def propagate():
        while prop_queue:
            lit = prop_queue.pop()
            v = abs(lit)
            val = 1 if lit > 0 else -1
            
            if assignment[v] == val:
                continue
            if assignment[v] != 0:
                return None # Should not happen
                
            neg_lit = -lit
            watch_list = watches[neg_lit]
            
            i = 0
            while i < len(watch_list):
                c_idx = watch_list[i]
                w1, w2 = clause_watches[c_idx]
                
                if w1 == neg_lit:
                    other = w2
                    other_var = abs(other)
                    other_val = assignment[other_var]
                    
                    if other_val == (1 if other > 0 else -1):
                        watch_list[i] = watch_list[-1]
                        watch_list.pop()
                        continue
                    elif other_val != 0:
                        return c_idx
                    else:
                        clause_watches[c_idx][0] = lit
                        watches[lit].append(c_idx)
                        
                elif w2 == neg_lit:
                    other = w1
                    other_var = abs(other)
                    other_val = assignment[other_var]
                    
                    if other_val == (1 if other > 0 else -1):
                        watch_list[i] = watch_list[-1]
                        watch_list.pop()
                        continue
                    elif other_val != 0:
                        return c_idx
                    else:
                        clause_watches[c_idx][1] = lit
                        watches[lit].append(c_idx)
                else:
                    i += 1
                    continue
                
                # Do not increment i
            # End while i < len(watch_list)
            
        return None
        
    def select_var():
        best_var = -1
        best_act = -1.0
        for v in range(1, n+1):
            if assignment[v] == 0:
                if activity[v] > best_act:
                    best_act = activity[v]
                    best_var = v
        return best_var
        
    def bump_activity(v):
        activity[v] += var_inc
        
    # Main Loop
    # We use a stack of decisions to handle backtracking and flipping.
    # decision_stack: list of (var, val_tried, val_to_try_next)
    # Actually, simpler: 
    # We try a var with val 1. If conflict, backtrack and try val -1.
    # If conflict again, backtrack further.
    
    # We'll use a recursive approach internally? 
    # No, iterative with explicit stack.
    
    # Let's use a standard DPLL recursion.
    # def dpll():
    #    ...
    
    # Given the complexity, I will implement a simple recursive DPLL 
    # that relies on the efficient `propagate` and `select_var`.
    # To avoid state copying, I will use the trail and `cancel_until`.
    
    def dpll():
        # 1. Propagate
        conflict = propagate()
        if conflict is not None:
            return False
            
        # 2. Check if all assigned
        # If no unassigned vars, SAT
        # We can check if trail length == n? 
        # Or just try to select a var.
        
        v = select_var()
        if v == -1:
            return True # All assigned, no conflict -> SAT
            
        # 3. Branch
        # Try True
        trail_lim.append(len(trail))
        assign(v, 1)
        bump_activity(v)
        
        if dpll():
            return True
            
        # Backtrack
        cancel_until(len(trail_lim) - 1) # Wait, trail_lim was appended.
        # The level we want to return to is the one before this decision.
        # The decision was at level len(trail_lim)-1 (before append? no, after append it's the current level).
        # Let's be precise.
        
        # Before assign: level = L
        # trail_lim.append(L) -> trail_lim[-1] = L
        # assign(v, 1) -> level[v] = L
        # If dpll fails, we need to undo this decision.
        # We need to go back to level L-1? No, L is the index in trail_lim.
        # The previous level is trail_lim[-2] if exists, else 0.
        
        # Let's use the level index.
        # current_level = len(trail_lim) - 1
        # prev_level = trail_lim[current_level - 1] if current_level > 0 else 0
        
        # But we already appended to trail_lim.
        # So we can just pop it?
        # No, cancel_until handles the trail.
        
        # Let's redo the backtracking logic in the caller.
        
        # Try False
        trail_lim.append(len(trail))
        assign(v, -1)
        bump_activity(v)
        
        if dpll():
            return True
            
        # Backtrack
        # Undo both decisions?
        # No, we need to undo the current decision and return to the caller.
        
        # This recursive structure is tricky with manual backtracking.
        # Let's use a standard pattern:
        
        # return False
        
    # The above recursive structure is flawed because it doesn't properly restore state 
    # for the "Try False" branch if "Try True" modified the watches.
    # Watched literals are modified in place. 
    # When we backtrack, we undo assignments, but the watches might have been moved.
    # This is a known issue with non-CDCL DPLL using watched literals.
    # Usually, you need to "un-move" watches or restart.
    
    # Given the risk, I will use a simpler **Clause Simplification** approach 
    # which is safer but slightly slower, or just use the **Recursive DPLL with State Copying** 
    # but optimize the copying.
    
    # For n=50, copying lists of clauses (max 100-200 clauses) is cheap.
    # Let's go back to the original approach but with **VSIDS** and **Better Simplification**.
    
    # Original approach:
    # def dpll(assignment, clauses):
    
    # I will implement this cleanly.
    
    def dpll(assignment, clauses):
        # Unit Propagation and Simplification
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                simplified = []
                sat = False
                for lit in clause:
                    v = abs(lit)
                    val = assignment[v-1] # 0-based in assignment list? 
                    # Let's use 1-based assignment list for consistency with lit.
                    # assignment is 1-based.
                    val = assignment[v]
                    if val != 0:
                        if (val == 1 and lit > 0) or (val == -1 and lit < 0):
                            sat = True
                            break
                        else:
                            continue
                    else:
                        simplified.append(lit)
                
                if not sat:
                    if not simplified:
                        return False
                    if len(simplified) == 1:
                        # Unit clause
                        lit = simplified[0]
                        v = abs(lit)
                        val = 1 if lit > 0 else -1
                        if assignment[v] == 0:
                            assignment[v] = val
                            changed = True
                        elif assignment[v] != val:
                            return False
                    else:
                        new_clauses.append(simplified)
            clauses = new_clauses
            
        if not clauses:
            return True
            
        # Pure Literal Elimination
        var_polarities = defaultdict(set)
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                if assignment[v] == 0:
                    var_polarities[v].add(1 if lit > 0 else -1)
                    
        pure_vars = []
        for v, pols in var_polarities.items():
            if len(pols) == 1:
                pure_vars.append((v, list(pols)[0]))
                
        if pure_vars:
            for v, val in pure_vars:
                assignment[v] = val
                
            # Simplify again
            new_clauses = []
            for clause in clauses:
                simplified = []
                sat = False
                for lit in clause:
                    v = abs(lit)
                    val = assignment[v]
                    if val != 0:
                        if (val == 1 and lit > 0) or (val == -1 and lit < 0):
                            sat = True
                            break
                        else:
                            continue
                    else:
                        simplified.append(lit)
                if not sat:
                    if not simplified:
                        return False
                    new_clauses.append(simplified)
            clauses = new_clauses
            
            # Loop back to UP
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    simplified = []
                    sat = False
                    for lit in clause:
                        v = abs(lit)
                        val = assignment[v]
                        if val != 0:
                            if (val == 1 and lit > 0) or (val == -1 and lit < 0):
                                sat = True
                                break
                            else:
                                continue
                        else:
                            simplified.append(lit)
                    if not sat:
                        if not simplified:
                            return False
                        if len(simplified) == 1:
                            lit = simplified[0]
                            v = abs(lit)
                            val = 1 if lit > 0 else -1
                            if assignment[v] == 0:
                                assignment[v] = val
                                changed = True
                            elif assignment[v] != val:
                                return False
                        else:
                            new_clauses.append(simplified)
                clauses = new_clauses
                
        if not clauses:
            return True
            
        # Branching: VSIDS-like (Max Occurrence)
        var_counts = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                if assignment[v] == 0:
                    var_counts[v] += 1
                    
        if not var_counts:
            return True
            
        # Pick var with max count
        branch_var = max(var_counts, key=lambda v: var_counts[v])
        
        # Try True
        assignment[branch_var] = 1
        if dpll(assignment, clauses):
            return True
            
        # Try False
        assignment[branch_var] = -1
        if dpll(assignment, clauses):
            return True
            
        assignment[branch_var] = 0
        return False
        
    assignment = [0] * (n + 1)
    clauses_copy = [list(c) for c in clauses]
    
    if dpll(assignment, clauses_copy):
        return [assignment[i] == 1 for i in range(1, n+1)]
    else:
        return [False] * n

# EVOLVE-BLOCK-END
