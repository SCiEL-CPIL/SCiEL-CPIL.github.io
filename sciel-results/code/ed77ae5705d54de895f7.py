import time
import random
import sys

def solve(instance):
    START = time.time()
    DEADLINE = START + 290.0

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # 1. Preprocessing: Remove tautologies and duplicates, simplify
    # Convert to 0-based
    processed_clauses = []
    for c in clauses:
        # Remove duplicates within clause
        lits = list(set(c))
        # Check for tautology (x and -x)
        is_taut = False
        for l in lits:
            if -l in lits:
                is_taut = True
                break
        if not is_taut:
            if len(lits) == 0:
                # Empty clause -> UNSAT
                return None # Or handle unsat
            processed_clauses.append(lits)
    
    # If any clause is empty, it's unsat. But task says satisfiable.
    # If processed_clauses is empty, any assignment works.
    if not processed_clauses:
        return [False] * n

    # Data structures for DPLL
    # watches[var] = list of (clause_idx, literal)
    watches = [[] for _ in range(n)]
    clause_watches = [None] * len(processed_clauses)
    
    # Initialize watches
    for i, c in enumerate(processed_clauses):
        if len(c) >= 2:
            clause_watches[i] = [c[0], c[1]]
            watches[abs(c[0]) - 1].append((i, c[0]))
            watches[abs(c[1]) - 1].append((i, c[1]))
        elif len(c) == 1:
            clause_watches[i] = [c[0]]
            watches[abs(c[0]) - 1].append((i, c[0]))
        else:
            # Empty clause
            return None

    # Assignment: 0 unassigned, 1 true, -1 false
    assignment = [0] * n
    trail = []
    level = [0] * n
    current_level = 0
    
    # VSIDS
    activity = [1.0] * n
    inc = 1.0
    decay = 0.95 # Standard decay
    
    # Queue for unit propagation
    queue = []

    def assign(var, val, lvl):
        """Assign var to val. Returns False if conflict."""
        if assignment[var] != 0:
            return assignment[var] == val
        
        assignment[var] = val
        trail.append(var)
        level[var] = lvl
        
        # Process watches for this variable
        # We need to iterate over a copy or handle removal carefully
        # Since we might add to watches, we iterate over the current list
        # But we also need to remove the satisfied/falsified watch?
        # Standard approach: When a literal is assigned, if it satisfies a clause, 
        # we can leave the watch. If it falsifies, we need a new watch.
        
        # To avoid modifying list while iterating, we can use a local list
        # or just iterate and mark for removal?
        # Actually, in DPLL, we usually don't remove watches on backtrack.
        # We just check if the watch is still "active" (clause not satisfied by other means).
        
        # For each watch on `var`:
        # If the literal `lit` is satisfied by `val`, we are done with this watch.
        # If `lit` is falsified by `val`, we need to find a new literal in the clause.
        
        to_remove = []
        for (ci, lit) in watches[var]:
            # Check if this watch is still relevant
            # If the clause is already satisfied by another literal, we can ignore this watch?
            # No, we must maintain the invariant: 2 watched literals per clause.
            
            # If lit is satisfied by current assignment, do nothing.
            if (lit > 0 and val == 1) or (lit < 0 and val == -1):
                continue
            
            # lit is falsified. Find new watch.
            cw = clause_watches[ci]
            if cw is None:
                continue
            
            # Find the other watched literal
            other_idx = 0 if cw[0] == lit else 1
            other_lit = cw[other_idx]
            ov = abs(other_lit) - 1
            
            # If other_lit is satisfied, we are fine.
            if (other_lit > 0 and assignment[ov] == 1) or (other_lit < 0 and assignment[ov] == -1):
                continue
            
            # Need to find a new literal in the clause
            c = processed_clauses[ci]
            new_lit = 0
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v] == 1) or (l < 0 and assignment[v] == -1):
                    continue
                new_lit = l
                break
            
            if new_lit == 0:
                # Conflict
                return False
            
            # Replace other_idx with new_lit
            cw[other_idx] = new_lit
            # Add new watch
            watches[abs(new_lit) - 1].append((ci, new_lit))
            # Mark old watch for removal? 
            # If we don't remove it, we will see it again in future assignments of `var`.
            # If `var` is unassigned and re-assigned, we might process it again.
            # It's safer to remove it.
            to_remove.append((ci, lit))
        
        # Remove stale watches
        for wr in to_remove:
            if wr in watches[var]:
                watches[var].remove(wr)
                
        return True

    def propagate():
        """Process unit clauses. Returns False if conflict."""
        while queue:
            lit = queue.pop()
            var = abs(lit) - 1
            val = 1 if lit > 0 else -1
            
            if assignment[var] != 0:
                if assignment[var] != val:
                    return False
                continue
            
            if not assign(var, val, current_level):
                return False
            
            # If assign triggered new unit clauses, they would be added to queue?
            # In the standard algorithm, `assign` doesn't add to queue directly.
            # The loop in `propagate` handles the queue.
            # But `assign` might falsify a literal, leading to a new unit clause?
            # No, `assign` handles the watch replacement. If a clause becomes unit,
            # the *other* watch is the only unassigned one. 
            # Wait, the standard algorithm pushes the *new* watch to the queue?
            # No, the standard algorithm is:
            # When a literal is assigned, for each watch on that var:
            #   If lit is falsified, find new lit.
            #   If no new lit, conflict.
            #   If new lit found, add (clause, new_lit) to watches[new_var].
            #   If the clause is now unit (only 1 unassigned lit left), 
            #   that lit is the one we just watched? No.
            #   Actually, if we replace a falsified lit with a new one, 
            #   and the clause has no other unassigned lits, it's a conflict.
            #   If it has exactly 1 unassigned lit, that lit is NOT necessarily the new watch.
            #   The new watch is just one of the unassigned lits.
            #   To detect unit clauses, we need to know if the clause size (unassigned) is 1.
            #   The watched literal scheme doesn't explicitly track size.
            #   However, if we find a `new_lit` and it's the *only* unassigned lit, 
            #   we should enqueue it.
            #   How do we know? We can check the clause.
            
            # Let's refine `assign` to enqueue unit clauses.
            
        return True

    # The above `assign`/`propagate` is getting complex to get right in a patch.
    # Alternative: Use a simpler, correct DPLL for N=50.
    # Since N is small, we can afford a slightly slower propagation if it's correct.
    
    # Let's use a **Recursive DPLL with Simple Unit Propagation**.
    # This is O(N*M) per node, but with good heuristics, it's fast for N=50.
    
    # Reset state for simple DPLL
    assignment_simple = [0] * n
    
    def count_unsat_simple(assign):
        cnt = 0
        for c in processed_clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (lit > 0 and assign[v] == 1) or (lit < 0 and assign[v] == -1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def unit_propagate_simple(assign):
        """Returns False if conflict, True otherwise. Modifies assign in place."""
        changed = True
        while changed:
            changed = False
            for c in processed_clauses:
                unassigned = []
                satisfied = False
                for lit in c:
                    v = abs(lit) - 1
                    if assign[v] == 0:
                        unassigned.append(lit)
                    elif (lit > 0 and assign[v] == 1) or (lit < 0 and assign[v] == -1):
                        satisfied = True
                        break
                
                if satisfied:
                    continue
                
                if len(unassigned) == 0:
                    return False # Conflict
                
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    v = abs(lit) - 1
                    val = 1 if lit > 0 else -1
                    if assign[v] != 0:
                        if assign[v] != val:
                            return False
                    else:
                        assign[v] = val
                        changed = True
                        
        return True

    def dpll_simple(assign, depth=0):
        if time.time() > DEADLINE:
            return None
            
        # Unit Propagation
        if not unit_propagate_simple(assign):
            return None
            
        # Check if all assigned
        all_assigned = True
        for v in range(n):
            if assign[v] == 0:
                all_assigned = False
                break
                
        if all_assigned:
            # Verify
            if count_unsat_simple(assign) == 0:
                return assign[:]
            return None
            
        # Select variable
        # Heuristic: Variable appearing in most unsatisfied clauses
        var_counts = [0] * n
        for c in processed_clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (lit > 0 and assign[v] == 1) or (lit < 0 and assign[v] == -1):
                    sat = True
                    break
            if not sat:
                # Clause is unsatisfied
                for lit in c:
                    v = abs(lit) - 1
                    if assign[v] == 0:
                        var_counts[v] += 1
                        
        best_var = -1
        best_count = -1
        for v in range(n):
            if assign[v] == 0 and var_counts[v] > best_count:
                best_count = var_counts[v]
                best_var = v
                
        if best_var == -1:
            return None
            
        # Branch True
        assign[best_var] = 1
        res = dpll_simple(assign, depth + 1)
        if res:
            return res
        assign[best_var] = 0
        
        # Branch False
        assign[best_var] = -1
        res = dpll_simple(assign, depth + 1)
        if res:
            return res
        assign[best_var] = 0
        
        return None

    # Run Simple DPLL
    result = dpll_simple(assignment_simple)
    
    if result:
        return [a == 1 for a in result]
        
    # Fallback: WalkSAT
    # Use the remaining time
    remaining_time = DEADLINE - time.time()
    if remaining_time < 5:
        # Not enough time for WalkSAT, return random
        return [random.random() < 0.5 for _ in range(n)]
        
    def walk_sat():
        rng = random.Random(42)
        current_assign = [rng.random() < 0.5 for _ in range(n)]
        
        def count_unsat_ws(assign):
            cnt = 0
            for c in processed_clauses:
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):
                        sat = True
                        break
                if not sat:
                    cnt += 1
            return cnt

        best_assign = current_assign[:]
        best_uc = count_unsat_ws(current_assign)
        
        # Estimate steps based on time
        # Assume 1ms per step (conservative for Python)
        max_steps = int(remaining_time * 1000 * 0.8)
        
        for step in range(max_steps):
            if time.time() > DEADLINE - 2: # Safety margin
                break
            if best_uc == 0:
                return best_assign
                
            # Pick random unsat clause
            unsat_clauses = []
            for i, c in enumerate(processed_clauses):
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    if (lit > 0 and current_assign[v]) or (lit < 0 and not current_assign[v]):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
                    
            if not unsat_clauses:
                return current_assign
                
            ci = rng.choice(unsat_clauses)
            c = processed_clauses[ci]
            
            # Pick variable to flip
            # Min Flips Heuristic
            min_flips = 100
            best_lit = c[0]
            for lit in c:
                v = abs(lit) - 1
                # Count occurrences
                flips = 0
                for cc in processed_clauses:
                    for ll in cc:
                        if abs(ll) - 1 == v:
                            flips += 1
                            break
                if flips < min_flips:
                    min_flips = flips
                    best_lit = lit
                    
            v = abs(best_lit) - 1
            current_assign[v] = not current_assign[v]
            
            uc = count_unsat_ws(current_assign)
            if uc < best_uc:
                best_uc = uc
                best_assign = current_assign[:]
            else:
                # Noise
                if rng.random() > 0.5:
                    current_assign[v] = not current_assign[v]
                    
        return best_assign

    result = walk_sat()
    return result
