import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 290 seconds to be safe
    deadline = time.time() + 290
    
    # Pre-process clauses to use 0-based internal indices
    # lit: 1-based signed. 
    # Internal: v in [0, n-1], sign in {+1, -1}
    # We'll store clauses as lists of (var, sign)
    internal_clauses = []
    for clause in clauses:
        int_clause = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            int_clause.append((v, s))
        internal_clauses.append(int_clause)
        
    m = len(internal_clauses)
    
    # Helper to check if a clause is satisfied
    def is_satisfied(clause, assignment):
        for v, s in clause:
            val = assignment[v]
            if val is not None:
                if (val == True and s == 1) or (val == False and s == -1):
                    return True
        return False
        
    # Helper to check if a clause is falsified (all literals false)
    def is_falsified(clause, assignment):
        for v, s in clause:
            val = assignment[v]
            if val is None:
                return False
            # If val is assigned and does NOT satisfy the literal, it's false
            if (val == True and s == -1) or (val == False and s == 1):
                continue
            else:
                return False
        return True

    # Helper to get unassigned literals in a clause
    def get_unassigned(clause, assignment):
        unassigned = []
        for v, s in clause:
            if assignment[v] is None:
                unassigned.append((v, s))
        return unassigned

    # Unit Propagation and Pure Literal Elimination
    # Returns: (new_assignment, conflict_bool)
    # Modifies assignment in place
    def propagate(assignment, clauses):
        # We use a loop to handle cascading effects
        changed = True
        while changed:
            changed = False
            # 1. Unit Propagation
            # Find clauses with exactly 1 unassigned literal
            for c in clauses:
                unassigned = get_unassigned(c, assignment)
                if len(unassigned) == 0:
                    return assignment, True # Conflict
                if len(unassigned) == 1:
                    v, s = unassigned[0]
                    if assignment[v] is not None:
                        # Already assigned, check if it satisfies
                        if (assignment[v] == True and s == 1) or (assignment[v] == False and s == -1):
                            continue
                        else:
                            return assignment, True # Conflict
                    # Assign
                    assignment[v] = (s == 1)
                    changed = True
            
            # 2. Pure Literal Elimination
            # Find variables that appear only with one polarity in unassigned literals
            # This is O(n * m) which is fine for N=50
            for v in range(len(assignment)):
                if assignment[v] is not None:
                    continue
                pos = False
                neg = False
                for c in clauses:
                    for vv, s in c:
                        if vv == v and assignment[vv] is None:
                            if s == 1:
                                pos = True
                            else:
                                neg = True
                if pos and not neg:
                    assignment[v] = True
                    changed = True
                elif neg and not pos:
                    assignment[v] = False
                    changed = True
                # If both pos and neg, it's not pure
                # If neither, it doesn't appear in any unassigned clause (shouldn't happen if clause not sat/falsified)
        
        return assignment, False

    # Variable Selection Heuristic
    # Pick variable from the shortest unsatisfied clause
    def select_variable(assignment, clauses):
        best_clause = None
        best_len = float('inf')
        for c in clauses:
            if is_satisfied(c, assignment):
                continue
            unassigned = get_unassigned(c, assignment)
            if len(unassigned) < best_len:
                best_len = len(unassigned)
                best_clause = c
                if best_len == 1:
                    break
        if best_clause is None:
            return None
        # Pick the first unassigned variable in the best clause
        for v, s in best_clause:
            if assignment[v] is None:
                return v
        return None

    # Recursive DPLL
    sys.setrecursionlimit(10000)
    
    def dpll(assignment, clauses):
        if time.time() > deadline:
            return None
            
        # Propagate
        assignment, conflict = propagate(assignment, list(clauses))
        if conflict:
            return None
            
        # Check if all satisfied
        all_sat = True
        for c in clauses:
            if not is_satisfied(c, assignment):
                # Check if falsified? If not, we need to branch
                if is_falsified(c, assignment):
                    return None
                all_sat = False
                break
        
        if all_sat:
            # Ensure all variables are assigned for the return format?
            # The task says "list of booleans". Usually implies length n.
            # If some are None, we can assign them arbitrarily (e.g., False)
            # But wait, if all clauses are satisfied, the unassigned variables don't matter.
            # However, the output must be a list of booleans.
            for i in range(len(assignment)):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment
            
        # Select variable
        v = select_variable(assignment, clauses)
        if v is None:
            # No unassigned variables in unsatisfied clauses?
            # This implies we are stuck or satisfied.
            # If not satisfied, it's a conflict?
            # Actually, if select_variable returns None, it means no unsatisfied clause has unassigned vars.
            # This means all unsatisfied clauses are falsified.
            return None

        # Branch 1: v = True
        assignment[v] = True
        result = dpll(assignment, clauses)
        if result is not None:
            return result
        
        # Branch 2: v = False
        assignment[v] = False
        result = dpll(assignment, clauses)
        if result is not None:
            return result
            
        # Backtrack
        assignment[v] = None
        return None

    # Initial Assignment
    initial_assignment = [None] * n
    
    # Try DPLL
    result = dpll(initial_assignment, internal_clauses)
    
    if result is not None:
        return result
        
    # Fallback: WalkSAT if DPLL fails or times out
    # Use random restarts
    import random
    rng = random.Random()
    
    def walksat(assignment, clauses, max_steps):
        for _ in range(max_steps):
            if time.time() > deadline:
                return assignment
            unsat = []
            for c in clauses:
                if not is_satisfied(c, assignment):
                    unsat.append(c)
            if not unsat:
                return assignment
            clause = rng.choice(unsat)
            # Pick variable to flip
            # Heuristic: pick var that satisfies most other unsat clauses? 
            # Simple: random var from clause
            v, s = rng.choice(clause)
            assignment[v] = not assignment[v]
        return assignment

    # Try a few WalkSAT restarts
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        assignment = [rng.random() < 0.5 for _ in range(n)]
        result = walksat(assignment, internal_clauses, 10000)
        if result:
            # Verify
            if all(is_satisfied(c, result) for c in internal_clauses):
                return result
                
    # If still no solution, return the last assignment (will likely fail verification in evaluator, but better than crash)
    return assignment
