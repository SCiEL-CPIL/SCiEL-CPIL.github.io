import time
import sys
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 290 seconds to be safe (evaluator kills at 330s, contract says 300s limit)
    deadline = time.time() + 290

    # Pre-process clauses to use 0-based internal indices
    # lit: 1-based signed. 
    # Internal: v in [0, n-1], sign in {+1, -1}
    internal_clauses = []
    for clause in clauses:
        int_clause = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            int_clause.append((v, s))
        internal_clauses.append(int_clause)
        
    m = len(internal_clauses)
    
    # Build variable-to-clause incidence for efficient impact calculation
    var_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(internal_clauses):
        for v, s in clause:
            var_clauses[v].append((c_idx, s))

    # Helper to check if a clause is satisfied given an assignment of booleans
    def is_satisfied(clause, assignment):
        for v, s in clause:
            val = assignment[v]
            if (val == True and s == 1) or (val == False and s == -1):
                return True
        return False

    # Helper to check if a clause is falsified (all literals false)
    def is_falsified(clause, assignment):
        for v, s in clause:
            val = assignment[v]
            # If val is assigned and does NOT satisfy the literal, it's false
            if (val == True and s == -1) or (val == False and s == 1):
                continue
            else:
                return False
        return True

    # Helper to get unassigned literals in a clause (for DPLL)
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
        changed = True
        while changed:
            changed = False
            # 1. Unit Propagation
            for c in clauses:
                unassigned = get_unassigned(c, assignment)
                if len(unassigned) == 0:
                    return assignment, True # Conflict
                if len(unassigned) == 1:
                    v, s = unassigned[0]
                    if assignment[v] is not None:
                        if (assignment[v] == True and s == 1) or (assignment[v] == False and s == -1):
                            continue
                        else:
                            return assignment, True # Conflict
                    assignment[v] = (s == 1)
                    changed = True
            
            # 2. Pure Literal Elimination
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
        for v, s in best_clause:
            if assignment[v] is None:
                return v
        return None

    # Recursive DPLL
    sys.setrecursionlimit(10000)
    
    def dpll(assignment, clauses):
        if time.time() > deadline:
            return None
            
        assignment, conflict = propagate(assignment, list(clauses))
        if conflict:
            return None
            
        all_sat = True
        for c in clauses:
            if not is_satisfied(c, assignment):
                if is_falsified(c, assignment):
                    return None
                all_sat = False
                break
        
        if all_sat:
            for i in range(len(assignment)):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment
            
        v = select_variable(assignment, clauses)
        if v is None:
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
            
        assignment[v] = None
        return None

    # Initial Assignment
    initial_assignment = [None] * n
    
    # Try DPLL first. For planted SAT, DPLL is often very fast.
    result = dpll(initial_assignment, internal_clauses)
    
    if result is not None:
        return result
        
    # Fallback: Improved WalkSAT with frequency-based heuristic
    rng = random.Random()
    
    def walksat(assignment, clauses, max_steps):
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment
            
            # Find unsatisfied clauses
            unsat = []
            for c in clauses:
                if not is_satisfied(c, assignment):
                    unsat.append(c)
            
            if not unsat:
                return assignment
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat)
            
            # Heuristic: pick variable that satisfies the clause (i.e., currently False in this clause)
            # Use frequency-based selection: prefer variables appearing in more unsat clauses
            candidates = []
            for v, s in clause:
                val = assignment[v]
                # If val is False and s is 1, flipping makes it True (satisfies)
                # If val is True and s is -1, flipping makes it False (satisfies)
                if (val == False and s == 1) or (val == True and s == -1):
                    candidates.append((v, s))
            
            if not candidates:
                # Should not happen if clause is unsatisfied, but just in case
                candidates = [(v, s) for v, s in clause]
            
            # Count frequency of each candidate variable in all unsat clauses
            if len(candidates) > 1:
                freq = {}
                for v, s in candidates:
                    freq[v] = 0
                for c in unsat:
                    for vv, ss in c:
                        if vv in freq:
                            freq[vv] += 1
                
                # Choose variable with max frequency, tie-break random
                max_freq = max(freq.values())
                top_candidates = [(v, s) for v, s in candidates if freq[v] == max_freq]
                v, s = rng.choice(top_candidates)
            else:
                v, s = candidates[0]
            
            assignment[v] = not assignment[v]
            
        return assignment

    # Try WalkSAT with multiple restarts
    # Use time-based budget
    while time.time() < deadline - 5:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        # Adaptive steps: start with fewer, increase if not solved
        steps = 5000 + int(5000 * (1 - (deadline - time.time()) / 285.0))
        steps = max(steps, 2000)
        result = walksat(assignment, internal_clauses, steps)
        if result:
            # Verify
            if all(is_satisfied(c, result) for c in internal_clauses):
                return result
                
    # If still no solution, return the last assignment
    return assignment
