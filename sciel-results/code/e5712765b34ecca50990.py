"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280.0

    # Build watched literals
    # For each variable, we track which clauses are watching it.
    # A clause is "watched" by a literal if that literal is in the clause.
    # We maintain:
    #   watches[var] = list of (clause_idx, other_literal) for clauses where var is watched
    #   Actually, standard watched literal: each clause watches two literals.
    #   We'll use: for each variable, a list of clause indices that have this variable.
    #   And for each clause, which two variables it's watching.

    # Simpler approach: use a DPLL with unit propagation via counting.
    # For 50 vars and planted SAT, a simple DPLL with good heuristics should work.

    # Let's implement a proper DPLL with:
    # - Unit propagation
    # - Pure literal elimination
    # - VSIDS-like variable selection (or simpler: pick variable in most unsatisfied clauses)
    # - Backtracking

    # Actually, for planted 3-SAT with 50 vars, WalkSAT already solves all (score 1.0).
    # Let's try a hybrid: DPLL with limited backtracking, then fall back to WalkSAT.

    # Better: implement a clean DPLL with watched literals for efficiency.

    # Convert clauses to use 0-based indexing internally
    # Literals are 1-based signed; variable i is in [1, n]
    # We'll use: lit > 0 means var lit-1 is True, lit < 0 means var -lit-1 is False

    # State:
    #   assignment: list of None (unassigned), True, False for each variable
    #   For each clause, we track which literals are still "active" (not falsified)

    # Let's use a simpler but effective approach:
    # 1. Try DPLL with unit propagation and pure literal elimination, with a node limit
    # 2. If that fails, use WalkSAT with more iterations
    # 3. Return best found

    import random

    def try_dpll(clauses, n, max_nodes=50000):
        """DPLL with unit propagation and pure literal elimination."""
        # assignment: None, True, False
        assignment = [None] * n
        # For each clause, track which literals are still satisfiable (not assigned to false)
        # clause_lits[c] = set of literals in clause c that are not yet falsified
        # Actually, let's track: for each clause, the list of literals, and which are still "alive"
        
        # We'll represent state as:
        #   assignment: list of (None, True, False)
        #   For propagation, we need to know for each clause how many literals are still true-able
        
        # Simpler: maintain for each clause a count of unassigned literals, and if any is assigned true, clause is satisfied.
        
        # Let's use a different representation:
        #   active_clauses: clauses that are not yet satisfied
        #   For each active clause, track which literals are still unassigned
        
        # I'll implement a straightforward DPLL:
        
        def get_var(lit):
            return abs(lit) - 1
        
        def is_true(var, val):
            return assignment[var] == val
        
        def is_false(var, val):
            return assignment[var] is not None and assignment[var] != val
        
        def propagate():
            """Unit propagation. Returns (success, new_assignment_changes) or None if conflict."""
            changed = True
            while changed:
                changed = False
                for c_idx, clause in enumerate(clauses):
                    # Check if clause is already satisfied
                    sat = False
                    unassigned_count = 0
                    for lit in clause:
                        v = get_var(lit)
                        if assignment[v] is None:
                            unassigned_count += 1
                        elif (lit > 0 and assignment[v] == True) or (lit < 0 and assignment[v] == False):
                            sat = True
                            break
                        # else: literal is falsified
                    if sat:
                        continue
                    if unassigned_count == 0:
                        return False  # conflict
                    if unassigned_count == 1:
                        # Unit clause: assign the remaining literal
                        for lit in clause:
                            v = get_var(lit)
                            if assignment[v] is None:
                                assignment[v] = (lit > 0)
                                changed = True
                                break
            return True

        def pure_literal_elimination():
            """Eliminate pure literals. Returns True if any literal was pure."""
            changed = False
            for v in range(n):
                if assignment[v] is not None:
                    continue
                # Check if v appears only positively or only negatively in unsatisfied clauses
                pos = False
                neg = False
                for clause in clauses:
                    # Skip satisfied clauses
                    sat = False
                    for lit in clause:
                        vv = get_var(lit)
                        if assignment[vv] is not None:
                            if (lit > 0 and assignment[vv] == True) or (lit < 0 and assignment[vv] == False):
                                sat = True
                                break
                    if sat:
                        continue
                    for lit in clause:
                        if get_var(lit) == v:
                            if lit > 0:
                                pos = True
                            else:
                                neg = True
                if pos and not neg:
                    assignment[v] = True
                    changed = True
                elif neg and not pos:
                    assignment[v] = False
                    changed = True
            return changed

        # Main DPLL
        def dpll(node_count):
            # Check if all assigned
            for v in range(n):
                if assignment[v] is None:
                    break
            else:
                # All assigned, check if all clauses satisfied
                for clause in clauses:
                    for lit in clause:
                        v = get_var(lit)
                        if (lit > 0 and assignment[v] == True) or (lit < 0 and assignment[v] == False):
                            break
                    else:
                        return None  # Not satisfied
                return list(assignment)
            
            # Unit propagation
            if not propagate():
                return None
            
            # Pure literal elimination
            pure_literal_elimination()
            
            # Check if all assigned after propagation
            for v in range(n):
                if assignment[v] is None:
                    break
            else:
                for clause in clauses:
                    for lit in clause:
                        v = get_var(lit)
                        if (lit > 0 and assignment[v] == True) or (lit < 0 and assignment[v] == False):
                            break
                    else:
                        return None
                return list(assignment)
            
            # Pick variable: choose one that appears in most unsatisfied clauses
            var_counts = [0] * n
            for clause in clauses:
                sat = False
                for lit in clause:
                    v = get_var(lit)
                    if assignment[v] is not None:
                        if (lit > 0 and assignment[v] == True) or (lit < 0 and assignment[v] == False):
                            sat = True
                            break
                if sat:
                    continue
                seen = set()
                for lit in clause:
                    v = get_var(lit)
                    if assignment[v] is None and v not in seen:
                        var_counts[v] += 1
                        seen.add(v)
            
            # Pick variable with max count
            best_var = -1
            best_count = -1
            for v in range(n):
                if assignment[v] is None and var_counts[v] > best_count:
                    best_count = var_counts[v]
                    best_var = v
            
            if best_var == -1:
                return None
            
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return "TIMEOUT"
            
            # Try True first
            assignment[best_var] = True
            result = dpll(node_count)
            if result is not None and result != "TIMEOUT":
                return result
            if result == "TIMEOUT":
                assignment[best_var] = None
                return "TIMEOUT"
            
            # Try False
            assignment[best_var] = False
            result = dpll(node_count)
            if result is not None and result != "TIMEOUT":
                return result
            if result == "TIMEOUT":
                assignment[best_var] = None
                return "TIMEOUT"
            
            assignment[best_var] = None
            return None
        
        node_count = [0]
        result = dpll(node_count)
        return result

    # Try DPLL first
    result = try_dpll(clauses, n, max_nodes=100000)
    
    if result is not None and result != "TIMEOUT":
        return result
    
    # Fall back to WalkSAT
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    
    def count_unsat(clause):
        return sum(1 for l in clause if assignment[abs(l) - 1] != (l > 0))
    
    # WalkSAT with clause selection based on min unsat count
    max_steps = 50000
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        
        # Pick clause with minimum number of false literals
        best_clause = min(unsat, key=count_unsat)
        
        # Pick literal to flip: with probability 0.5, pick one that would satisfy the clause
        if rng.random() < 0.5:
            # Pick a literal in the clause that is currently false
            false_lits = [l for l in best_clause if assignment[abs(l) - 1] != (l > 0)]
            if false_lits:
                literal = rng.choice(false_lits)
            else:
                literal = rng.choice(best_clause)
        else:
            literal = rng.choice(best_clause)
        
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
    
    return assignment
# EVOLVE-BLOCK-END
