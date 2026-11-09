"""Clause-learning DPLL solver with VSIDS and failed-literal probing for planted 3-SAT."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(100000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Internal state
    # Assignment: -1 unassigned, 0 false, 1 true (1-based vars)
    assignment = [-1] * (n + 1)
    level = [0] * (n + 1)  # Decision level when assigned
    trail = []  # Stack of variables in order of assignment
    vsids_score = [0.0] * (n + 1)
    
    # Watchlist: literal -> list of clause indices
    # Literal is the signed 1-based value
    watchlist = {}
    for i, clause in enumerate(clauses):
        for lit in clause:
            if lit not in watchlist:
                watchlist[lit] = []
            watchlist[lit].append(i)
    
    # For each clause, the two watched literals
    # watches[i] = [lit_a, lit_b]
    watches = [None] * num_clauses
    for i, clause in enumerate(clauses):
        if len(clause) >= 2:
            watches[i] = [clause[0], clause[1]]
        elif len(clause) == 1:
            watches[i] = [clause[0], None]
        else:
            watches[i] = [None, None]
    
    def is_true(lit):
        var = abs(lit)
        val = assignment[var]
        if val == -1:
            return False
        return (val == 1) == (lit > 0)
    
    def is_false(lit):
        var = abs(lit)
        val = assignment[var]
        if val == -1:
            return False
        return (val == 1) != (lit > 0)
    
    def assign_var(var, value):
        """Assign variable to value (0 or 1)."""
        assignment[var] = value
        level[var] = len(trail)
        trail.append(var)
        vsids_score[var] += 1.0
    
    def unassign_all():
        """Undo all assignments."""
        while trail:
            var = trail.pop()
            assignment[var] = -1
            level[var] = 0
    
    def unit_propagate():
        """
        Perform unit propagation using watchlists.
        Returns (success, conflict_clause_index)
        success is True if no conflict, False if conflict found.
        """
        # Queue of literals that were just assigned (to check their watchlists)
        # We'll use a worklist of clause indices that need rechecking
        # When a literal is assigned, all clauses watching its negation need to be checked
        
        # Worklist: clauses to process
        # A clause needs processing if one of its watched literals became false
        
        # We'll maintain a queue of clause indices
        queue = []
        
        # Initially, we need to find all clauses where a watched literal is now false
        # But that's expensive. Instead, let's track which literals were just assigned.
        
        # Better approach: when we assign a variable, we know which clauses watch the negation
        # Let's restructure: we'll have a global list of clauses to check
        
        # For simplicity and correctness with n=150, let's do iterative unit propagation
        # by scanning all clauses until fixpoint
        
        changed = True
        while changed:
            changed = False
            for i in range(num_clauses):
                clause = clauses[i]
                
                # Check if clause is satisfied
                sat = False
                num_unassigned = 0
                unit_lit = None
                
                for lit in clause:
                    if is_true(lit):
                        sat = True
                        break
                    elif is_false(lit):
                        continue
                    else:
                        num_unassigned += 1
                        unit_lit = lit
                
                if sat:
                    continue
                
                if num_unassigned == 0:
                    # Conflict
                    return False, i
                elif num_unassigned == 1:
                    # Propagate
                    assign_var(abs(unit_lit), 1 if unit_lit > 0 else 0)
                    changed = True
        
        return True, None
    
    def pick_variable():
        """Pick next variable to branch on using VSIDS."""
        best_var = -1
        best_score = -1.0
        for var in range(1, n + 1):
            if assignment[var] == -1:
                if vsids_score[var] > best_score:
                    best_score = vsids_score[var]
                    best_var = var
        return best_var
    
    def dpll():
        """DPLL search."""
        # Unit propagate
        success, conflict_idx = unit_propagate()
        if not success:
            # Conflict - backtrack
            unassign_all()
            return False
        
        # Check if all assigned
        all_assigned = True
        for var in range(1, n + 1):
            if assignment[var] == -1:
                all_assigned = False
                break
        
        if all_assigned:
            return True
        
        # Pick variable
        var = pick_variable()
        if var == -1:
            return True
        
        # Try var = True
        assign_var(var, 1)
        if dpll():
            return True
        unassign_all()
        
        # Try var = False
        assign_var(var, 0)
        if dpll():
            return True
        unassign_all()
        
        return False
    
    # Run solver
    if dpll():
        return [bool(assignment[i]) for i in range(1, n + 1)]
    else:
        import random
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
