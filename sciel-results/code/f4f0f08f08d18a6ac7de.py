"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: evaluator kills at 330s, use 290s as safety margin
    deadline = time.time() + 290
    
    # Preprocess: convert to 0-based internal representation
    # clauses is list of lists of signed 1-based literals
    # We'll work with a list of clauses where each clause is a list of (var, sign)
    # var is 0-based, sign is +1 or -1
    
    # Convert clauses to internal format for speed
    internal_clauses = []
    for clause in clauses:
        internal_clause = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            internal_clause.append((var, sign))
        internal_clauses.append(internal_clause)
    
    # Build initial data structures for DPLL
    # assignment: list of None (unassigned), True, or False
    # We'll use a simple recursive DPLL with unit propagation and pure literal elimination
    
    def dpll(assignment, clauses):
        """
        assignment: list of length n with None, True, False
        clauses: list of lists of (var, sign)
        Returns: True if satisfiable, and updates assignment
        """
        # Check deadline
        if time.time() > deadline:
            return False
        
        # Unit propagation and pure literal elimination
        changed = True
        while changed:
            changed = False
            
            # Unit propagation: find clauses with exactly one unassigned literal
            new_clauses = []
            for clause in clauses:
                # Count unassigned literals and check for satisfied
                unassigned_lits = []
                satisfied = False
                for var, sign in clause:
                    val = assignment[var]
                    if val is not None:
                        if (val == True and sign == 1) or (val == False and sign == -1):
                            satisfied = True
                            break
                    else:
                        unassigned_lits.append((var, sign))
                
                if satisfied:
                    continue  # Clause is satisfied, remove it
                
                if len(unassigned_lits) == 0:
                    # Clause is unsatisfied
                    return False
                
                if len(unassigned_lits) == 1:
                    # Unit clause: force assignment
                    var, sign = unassigned_lits[0]
                    assignment[var] = (sign == 1)
                    changed = True
                else:
                    new_clauses.append(unassigned_lits)
            
            clauses = new_clauses
            
            if not clauses:
                # All clauses satisfied
                return True
            
            # Pure literal elimination
            # Find literals that appear in only one polarity
            pos_vars = set()
            neg_vars = set()
            for clause in clauses:
                for var, sign in clause:
                    if sign == 1:
                        pos_vars.add(var)
                    else:
                        neg_vars.add(var)
            
            pure_pos = pos_vars - neg_vars
            pure_neg = neg_vars - pos_vars
            
            if pure_pos or pure_neg:
                for var in pure_pos:
                    assignment[var] = True
                    changed = True
                for var in pure_neg:
                    assignment[var] = False
                    changed = True
                
                # Re-filter clauses
                new_clauses = []
                for clause in clauses:
                    unassigned_lits = []
                    satisfied = False
                    for var, sign in clause:
                        val = assignment[var]
                        if val is not None:
                            if (val == True and sign == 1) or (val == False and sign == -1):
                                satisfied = True
                                break
                        else:
                            unassigned_lits.append((var, sign))
                    
                    if satisfied:
                        continue
                    
                    if len(unassigned_lits) == 0:
                        return False
                    
                    if len(unassigned_lits) == 1:
                        var, sign = unassigned_lits[0]
                        assignment[var] = (sign == 1)
                        changed = True
                    else:
                        new_clauses.append(unassigned_lits)
                
                clauses = new_clauses
                
                if not clauses:
                    return True
        
        # No more unit propagation or pure literal elimination
        # Choose a variable to branch on
        # Use a simple heuristic: pick the variable appearing in the most clauses
        var_count = {}
        for clause in clauses:
            for var, sign in clause:
                var_count[var] = var_count.get(var, 0) + 1
        
        if not var_count:
            # All clauses satisfied
            return True
        
        # Pick the variable with highest frequency
        branch_var = max(var_count, key=lambda v: var_count[v])
        
        # Try True first
        save = assignment[branch_var]
        assignment[branch_var] = True
        
        # Filter clauses
        new_clauses = []
        for clause in clauses:
            unassigned_lits = []
            satisfied = False
            for var, sign in clause:
                val = assignment[var]
                if val is not None:
                    if (val == True and sign == 1) or (val == False and sign == -1):
                        satisfied = True
                        break
                else:
                    unassigned_lits.append((var, sign))
            
            if satisfied:
                continue
            
            if len(unassigned_lits) == 0:
                assignment[branch_var] = save
                return False
            
            if len(unassigned_lits) == 1:
                var, sign = unassigned_lits[0]
                assignment[var] = (sign == 1)
            else:
                new_clauses.append(unassigned_lits)
        
        if dpll(assignment, new_clauses):
            return True
        
        # Backtrack: try False
        assignment[branch_var] = False
        
        new_clauses = []
        for clause in clauses:
            unassigned_lits = []
            satisfied = False
            for var, sign in clause:
                val = assignment[var]
                if val is not None:
                    if (val == True and sign == 1) or (val == False and sign == -1):
                        satisfied = True
                        break
                else:
                    unassigned_lits.append((var, sign))
            
            if satisfied:
                continue
            
            if len(unassigned_lits) == 0:
                assignment[branch_var] = save
                return False
            
            if len(unassigned_lits) == 1:
                var, sign = unassigned_lits[0]
                assignment[var] = (sign == 1)
            else:
                new_clauses.append(unassigned_lits)
        
        if dpll(assignment, new_clauses):
            return True
        
        assignment[branch_var] = save
        return False
    
    # Start DPLL
    assignment = [None] * n
    if dpll(assignment, internal_clauses):
        # Convert None to False (shouldn't happen if satisfiable)
        return [bool(a) for a in assignment]
    
    # Fallback: WalkSAT if DPLL fails (shouldn't happen for satisfiable instances)
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    
    for _ in range(20000):
        if time.time() > deadline:
            break
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        literal = rng.choice(clause)
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
    
    return assignment

# EVOLVE-BLOCK-END
