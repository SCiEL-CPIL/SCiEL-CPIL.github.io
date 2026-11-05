import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    start_time = time.time()
    time_limit = 250
    
    # Convert clauses to a format easier to manipulate
    # We will maintain a list of clauses. Each clause is a list of literals.
    # We need to track assignments.
    
    class DPLL:
        def __init__(self, n, clauses, limit_nodes=100000, time_limit=250):
            self.n = n
            self.initial_clauses = clauses
            self.limit_nodes = limit_nodes
            self.time_limit = time_limit
            self.start_time = time.time()
            self.nodes = 0
            self.aborted = False
            
        def check_time(self):
            if time.time() - self.start_time > self.time_limit:
                self.aborted = True
                return True
            return False

        def unit_propagate(self, clauses, assignment):
            # clauses: list of lists of literals
            # assignment: list of None, True, False
            changed = True
            while changed:
                changed = False
                if self.check_time():
                    return None
                new_clauses = []
                for c in clauses:
                    # Filter out satisfied literals
                    remaining = []
                    sat = False
                    for lit in c:
                        v = abs(lit) - 1
                        val = assignment[v]
                        if val is not None:
                            if (lit > 0 and val) or (lit < 0 and not val):
                                sat = True
                                break
                        else:
                            remaining.append(lit)
                    
                    if sat:
                        continue # Clause satisfied, discard
                    if not remaining:
                        return None # Contradiction
                    new_clauses.append(remaining)
                
                # Check for unit clauses
                for c in new_clauses:
                    if len(c) == 1:
                        lit = c[0]
                        v = abs(lit) - 1
                        val = lit > 0
                        if assignment[v] is None:
                            assignment[v] = val
                            changed = True
                        elif assignment[v] != val:
                            return None # Contradiction
                
                clauses = new_clauses
                # Note: We must re-evaluate all clauses if a new assignment is made
                # The loop above handles this by repeating if 'changed' is True
                
            return clauses

        def pure_literal_elimination(self, clauses, assignment):
            # Find variables that appear in only one polarity in 'clauses'
            # among unassigned variables
            if self.check_time():
                return None, False
            
            pos_vars = set()
            neg_vars = set()
            for c in clauses:
                for lit in c:
                    v = abs(lit) - 1
                    if assignment[v] is None:
                        if lit > 0:
                            pos_vars.add(v)
                        else:
                            neg_vars.add(v)
            
            changed = False
            # Pure positive: in pos_vars, not in neg_vars
            for v in pos_vars:
                if v not in neg_vars:
                    assignment[v] = True
                    changed = True
            # Pure negative: in neg_vars, not in pos_vars
            for v in neg_vars:
                if v not in pos_vars:
                    assignment[v] = False
                    changed = True
            
            return assignment, changed

        def select_variable(self, clauses, assignment):
            # Pick variable from shortest clause
            # Tie-break: highest degree
            if not clauses:
                return None
            
            # Find shortest clause
            min_len = min(len(c) for c in clauses)
            # Collect candidates from shortest clauses
            candidates = []
            for c in clauses:
                if len(c) == min_len:
                    for lit in c:
                        v = abs(lit) - 1
                        if assignment[v] is None:
                            candidates.append(v)
            
            if not candidates:
                return None
            
            # Count degrees in remaining clauses
            degrees = {v: 0 for v in candidates}
            for c in clauses:
                for lit in c:
                    v = abs(lit) - 1
                    if v in degrees:
                        degrees[v] += 1
            
            max_deg = max(degrees.values())
            best_vars = [v for v, d in degrees.items() if d == max_deg]
            return random.choice(best_vars)

        def solve(self):
            assignment = [None] * self.n
            clauses = [list(c) for c in self.initial_clauses]
            
            # Initial unit propagation and pure literal elimination
            clauses = self.unit_propagate(clauses, assignment)
            if clauses is None:
                return None
            if self.check_time():
                return None
                
            while True:
                if self.check_time():
                    return None
                self.nodes += 1
                if self.nodes > self.limit_nodes:
                    self.aborted = True
                    return None
                
                # Check if all satisfied
                if not clauses:
                    return assignment
                
                # Pure literal elimination
                assignment, changed = self.pure_literal_elimination(clauses, assignment)
                if self.check_time():
                    return None
                if changed:
                    clauses = self.unit_propagate(clauses, assignment)
                    if clauses is None:
                        return None
                    continue
                
                # Select variable to branch
                v = self.select_variable(clauses, assignment)
                if v is None:
                    # No unassigned variables in clauses? Should be satisfied or empty
                    return assignment if not clauses else None
                
                # Branch True
                assignment[v] = True
                clauses_true = self.unit_propagate(clauses, assignment)
                if clauses_true is not None:
                    # Recurse
                    res = self._solve_recursive(clauses_true, assignment)
                    if res is not None:
                        return res
                
                # Branch False
                assignment[v] = False
                clauses_false = self.unit_propagate(clauses, assignment)
                if clauses_false is not None:
                    res = self._solve_recursive(clauses_false, assignment)
                    if res is not None:
                        return res
                
                # Backtrack
                assignment[v] = None
                return None # Failure at this level

        def _solve_recursive(self, clauses, assignment):
            # This is a helper to avoid deep recursion issues or to manage state
            # Actually, standard DPLL is recursive. Let's make the main logic recursive.
            # The structure above is a bit mixed. Let's refactor to a clean recursive function.
            pass

    # Let's implement a cleaner recursive DPLL
    def dpll(clauses, assignment):
        if solver.check_time():
            return None
        solver.nodes += 1
        if solver.nodes > solver.limit_nodes:
            solver.aborted = True
            return None

        # Unit Propagation
        clauses, assignment = unit_prop(clauses, assignment)
        if clauses is None:
            return None
        if solver.check_time():
            return None

        # Check for empty clause set (all satisfied)
        if not clauses:
            return assignment

        # Pure Literal Elimination
        assignment, changed = pure_lit(clauses, assignment)
        if changed:
            return dpll(clauses, assignment)

        # Select variable
        v = select_var(clauses, assignment)
        if v is None:
            return None # Should not happen if clauses not empty

        # Branch True
        assignment[v] = True
        res = dpll(clauses, assignment)
        if res is not None:
            return res
        
        # Branch False
        assignment[v] = False
        res = dpll(clauses, assignment)
        if res is not None:
            return res
        
        # Backtrack
        assignment[v] = None
        return None

    # Implement helper functions for dpll
    def unit_prop(clauses, assignment):
        changed = True
        while changed:
            changed = False
            if solver.check_time():
                return None, None
            new_clauses = []
            for c in clauses:
                remaining = []
                sat = False
                for lit in c:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if val is not None:
                        if (lit > 0 and val) or (lit < 0 and not val):
                            sat = True
                            break
                    else:
                        remaining.append(lit)
                
                if sat:
                    continue
                if not remaining:
                    return None, None
                new_clauses.append(remaining)
            
            # Unit clauses
            for c in new_clauses:
                if len(c) == 1:
                    lit = c[0]
                    v = abs(lit) - 1
                    val = lit > 0
                    if assignment[v] is None:
                        assignment[v] = val
                        changed = True
                    elif assignment[v] != val:
                        return None, None
            
            clauses = new_clauses
        return clauses, assignment

    def pure_lit(clauses, assignment):
        if solver.check_time():
            return assignment, False
        pos = set()
        neg = set()
        for c in clauses:
            for lit in c:
                v = abs(lit) - 1
                if assignment[v] is None:
                    if lit > 0:
                        pos.add(v)
                    else:
                        neg.add(v)
        
        changed = False
        for v in pos:
            if v not in neg:
                assignment[v] = True
                changed = True
        for v in neg:
            if v not in pos:
                assignment[v] = False
                changed = True
        return assignment, changed

    def select_var(clauses, assignment):
        if not clauses:
            return None
        min_len = min(len(c) for c in clauses)
        cands = []
        for c in clauses:
            if len(c) == min_len:
                for lit in c:
                    v = abs(lit) - 1
                    if assignment[v] is None:
                        cands.append(v)
        if not cands:
            return None
        
        degs = {v: 0 for v in cands}
        for c in clauses:
            for lit in c:
                v = abs(lit) - 1
                if v in degs:
                    degs[v] += 1
        max_d = max(degs.values())
        best = [v for v, d in degs.items() if d == max_d]
        return random.choice(best)

    # Main execution
    solver = DPLL(n, clauses, limit_nodes=500000, time_limit=time_limit)
    assignment = [None] * n
    result = dpll([list(c) for c in clauses], assignment)
    
    if result is not None:
        # Verify
        for c in clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if result[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                return [False] * n # Fallback
        return result
    
    # Fallback: Random assignment
    return [random.random() < 0.5 for _ in range(n)]
