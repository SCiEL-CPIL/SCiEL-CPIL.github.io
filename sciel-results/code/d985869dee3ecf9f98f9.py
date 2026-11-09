import random
import time
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0  # Safety margin
    
    # Convert clauses to (var, pos) tuples for DPLL
    # var: 0..n-1, pos: True/False
    def to_internal(clause):
        return [(abs(l) - 1, l > 0) for l in clause]
    
    internal_clauses = [to_internal(c) for c in clauses]
    
    # DPLL solver with unit propagation and pure literal elimination
    def dpll(clauses, assign):
        """
        clauses: list of lists of (var, pos)
        assign: list of -1 (unassigned), 0 (False), 1 (True)
        Returns: (solved, final_assign) where solved is True/False/None (timeout)
        """
        
        def propagate(clauses, assign):
            """Apply unit propagation. Returns (ok, new_clauses, new_assign, conflict)"""
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    sat = False
                    unassigned = []
                    for var, pos in clause:
                        if assign[var] == 1:
                            if pos:
                                sat = True
                                break
                        elif assign[var] == 0:
                            if not pos:
                                sat = True
                                break
                        else:
                            unassigned.append((var, pos))
                    if sat:
                        continue
                    if not unassigned:
                        return False, None, None, True
                    if len(unassigned) == 1:
                        var, pos = unassigned[0]
                        assign[var] = 1 if pos else 0
                        changed = True
                    else:
                        new_clauses.append(unassigned)
            return True, new_clauses, assign, False

        def pure_literal_elim(clauses, assign):
            """Apply pure literal elimination. Returns (new_clauses, new_assign) or (None, None) on conflict"""
            if not clauses:
                return clauses, assign
            
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] != -1:
                        continue
                    if pos:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1

            # Find pure literals
            pure_assignments = []
            for var in range(n):
                if assign[var] != -1:
                    continue
                if pos_count[var] > 0 and neg_count[var] == 0:
                    pure_assignments.append((var, 1))
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    pure_assignments.append((var, 0))

            if not pure_assignments:
                return clauses, assign

            # Apply pure literal assignments
            for var, val in pure_assignments:
                assign[var] = val

            # Simplify clauses
            new_clauses = []
            for clause in clauses:
                sat = False
                remaining = []
                for var, pos in clause:
                    if assign[var] == 1:
                        if pos:
                            sat = True
                            break
                    elif assign[var] == 0:
                        if not pos:
                            sat = True
                            break
                    else:
                        remaining.append((var, pos))
                if sat:
                    continue
                if not remaining:
                    return None, None
                new_clauses.append(remaining)
            return new_clauses, assign

        # Main DPLL loop
        while True:
            if time.time() > deadline:
                return None, None

            # Unit propagation
            ok, clauses, assign, conflict = propagate(clauses, assign)
            if conflict:
                return False, None
            if not clauses:
                return True, assign

            # Pure literal elimination
            clauses, assign = pure_literal_elim(clauses, assign)
            if clauses is None:
                return False, None
            if not clauses:
                return True, assign

            # If all variables assigned, check satisfaction
            unassigned_vars = [v for v in range(n) if assign[v] == -1]
            if not unassigned_vars:
                # Verify all clauses satisfied
                for clause in clauses:
                    sat = False
                    for var, pos in clause:
                        if assign[var] == (1 if pos else 0):
                            sat = True
                            break
                    if not sat:
                        return False, None
                return True, assign

            # Select variable: most frequent in remaining clauses
            var_freq = {}
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] == -1:
                        var_freq[var] = var_freq.get(var, 0) + 1

            if not var_freq:
                return True, assign

            # Pick variable with highest frequency
            var = max(var_freq, key=lambda v: var_freq[v])

            # Try True first, then False
            for val in [1, 0]:
                if time.time() > deadline:
                    return None, None

                new_assign = assign[:]
                new_assign[var] = val
                
                # Simplify clauses with this assignment
                new_clauses = []
                conflict_found = False
                for clause in clauses:
                    sat = False
                    remaining = []
                    for v, pos in clause:
                        if v == var:
                            if (val == 1 and pos) or (val == 0 and not pos):
                                sat = True
                                break
                        elif new_assign[v] == -1:
                            remaining.append((v, pos))
                        else:
                            if (new_assign[v] == 1 and pos) or (new_assign[v] == 0 and not pos):
                                sat = True
                                break
                    if sat:
                        continue
                    if not remaining:
                        conflict_found = True
                        break
                    new_clauses.append(remaining)

                if conflict_found:
                    continue

                result, new_assign2 = dpll(new_clauses, new_assign)
                if result is True:
                    return True, new_assign2
                elif result is None:
                    return None, None
                # result is False, try other value

            return False, None

    # Run DPLL
    initial_assign = [-1] * n
    success, dpll_assign = dpll(internal_clauses, initial_assign)
    
    if success and dpll_assign is not None:
        result = []
        for v in range(n):
            if dpll_assign[v] == -1:
                result.append(random.random() < 0.5)
            else:
                result.append(dpll_assign[v] == 1)
        return result

    # Fallback: WalkSAT with improved efficiency
    rng = random.Random(42)
    
    # Build index for fast lookup
    # For each variable, store which clauses contain it and in what polarity
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for l in clause:
            v = abs(l) - 1
            var_to_clauses[v].append((i, l > 0))
    
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def is_satisfied(clause_idx):
        for l in clauses[clause_idx]:
            v = abs(l) - 1
            if assignment[v] == (l > 0):
                return True
        return False
    
    def count_unsat_in_clause(clause_idx):
        count = 0
        for l in clauses[clause_idx]:
            v = abs(l) - 1
            if assignment[v] != (l > 0):
                count += 1
        return count
    
    def get_unsatisfied_clauses():
        return [i for i in range(len(clauses)) if not is_satisfied(i)]
    
    max_restarts = 50
    flips_per_restart = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        if restart > 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]

        for _ in range(flips_per_restart):
            if time.time() > deadline - 5:
                return assignment

            unsat_clauses = get_unsatisfied_clauses()
            if not unsat_clauses:
                return assignment

            # Pick clause with fewest unsatisfied literals
            best_clause = min(unsat_clauses, key=count_unsat_in_clause)

            # Pick literal to flip: minimize number of newly unsatisfied clauses
            best_var = -1
            best_new_unsat = float('inf')
            for l in clauses[best_clause]:
                var = abs(l) - 1
                # Count how many currently unsatisfied clauses would become satisfied
                # and how many satisfied clauses would become unsatisfied
                would_satisfy = 0
                would_break = 0
                for ci, pos in var_to_clauses[var]:
                    if is_satisfied(ci):
                        # This clause is currently satisfied
                        # Check if flipping var would make it unsatisfied
                        # The clause is satisfied by some literal. If that literal is this var,
                        # flipping might break it. But we need to check if var is the only
                        # satisfying literal.
                        # Simpler: just count if this var appears in the clause
                        # and if flipping would make all literals false
                        all_false_after_flip = True
                        for ll in clauses[ci]:
                            vv = abs(ll) - 1
                            new_val = (not assignment[vv]) if vv == var else assignment[vv]
                            if new_val == (ll > 0):
                                all_false_after_flip = False
                                break
                        if all_false_after_flip:
                            would_break += 1
                    else:
                        # Clause is currently unsatisfied
                        # Check if flipping var would satisfy it
                        # The clause becomes satisfied if the literal with this var becomes true
                        # and all other literals can be anything
                        # Actually, a clause is satisfied if ANY literal is true
                        # So if we flip var to make this literal true, the clause is satisfied
                        if (l > 0 and not assignment[var]) or (l < 0 and assignment[var]):
                            # After flip, this literal will be true
                            would_satisfy += 1
                
                net_change = would_break - would_satisfy
                if net_change < best_new_unsat:
                    best_new_unsat = net_change
                    best_var = var

            if best_var != -1:
                assignment[best_var] = not assignment[best_var]

    return assignment
