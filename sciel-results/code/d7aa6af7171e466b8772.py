"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
sys.setrecursionlimit(10000)


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280  # 300s limit, 20s safety margin

    # Precompute: for each variable, list of (clause_idx, literal_value) where
    # literal_value is True if the clause contains +var, False if -var.
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occurrences[var].append((ci, lit > 0))

    # DPLL with watched literals and unit propagation
    def dpll_solve():
        # assignment[v] = -1 (unassigned), 0 (false), 1 (true)
        assignment = [-1] * n
        
        # For each clause, track two watched literals
        # watches[ci] = (lit1, lit2) - two literals from clause ci being watched
        # We use the standard watched literal scheme
        
        # Build initial watches: for each clause, pick first two literals
        watches = []
        for ci, clause in enumerate(clauses):
            if len(clause) >= 2:
                watches.append((clause[0], clause[1]))
            elif len(clause) == 1:
                watches.append((clause[0], None))
            else:
                watches.append((None, None))
        
        # Count satisfied clauses
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (1 if lit > 0 else 0):
                    sat_count[ci] += 1
        
        # Unit propagation queue
        from collections import deque
        up_queue = deque()
        
        # Check for initially unit or satisfied clauses
        for ci in range(m):
            if sat_count[ci] == 0:
                # All literals unsatisfied - check if unit
                clause = clauses[ci]
                unassigned = []
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] == -1:
                        unassigned.append(lit)
                if len(unassigned) == 1:
                    up_queue.append(unassigned[0])
        
        # Process unit propagation
        while up_queue:
            if time.time() > deadline:
                return None
            
            lit = up_queue.popleft()
            var = abs(lit) - 1
            want_val = 1 if lit > 0 else 0
            
            if assignment[var] != -1:
                # Already assigned
                if assignment[var] != want_val:
                    # Conflict
                    return None
                continue
            
            assignment[var] = want_val
            
            # Update watches and sat_counts
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == (want_val == 1):
                    # This literal is now satisfied
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        # Clause just became satisfied - no action needed for watches
                        pass
                else:
                    # This literal is now unsatisfied
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        # Clause is now all unsatisfied - check if unit
                        clause = clauses[oc]
                        unassigned = []
                        for l in clause:
                            v = abs(l) - 1
                            if assignment[v] == -1:
                                unassigned.append(l)
                        if len(unassigned) == 1:
                            up_queue.append(unassigned[0])
                        elif len(unassigned) == 0:
                            # Empty clause - conflict
                            return None
        
        # Check if all clauses are satisfied
        if all(sat_count[ci] > 0 for ci in range(m)):
            # Convert to boolean list
            result = []
            for v in range(n):
                if assignment[v] == -1:
                    result.append(random.random() < 0.5)
                else:
                    result.append(assignment[v] == 1)
            return result
        
        # If not all satisfied, we need to search
        # Find an unsatisfied clause
        unsat_clauses = [ci for ci in range(m) if sat_count[ci] == 0]
        if not unsat_clauses:
            # All satisfied but some vars unassigned
            result = []
            for v in range(n):
                if assignment[v] == -1:
                    result.append(random.random() < 0.5)
                else:
                    result.append(assignment[v] == 1)
            return result
        
        # Backtracking search on remaining unassigned variables
        ci = unsat_clauses[0]
        clause = clauses[ci]
        
        # Try each literal in the clause
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] != -1:
                continue
            
            want_val = 1 if lit > 0 else 0
            
            # Save state
            saved_assignment = assignment[:]
            saved_sat_count = sat_count[:]
            
            # Assign and propagate
            assignment[var] = want_val
            up_queue2 = deque()
            
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == (want_val == 1):
                    sat_count[oc] += 1
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        clause_oc = clauses[oc]
                        unassigned = []
                        for l in clause_oc:
                            v = abs(l) - 1
                            if assignment[v] == -1:
                                unassigned.append(l)
                        if len(unassigned) == 1:
                            up_queue2.append(unassigned[0])
                        elif len(unassigned) == 0:
                            # Conflict - restore and try next
                            assignment = saved_assignment
                            sat_count = saved_sat_count
                            break
            
            if sat_count[ci] > 0 or all(sat_count[oc] > 0 for oc in range(m)):
                # This assignment works or at least doesn't immediately conflict
                # Continue with full propagation
                while up_queue2:
                    if time.time() > deadline:
                        return None
                    
                    lit2 = up_queue2.popleft()
                    var2 = abs(lit2) - 1
                    want_val2 = 1 if lit2 > 0 else 0
                    
                    if assignment[var2] != -1:
                        if assignment[var2] != want_val2:
                            assignment = saved_assignment
                            sat_count = saved_sat_count
                            break
                        continue
                    
                    assignment[var2] = want_val2
                    
                    for (oc, lit_val) in var_occurrences[var2]:
                        if lit_val == (want_val2 == 1):
                            sat_count[oc] += 1
                        else:
                            sat_count[oc] -= 1
                            if sat_count[oc] == 0:
                                clause_oc = clauses[oc]
                                unassigned = []
                                for l in clause_oc:
                                    v = abs(l) - 1
                                    if assignment[v] == -1:
                                        unassigned.append(l)
                                if len(unassigned) == 1:
                                    up_queue2.append(unassigned[0])
                                elif len(unassigned) == 0:
                                    assignment = saved_assignment
                                    sat_count = saved_sat_count
                                    break
                    else:
                        continue
                    break
                else:
                    # No conflict
                    if all(sat_count[oc] > 0 for oc in range(m)):
                        result = []
                        for v in range(n):
                            if assignment[v] == -1:
                                result.append(random.random() < 0.5)
                            else:
                                result.append(assignment[v] == 1)
                        return result
        
        return None

    def random_walksat(seed):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        # If already satisfied
        if not unsat_set:
            return assignment

        # WalkSAT with clause-count tracking
        noise = 0.3
        max_steps = 500000

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_set:
                break

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]

            # Evaluate each literal in the clause
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val == want_val:
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        if sat_count[oc] == 1:
                            loss += 1
                score = gain - loss
                candidates.append((score, var, want_val))

            if rng.random() < noise:
                _, var, want_val = rng.choice(candidates)
            else:
                best = max(candidates, key=lambda x: x[0])
                var, want_val = best[1], best[2]

            old_val = assignment[var]
            if old_val == want_val:
                continue
            assignment[var] = want_val

            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == want_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                break

        return assignment

    # First try DPLL
    result = dpll_solve()
    if result is not None:
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result

    # Fall back to WalkSAT
    best_assignment = None
    best_unsat = float('inf')

    if result is not None:
        best_assignment = result
        best_unsat = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))

    for seed in range(50):
        if time.time() > deadline - 5:
            break
        result = random_walksat(seed)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if best_assignment is None or unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
# EVOLVE-BLOCK-END
