"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s limit, use 295s to be safe
    start_time = time.time()
    deadline = start_time + 295.0

    # Pre-compute clause structure for speed
    # clause_vars: list of lists of 0-based variable indices
    # clause_signs: list of lists of 1 (positive) or -1 (negative)
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[1 if l > 0 else -1 for l in c] for c in clauses]
    
    # Build variable to clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)

    def is_satisfied(assign):
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            ok = False
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def count_unsat(assign):
        cnt = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            ok = False
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    ok = True
                    break
            if not ok:
                cnt += 1
        return cnt

    def polish(assign):
        """Deterministic local search: flip any variable that strictly reduces unsat count."""
        changed = True
        while changed:
            changed = False
            base = count_unsat(assign)
            for v in range(n):
                old_val = assign[v]
                assign[v] = not old_val
                c = count_unsat(assign)
                if c < base:
                    base = c
                    changed = True
                else:
                    assign[v] = old_val
        return assign

    def walk_sat(assign, rng, flip_budget):
        """Run WalkSAT with a smart heuristic; return (assign, solved_bool)."""
        for _ in range(flip_budget):
            if time.time() > deadline - 5.0:
                return assign, False
            
            # Find unsatisfied clauses
            unsat_idx = []
            for i in range(m):
                cv = clause_vars[i]
                cs = clause_signs[i]
                ok = False
                for j in range(3):
                    if assign[cv[j]] == (cs[j] > 0):
                        ok = True
                        break
                if not ok:
                    unsat_idx.append(i)
            
            if not unsat_idx:
                return assign, True
            
            # Pick a random unsatisfied clause
            ci = unsat_idx[rng.randrange(len(unsat_idx))]
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            # For each literal in this clause, estimate the benefit of flipping it
            # Benefit = number of other unsat clauses that would be satisfied
            best_score = -1
            best_j = -1
            for j in range(3):
                v = cv[j]
                s = cs[j]
                new_val = (s > 0)
                score = 0
                for ui in unsat_idx:
                    if ui == ci:
                        continue
                    uc = clause_vars[ui]
                    us = clause_signs[ui]
                    for k in range(3):
                        if uc[k] == v:
                            if new_val == (us[k] > 0):
                                score += 1
                            break
                # Add small random noise to break ties
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_j = j
            
            v = cv[best_j]
            assign[v] = (cs[best_j] > 0)
        
        return assign, is_satisfied(assign)

    # --- Main Search Loop ---
    rng = random.Random(0)
    best_assign = None
    min_unsat = float('inf')
    
    max_restarts = 100
    steps_per_run = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize assignment
        if best_assign is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb the best assignment slightly
            assignment = list(best_assign)
            # Flip a few random variables
            for _ in range(n // 5):
                idx = rng.randint(0, n - 1)
                assignment[idx] = not assignment[idx]
        
        # Run WalkSAT
        assignment, solved = walk_sat(assignment, rng, steps_per_run)
        
        if solved:
            return assignment
        
        # Track best
        current_unsat = count_unsat(assignment)
        if current_unsat < min_unsat:
            min_unsat = current_unsat
            best_assign = list(assignment)
            # Try to polish the best assignment found so far
            polished = polish(list(best_assign))
            pol_unsat = count_unsat(polished)
            if pol_unsat < min_unsat:
                min_unsat = pol_unsat
                best_assign = polished
            if min_unsat == 0:
                return best_assign

    # Final attempt: polish the best assignment one more time
    if best_assign is not None:
        final = polish(list(best_assign))
        if count_unsat(final) == 0:
            return final
        return final
    
    # Fallback
    return [False] * n
# EVOLVE-BLOCK-END
