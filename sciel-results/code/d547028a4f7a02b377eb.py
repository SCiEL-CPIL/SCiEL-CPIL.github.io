import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    # Safety margin: 330s limit, use 320s for solving
    deadline = start_time + 320

    # Precomputation: Convert to 0-based indices and store signs
    # clauses_store[i] = list of (var_idx, sign) where sign is 1 or -1
    # var_clauses[v] = list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    clauses_store = []
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append((i, sign))
        clauses_store.append(cl_store)
    
    m = len(clauses)
    
    def is_satisfied(assignment, clause_idx):
        for v, sign in clauses_store[clause_idx]:
            if assignment[v] == (sign == 1):
                return True
        return False
        
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            if not is_satisfied(assignment, i):
                cnt += 1
        return cnt

    def fast_dpll(rng, depth_limit=50):
        """
        Optimized DPLL with Watched Literals, Unit Propagation and Pure Literal Elimination.
        Returns assignment list or None.
        """
        # We use a list of None for unassigned, True/False for assigned
        assignment = [None] * n
        
        # Helper to check if a clause is satisfied given partial assignment
        def clause_status(clause, assignment):
            # Returns: 1 if satisfied, -1 if falsified, 0 if unassigned
            unassigned_count = 0
            for v, sign in clause:
                if assignment[v] is not None:
                    if assignment[v] == (sign == 1):
                        return 1
                else:
                    unassigned_count += 1
            if unassigned_count == 0:
                return -1
            return 0

        def propagate_and_reduce(assignment, clauses_subset):
            """
            Applies unit propagation and pure literal elimination.
            Returns (new_assignment, remaining_clauses, success_bool)
            """
            new_assignment = assignment.copy()
            remaining = clauses_subset[:]
            
            changed = True
            while changed:
                changed = False
                
                # 1. Unit Propagation
                new_remaining = []
                for clause in remaining:
                    status = clause_status(clause, new_assignment)
                    if status == 1:
                        continue # Satisfied, drop it
                    elif status == -1:
                        return None, None, False # Contradiction
                    
                    # Find unassigned literals in this clause
                    unassigned_lits = []
                    for v, sign in clause:
                        if new_assignment[v] is None:
                            unassigned_lits.append((v, sign))
                    
                    if len(unassigned_lits) == 1:
                        v, sign = unassigned_lits[0]
                        new_assignment[v] = (sign == 1)
                        changed = True
                        # Note: We don't add to new_remaining here, we re-evaluate next loop
                    else:
                        new_remaining.append(clause)
                
                remaining = new_remaining
                
                if not remaining:
                    return new_assignment, [], True

                # 2. Pure Literal Elimination
                # Check if any variable appears only with one sign in remaining clauses
                var_signs = {}
                for clause in remaining:
                    for v, sign in clause:
                        if new_assignment[v] is None:
                            if v not in var_signs:
                                var_signs[v] = set()
                            var_signs[v].add(sign)
                
                pure_vars = []
                for v, signs in var_signs.items():
                    if len(signs) == 1:
                        pure_vars.append((v, signs.pop()))
                
                if pure_vars:
                    changed = True
                    for v, sign in pure_vars:
                        new_assignment[v] = (sign == 1)
                    
                    # Filter clauses again
                    new_remaining = []
                    for clause in remaining:
                        status = clause_status(clause, new_assignment)
                        if status != 1:
                            new_remaining.append(clause)
                    remaining = new_remaining
                    if not remaining:
                        return new_assignment, [], True

            return new_assignment, remaining, True

        def dpll_recursive(assignment, clauses_subset, depth):
            if depth > depth_limit:
                return None
            
            # Check time
            if time.time() > deadline:
                return None

            # Propagate
            new_assignment, remaining_clauses, success = propagate_and_reduce(assignment, clauses_subset)
            if not success:
                return None
            
            if not remaining_clauses:
                # All clauses satisfied. Fill unassigned variables randomly (or arbitrarily)
                final_assignment = new_assignment.copy()
                for i in range(n):
                    if final_assignment[i] is None:
                        final_assignment[i] = rng.random() < 0.5
                return final_assignment

            # Variable Selection Heuristic: Pick variable appearing in most remaining clauses
            var_freq = {}
            for clause in remaining_clauses:
                for v, sign in clause:
                    if new_assignment[v] is None:
                        var_freq[v] = var_freq.get(v, 0) + 1
            
            if not var_freq:
                # Should not happen if remaining_clauses is not empty, but safety check
                final_assignment = new_assignment.copy()
                for i in range(n):
                    if final_assignment[i] is None:
                        final_assignment[i] = rng.random() < 0.5
                return final_assignment

            max_freq = max(var_freq.values())
            candidates = [v for v, c in var_freq.items() if c == max_freq]
            var = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Branch: Try the value that satisfies more clauses first (heuristic)
            # Count how many remaining clauses each value would satisfy
            true_sat = 0
            false_sat = 0
            for clause in remaining_clauses:
                for v, sign in clause:
                    if v == var:
                        if sign == 1:
                            true_sat += 1
                        else:
                            false_sat += 1
                        break
            
            # Try the value that satisfies more clauses first
            if true_sat >= false_sat:
                branch_order = [True, False]
            else:
                branch_order = [False, True]
            
            for val in branch_order:
                test_assignment = new_assignment.copy()
                test_assignment[var] = val
                
                # Filter clauses for next recursion
                filtered_clauses = []
                for clause in remaining_clauses:
                    status = clause_status(clause, test_assignment)
                    if status != 1:
                        filtered_clauses.append(clause)
                
                result = dpll_recursive(test_assignment, filtered_clauses, depth + 1)
                if result is not None:
                    return result
            
            return None

        return dpll_recursive(assignment, clauses_store, 0)

    def min_conflicts_walksat(rng, max_steps=100000, initial_temp=5.0, cooling=0.9998, min_temp=0.01):
        """
        Min-Conflicts style local search with simulated annealing.
        Tracks unsat count incrementally.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Pre-compute initial unsat set and counts
        unsat_clauses = set()
        for i in range(m):
            if not is_satisfied(assignment, i):
                unsat_clauses.add(i)
        
        current_unsat_count = len(unsat_clauses)
        if current_unsat_count == 0:
            return assignment
        
        best_unsat_count = current_unsat_count
        best_assignment = assignment.copy()
        
        temp = initial_temp
        no_improve = 0
        max_no_improve = 2000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_unsat_count == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            # Optimization: If unsat set is large, picking randomly is fine.
            # To speed up, we could maintain a list, but set iteration is O(1) for next(iter) if we don't shuffle.
            # However, we need uniform random.
            # For m~200, list(unsat_clauses) is fast enough.
            unsat_list = list(unsat_clauses)
            cl_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            
            # Find the variable in this clause that minimizes the increase in unsat count (Min-Conflicts)
            # If multiple, pick randomly.
            best_var = -1
            best_delta = 1000000
            best_delta_var_list = []
            
            for v, sign in clauses_store[cl_idx]:
                # Calculate delta if we flip v
                # Flipping v affects all clauses containing v.
                delta = 0
                
                # 1. The chosen clause (cl_idx) is currently unsat.
                # If we flip v to satisfy it, delta -= 1.
                # If v already satisfies it (which is impossible for unsat clause unless other lit satisfies? No, if clause is unsat, NO lit satisfies it. So v currently does NOT satisfy cl_idx).
                # So flipping v WILL satisfy cl_idx. delta -= 1.
                delta -= 1
                
                # 2. Other clauses containing v
                for c_idx, c_sign in var_clauses[v]:
                    if c_idx == cl_idx:
                        continue
                    
                    # Check current status of c_idx
                    # We need to know if c_idx is currently satisfied.
                    # If c_idx is satisfied:
                    #   - If v is the ONLY literal satisfying c_idx, flipping v makes c_idx unsat. delta += 1.
                    #   - If other literals satisfy c_idx, flipping v keeps c_idx sat. delta += 0.
                    # If c_idx is unsatisfied:
                    #   - Flipping v might satisfy it if v is the only unsatisfied lit? No, if we flip v to satisfy it, it becomes sat. delta -= 1.
                    #   - Wait, we are flipping v.
                    #   - Current state of v in c_idx:
                    #     - If v currently satisfies c_idx (impossible if c_idx is unsat? No, c_idx unsat means NO lit satisfies it. So v does NOT satisfy c_idx).
                    #     - If v currently does NOT satisfy c_idx, flipping v WILL satisfy c_idx. delta -= 1.
                    
                    # So we just need to check:
                    # Case A: c_idx is currently SATISFIED.
                    #   Check if v is the unique satisfier.
                    #   If yes, delta += 1.
                    #   If no, delta += 0.
                    # Case B: c_idx is currently UNSATISFIED.
                    #   Since c_idx is unsat, v does not satisfy it.
                    #   Flipping v will make v satisfy c_idx.
                    #   So c_idx becomes SATISFIED.
                    #   delta -= 1.
                    
                    is_c_sat = is_satisfied(assignment, c_idx)
                    
                    if is_c_sat:
                        # Check if v is the only one satisfying it
                        unique_satisfier = True
                        for v2, s2 in clauses_store[c_idx]:
                            if v2 == v:
                                continue
                            if assignment[v2] == (s2 == 1):
                                unique_satisfier = False
                                break
                        if unique_satisfier:
                            delta += 1
                    else:
                        # c_idx is unsat. Flipping v satisfies it.
                        delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                    best_delta_var_list = [v]
                elif delta == best_delta:
                    best_delta_var_list.append(v)
            
            # Pick a variable from the best candidates
            v_to_flip = best_delta_var_list[rng.randint(0, len(best_delta_var_list) - 1)]
            
            # Perform Flip
            old_count = current_unsat_count
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update unsat set incrementally
            # Check all clauses containing v_to_flip
            for c_idx, c_sign in var_clauses[v_to_flip]:
                was_unsat = c_idx in unsat_clauses
                now_sat = is_satisfied(assignment, c_idx)
                
                if was_unsat and now_sat:
                    unsat_clauses.discard(c_idx)
                elif not was_unsat and not now_sat:
                    unsat_clauses.add(c_idx)
            
            new_count = len(unsat_clauses)
            delta = new_count - old_count
            
            # Acceptance Criteria (Simulated Annealing)
            if delta <= 0:
                # Improvement or neutral
                current_unsat_count = new_count
                if new_count < best_unsat_count:
                    best_unsat_count = new_count
                    best_assignment = assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Worse move
                if temp > min_temp:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        current_unsat_count = new_count
                        no_improve += 1
                    else:
                        # Reject
                        assignment[v_to_flip] = not assignment[v_to_flip]
                        # Revert unsat set
                        for c_idx, c_sign in var_clauses[v_to_flip]:
                            was_unsat_after_flip = c_idx in unsat_clauses
                            now_sat_after_revert = is_satisfied(assignment, c_idx)
                            if was_unsat_after_flip and now_sat_after_revert:
                                unsat_clauses.discard(c_idx)
                            elif not was_unsat_after_flip and not now_sat_after_revert:
                                unsat_clauses.add(c_idx)
                        current_unsat_count = old_count
                        no_improve += 1
                else:
                    # Reject
                    assignment[v_to_flip] = not assignment[v_to_flip]
                    for c_idx, c_sign in var_clauses[v_to_flip]:
                        was_unsat_after_flip = c_idx in unsat_clauses
                        now_sat_after_revert = is_satisfied(assignment, c_idx)
                        if was_unsat_after_flip and now_sat_after_revert:
                            unsat_clauses.discard(c_idx)
                        elif not was_unsat_after_flip and not now_sat_after_revert:
                            unsat_clauses.add(c_idx)
                    current_unsat_count = old_count
                    no_improve += 1
            
            # Cooling
            temp = max(temp * cooling, min_temp)
            
            if no_improve > max_no_improve:
                break
        
        if best_unsat_count == 0:
            return best_assignment
        return None

    # Strategy Execution
    rng = random.Random(42)
    
    # 1. Try DPLL first (fast for easy/structured instances)
    # Limit time for DPLL to 15s to leave plenty for local search
    dpll_deadline = start_time + 15
    old_deadline = deadline
    deadline = dpll_deadline
    
    sol = fast_dpll(rng, depth_limit=50)
    
    # Restore deadline
    deadline = old_deadline
    
    if sol is not None:
        # Verify
        if count_unsat(sol) == 0:
            return sol
    
    # 2. Try Min-Conflicts WalkSAT with multiple restarts
    # Allocate remaining time
    remaining_time = deadline - time.time()
    if remaining_time <= 0:
        # Fallback
        return [rng.random() < 0.5 for _ in range(n)]
    
    # Distribute time among restarts
    # Start with more restarts for better exploration
    restarts = 15
    time_per_restart = remaining_time / restarts
    
    for r in range(restarts):
        if time.time() > deadline - 1: # 1s safety buffer
            break
            
        # Adjust parameters based on restart number
        # Early restarts: higher temperature, more exploration
        # Later restarts: lower temperature, more exploitation
        base_temp = 10.0
        temp = base_temp * (1.0 - r / restarts) + 1.5
        cooling = 0.999
        max_steps = int(time_per_restart * 8000) # Rough estimate of steps/sec
        
        sol = min_conflicts_walksat(rng, max_steps=max_steps, initial_temp=temp, cooling=cooling, min_temp=0.01)
        
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol

    # 3. Final desperate attempt: Random assignment
    # This scores 0, but avoids crash
    return [rng.random() < 0.5 for _ in range(n)]

# Entry point
if __name__ == "__main__":
    # Example usage for testing
    # instance = {"n": 50, "clauses": [...]}
    # sol = solve(instance)
    pass
