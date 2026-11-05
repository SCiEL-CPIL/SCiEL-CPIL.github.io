"""Hybrid 3-SAT solver: Incremental WalkSAT with Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # Safety margin
    num_clauses = len(clauses)

    if num_clauses == 0:
        return [True] * n

    # Build variable-to-clauses index for fast incremental updates
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    # Precompute clause info for fast satisfaction check
    # clause_info[ci] = list of (var_idx, target_val)
    clause_info = []
    for clause in clauses:
        info = []
        for lit in clause:
            var_idx = abs(lit) - 1
            target_val = (lit > 0)
            info.append((var_idx, target_val))
        clause_info.append(info)

    def is_clause_satisfied(ci, assignment):
        for var_idx, target_val in clause_info[ci]:
            if assignment[var_idx] == target_val:
                return True
        return False

    def compute_unsat_list(assignment):
        unsat = []
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat.append(ci)
        return unsat

    def update_unsat_after_flip(var_idx, old_val, new_val, unsat_list, assignment):
        """Update unsat list after flipping variable var_idx from old_val to new_val.
        Returns new_unsat_list and delta (change in count)."""
        # Use a set for O(1) membership, but convert back to list at end for compatibility
        unsat_set = set(unsat_list)
        old_count = len(unsat_set)
        
        for ci in var_to_clauses[var_idx]:
            was_unsat = ci in unsat_set
            # Check if clause is now satisfied
            now_sat = False
            for v_idx, t_val in clause_info[ci]:
                if assignment[v_idx] == t_val:
                    now_sat = True
                    break
            
            if now_sat:
                if was_unsat:
                    unsat_set.discard(ci)
            else:
                if not was_unsat:
                    unsat_set.add(ci)
        
        new_count = len(unsat_set)
        return list(unsat_set), new_count - old_count

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def min_confl_score(var_idx, target_val, assignment, unsat_set, current_clause_ci):
        """
        Calculate the 'conflict' score for flipping var_idx to target_val.
        We want to minimize the number of currently satisfied clauses that would become unsatisfied.
        """
        conflict = 0
        for other_ci in var_to_clauses[var_idx]:
            if other_ci == current_clause_ci:
                continue
            
            # If the clause is already unsatisfied, flipping this variable might fix it or keep it unsat.
            # We only care about breaking SATISFIED clauses (increasing unsat count).
            if other_ci in unsat_set:
                continue
            
            # Check if other_ci is currently satisfied (it is, by definition of the 'else' branch above)
            # Check if flipping var_idx would break it.
            # A clause is satisfied if at least one literal is true.
            # It becomes unsatisfied if ALL literals are false.
            # Since it is currently satisfied, at least one literal is true.
            # If the literal for var_idx in other_ci is the ONLY true literal, flipping breaks it.
            
            other_sat_lit_exists = False
            other_sat_lit_is_var = False
            
            for v_idx, t_val in clause_info[other_ci]:
                if assignment[v_idx] == t_val:
                    other_sat_lit_exists = True
                    if v_idx == var_idx:
                        other_sat_lit_is_var = True
                    else:
                        # Another variable satisfies the clause, so flipping var_idx won't break it
                        other_sat_lit_is_var = False 
                        break
            
            if other_sat_lit_exists and other_sat_lit_is_var:
                # The only satisfied literal in this clause belongs to var_idx.
                # Flipping var_idx will make this clause unsatisfied.
                conflict += 1
        
        return conflict

    def sa_search(assignment, unsat_list, rng, max_steps, T_start, T_end, cooling_rate):
        """
        Simulated Annealing search.
        Returns (assignment, unsat_count, unsat_list)
        """
        current_unsat = len(unsat_list)
        T = T_start
        
        # If we start with 0 unsat, we are done
        if current_unsat == 0:
            return assignment, 0, unsat_list

        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_unsat_list = unsat_list[:]
        
        steps_since_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return assignment, 0, unsat_list

            # Select a random unsatisfied clause
            if not unsat_list:
                break
            ci = rng.choice(unsat_list)
            clause = clause_info[ci]
            
            # Select a variable to flip from the clause
            # We can use min-confl to bias selection, or pick randomly.
            # Let's pick randomly among the 3 literals for diversity, 
            # but we will evaluate the move.
            var_idx, target_val = rng.choice(clause)
            
            old_val = assignment[var_idx]
            new_val = not old_val
            
            # Apply flip temporarily to calculate delta
            assignment[var_idx] = new_val
            new_unsat_list, delta = update_unsat_after_flip(var_idx, old_val, new_val, unsat_list, assignment)
            new_unsat_count = current_unsat + delta
            
            # Acceptance criterion
            accepted = False
            if delta < 0:
                accepted = True
            elif delta == 0:
                accepted = True
            else:
                # Worsening move
                # P(accept) = exp(-delta / T)
                if T > 0:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accepted = True
                # else T is 0, only accept improvements
            
            if accepted:
                current_unsat = new_unsat_count
                unsat_list = new_unsat_list
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                    best_unsat_list = unsat_list[:]
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
                    
                # Cooling
                T *= cooling_rate
                if T < T_end:
                    T = T_end
            else:
                # Revert flip
                assignment[var_idx] = old_val
                steps_since_improve += 1
                T *= cooling_rate
                if T < T_end:
                    T = T_end
                
                # If stuck for too long, restart or break
                if steps_since_improve > 1000:
                    break

        return best_assignment, best_unsat, best_unsat_list

    def greedy_polish(assignment, unsat_list, rng, max_steps):
        """
        Greedy hill climbing using min-confl.
        Used for final polishing when close to solution.
        """
        current_unsat = len(unsat_list)
        if current_unsat == 0:
            return assignment, 0, unsat_list
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_unsat_list = unsat_list[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_unsat == 0:
                return assignment, 0, unsat_list
            
            if not unsat_list:
                break
                
            ci = rng.choice(unsat_list)
            clause = clause_info[ci]
            
            # Find the literal with minimum conflict
            best_var = None
            best_score = float('inf')
            best_lit_val = None
            
            for var_idx, target_val in clause:
                # Calculate conflict score
                # We need unsat_set for min_confl_score
                unsat_set = set(unsat_list)
                score = min_confl_score(var_idx, target_val, assignment, unsat_set, ci)
                
                # Add small random noise to break ties
                score += rng.random() * 0.01
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
                    best_lit_val = target_val
            
            if best_var is None:
                break
                
            old_val = assignment[best_var]
            new_val = not old_val
            
            assignment[best_var] = new_val
            new_unsat_list, delta = update_unsat_after_flip(best_var, old_val, new_val, unsat_list, assignment)
            new_unsat_count = current_unsat + delta
            
            if delta <= 0:
                current_unsat = new_unsat_count
                unsat_list = new_unsat_list
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                    best_unsat_list = unsat_list[:]
            else:
                # Revert if it worsens (greedy)
                assignment[best_var] = old_val
                # If no improvement possible with this heuristic, break to avoid infinite loop
                # In strict greedy, if we can't improve, we are stuck.
                # But we might have picked the "least bad" move. 
                # For polish, we only accept improvements.
                # If we can't improve, break.
                break

        return best_assignment, best_unsat, best_unsat_list

    best_global_assignment = None
    best_global_unsat = num_clauses + 1

    # Main loop: Multiple restarts
    # Strategy: 
    # 1. Random assignment
    # 2. SA phase to explore landscape
    # 3. Greedy polish to converge
    # 4. If stuck, restart
    
    max_restarts = 200
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = attempt * 7919 + 13
        assignment = make_random_assignment(seed)
        unsat_list = compute_unsat_list(assignment)
        cur_unsat = len(unsat_list)
        
        if cur_unsat < best_global_unsat:
            best_global_unsat = cur_unsat
            best_global_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        rng = random.Random(attempt * 31337 + 42)
        
        # Phase 1: Simulated Annealing
        # Parameters:
        # T_start: High enough to accept large worsening moves
        # T_end: Low enough to converge
        # Cooling: Exponential
        # Max steps: Depends on n and num_clauses
        
        # Heuristic for steps:
        steps_sa = 20000 
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.9995 # Slow cooling for exploration
        
        # Adjust steps based on problem size
        if n > 40:
            steps_sa = 30000
            cooling_rate = 0.9997
            
        assignment, cur_unsat, unsat_list = sa_search(
            assignment, unsat_list, rng, 
            steps_sa, T_start, T_end, cooling_rate
        )
        
        if cur_unsat < best_global_unsat:
            best_global_unsat = cur_unsat
            best_global_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        # Phase 2: Greedy Polish
        # If we are close to a solution, use greedy to finish
        if cur_unsat <= 10:
            polish_steps = 5000
            assignment, cur_unsat, unsat_list = greedy_polish(
                assignment, unsat_list, rng, polish_steps
            )
            
            if cur_unsat < best_global_unsat:
                best_global_unsat = cur_unsat
                best_global_assignment = assignment[:]
                
            if cur_unsat == 0:
                return assignment

        # If we are still stuck, we could try a "kick" (random flips) and restart SA,
        # but for simplicity, we just restart with a new random assignment.
        # The diversity of seeds handles this.

    # Final fallback: If we have a good partial solution, try one last long polish
    if best_global_assignment is not None and best_global_unsat <= 5:
        rng = random.Random(99999)
        final_unsat_list = compute_unsat_list(best_global_assignment)
        final_assignment, final_unsat, _ = greedy_polish(best_global_assignment, final_unsat_list, rng, 20000)
        if final_unsat == 0:
            return final_assignment

    return best_global_assignment if best_global_assignment else make_random_assignment(0)

# EVOLVE-BLOCK-END
