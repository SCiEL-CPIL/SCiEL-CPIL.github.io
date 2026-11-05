"""
Hybrid 3-SAT solver: Fast incremental WalkSAT with Simulated Annealing and
clause-weighting (min-confl) for planted satisfiable instances.
"""
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

    # --- Preprocessing ---
    # var_to_clauses[var_idx] = list of clause indices containing var
    var_to_clauses = defaultdict(list)
    # clause_lits[ci] = list of (var_idx, is_positive)
    clause_lits = []
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(ci)
        clause_lits.append([(abs(lit) - 1, lit > 0) for lit in clause])

    # --- Core Helpers ---

    def is_clause_satisfied(ci, assignment):
        for var_idx, is_pos in clause_lits[ci]:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def compute_unsat_set(assignment):
        unsat = set()
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat.add(ci)
        return unsat

    def update_unsat_after_flip(var_idx, old_val, new_val, unsat_set, assignment):
        """
        Updates unsat_set in-place after flipping var_idx.
        Returns delta: change in number of unsatisfied clauses.
        """
        delta = 0
        for ci in var_to_clauses[var_idx]:
            was_unsat = ci in unsat_set
            # Check satisfaction of clause ci with the NEW assignment
            now_sat = False
            for v_idx, is_pos in clause_lits[ci]:
                if assignment[v_idx] == is_pos:
                    now_sat = True
                    break
            
            if now_sat:
                if was_unsat:
                    unsat_set.discard(ci)
                    delta -= 1
            else:
                if not was_unsat:
                    unsat_set.add(ci)
                    delta += 1
        return delta

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def min_confl_score(var_idx, target_val, assignment, unsat_set, current_clause_ci):
        """
        Calculate the number of currently satisfied clauses that would become
        unsatisfied if var_idx is flipped to target_val.
        """
        conflict = 0
        for other_ci in var_to_clauses[var_idx]:
            if other_ci == current_clause_ci:
                continue
            
            # If the clause is already unsatisfied, flipping this variable might fix it or keep it unsat.
            # We only care about breaking SATISFIED clauses (increasing unsat count).
            if other_ci in unsat_set:
                continue
            
            # Check if other_ci is currently satisfied (it is, by definition)
            # Check if flipping var_idx would break it.
            # It becomes unsatisfied if ALL literals are false.
            # Since it is currently satisfied, at least one literal is true.
            # If the literal for var_idx in other_ci is the ONLY true literal, flipping breaks it.
            
            other_sat_lit_exists = False
            other_sat_lit_is_var = False
            
            for v_idx, is_pos in clause_lits[other_ci]:
                if assignment[v_idx] == is_pos:
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

    def sa_search(assignment, unsat_set, rng, max_steps, T_start, T_end, cooling_rate):
        """
        Simulated Annealing search.
        Returns (assignment, unsat_count, unsat_set)
        """
        current_unsat = len(unsat_set)
        T = T_start
        
        if current_unsat == 0:
            return assignment, 0, unsat_set

        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_unsat_set = unsat_set.copy()
        
        steps_since_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return assignment, 0, unsat_set

            # Select a random unsatisfied clause
            if not unsat_set:
                break
            ci = rng.choice(list(unsat_set))
            clause = clause_lits[ci]
            
            # Select a variable to flip from the clause
            var_idx, target_val = rng.choice(clause)
            
            old_val = assignment[var_idx]
            new_val = not old_val
            
            # Apply flip temporarily to calculate delta
            assignment[var_idx] = new_val
            delta = update_unsat_after_flip(var_idx, old_val, new_val, unsat_set, assignment)
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
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                    best_unsat_set = unsat_set.copy()
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
                
                # If stuck for too long, break
                if steps_since_improve > 1000:
                    break

        return best_assignment, best_unsat, best_unsat_set

    def greedy_polish(assignment, unsat_set, rng, max_steps):
        """
        Greedy hill climbing using min-confl.
        Used for final polishing when close to solution.
        """
        current_unsat = len(unsat_set)
        if current_unsat == 0:
            return assignment, 0, unsat_set
        
        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_unsat_set = unsat_set.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_unsat == 0:
                return assignment, 0, unsat_set
            
            if not unsat_set:
                break
                
            ci = rng.choice(list(unsat_set))
            clause = clause_lits[ci]
            
            # Find the literal with minimum conflict
            best_var = None
            best_score = float('inf')
            best_lit_val = None
            
            for var_idx, target_val in clause:
                # Calculate conflict score
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
            delta = update_unsat_after_flip(best_var, old_val, new_val, unsat_set, assignment)
            new_unsat_count = current_unsat + delta
            
            if delta <= 0:
                current_unsat = new_unsat_count
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                    best_unsat_set = unsat_set.copy()
            else:
                # Revert if it worsens (greedy)
                assignment[best_var] = old_val
                # If no improvement possible with this heuristic, break to avoid infinite loop
                break

        return best_assignment, best_unsat, best_unsat_set

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
        unsat_set = compute_unsat_set(assignment)
        cur_unsat = len(unsat_set)
        
        if cur_unsat < best_global_unsat:
            best_global_unsat = cur_unsat
            best_global_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        rng = random.Random(attempt * 31337 + 42)
        
        # Phase 1: Simulated Annealing
        steps_sa = 20000 
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.9995 # Slow cooling for exploration
        
        if n > 40:
            steps_sa = 30000
            cooling_rate = 0.9997
            
        assignment, cur_unsat, unsat_set = sa_search(
            assignment, unsat_set, rng, 
            steps_sa, T_start, T_end, cooling_rate
        )
        
        if cur_unsat < best_global_unsat:
            best_global_unsat = cur_unsat
            best_global_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        # Phase 2: Greedy Polish
        if cur_unsat <= 10:
            polish_steps = 5000
            assignment, cur_unsat, unsat_set = greedy_polish(
                assignment, unsat_set, rng, polish_steps
            )
            
            if cur_unsat < best_global_unsat:
                best_global_unsat = cur_unsat
                best_global_assignment = assignment[:]
                
            if cur_unsat == 0:
                return assignment

    # Final fallback: If we have a good partial solution, try one last long polish
    if best_global_assignment is not None and best_global_unsat <= 5:
        rng = random.Random(99999)
        final_unsat_set = compute_unsat_set(best_global_assignment)
        final_assignment, final_unsat, _ = greedy_polish(best_global_assignment, final_unsat_set, rng, 20000)
        if final_unsat == 0:
            return final_assignment

    return best_global_assignment if best_global_assignment else make_random_assignment(0)

# EVOLVE-BLOCK-END
